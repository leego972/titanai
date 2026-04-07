"""
TitanAI Corpus Rebalancer
==========================
Adds more content to underweight buckets C (Technical) and D (Cyber).
Also trims Bucket E to a representative subset to restore balance.

Strategy:
  C: Add more Python code from The Stack + technical Wikipedia articles
  D: Add more security-focused content: OWASP, security Wikipedia (already loaded),
     more NVD CVE pages, CWE descriptions
  E: The 45K CMU summaries are short (~200 words each) — keep them but note
     the pipeline's weighted sampler will enforce the 10% ratio at shard time.
     No trimming needed — the pipeline handles it.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "rebalance.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("rebalancer")

INVENTORY     = BASE / "data" / "source_inventory.json"
EXCLUSION_LOG = BASE / "data" / "exclusions.jsonl"

try:
    with open(INVENTORY) as f:
        inventory = json.load(f)
except Exception:
    inventory = {}

exclusions = []


def record_source(bucket, source_name, source_url, size_mb, reason, n_docs):
    inventory.setdefault(bucket, []).append({
        "source": source_name, "url": source_url,
        "size_mb": round(size_mb, 2), "n_documents": n_docs,
        "reason_for_inclusion": reason,
        "loaded_at": datetime.utcnow().isoformat()
    })


def record_exclusion(source_name, reason):
    exclusions.append({"source": source_name, "reason": reason,
                        "timestamp": datetime.utcnow().isoformat()})
    log.warning(f"EXCLUDED: {source_name} — {reason}")


def write_docs(bucket_dir: Path, docs: list, source_tag: str):
    bucket_dir.mkdir(parents=True, exist_ok=True)
    existing = list(bucket_dir.glob(f"{source_tag}_*.txt"))
    start_idx = len(existing)
    written = 0
    for i, text in enumerate(docs):
        if not text or len(text.strip()) < 100:
            continue
        fname = bucket_dir / f"{source_tag}_{start_idx + i:06d}.txt"
        fname.write_text(text.strip(), encoding="utf-8")
        written += 1
    log.info(f"  Wrote {written} documents to {bucket_dir.name}/")
    return written


def approx_mb(docs):
    return sum(len(d.encode("utf-8")) for d in docs) / 1_048_576


# ── Bucket C supplement: more Python code + technical Wikipedia ───────────────

def supplement_bucket_c():
    log.info("=== Supplementing Bucket C: Technical ===")
    bucket = RAW / "corpus_C_technical"
    total = 0

    # C-supp-1: More Python from The Stack (remove docstring filter to get more volume)
    try:
        from datasets import load_dataset
        log.info("  Loading more Python from The Stack (relaxed filter)...")
        stack = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                             split="train", streaming=True, trust_remote_code=False)
        docs = []
        for i, item in enumerate(stack):
            if i >= 50000:
                break
            content = item.get("content", "").strip()
            # Relaxed: accept any file with at least one function/class definition
            if ("def " in content or "class " in content):
                if 200 < len(content) < 15000:
                    docs.append(content)
            if len(docs) >= 15000:
                break
        n = write_docs(bucket, docs, "stack_py2")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "The Stack (Python, relaxed filter)",
                      "https://huggingface.co/datasets/bigcode/the-stack-dedup",
                      mb, "Python code with function/class definitions", n)
        total += n
        log.info(f"  Stack Python (relaxed): {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Stack Python relaxed failed: {e}")
        record_exclusion("The Stack Python relaxed", str(e))

    # C-supp-2: Technical Wikipedia articles
    try:
        from datasets import load_dataset
        log.info("  Loading technical Wikipedia articles...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        tech_keywords = {
            "algorithm", "data structure", "operating system", "compiler",
            "database", "computer network", "software engineering",
            "machine learning", "artificial intelligence", "distributed system",
            "cloud computing", "microservices", "kubernetes", "docker",
            "linux kernel", "tcp/ip", "http", "rest api", "sql", "nosql",
            "programming language", "software architecture", "devops",
            "continuous integration", "version control", "git", "agile",
            "object-oriented", "functional programming", "concurrency",
            "parallel computing", "memory management", "cpu architecture"
        }
        docs = []
        scanned = 0
        for item in wiki:
            scanned += 1
            if scanned > 500000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in tech_keywords):
                if len(text) > 300:
                    docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= 3000:
                break
        n = write_docs(bucket, docs, "wiki_tech")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "Wikipedia Technical Articles",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      mb, "CS, software engineering, systems, networking", n)
        total += n
        log.info(f"  Wikipedia Technical: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia Technical failed: {e}")
        record_exclusion("Wikipedia Technical", str(e))

    log.info(f"  Bucket C supplement total: {total} new documents")
    return total


# ── Bucket D supplement: more cyber content ───────────────────────────────────

def supplement_bucket_d():
    log.info("=== Supplementing Bucket D: Cybersecurity ===")
    bucket = RAW / "corpus_D_cyber"
    total = 0

    # D-supp-1: CWE (Common Weakness Enumeration) descriptions from MITRE
    try:
        import requests
        log.info("  Loading CWE descriptions from MITRE...")
        # CWE XML feed
        url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
        import zipfile, io
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        xml_name = [n for n in z.namelist() if n.endswith(".xml")][0]
        xml_content = z.read(xml_name).decode("utf-8", errors="replace")

        # Simple XML parsing for CWE descriptions
        import re
        weaknesses = re.findall(
            r'<Weakness[^>]*ID="(\d+)"[^>]*Name="([^"]+)"[^>]*>.*?'
            r'<Description>(.*?)</Description>',
            xml_content, re.DOTALL
        )
        docs = []
        for cwe_id, name, desc in weaknesses:
            # Clean XML tags
            desc_clean = re.sub(r'<[^>]+>', ' ', desc).strip()
            desc_clean = re.sub(r'\s+', ' ', desc_clean)
            if len(desc_clean) > 80:
                text = f"CWE-{cwe_id}: {name}\n\nDescription:\n{desc_clean}"
                docs.append(text)

        n = write_docs(bucket, docs, "cwe")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "MITRE CWE (Common Weakness Enumeration)",
                      "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
                      mb, "Software weakness taxonomy for secure coding and remediation", n)
        total += n
        log.info(f"  CWE: {n} entries, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  CWE load failed: {e}")
        record_exclusion("MITRE CWE", str(e))

    # D-supp-2: More NVD CVE (2022 and 2021)
    try:
        import requests
        log.info("  Loading additional NVD CVE (2022)...")
        docs = []
        headers = {"User-Agent": "TitanAI-Corpus-Loader/1.0"}
        start_index = 0
        results_per_page = 2000
        total_fetched = 0
        max_cves = 6000

        while total_fetched < max_cves:
            # Use pubStartDate filter for 2022
            url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
                   f"?resultsPerPage={results_per_page}&startIndex={start_index}"
                   f"&pubStartDate=2022-01-01T00:00:00.000&pubEndDate=2022-12-31T23:59:59.999")
            try:
                resp = requests.get(url, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.error(f"  NVD 2022 API error: {e}")
                break

            vulnerabilities = data.get("vulnerabilities", [])
            if not vulnerabilities:
                break

            for v in vulnerabilities:
                cve = v.get("cve", {})
                cve_id = cve.get("id", "")
                descriptions = cve.get("descriptions", [])
                desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
                if desc and len(desc) > 80:
                    docs.append(f"CVE ID: {cve_id}\n\nDescription:\n{desc}")

            total_fetched += len(vulnerabilities)
            start_index += results_per_page
            log.info(f"  NVD 2022: fetched {total_fetched} CVEs...")

            if len(vulnerabilities) < results_per_page or total_fetched >= max_cves:
                break
            time.sleep(0.6)

        n = write_docs(bucket, docs, "nvd_cve_2022")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "NVD CVE Descriptions (2022)",
                      "https://nvd.nist.gov/developers/vulnerabilities",
                      mb, "Vulnerability descriptions 2022 — defensive context only", n)
        total += n
        log.info(f"  NVD CVE 2022: {n} entries, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  NVD CVE 2022 failed: {e}")
        record_exclusion("NVD CVE 2022", str(e))

    # D-supp-3: Security-focused dataset from HuggingFace
    try:
        from datasets import load_dataset
        log.info("  Loading security papers/abstracts dataset...")
        # Use a security-focused Q&A or paper dataset
        sec_ds = load_dataset("mrm8488/bert-tiny-finetuned-squadv2",
                              split="train", streaming=True)
        # This won't work — use a real security dataset
        raise ValueError("placeholder — use real source below")
    except Exception:
        pass

    # D-supp-3 (real): SecQA security Q&A dataset
    try:
        from datasets import load_dataset
        log.info("  Loading SecQA security Q&A dataset...")
        secqa = load_dataset("zefang-liu/secqa", split="train",
                             streaming=True, trust_remote_code=False)
        docs = []
        for i, item in enumerate(secqa):
            if i >= 5000:
                break
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a and len(q) > 20:
                docs.append(f"Security Question: {q}\n\nAnswer: {a}")
        n = write_docs(bucket, docs, "secqa")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "SecQA Security Q&A",
                      "https://huggingface.co/datasets/zefang-liu/secqa",
                      mb, "Security knowledge Q&A — defensive focus", n)
        total += n
        log.info(f"  SecQA: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  SecQA failed: {e}")
        record_exclusion("SecQA", str(e))

    log.info(f"  Bucket D supplement total: {total} new documents")
    return total


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("TitanAI Corpus Rebalancer Starting")

    c_added = supplement_bucket_c()
    d_added = supplement_bucket_d()

    # Save updated inventory
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)

    with open(EXCLUSION_LOG, "a") as f:
        for exc in exclusions:
            f.write(json.dumps(exc) + "\n")

    log.info("\n=== FINAL DISK USAGE BY BUCKET ===")
    for bucket_name in ["corpus_A_general", "corpus_B_reasoning",
                         "corpus_C_technical", "corpus_D_cyber", "corpus_E_cinema"]:
        bucket_path = RAW / bucket_name
        if bucket_path.exists():
            files = list(bucket_path.glob("*.txt"))
            total_bytes = sum(f.stat().st_size for f in files)
            log.info(f"  {bucket_name}: {len(files)} files, {total_bytes/1_048_576:.1f} MB")

    log.info(f"Rebalancing complete. Added C:{c_added}, D:{d_added} documents.")


if __name__ == "__main__":
    main()
