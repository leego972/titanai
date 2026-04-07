"""
TitanAI Corpus Rebalancer — v2 (Pre-Flight Fix)
================================================
Fixes the severe bucket imbalance identified in the staging audit:
  - Bucket C (Technical): raw docs only 2% of corpus → must reach ≥15% raw volume
  - Bucket D (Cyber):     raw docs only 5.7% of corpus → must reach ≥15% raw volume
  - Bucket E (Cinema):    raw docs 31.9% of corpus → hard-capped at CINEMA_MAX_DOCS

Strategy:
  C: Add Python code (The Stack), technical Wikipedia, StackOverflow Q&A
  D: Add CWE descriptions, NVD CVE (2022), SecQA security Q&A, security Wikipedia
  E: TRIM to a representative hard cap by archiving excess Cinema files

This script MUST be run before generate_shards.py on any serious training run.
After running, validate with: python scripts/validate_corpus_quality.py

OVERSAMPLING GUARD:
  The sharding pipeline can oversample underweight buckets, but >20x oversampling
  causes memorization. This script enforces minimum raw doc volume so that
  no bucket requires >10x oversampling to hit its token target.

  Required minimum raw docs per bucket (for 100M token target):
    A: ~6,000 docs   (35M tokens)
    B: ~7,000 docs   (15M tokens)
    C: ~15,000 docs  (20M tokens) ← was 5,529 — CRITICAL FIX
    D: ~15,000 docs  (20M tokens) ← was 11,398 — CRITICAL FIX
    E: ≤10,000 docs  (10M tokens) ← was 45,304 — TRIM REQUIRED
"""

import os
import sys
import json
import time
import random
import shutil
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

# ── Target minimum raw document counts ────────────────────────────────────────
# These ensure no bucket requires >10x oversampling in generate_shards.py
BUCKET_MIN_DOCS = {
    "corpus_A_general":   6_000,
    "corpus_B_reasoning": 7_000,
    "corpus_C_technical": 15_000,   # FIX: was ~5,500
    "corpus_D_cyber":     15_000,   # FIX: was ~11,400
    "corpus_E_cinema":    8_000,    # FIX: was ~45,300 (TRIM)
}

# Hard cap for Cinema to prevent overrepresentation
CINEMA_MAX_DOCS = 10_000


def count_bucket_docs(bucket_name: str) -> int:
    bucket_dir = RAW / bucket_name
    if not bucket_dir.exists():
        return 0
    return len(list(bucket_dir.glob("*.txt")))


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


# ── FIX 1: Trim Bucket E (Cinema) to hard cap ───────────────────────────────────

def trim_bucket_e():
    """
    CRITICAL FIX: Cinema bucket has 45,304 docs vs target 10% of corpus.
    Hard-cap at CINEMA_MAX_DOCS by archiving excess files (not deleting).
    """
    log.info("=== FIX: Trimming Bucket E (Cinema) to hard cap ===")
    bucket = RAW / "corpus_E_cinema"
    if not bucket.exists():
        log.warning("  Bucket E does not exist — skipping trim")
        return 0

    all_files = list(bucket.glob("*.txt"))
    current_count = len(all_files)
    log.info(f"  Current Cinema docs: {current_count}")
    log.info(f"  Hard cap: {CINEMA_MAX_DOCS}")

    if current_count <= CINEMA_MAX_DOCS:
        log.info(f"  Cinema is within cap ({current_count} ≤ {CINEMA_MAX_DOCS}) — no trim needed")
        return 0

    archive_dir = RAW / "corpus_E_cinema_archived"
    archive_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(all_files)
    keep_files = set(f.name for f in all_files[:CINEMA_MAX_DOCS])
    excess_files = [f for f in all_files if f.name not in keep_files]

    archived = 0
    for f in excess_files:
        dest = archive_dir / f.name
        shutil.move(str(f), str(dest))
        archived += 1

    remaining = len(list(bucket.glob("*.txt")))
    log.info(f"  Archived {archived} excess Cinema files to corpus_E_cinema_archived/")
    log.info(f"  Cinema docs remaining: {remaining}")
    return archived


# ── FIX 2: Supplement Bucket C (Technical) ─────────────────────────────────────

def supplement_bucket_c():
    """
    Add Python code and technical content to reach BUCKET_MIN_DOCS['corpus_C_technical'].
    """
    log.info("=== Supplementing Bucket C: Technical ===")
    bucket = RAW / "corpus_C_technical"
    current = count_bucket_docs("corpus_C_technical")
    needed  = max(0, BUCKET_MIN_DOCS["corpus_C_technical"] - current)
    log.info(f"  Current: {current} docs | Target min: {BUCKET_MIN_DOCS['corpus_C_technical']} | Need: {needed}")

    if needed == 0:
        log.info("  Bucket C already at target — skipping")
        return 0

    total = 0

    # C-1: Python from The Stack
    try:
        from datasets import load_dataset
        log.info("  Loading Python from The Stack...")
        stack = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                             split="train", streaming=True, trust_remote_code=False)
        docs = []
        for i, item in enumerate(stack):
            if i >= 100_000:
                break
            content = item.get("content", "").strip()
            if ("def " in content or "class " in content) and len(content) >= 200:
                docs.append(content)
            if len(docs) >= min(8_000, needed):
                break
        n = write_docs(bucket, docs, "stack_py")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "The Stack (Python)",
                      "https://huggingface.co/datasets/bigcode/the-stack-dedup",
                      mb, "Python code files with function/class definitions", n)
        total += n
        log.info(f"  Stack Python: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Stack Python failed: {e}")
        record_exclusion("The Stack Python", str(e))

    # C-2: Technical Wikipedia articles
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
            "parallel computing", "memory management", "cpu architecture",
            "cryptography", "encryption", "hash function", "digital signature",
        }
        docs = []
        scanned = 0
        still_needed = max(0, BUCKET_MIN_DOCS["corpus_C_technical"] - count_bucket_docs("corpus_C_technical") - total)
        for item in wiki:
            scanned += 1
            if scanned > 600_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in tech_keywords) and len(text) > 300:
                docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= min(5_000, still_needed):
                break
        n = write_docs(bucket, docs, "wiki_tech")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "Wikipedia Technical Articles",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      mb, "CS, software engineering, systems, networking articles", n)
        total += n
        log.info(f"  Wikipedia Technical: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia Technical failed: {e}")
        record_exclusion("Wikipedia Technical", str(e))

    # C-3: StackOverflow Q&A
    try:
        from datasets import load_dataset
        log.info("  Loading StackOverflow Q&A...")
        so = load_dataset("koutch/stackoverflow_python", split="train",
                          streaming=True, trust_remote_code=False)
        docs = []
        still_needed = max(0, BUCKET_MIN_DOCS["corpus_C_technical"] - count_bucket_docs("corpus_C_technical") - total)
        for i, item in enumerate(so):
            if i >= 50_000:
                break
            q = item.get("question_body", item.get("title", "")).strip()
            a = item.get("answer_body", item.get("body", "")).strip()
            if q and a and len(q) > 50 and len(a) > 100:
                docs.append(f"Question: {q}\n\nAnswer: {a}")
            if len(docs) >= min(4_000, still_needed):
                break
        n = write_docs(bucket, docs, "stackoverflow")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "StackOverflow Python Q&A",
                      "https://huggingface.co/datasets/koutch/stackoverflow_python",
                      mb, "Python programming Q&A — technical problem solving", n)
        total += n
        log.info(f"  StackOverflow: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  StackOverflow failed: {e}")
        record_exclusion("StackOverflow Python", str(e))

    final_count = count_bucket_docs("corpus_C_technical")
    log.info(f"  Bucket C final count: {final_count} docs (target: {BUCKET_MIN_DOCS['corpus_C_technical']})")
    if final_count < BUCKET_MIN_DOCS["corpus_C_technical"]:
        log.warning(f"  [WARN] Bucket C still below target: {final_count} < {BUCKET_MIN_DOCS['corpus_C_technical']}")
    return total


# ── FIX 3: Supplement Bucket D (Cybersecurity) ─────────────────────────────────────

def supplement_bucket_d():
    """
    Add cybersecurity content to reach BUCKET_MIN_DOCS['corpus_D_cyber'].
    All content is defensive/educational — no offensive exploit code.
    """
    log.info("=== Supplementing Bucket D: Cybersecurity ===")
    bucket = RAW / "corpus_D_cyber"
    current = count_bucket_docs("corpus_D_cyber")
    needed  = max(0, BUCKET_MIN_DOCS["corpus_D_cyber"] - current)
    log.info(f"  Current: {current} docs | Target min: {BUCKET_MIN_DOCS['corpus_D_cyber']} | Need: {needed}")

    if needed == 0:
        log.info("  Bucket D already at target — skipping")
        return 0

    total = 0

    # D-1: MITRE CWE (Common Weakness Enumeration)
    try:
        import requests, zipfile, io, re
        log.info("  Loading CWE descriptions from MITRE...")
        url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        xml_name = [n for n in z.namelist() if n.endswith(".xml")][0]
        xml_content = z.read(xml_name).decode("utf-8", errors="replace")
        weaknesses = re.findall(
            r'<Weakness[^>]*ID="(\d+)"[^>]*Name="([^"]+)"[^>]*>.*?'
            r'<Description>(.*?)</Description>',
            xml_content, re.DOTALL
        )
        docs = []
        for cwe_id, name, desc in weaknesses:
            desc_clean = re.sub(r'<[^>]+>', ' ', desc).strip()
            desc_clean = re.sub(r'\s+', ' ', desc_clean)
            if len(desc_clean) > 80:
                docs.append(f"CWE-{cwe_id}: {name}\n\nDescription:\n{desc_clean}")
        n = write_docs(bucket, docs, "cwe")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "MITRE CWE",
                      "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
                      mb, "Software weakness taxonomy — secure coding and remediation", n)
        total += n
        log.info(f"  CWE: {n} entries, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  CWE failed: {e}")
        record_exclusion("MITRE CWE", str(e))

    # D-2: NVD CVE descriptions (2022)
    try:
        import requests
        log.info("  Loading NVD CVE descriptions (2022)...")
        docs = []
        headers = {"User-Agent": "TitanAI-Corpus-Loader/1.0"}
        start_index = 0
        results_per_page = 2000
        total_fetched = 0
        max_cves = 6_000
        while total_fetched < max_cves:
            url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
                   f"?resultsPerPage={results_per_page}&startIndex={start_index}"
                   f"&pubStartDate=2022-01-01T00:00:00.000"
                   f"&pubEndDate=2022-12-31T23:59:59.999")
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
            log.info(f"  NVD 2022: fetched {total_fetched} CVEs so far...")
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

    # D-3: SecQA security Q&A dataset
    try:
        from datasets import load_dataset
        log.info("  Loading SecQA security Q&A dataset...")
        secqa = load_dataset("zefang-liu/secqa", split="train",
                             streaming=True, trust_remote_code=False)
        docs = []
        for i, item in enumerate(secqa):
            if i >= 8_000:
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

    # D-4: Security Wikipedia articles
    try:
        from datasets import load_dataset
        log.info("  Loading security-focused Wikipedia articles...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        sec_keywords = {
            "cryptography", "encryption", "firewall", "intrusion detection",
            "vulnerability", "exploit", "penetration testing", "malware",
            "ransomware", "phishing", "authentication", "authorization",
            "zero trust", "incident response", "threat modeling", "owasp",
            "network security", "application security", "security audit",
            "digital forensics", "cyber threat intelligence", "siem",
            "security information", "access control", "privilege escalation",
        }
        docs = []
        scanned = 0
        still_needed = max(0, BUCKET_MIN_DOCS["corpus_D_cyber"] - count_bucket_docs("corpus_D_cyber") - total)
        for item in wiki:
            scanned += 1
            if scanned > 600_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in sec_keywords) and len(text) > 300:
                docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= min(3_000, still_needed):
                break
        n = write_docs(bucket, docs, "wiki_security2")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "Wikipedia Security Articles (supplement)",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      mb, "Security architecture, cryptography, defensive techniques", n)
        total += n
        log.info(f"  Wikipedia Security: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia Security failed: {e}")
        record_exclusion("Wikipedia Security supplement", str(e))

    final_count = count_bucket_docs("corpus_D_cyber")
    log.info(f"  Bucket D final count: {final_count} docs (target: {BUCKET_MIN_DOCS['corpus_D_cyber']})")
    if final_count < BUCKET_MIN_DOCS["corpus_D_cyber"]:
        log.warning(f"  [WARN] Bucket D still below target: {final_count} < {BUCKET_MIN_DOCS['corpus_D_cyber']}")
    return total


# ── Ratio Verification Gate ────────────────────────────────────────────────────────────────

def verify_ratios_gate():
    """
    After rebalancing, verify that raw document counts are within acceptable
    bounds so that generate_shards.py will not require >10x oversampling
    for any bucket. Returns True if all buckets pass, False if any blocker remains.
    """
    log.info("\n=== POST-REBALANCE RATIO VERIFICATION GATE ===")
    total_docs = sum(count_bucket_docs(b) for b in BUCKET_MIN_DOCS)
    if total_docs == 0:
        log.error("  GATE FAIL: No documents found in any bucket")
        return False

    all_pass = True
    for bucket, min_docs in BUCKET_MIN_DOCS.items():
        count = count_bucket_docs(bucket)
        raw_ratio = count / total_docs if total_docs > 0 else 0
        status = "PASS" if count >= min_docs else "FAIL"
        if status == "FAIL":
            all_pass = False
        log.info(f"  {bucket:30s}: {count:6d} docs ({raw_ratio:.1%} of total) [{status}]")

    # Cinema-specific cap check
    cinema_count = count_bucket_docs("corpus_E_cinema")
    if cinema_count > CINEMA_MAX_DOCS:
        log.error(f"  GATE FAIL: Cinema still exceeds cap: {cinema_count} > {CINEMA_MAX_DOCS}")
        all_pass = False

    if all_pass:
        log.info("  *** ALL RATIO GATES PASSED — CORPUS REBALANCING COMPLETE ***")
    else:
        log.error("  *** RATIO GATE FAILED — DO NOT PROCEED TO generate_shards.py ***")

    return all_pass


# ── Main ───────────────────────────────────────────────────────────────────────────────────

def main():
    log.info("TitanAI Corpus Rebalancer v2 Starting")
    log.info("=" * 60)

    # Step 1: Trim Cinema first (before counting totals)
    archived = trim_bucket_e()
    log.info(f"  Cinema trim: {archived} files archived")

    # Step 2: Supplement underweight buckets
    c_added = supplement_bucket_c()
    d_added = supplement_bucket_d()

    # Step 3: Save updated inventory and exclusions
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    with open(EXCLUSION_LOG, "a") as f:
        for exc in exclusions:
            f.write(json.dumps(exc) + "\n")

    # Step 4: Print final disk usage
    log.info("\n=== FINAL DISK USAGE BY BUCKET ===")
    for bucket_name in BUCKET_MIN_DOCS:
        bucket_path = RAW / bucket_name
        if bucket_path.exists():
            files = list(bucket_path.glob("*.txt"))
            total_bytes = sum(f.stat().st_size for f in files)
            log.info(f"  {bucket_name}: {len(files)} files, {total_bytes/1_048_576:.1f} MB")

    # Step 5: Run ratio verification gate
    gate_passed = verify_ratios_gate()

    log.info("=" * 60)
    log.info(f"Rebalancing complete. Added C:{c_added}, D:{d_added} docs. Cinema trimmed: {archived}.")
    if not gate_passed:
        log.error("RATIO GATE FAILED — run validate_corpus_quality.py and inspect logs")
        sys.exit(1)

    log.info("Next step: python scripts/validate_corpus_quality.py")
    log.info("Then:      python scripts/generate_shards.py --config configs/titan_probe_v015.yaml")


if __name__ == "__main__":
    main()
