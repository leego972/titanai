"""
TitanAI Corpus Loader — Phase 1
=================================
Downloads real, high-quality, publicly available data into the approved
corpus buckets A–E. All sources are clearly documented with provenance.

Sources selected per bucket taxonomy:
  A (General)    : Wikipedia (en), BookCorpus-style open books (gutenberg)
  B (Reasoning)  : OpenWebMath (math reasoning), FLAN reasoning subset
  C (Technical)  : The Stack (Python/docs), StackOverflow Q&A
  D (Cyber)      : SecWiki papers, MITRE ATT&CK descriptions, NIST docs,
                   CVE descriptions (defensive/architecture focus only)
  E (Cinema)     : Wikipedia film articles, screenplay craft essays

Rules:
  - No fake data
  - No offensive-abuse cyber content
  - Log all exclusions
  - Record provenance in source_inventory.json
  - Target approximate token ratios: A:35 B:15 C:20 D:20 E:10
"""

import os
import sys
import json
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"
LOG  = BASE / "data" / "corpus_load.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("corpus_loader")

EXCLUSION_LOG = BASE / "data" / "exclusions.jsonl"
INVENTORY     = BASE / "data" / "source_inventory.json"

inventory = {}
exclusions = []


def record_source(bucket, source_name, source_url, size_mb, reason, n_docs):
    inventory.setdefault(bucket, []).append({
        "source": source_name,
        "url": source_url,
        "size_mb": round(size_mb, 2),
        "n_documents": n_docs,
        "reason_for_inclusion": reason,
        "loaded_at": datetime.utcnow().isoformat()
    })


def record_exclusion(source_name, reason):
    exclusions.append({"source": source_name, "reason": reason,
                        "timestamp": datetime.utcnow().isoformat()})
    log.warning(f"EXCLUDED: {source_name} — {reason}")


def write_docs(bucket_dir: Path, docs: list, source_tag: str):
    """Write a list of text strings to individual .txt files in the bucket dir."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, text in enumerate(docs):
        if not text or len(text.strip()) < 100:
            continue
        fname = bucket_dir / f"{source_tag}_{i:06d}.txt"
        fname.write_text(text.strip(), encoding="utf-8")
        written += 1
    log.info(f"  Wrote {written} documents to {bucket_dir.name}/")
    return written


def approx_mb(docs):
    return sum(len(d.encode("utf-8")) for d in docs) / 1_048_576


# ── Bucket A: General ──────────────────────────────────────────────────────────

def load_bucket_a():
    log.info("=== Bucket A: General Language ===")
    bucket = RAW / "corpus_A_general"
    total_docs = 0

    # Source A1: Wikipedia (en) — 20K articles via HuggingFace datasets
    try:
        from datasets import load_dataset
        log.info("  Loading Wikipedia (en) — 20K articles...")
        wiki = load_dataset("wikipedia", "20220301.en", split="train",
                            streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(wiki):
            if i >= 20000:
                break
            text = item.get("text", "").strip()
            if len(text) > 200:
                docs.append(text)
        n = write_docs(bucket, docs, "wiki_en")
        mb = approx_mb(docs)
        record_source("corpus_A_general", "Wikipedia (en) 20220301",
                      "https://huggingface.co/datasets/wikipedia",
                      mb, "High-quality encyclopedic general knowledge, clean text", n)
        total_docs += n
        log.info(f"  Wikipedia: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia load failed: {e}")
        record_exclusion("Wikipedia (en)", f"Load error: {e}")

    # Source A2: Project Gutenberg (open books via gutenberg_dammit or direct)
    try:
        log.info("  Loading Project Gutenberg books (via datasets)...")
        from datasets import load_dataset
        pg = load_dataset("sedthh/gutenberg_english", split="train",
                          streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(pg):
            if i >= 3000:
                break
            text = item.get("TEXT", "").strip()
            if len(text) > 500:
                docs.append(text[:50000])  # cap per book
        n = write_docs(bucket, docs, "gutenberg")
        mb = approx_mb(docs)
        record_source("corpus_A_general", "Project Gutenberg (English)",
                      "https://huggingface.co/datasets/sedthh/gutenberg_english",
                      mb, "Classic English literature, public domain, high linguistic quality", n)
        total_docs += n
        log.info(f"  Gutenberg: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Gutenberg load failed: {e}")
        record_exclusion("Project Gutenberg", f"Load error: {e}")

    log.info(f"  Bucket A total: {total_docs} documents")
    return total_docs


# ── Bucket B: Reasoning ────────────────────────────────────────────────────────

def load_bucket_b():
    log.info("=== Bucket B: Reasoning and Planning ===")
    bucket = RAW / "corpus_B_reasoning"
    total_docs = 0

    # Source B1: OpenWebMath — high-quality mathematical reasoning text
    try:
        from datasets import load_dataset
        log.info("  Loading OpenWebMath (math reasoning)...")
        owm = load_dataset("open-web-math/open-web-math", split="train",
                           streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(owm):
            if i >= 8000:
                break
            text = item.get("text", "").strip()
            if len(text) > 200:
                docs.append(text)
        n = write_docs(bucket, docs, "openwebmath")
        mb = approx_mb(docs)
        record_source("corpus_B_reasoning", "OpenWebMath",
                      "https://huggingface.co/datasets/open-web-math/open-web-math",
                      mb, "High-quality mathematical reasoning, proofs, structured problem solving", n)
        total_docs += n
        log.info(f"  OpenWebMath: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  OpenWebMath load failed: {e}")
        record_exclusion("OpenWebMath", f"Load error: {e}")

    # Source B2: FLAN reasoning tasks (chain-of-thought formatted)
    try:
        from datasets import load_dataset
        log.info("  Loading FLAN CoT reasoning subset...")
        flan = load_dataset("Muennighoff/flan", split="train",
                            streaming=True, trust_remote_code=True)
        docs = []
        cot_tasks = {"cot_gsm8k", "cot_strategyqa", "cot_creak",
                     "cot_ecqa", "cot_esnli", "cot_qasc", "cot_sensemaking"}
        for i, item in enumerate(flan):
            if len(docs) >= 5000:
                break
            if item.get("task", "") in cot_tasks:
                inp = item.get("inputs", "")
                tgt = item.get("targets", "")
                if inp and tgt:
                    docs.append(f"Question: {inp}\nAnswer: {tgt}")
        n = write_docs(bucket, docs, "flan_cot")
        mb = approx_mb(docs)
        record_source("corpus_B_reasoning", "FLAN Chain-of-Thought subset",
                      "https://huggingface.co/datasets/Muennighoff/flan",
                      mb, "Structured reasoning traces, planning, multi-step problem solving", n)
        total_docs += n
        log.info(f"  FLAN CoT: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  FLAN CoT load failed: {e}")
        record_exclusion("FLAN CoT", f"Load error: {e}")

    log.info(f"  Bucket B total: {total_docs} documents")
    return total_docs


# ── Bucket C: Technical ────────────────────────────────────────────────────────

def load_bucket_c():
    log.info("=== Bucket C: Technical / Systems ===")
    bucket = RAW / "corpus_C_technical"
    total_docs = 0

    # Source C1: The Stack (Python only — clean, documented code)
    try:
        from datasets import load_dataset
        log.info("  Loading The Stack (Python subset)...")
        stack = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                             split="train", streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(stack):
            if i >= 10000:
                break
            content = item.get("content", "").strip()
            # Only include files with docstrings (higher quality)
            if '"""' in content or "'''" in content:
                if len(content) > 200 and len(content) < 20000:
                    docs.append(content)
            if len(docs) >= 8000:
                break
        n = write_docs(bucket, docs, "stack_python")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "The Stack (Python, deduplicated)",
                      "https://huggingface.co/datasets/bigcode/the-stack-dedup",
                      mb, "High-quality Python code with documentation, systems and infrastructure focus", n)
        total_docs += n
        log.info(f"  The Stack (Python): {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  The Stack load failed: {e}")
        record_exclusion("The Stack (Python)", f"Load error: {e}")

    # Source C2: StackOverflow Q&A (technical problem solving)
    try:
        from datasets import load_dataset
        log.info("  Loading StackOverflow Q&A...")
        so = load_dataset("koutch/stackoverflow_python", split="train",
                          streaming=True, trust_remote_code=True)
        docs = []
        for i, item in enumerate(so):
            if i >= 5000:
                break
            q = item.get("question_body", "").strip()
            a = item.get("answer_body", "").strip()
            if q and a and len(q) > 50 and len(a) > 100:
                docs.append(f"Question: {q}\n\nAnswer: {a}")
        n = write_docs(bucket, docs, "stackoverflow")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "StackOverflow Python Q&A",
                      "https://huggingface.co/datasets/koutch/stackoverflow_python",
                      mb, "Real-world technical problem solving, software engineering knowledge", n)
        total_docs += n
        log.info(f"  StackOverflow: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  StackOverflow load failed: {e}")
        record_exclusion("StackOverflow Q&A", f"Load error: {e}")

    log.info(f"  Bucket C total: {total_docs} documents")
    return total_docs


# ── Bucket D: Cybersecurity ────────────────────────────────────────────────────

def load_bucket_d():
    """
    Cybersecurity bucket — DEFENSIVE focus only.
    Sources: MITRE ATT&CK (descriptions only), CVE summaries (defensive context),
             security architecture papers, NIST guidelines.
    Excluded: exploit code, offensive tooling, evasion techniques as primary content.
    """
    log.info("=== Bucket D: Cybersecurity (Defensive / Architecture) ===")
    bucket = RAW / "corpus_D_cyber"
    total_docs = 0

    # Source D1: MITRE ATT&CK technique descriptions (detection + mitigation focus)
    try:
        import requests
        log.info("  Loading MITRE ATT&CK technique descriptions...")
        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        attack_data = resp.json()

        docs = []
        for obj in attack_data.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            name = obj.get("name", "")
            desc = obj.get("description", "").strip()
            # Include detection and mitigation context
            detection = ""
            for ref in obj.get("x_mitre_detection", ""):
                detection += ref
            if len(desc) > 100:
                text = f"Technique: {name}\n\nDescription:\n{desc}"
                if detection:
                    text += f"\n\nDetection:\n{detection}"
                docs.append(text)

        n = write_docs(bucket, docs, "mitre_attack")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "MITRE ATT&CK Enterprise (technique descriptions)",
                      "https://github.com/mitre/cti",
                      mb,
                      "Authoritative threat taxonomy with detection and mitigation context. "
                      "Descriptions only — no exploit code.", n)
        total_docs += n
        log.info(f"  MITRE ATT&CK: {n} techniques, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  MITRE ATT&CK load failed: {e}")
        record_exclusion("MITRE ATT&CK", f"Load error: {e}")

    # Source D2: CVE descriptions (NVD) — defensive context, no exploit code
    try:
        import requests
        log.info("  Loading NVD CVE descriptions (2023 dataset)...")
        # Use NVD JSON feed for 2023
        nvd_url = "https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-2023.json.gz"
        import gzip, io
        resp = requests.get(nvd_url, timeout=120, stream=True)
        resp.raise_for_status()
        content = gzip.decompress(resp.content)
        nvd_data = json.loads(content)

        docs = []
        for item in nvd_data.get("CVE_Items", []):
            cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID", "")
            descs = item.get("cve", {}).get("description", {}).get("description_data", [])
            desc = next((d["value"] for d in descs if d.get("lang") == "en"), "")
            impact = item.get("impact", {})
            cvss = (impact.get("baseMetricV3", {}).get("cvssV3", {}).get("baseScore") or
                    impact.get("baseMetricV2", {}).get("cvssV2", {}).get("baseScore"))
            if desc and len(desc) > 80 and cve_id:
                text = f"CVE ID: {cve_id}\n"
                if cvss:
                    text += f"CVSS Score: {cvss}\n"
                text += f"\nDescription:\n{desc}"
                docs.append(text)
            if len(docs) >= 8000:
                break

        n = write_docs(bucket, docs, "nvd_cve")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "NVD CVE Descriptions (2023)",
                      "https://nvd.nist.gov/vuln/data-feeds",
                      mb,
                      "Vulnerability descriptions for security awareness and remediation knowledge. "
                      "Descriptions only — no exploit code or PoC.", n)
        total_docs += n
        log.info(f"  NVD CVE: {n} entries, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  NVD CVE load failed: {e}")
        record_exclusion("NVD CVE 2023", f"Load error: {e}")

    # Source D3: Security-focused Wikipedia articles
    try:
        from datasets import load_dataset
        log.info("  Loading security-focused Wikipedia articles...")
        wiki = load_dataset("wikipedia", "20220301.en", split="train",
                            streaming=True, trust_remote_code=True)
        security_keywords = {
            "cybersecurity", "information security", "network security",
            "cryptography", "firewall", "intrusion detection", "penetration testing",
            "vulnerability", "threat modeling", "security architecture",
            "incident response", "malware analysis", "digital forensics",
            "zero trust", "SIEM", "SOC", "NIST", "ISO 27001", "risk assessment",
            "secure coding", "authentication", "authorization", "encryption",
            "public key infrastructure", "certificate authority", "VPN",
            "security operations", "threat intelligence", "CVSS", "CVE", "CWE"
        }
        docs = []
        for i, item in enumerate(wiki):
            if i > 500000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in security_keywords):
                if len(text) > 300:
                    docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= 2000:
                break
        n = write_docs(bucket, docs, "wiki_security")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "Wikipedia Security Articles",
                      "https://huggingface.co/datasets/wikipedia",
                      mb,
                      "Encyclopedic security knowledge: architecture, concepts, standards, frameworks", n)
        total_docs += n
        log.info(f"  Wikipedia Security: {n} articles, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia Security load failed: {e}")
        record_exclusion("Wikipedia Security Articles", f"Load error: {e}")

    # Explicit exclusions for Bucket D
    record_exclusion("Exploit-DB full database",
                     "Contains working exploit code — offensive abuse risk. "
                     "Excluded per corpus policy.")
    record_exclusion("Metasploit module source",
                     "Offensive tooling source code — excluded per corpus policy.")
    record_exclusion("Malware sample repositories",
                     "Malware source code — excluded per corpus policy.")
    record_exclusion("Phishing kit datasets",
                     "Phishing templates — excluded per corpus policy.")

    log.info(f"  Bucket D total: {total_docs} documents")
    return total_docs


# ── Bucket E: Cinema ───────────────────────────────────────────────────────────

def load_bucket_e():
    log.info("=== Bucket E: Film and Cinema Production ===")
    bucket = RAW / "corpus_E_cinema"
    total_docs = 0

    # Source E1: Wikipedia film articles
    try:
        from datasets import load_dataset
        log.info("  Loading Wikipedia film and cinema articles...")
        wiki = load_dataset("wikipedia", "20220301.en", split="train",
                            streaming=True, trust_remote_code=True)
        cinema_keywords = {
            "film", "movie", "cinema", "screenplay", "director", "cinematography",
            "actor", "actress", "producer", "editing", "production design",
            "visual effects", "score", "soundtrack", "animation", "documentary",
            "box office", "academy award", "cannes", "sundance", "film festival",
            "screenwriter", "storyboard", "mise-en-scène", "montage",
            "cinematographer", "film noir", "new wave", "auteur"
        }
        docs = []
        for i, item in enumerate(wiki):
            if i > 600000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in cinema_keywords):
                if len(text) > 300:
                    docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= 3000:
                break
        n = write_docs(bucket, docs, "wiki_cinema")
        mb = approx_mb(docs)
        record_source("corpus_E_cinema", "Wikipedia Film and Cinema Articles",
                      "https://huggingface.co/datasets/wikipedia",
                      mb,
                      "Film history, production craft, directors, cinematography, screenwriting", n)
        total_docs += n
        log.info(f"  Wikipedia Cinema: {n} articles, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia Cinema load failed: {e}")
        record_exclusion("Wikipedia Cinema Articles", f"Load error: {e}")

    # Source E2: CMU Movie Summary Corpus (plot summaries + metadata)
    try:
        import requests
        log.info("  Loading CMU Movie Summary Corpus...")
        url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
        import tarfile, io
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        tar = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
        summaries_file = tar.extractfile("MovieSummaries/plot_summaries.txt")
        if summaries_file:
            lines = summaries_file.read().decode("utf-8", errors="replace").splitlines()
            docs = []
            for line in lines:
                parts = line.split("\t", 1)
                if len(parts) == 2 and len(parts[1]) > 100:
                    docs.append(f"Movie Plot Summary:\n{parts[1].strip()}")
            n = write_docs(bucket, docs, "cmu_movie_summaries")
            mb = approx_mb(docs)
            record_source("corpus_E_cinema", "CMU Movie Summary Corpus",
                          "http://www.cs.cmu.edu/~ark/personas/",
                          mb, "42K movie plot summaries — narrative structure and cinema knowledge", n)
            total_docs += n
            log.info(f"  CMU Movie Summaries: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  CMU Movie Summaries load failed: {e}")
        record_exclusion("CMU Movie Summary Corpus", f"Load error: {e}")

    log.info(f"  Bucket E total: {total_docs} documents")
    return total_docs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("TitanAI Corpus Loader — Phase 1 Starting")
    log.info(f"Target directory: {RAW}")

    results = {}
    results["corpus_A_general"]   = load_bucket_a()
    results["corpus_B_reasoning"] = load_bucket_b()
    results["corpus_C_technical"] = load_bucket_c()
    results["corpus_D_cyber"]     = load_bucket_d()
    results["corpus_E_cinema"]    = load_bucket_e()

    # Save inventory
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    log.info(f"Source inventory saved: {INVENTORY}")

    # Save exclusions
    with open(EXCLUSION_LOG, "w") as f:
        for exc in exclusions:
            f.write(json.dumps(exc) + "\n")
    log.info(f"Exclusion log saved: {EXCLUSION_LOG}")

    # Summary
    log.info("\n=== CORPUS LOAD SUMMARY ===")
    total = 0
    for bucket, n in results.items():
        log.info(f"  {bucket}: {n} documents")
        total += n
    log.info(f"  TOTAL: {total} documents")

    # Check bucket sizes on disk
    log.info("\n=== DISK USAGE BY BUCKET ===")
    for bucket_name in ["corpus_A_general", "corpus_B_reasoning",
                         "corpus_C_technical", "corpus_D_cyber", "corpus_E_cinema"]:
        bucket_path = RAW / bucket_name
        if bucket_path.exists():
            files = list(bucket_path.glob("*.txt"))
            total_bytes = sum(f.stat().st_size for f in files)
            log.info(f"  {bucket_name}: {len(files)} files, {total_bytes/1_048_576:.1f} MB")

    log.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
