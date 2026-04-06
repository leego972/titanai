"""
TitanAI Corpus Supplement Loader
==================================
Fixes failed sources from Phase 1 and supplements thin buckets:
  - Wikipedia (using correct parquet-based API, no trust_remote_code)
  - NVD CVE (using NVD 2.0 REST API instead of deprecated JSON feed)
  - Bucket D supplement: CISA advisories, security-focused Wikipedia
  - Bucket E supplement: Wikipedia film articles via correct API
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
LOG  = BASE / "data" / "corpus_supplement.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(str(LOG))]
)
log = logging.getLogger("supplement")

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
    bucket_dir.mkdir(parents=True, exist_ok=True)
    # Find next available index
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


# ── Wikipedia via correct parquet API ─────────────────────────────────────────

def load_wikipedia_filtered(bucket_dir, source_tag, keywords, max_docs, bucket_key, reason):
    """Load Wikipedia articles filtered by title keywords using the parquet-based API."""
    try:
        from datasets import load_dataset
        log.info(f"  Loading Wikipedia filtered ({source_tag}, max {max_docs})...")
        # Use the correct dataset name without trust_remote_code
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
        docs = []
        scanned = 0
        for item in wiki:
            scanned += 1
            if scanned > 2_000_000:
                break
            title = item.get("title", "").lower()
            text  = item.get("text", "").strip()
            if any(kw in title for kw in keywords):
                if len(text) > 300:
                    docs.append(f"Title: {item['title']}\n\n{text}")
            if len(docs) >= max_docs:
                break
        n = write_docs(bucket_dir, docs, source_tag)
        mb = approx_mb(docs)
        record_source(bucket_key, f"Wikipedia (wikimedia/wikipedia) — {source_tag}",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      mb, reason, n)
        log.info(f"  {source_tag}: {n} articles, ~{mb:.1f} MB (scanned {scanned} articles)")
        return n
    except Exception as e:
        log.error(f"  Wikipedia {source_tag} failed: {e}")
        record_exclusion(f"Wikipedia {source_tag}", str(e))
        return 0


# ── NVD CVE via REST API 2.0 ──────────────────────────────────────────────────

def load_nvd_cve():
    """Load CVE descriptions from NVD REST API 2.0 (replaces deprecated JSON feed)."""
    import requests
    bucket = RAW / "corpus_D_cyber"
    log.info("  Loading NVD CVE via REST API 2.0...")
    docs = []
    start_index = 0
    results_per_page = 2000
    total_fetched = 0
    max_cves = 8000

    headers = {"User-Agent": "TitanAI-Corpus-Loader/1.0"}

    while total_fetched < max_cves:
        url = (f"https://services.nvd.nist.gov/rest/json/cves/2.0"
               f"?resultsPerPage={results_per_page}&startIndex={start_index}")
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"  NVD API error at startIndex={start_index}: {e}")
            break

        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            break

        for v in vulnerabilities:
            cve = v.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            desc = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
            metrics = cve.get("metrics", {})
            cvss_score = None
            if "cvssMetricV31" in metrics:
                cvss_score = metrics["cvssMetricV31"][0].get("cvssData", {}).get("baseScore")
            elif "cvssMetricV30" in metrics:
                cvss_score = metrics["cvssMetricV30"][0].get("cvssData", {}).get("baseScore")
            elif "cvssMetricV2" in metrics:
                cvss_score = metrics["cvssMetricV2"][0].get("cvssData", {}).get("baseScore")

            if desc and len(desc) > 80:
                text = f"CVE ID: {cve_id}\n"
                if cvss_score:
                    text += f"CVSS Score: {cvss_score}\n"
                text += f"\nDescription:\n{desc}"
                docs.append(text)

        total_fetched += len(vulnerabilities)
        start_index += results_per_page
        log.info(f"  NVD: fetched {total_fetched} CVEs so far...")

        if len(vulnerabilities) < results_per_page:
            break
        if total_fetched >= max_cves:
            break

        time.sleep(0.6)  # NVD rate limit: ~100 req/min without API key

    n = write_docs(bucket, docs, "nvd_cve")
    mb = approx_mb(docs)
    record_source("corpus_D_cyber", "NVD CVE Descriptions (REST API 2.0)",
                  "https://nvd.nist.gov/developers/vulnerabilities",
                  mb,
                  "Vulnerability descriptions for security awareness and remediation. "
                  "No exploit code or PoC.", n)
    log.info(f"  NVD CVE: {n} entries, ~{mb:.1f} MB")
    return n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log.info("TitanAI Corpus Supplement Loader Starting")

    total_added = 0

    # ── Supplement Bucket D: Cybersecurity ────────────────────────────────────
    log.info("=== Supplementing Bucket D: Cybersecurity ===")

    # D-supp-1: NVD CVE via REST API
    total_added += load_nvd_cve()

    # D-supp-2: Security Wikipedia articles
    security_keywords = {
        "cybersecurity", "information security", "network security",
        "cryptography", "firewall", "intrusion detection", "penetration testing",
        "vulnerability", "threat modeling", "security architecture",
        "incident response", "malware analysis", "digital forensics",
        "zero trust", "siem", "nist", "iso 27001", "risk assessment",
        "secure coding", "authentication", "authorization", "encryption",
        "public key infrastructure", "certificate authority", "vpn",
        "security operations", "threat intelligence", "cvss", "cve", "cwe",
        "hardening", "patch management", "security audit", "penetration test"
    }
    total_added += load_wikipedia_filtered(
        RAW / "corpus_D_cyber", "wiki_security", security_keywords, 2000,
        "corpus_D_cyber",
        "Encyclopedic security knowledge: architecture, concepts, standards, frameworks"
    )

    # ── Supplement Bucket E: Cinema ───────────────────────────────────────────
    log.info("=== Supplementing Bucket E: Cinema ===")

    cinema_keywords = {
        "film", "movie", "cinema", "screenplay", "director", "cinematography",
        "actor", "actress", "producer", "film editing", "production design",
        "visual effects", "film score", "animation", "documentary",
        "box office", "academy award", "cannes", "sundance", "film festival",
        "screenwriter", "storyboard", "film noir", "new wave", "auteur",
        "cinematographer", "mise en scène", "montage", "film theory"
    }
    total_added += load_wikipedia_filtered(
        RAW / "corpus_E_cinema", "wiki_cinema", cinema_keywords, 3000,
        "corpus_E_cinema",
        "Film history, production craft, directors, cinematography, screenwriting"
    )

    # ── Supplement Bucket A: General (Wikipedia general) ─────────────────────
    log.info("=== Supplementing Bucket A: General (Wikipedia) ===")
    # Check if wiki_en already loaded enough
    existing_a = list((RAW / "corpus_A_general").glob("wiki_en_*.txt"))
    if len(existing_a) < 5000:
        general_keywords = {
            "history", "science", "geography", "economics", "philosophy",
            "mathematics", "physics", "chemistry", "biology", "medicine",
            "politics", "culture", "art", "music", "literature", "religion",
            "technology", "engineering", "architecture", "psychology",
            "sociology", "anthropology", "linguistics", "education"
        }
        total_added += load_wikipedia_filtered(
            RAW / "corpus_A_general", "wiki_general", general_keywords, 5000,
            "corpus_A_general",
            "Broad encyclopedic knowledge across academic and general domains"
        )

    # Save updated inventory and exclusions
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)
    log.info(f"Updated source inventory saved: {INVENTORY}")

    with open(EXCLUSION_LOG, "a") as f:
        for exc in exclusions:
            f.write(json.dumps(exc) + "\n")

    # Final disk summary
    log.info("\n=== UPDATED DISK USAGE BY BUCKET ===")
    for bucket_name in ["corpus_A_general", "corpus_B_reasoning",
                         "corpus_C_technical", "corpus_D_cyber", "corpus_E_cinema"]:
        bucket_path = RAW / bucket_name
        if bucket_path.exists():
            files = list(bucket_path.glob("*.txt"))
            total_bytes = sum(f.stat().st_size for f in files)
            log.info(f"  {bucket_name}: {len(files)} files, {total_bytes/1_048_576:.1f} MB")

    log.info(f"Supplement complete. Added {total_added} new documents.")


if __name__ == "__main__":
    main()
