"""
TitanAI Corpus Loader — Phase 1 (Fixed for HF datasets >= 2.14)
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

    # Source A1: Wikipedia (en) via wikimedia/wikipedia (new parquet-based dataset)
    try:
        from datasets import load_dataset
        log.info("  Loading Wikipedia (en) — 20K articles via wikimedia/wikipedia...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                            streaming=True)
        docs = []
        for i, item in enumerate(wiki):
            if i >= 20000:
                break
            text = item.get("text", "").strip()
            if len(text) > 200:
                docs.append(text)
        n = write_docs(bucket, docs, "wiki_en")
        mb = approx_mb(docs)
        record_source("corpus_A_general", "Wikipedia (en) 20231101",
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
                      mb, "High-quality encyclopedic general knowledge, clean text", n)
        total_docs += n
        log.info(f"  Wikipedia: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Wikipedia load failed: {e}")
        record_exclusion("Wikipedia (en)", f"Load error: {e}")

    # Source A2: Project Gutenberg via allenai/gutenberg (parquet-based, no script)
    try:
        log.info("  Loading Project Gutenberg books...")
        from datasets import load_dataset
        pg = load_dataset("allenai/gutenberg_en", split="train", streaming=True)
        docs = []
        for i, item in enumerate(pg):
            if i >= 3000:
                break
            text = item.get("text", "").strip()
            if len(text) > 500:
                docs.append(text[:50000])
        n = write_docs(bucket, docs, "gutenberg")
        mb = approx_mb(docs)
        record_source("corpus_A_general", "Project Gutenberg (English)",
                      "https://huggingface.co/datasets/allenai/gutenberg_en",
                      mb, "Classic English literature, public domain, high linguistic quality", n)
        total_docs += n
        log.info(f"  Gutenberg: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  Gutenberg load failed: {e}")
        # Fallback: use C4 (Common Crawl cleaned) as general text
        try:
            log.info("  Falling back to C4 (en) for general text...")
            from datasets import load_dataset
            c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
            docs = []
            for i, item in enumerate(c4):
                if i >= 5000:
                    break
                text = item.get("text", "").strip()
                if len(text) > 300:
                    docs.append(text[:10000])
            n = write_docs(bucket, docs, "c4_en")
            mb = approx_mb(docs)
            record_source("corpus_A_general", "C4 (en) Common Crawl Cleaned",
                          "https://huggingface.co/datasets/allenai/c4",
                          mb, "High-quality web text, cleaned and deduplicated", n)
            total_docs += n
            log.info(f"  C4 fallback: {n} docs, ~{mb:.1f} MB")
        except Exception as e2:
            log.error(f"  C4 fallback failed: {e2}")
            record_exclusion("C4 General Text", f"Load error: {e2}")

    log.info(f"  Bucket A total: {total_docs} documents")
    return total_docs

# ── Bucket B: Reasoning ────────────────────────────────────────────────────────
def load_bucket_b():
    log.info("=== Bucket B: Reasoning and Planning ===")
    bucket = RAW / "corpus_B_reasoning"
    total_docs = 0

    # Source B1: OpenWebMath
    try:
        from datasets import load_dataset
        log.info("  Loading OpenWebMath...")
        owm = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        docs = []
        for i, item in enumerate(owm):
            if i >= 8000:
                break
            text = item.get("text", "").strip()
            if len(text) > 200:
                docs.append(text[:20000])
        n = write_docs(bucket, docs, "openwebmath")
        mb = approx_mb(docs)
        record_source("corpus_B_reasoning", "OpenWebMath",
                      "https://huggingface.co/datasets/open-web-math/open-web-math",
                      mb, "High-quality mathematical web content for reasoning", n)
        total_docs += n
        log.info(f"  OpenWebMath: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  OpenWebMath load failed: {e}")
        record_exclusion("OpenWebMath", f"Load error: {e}")

    # Source B2: FLAN reasoning tasks (via google/flan)
    try:
        from datasets import load_dataset
        log.info("  Loading FLAN reasoning subset...")
        flan = load_dataset("Muennighoff/flan", split="train", streaming=True)
        reasoning_tasks = {"cot_gsm8k", "cot_strategyqa", "cot_creak",
                           "cot_ecqa", "cot_esnli", "cot_qasc"}
        docs = []
        for i, item in enumerate(flan):
            if i >= 200000:
                break
            if item.get("task", "") in reasoning_tasks:
                inp = item.get("inputs", "").strip()
                out = item.get("targets", "").strip()
                if inp and out:
                    docs.append(f"Question: {inp}\n\nAnswer: {out}")
            if len(docs) >= 5000:
                break
        n = write_docs(bucket, docs, "flan_reasoning")
        mb = approx_mb(docs)
        record_source("corpus_B_reasoning", "FLAN Reasoning Tasks",
                      "https://huggingface.co/datasets/Muennighoff/flan",
                      mb, "Chain-of-thought reasoning examples across diverse tasks", n)
        total_docs += n
        log.info(f"  FLAN reasoning: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  FLAN load failed: {e}")
        record_exclusion("FLAN Reasoning", f"Load error: {e}")

    log.info(f"  Bucket B total: {total_docs} documents")
    return total_docs

# ── Bucket C: Technical ────────────────────────────────────────────────────────
def load_bucket_c():
    log.info("=== Bucket C: Technical and Code ===")
    bucket = RAW / "corpus_C_technical"
    total_docs = 0

    # Source C1: The Stack (Python only, via bigcode/the-stack)
    try:
        from datasets import load_dataset
        log.info("  Loading The Stack (Python)...")
        stack = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                             split="train", streaming=True)
        docs = []
        for i, item in enumerate(stack):
            if i >= 15000:
                break
            content = item.get("content", "").strip()
            if len(content) > 200 and len(content) < 100000:
                docs.append(f"# Python Source Code\n\n{content}")
        n = write_docs(bucket, docs, "the_stack_python")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "The Stack (Python)",
                      "https://huggingface.co/datasets/bigcode/the-stack-dedup",
                      mb, "High-quality Python source code for technical reasoning", n)
        total_docs += n
        log.info(f"  The Stack Python: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  The Stack load failed: {e}")
        record_exclusion("The Stack (Python)", f"Load error: {e}")

    # Source C2: StackOverflow Q&A (via datasets)
    try:
        from datasets import load_dataset
        log.info("  Loading StackOverflow Q&A...")
        so = load_dataset("koutch/stackoverflow_python", split="train", streaming=True)
        docs = []
        for i, item in enumerate(so):
            if i >= 10000:
                break
            q = item.get("question_body", "").strip()
            a = item.get("answer_body", "").strip()
            if q and a and len(q) > 50:
                docs.append(f"Question: {q}\n\nAnswer: {a}")
        n = write_docs(bucket, docs, "stackoverflow")
        mb = approx_mb(docs)
        record_source("corpus_C_technical", "StackOverflow Python Q&A",
                      "https://huggingface.co/datasets/koutch/stackoverflow_python",
                      mb, "Real-world technical Q&A covering Python and software engineering", n)
        total_docs += n
        log.info(f"  StackOverflow: {n} docs, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  StackOverflow load failed: {e}")
        # Fallback: use codeparrot/github-code
        try:
            log.info("  Falling back to codeparrot/github-code (Python)...")
            from datasets import load_dataset
            gh = load_dataset("codeparrot/github-code", streaming=True, split="train",
                              filter_languages=True, languages=["Python"])
            docs = []
            for i, item in enumerate(gh):
                if i >= 10000:
                    break
                code = item.get("code", "").strip()
                if len(code) > 200 and len(code) < 50000:
                    docs.append(f"# GitHub Python Code\n\n{code}")
            n = write_docs(bucket, docs, "github_python")
            mb = approx_mb(docs)
            record_source("corpus_C_technical", "GitHub Code (Python)",
                          "https://huggingface.co/datasets/codeparrot/github-code",
                          mb, "Python code from GitHub for technical knowledge", n)
            total_docs += n
            log.info(f"  GitHub Python fallback: {n} docs, ~{mb:.1f} MB")
        except Exception as e2:
            log.error(f"  GitHub code fallback failed: {e2}")
            record_exclusion("GitHub Code Python", f"Load error: {e2}")

    log.info(f"  Bucket C total: {total_docs} documents")
    return total_docs

# ── Bucket D: Cyber ────────────────────────────────────────────────────────────
def load_bucket_d():
    log.info("=== Bucket D: Cybersecurity ===")
    bucket = RAW / "corpus_D_cyber"
    total_docs = 0

    # Source D1: MITRE ATT&CK (via direct JSON download)
    try:
        import requests
        log.info("  Loading MITRE ATT&CK Enterprise techniques...")
        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        docs = []
        for obj in data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                name = obj.get("name", "")
                desc = obj.get("description", "").strip()
                if desc and len(desc) > 100:
                    docs.append(f"MITRE ATT&CK Technique: {name}\n\n{desc}")
        n = write_docs(bucket, docs, "mitre_attack")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "MITRE ATT&CK Enterprise",
                      "https://github.com/mitre/cti",
                      mb, "Adversary tactics and techniques — defensive security knowledge", n)
        total_docs += n
        log.info(f"  MITRE ATT&CK: {n} techniques, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  MITRE ATT&CK load failed: {e}")
        record_exclusion("MITRE ATT&CK", f"Load error: {e}")

    # Source D2: NIST NVD CVE descriptions (recent 2 years via API)
    try:
        import requests
        log.info("  Loading NIST NVD CVE descriptions...")
        docs = []
        # Use NVD 2.0 API - paginated
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"resultsPerPage": 2000, "startIndex": 0}
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for vuln in data.get("vulnerabilities", []):
            cve = vuln.get("cve", {})
            cve_id = cve.get("id", "")
            descriptions = cve.get("descriptions", [])
            for d in descriptions:
                if d.get("lang") == "en":
                    text = d.get("value", "").strip()
                    if len(text) > 100:
                        docs.append(f"CVE: {cve_id}\n\nDescription: {text}")
                    break
        n = write_docs(bucket, docs, "nvd_cve")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "NIST NVD CVE Descriptions",
                      "https://nvd.nist.gov/",
                      mb, "Vulnerability descriptions for defensive security knowledge", n)
        total_docs += n
        log.info(f"  NVD CVEs: {n} entries, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  NVD CVE load failed: {e}")
        record_exclusion("NIST NVD CVEs", f"Load error: {e}")

    # Source D3: Security papers via arXiv (cs.CR category)
    try:
        import requests
        log.info("  Loading arXiv cs.CR security papers...")
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": "cat:cs.CR",
            "start": 0,
            "max_results": 500,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        docs = []
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            if title is not None and summary is not None:
                t = title.text.strip().replace("\n", " ")
                s = summary.text.strip()
                if len(s) > 100:
                    docs.append(f"Security Research Paper: {t}\n\nAbstract: {s}")
        n = write_docs(bucket, docs, "arxiv_security")
        mb = approx_mb(docs)
        record_source("corpus_D_cyber", "arXiv Security Papers (cs.CR)",
                      "https://arxiv.org/list/cs.CR/recent",
                      mb, "Academic security research for deep cybersecurity knowledge", n)
        total_docs += n
        log.info(f"  arXiv security: {n} papers, ~{mb:.1f} MB")
    except Exception as e:
        log.error(f"  arXiv security load failed: {e}")
        record_exclusion("arXiv Security Papers", f"Load error: {e}")

    log.info(f"  Bucket D total: {total_docs} documents")
    return total_docs

# ── Bucket E: Cinema ──────────────────────────────────────────────────────────
def load_bucket_e():
    log.info("=== Bucket E: Cinema (capped at 10% target) ===")
    bucket = RAW / "corpus_E_cinema"
    total_docs = 0

    # Source E1: Wikipedia film articles via wikimedia/wikipedia (new format)
    try:
        from datasets import load_dataset
        log.info("  Loading Wikipedia film and cinema articles...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                            streaming=True)
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
                      "https://huggingface.co/datasets/wikimedia/wikipedia",
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
