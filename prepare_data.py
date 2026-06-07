#!/usr/bin/env python3
"""
TitanAI 1B — Data Preparation Pipeline
=======================================
Downloads, cleans, tokenizes, and saves all training datasets in the correct mix.

Dataset mix (Phase 1 - Continued Pretraining):
  35%  FineWeb          — High-quality web text (HuggingFace)
  25%  The Stack v2     — Python, TypeScript, Go, C, Bash code (BigCode)
  12%  NVD/CVE          — NIST vulnerability database (free JSON)
  10%  Exploit-DB       — Public exploit archive (GitHub)
   8%  MITRE ATT&CK     — Adversary tactics/techniques (JSON)
   5%  HackTricks       — Penetration testing knowledge base (GitHub)
   5%  OpenOrca         — GPT-4-style instruction pairs (HuggingFace)

Dataset mix (Phase 2 - Instruction Fine-Tuning):
  50%  OpenOrca
  30%  WizardCoder-Evol
  20%  Custom Titan pairs (add your own to data/raw/custom_instruct.jsonl)

Usage:
  pip install datasets transformers tokenizers tqdm requests
  python prepare_data.py --phase 1       # pretraining data
  python prepare_data.py --phase 2       # instruction fine-tuning data
  python prepare_data.py --phase all     # everything
  python prepare_data.py --check         # verify existing data
"""

import os
import json
import gzip
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
RAW_DIR     = ROOT / "data" / "raw"
PROC_DIR    = ROOT / "data" / "processed"
TOK_PATH    = ROOT / "tokenizer" / "artifacts_v32k" / "tokenizer.json"

# ── Token targets ──────────────────────────────────────────────────────────────
# Phase 1: ~12B tokens total for continued pretraining
PHASE1_TARGETS = {
    "fineweb":    4_200_000_000,   # 35% of 12B
    "thestack":   3_000_000_000,   # 25%
    "nvd_cve":    1_440_000_000,   # 12%
    "exploitdb":  1_200_000_000,   # 10%
    "mitre":        960_000_000,   #  8%
    "hacktricks":   600_000_000,   #  5%
    "openorca":     600_000_000,   #  5%
    "hackerone":     400_000_000,   #  3.3% of 12B — HackerOne bug bounty reports
    "owasp":         200_000_000,   #  1.7% — OWASP documentation
    "defcon":        200_000_000,   #  1.7% — DEF CON talk transcripts
    "arxiv_security":400_000_000,   #  3.3% — arXiv security papers (cs.CR)
}

# Phase 2: ~2B tokens for instruction fine-tuning
PHASE2_TARGETS = {
    "openorca":       1_000_000_000,   # 50%
    "wizardcoder":      600_000_000,   # 30%
    "custom_instruct":  400_000_000,   # 20%
}


def setup_dirs():
    for subdir in list(PHASE1_TARGETS.keys()) + list(PHASE2_TARGETS.keys()) + ["hackerone","owasp","defcon","arxiv_security"]:
        (RAW_DIR / subdir).mkdir(parents=True, exist_ok=True)
        (PROC_DIR / subdir).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created.")


def load_tokenizer():
    if not TOK_PATH.exists():
        print(f"⚠️  Tokenizer not found at {TOK_PATH}. Install it first:")
        print("   python tokenizer/train_tokenizer.py")
        return None
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def tokenize_and_save(texts, output_path, tokenizer, max_seq_len=2048):
    """Tokenize a list of strings and save as flat uint16 binary."""
    import numpy as np
    all_ids = []
    for text in tqdm(texts, desc=f"Tokenizing → {output_path.name}"):
        ids = tokenizer.encode(text).ids
        # Chunk into max_seq_len sequences
        for i in range(0, len(ids), max_seq_len):
            chunk = ids[i:i + max_seq_len]
            if len(chunk) == max_seq_len:
                all_ids.extend(chunk)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_path)
    tokens = len(arr)
    print(f"   Saved {tokens:,} tokens → {output_path}")
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 DATASETS
# ──────────────────────────────────────────────────────────────────────────────

def download_fineweb(tokenizer):
    """FineWeb — HuggingFace high-quality web text (10BT sample subset)."""
    print("\n📥 [1/7] Downloading FineWeb...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        texts = []
        token_count = 0
        target = PHASE1_TARGETS["fineweb"]
        for sample in ds:
            texts.append(sample["text"])
            if len(texts) % 10000 == 0:
                out = PROC_DIR / "fineweb" / f"chunk_{len(texts)//10000:04d}.bin"
                token_count += tokenize_and_save(texts, out, tokenizer)
                texts = []
                if token_count >= target:
                    break
        if texts:
            out = PROC_DIR / "fineweb" / "chunk_final.bin"
            tokenize_and_save(texts, out, tokenizer)
        print(f"✅ FineWeb done. ~{token_count/1e9:.1f}B tokens.")
    except Exception as e:
        print(f"❌ FineWeb failed: {e}")


def download_thestack(tokenizer):
    """The Stack v2 — Python, TypeScript, Go, C, Bash."""
    print("\n📥 [2/7] Downloading The Stack v2 (Python + TypeScript + Go + C + Bash)...")
    try:
        from datasets import load_dataset
        languages = ["python", "typescript", "go", "c", "shell"]
        for lang in languages:
            print(f"   Fetching {lang}...")
            ds = load_dataset(
                "bigcode/the-stack-v2-train-smol-ids",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            texts, count = [], 0
            for sample in ds:
                if sample.get("content"):
                    texts.append(sample["content"])
                if len(texts) >= 5000:
                    out = PROC_DIR / "thestack" / f"{lang}_{count:04d}.bin"
                    tokenize_and_save(texts, out, tokenizer)
                    count += 1
                    texts = []
                    if count >= 20:   # ~100k files per language
                        break
            if texts:
                out = PROC_DIR / "thestack" / f"{lang}_final.bin"
                tokenize_and_save(texts, out, tokenizer)
        print("✅ The Stack done.")
    except Exception as e:
        print(f"❌ The Stack failed: {e}")


def download_nvd_cve(tokenizer):
    """NVD/CVE — NIST National Vulnerability Database (free JSON feeds)."""
    print("\n📥 [3/7] Downloading NVD/CVE database...")
    import urllib.request
    years = list(range(2002, 2026))
    texts = []
    for year in years:
        url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.gz"
        gz_path = RAW_DIR / "nvd_cve" / f"nvdcve-{year}.json.gz"
        try:
            if not gz_path.exists():
                print(f"   Downloading CVE {year}...")
                urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("CVE_Items", []):
                cve_id = item["cve"]["CVE_data_meta"]["ID"]
                descs = item["cve"]["description"]["description_data"]
                desc = " ".join(d["value"] for d in descs if d["lang"] == "en")
                cvss = ""
                impact = item.get("impact", {})
                if "baseMetricV3" in impact:
                    cvss = f"CVSS v3: {impact['baseMetricV3']['cvssV3']['baseScore']} ({impact['baseMetricV3']['cvssV3']['baseSeverity']})"
                elif "baseMetricV2" in impact:
                    cvss = f"CVSS v2: {impact['baseMetricV2']['cvssV2']['baseScore']}"
                text = f"Vulnerability {cve_id}\n{cvss}\nDescription: {desc}\n"
                texts.append(text)
        except Exception as e:
            print(f"   ⚠️  Year {year}: {e}")
    out = PROC_DIR / "nvd_cve" / "nvd_all.bin"
    tokenize_and_save(texts, out, tokenizer)
    print("✅ NVD/CVE done.")


def download_exploitdb(tokenizer):
    """Exploit-DB — Offensive Security public exploit archive."""
    print("\n📥 [4/7] Cloning Exploit-DB...")
    clone_path = RAW_DIR / "exploitdb"
    if not clone_path.exists():
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/offensive-security/exploitdb",
            str(clone_path)
        ], check=True)
    texts = []
    exploits_dir = clone_path / "exploits"
    for fpath in tqdm(list(exploits_dir.rglob("*.py")) +
                      list(exploits_dir.rglob("*.c")) +
                      list(exploits_dir.rglob("*.rb")),
                      desc="Reading exploits"):
        try:
            text = fpath.read_text(errors="replace")
            if len(text) > 100:
                texts.append(f"# Exploit: {fpath.name}\n{text}")
        except Exception:
            pass
    out = PROC_DIR / "exploitdb" / "exploits.bin"
    tokenize_and_save(texts, out, tokenizer)
    print("✅ Exploit-DB done.")


def download_mitre(tokenizer):
    """MITRE ATT&CK — Adversary tactics, techniques and procedures."""
    print("\n📥 [5/7] Downloading MITRE ATT&CK...")
    import urllib.request
    url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    raw_path = RAW_DIR / "mitre" / "enterprise-attack.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        urllib.request.urlretrieve(url, raw_path)
    with open(raw_path) as f:
        data = json.load(f)
    texts = []
    for obj in data.get("objects", []):
        obj_type = obj.get("type", "")
        if obj_type not in ("attack-pattern", "malware", "tool", "course-of-action"):
            continue
        name = obj.get("name", "")
        desc = obj.get("description", "")
        kill_chain = ", ".join(
            phase["phase_name"]
            for phase in obj.get("kill_chain_phases", [])
        )
        text = f"MITRE ATT&CK — {obj_type.upper()}: {name}\n"
        if kill_chain:
            text += f"Kill chain phases: {kill_chain}\n"
        text += f"Description:\n{desc}\n"
        texts.append(text)
    out = PROC_DIR / "mitre" / "mitre_attack.bin"
    tokenize_and_save(texts, out, tokenizer)
    print("✅ MITRE ATT&CK done.")


def download_hacktricks(tokenizer):
    """HackTricks — Penetration testing techniques from GitBook."""
    print("\n📥 [6/7] Cloning HackTricks...")
    clone_path = RAW_DIR / "hacktricks"
    if not clone_path.exists():
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/carlospolop/hacktricks",
            str(clone_path)
        ], check=True)
    texts = []
    for fpath in tqdm(list(clone_path.rglob("*.md")), desc="Reading HackTricks"):
        try:
            text = fpath.read_text(errors="replace")
            if len(text) > 200:
                texts.append(text)
        except Exception:
            pass
    out = PROC_DIR / "hacktricks" / "hacktricks.bin"
    tokenize_and_save(texts, out, tokenizer)
    print("✅ HackTricks done.")


def download_openorca(tokenizer):
    """OpenOrca — 4M GPT-4-style instruction pairs."""
    print("\n📥 [7/7] Downloading OpenOrca...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Open-Orca/OpenOrca", split="train")
        texts = []
        for item in tqdm(ds, desc="Formatting OpenOrca"):
            prompt = item.get("question", "")
            response = item.get("response", "")
            system = item.get("system_prompt", "")
            text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
            texts.append(text)
        out = PROC_DIR / "openorca" / "openorca.bin"
        tokenize_and_save(texts, out, tokenizer)
        print("✅ OpenOrca done.")
    except Exception as e:
        print(f"❌ OpenOrca failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 DATASETS
# ──────────────────────────────────────────────────────────────────────────────

def download_wizardcoder(tokenizer):
    """WizardCoder-Evol — Code instruction pairs with complex reasoning."""
    print("\n📥 [P2-1] Downloading WizardCoder...")
    try:
        from datasets import load_dataset
        ds = load_dataset("WizardLM/WizardCoder_evol_instruct_110k", split="train")
        texts = []
        for item in tqdm(ds, desc="Formatting WizardCoder"):
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            text = f"<|user|>\n{instruction}\n<|assistant|>\n{output}\n"
            texts.append(text)
        out = PROC_DIR / "wizardcoder" / "wizardcoder.bin"
        tokenize_and_save(texts, out, tokenizer)
        print("✅ WizardCoder done.")
    except Exception as e:
        print(f"❌ WizardCoder failed: {e}")


def process_custom_instruct(tokenizer):
    """Custom Titan instruction pairs — add your own to data/raw/custom_instruct.jsonl."""
    print("\n📥 [P2-2] Processing custom instruction pairs...")
    custom_path = RAW_DIR / "custom_instruct.jsonl"
    if not custom_path.exists():
        print(f"   ⚠️  No custom data found at {custom_path}")
        print("   Create it with lines like:")
        print('   {"prompt": "Explain CVE-2024-1234", "response": "This CVE..."}')
        return
    texts = []
    with open(custom_path) as f:
        for line in f:
            item = json.loads(line.strip())
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
            texts.append(text)
    out = PROC_DIR / "custom_instruct" / "custom.bin"
    tokenize_and_save(texts, out, tokenizer)
    print(f"✅ Custom instruct done. {len(texts):,} pairs.")


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def verify_data():
    """Count tokens across all processed datasets."""
    import numpy as np
    print("\n📊 Data verification:")
    total = 0
    for name in list(PHASE1_TARGETS.keys()) + list(PHASE2_TARGETS.keys()):
        d = PROC_DIR / name
        if not d.exists():
            print(f"   {name:<20} — NOT FOUND")
            continue
        tokens = sum(
            np.fromfile(f, dtype=np.uint16).size
            for f in d.glob("*.bin")
        )
        target = PHASE1_TARGETS.get(name) or PHASE2_TARGETS.get(name, 0)
        pct = tokens / target * 100 if target else 0
        status = "✅" if pct >= 80 else "⚠️ " if pct >= 30 else "❌"
        print(f"   {status} {name:<20} {tokens/1e9:6.2f}B tokens  ({pct:.0f}% of target)")
        total += tokens
    print(f"\n   Total: {total/1e9:.1f}B tokens")




def download_hackerone(tokenizer):
    """HackerOne public bug bounty reports."""
    print("\n📥 [8/11] Cloning HackerOne public reports...")
    import urllib.request
    clone_path = RAW_DIR / "hackerone"
    if not clone_path.exists():
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/reddelexc/hackerone-reports",
            str(clone_path)
        ], check=True)
    texts = []
    for fpath in tqdm(list(clone_path.rglob("*.md")), desc="Reading HackerOne reports"):
        try:
            text = fpath.read_text(errors="replace")
            if len(text) > 200:
                texts.append(text)
        except Exception:
            pass
    out = PROC_DIR / "hackerone" / "hackerone.bin"
    (PROC_DIR / "hackerone").mkdir(parents=True, exist_ok=True)
    tokenize_and_save(texts, out, tokenizer)
    print("✅ HackerOne done.")


def download_owasp(tokenizer):
    """OWASP documentation — top 10, testing guide, etc."""
    print("\n📥 [9/11] Cloning OWASP documentation...")
    clone_path = RAW_DIR / "owasp"
    if not clone_path.exists():
        subprocess.run([
            "git", "clone", "--depth=1",
            "https://github.com/OWASP/www-project-top-ten",
            str(clone_path)
        ], check=True)
    texts = []
    for fpath in tqdm(list(clone_path.rglob("*.md")) + list(clone_path.rglob("*.html")),
                      desc="Reading OWASP docs"):
        try:
            text = fpath.read_text(errors="replace")
            if len(text) > 200:
                texts.append(text)
        except Exception:
            pass
    out = PROC_DIR / "owasp" / "owasp.bin"
    (PROC_DIR / "owasp").mkdir(parents=True, exist_ok=True)
    tokenize_and_save(texts, out, tokenizer)
    print("✅ OWASP done.")


def download_defcon(tokenizer):
    """DEF CON talk metadata and descriptions."""
    print("\n📥 [10/11] Fetching DEF CON talk data...")
    import urllib.request, json as _json
    texts = []
    # Use the DEF CON media archive (JSON metadata)
    url = "https://raw.githubusercontent.com/recon-infosec/defcon-media-parser/master/defcon_talks.json"
    raw_path = RAW_DIR / "defcon" / "defcon_talks.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not raw_path.exists():
            urllib.request.urlretrieve(url, raw_path)
        with open(raw_path) as f:
            talks = _json.load(f)
        for talk in talks:
            title = talk.get("title", "")
            desc = talk.get("description", "")
            speakers = ", ".join(talk.get("speakers", []))
            year = talk.get("year", "")
            text = f"DEF CON {year} Talk: {title}\nSpeakers: {speakers}\nDescription: {desc}\n"
            texts.append(text)
    except Exception as e:
        print(f"   ⚠️  DEF CON fetch failed: {e}. Using placeholder.")
        texts = [f"DEF CON security conference talk data — download from https://media.defcon.org"]
    out = PROC_DIR / "defcon" / "defcon.bin"
    (PROC_DIR / "defcon").mkdir(parents=True, exist_ok=True)
    tokenize_and_save(texts, out, tokenizer)
    print("✅ DEF CON done.")


def download_arxiv_security(tokenizer):
    """arXiv security and cryptography papers (cs.CR category)."""
    print("\n📥 [11/11] Downloading arXiv security papers (cs.CR)...")
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query="cat:cs.CR",
            max_results=5000,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        texts = []
        for paper in tqdm(client.results(search), desc="Fetching arXiv cs.CR", total=5000):
            text = (
                f"arXiv Security Paper: {paper.title}\n"
                f"Authors: {', '.join(str(a) for a in paper.authors)}\n"
                f"Published: {paper.published.date()}\n"
                f"Abstract: {paper.summary}\n"
            )
            texts.append(text)
        out = PROC_DIR / "arxiv_security" / "arxiv_security.bin"
        (PROC_DIR / "arxiv_security").mkdir(parents=True, exist_ok=True)
        tokenize_and_save(texts, out, tokenizer)
        print(f"✅ arXiv security done. {len(texts):,} papers.")
    except ImportError:
        print("   ⚠️  arxiv package not found. Run: pip install arxiv")
    except Exception as e:
        print(f"   ❌ arXiv failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TitanAI Data Prep Pipeline")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--check", action="store_true", help="Verify existing data only")
    args = parser.parse_args()

    if args.check:
        verify_data()
        return

    setup_dirs()
    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("❌ Cannot proceed without tokenizer.")
        return

    if args.phase in ("1", "all"):
        print("\n" + "="*60)
        print("PHASE 1 — Continued Pretraining Data (~12B tokens)")
        print("="*60)
        download_fineweb(tokenizer)
        download_thestack(tokenizer)
        download_nvd_cve(tokenizer)
        download_exploitdb(tokenizer)
        download_mitre(tokenizer)
        download_hacktricks(tokenizer)
        download_openorca(tokenizer)
        download_hackerone(tokenizer)
        download_owasp(tokenizer)
        download_defcon(tokenizer)
        download_arxiv_security(tokenizer)

    if args.phase in ("2", "all"):
        print("\n" + "="*60)
        print("PHASE 2 — Instruction Fine-Tuning Data (~2B tokens)")
        print("="*60)
        download_wizardcoder(tokenizer)
        process_custom_instruct(tokenizer)

    verify_data()
    print("\n✅ All done! Update data.mix in titan_1b.yaml and run train_1b.py")


if __name__ == "__main__":
    main()
