#!/usr/bin/env python3
"""
TitanAI — SFT v2 Data Preparation
====================================
Downloads and formats two open-source instruction datasets into TitanAI's
JSONL format ({"instruction": ..., "input": ..., "response": ...}).

Datasets:
  1. Alpaca Cleaned  — ~52K instruction-response pairs (cleaned version of
                       Stanford Alpaca, removing low-quality generations)
  2. Dolly 15K       — 15K human-written instruction pairs by Databricks

Output:
  data/sft/alpaca_cleaned.jsonl
  data/sft/dolly_15k.jsonl

Usage:
  python scripts/prepare_sft_v2_data.py
  python scripts/prepare_sft_v2_data.py --validate-only
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent.parent
SFT_DIR = BASE / "data" / "sft"

# ── Dataset sources ───────────────────────────────────────────────────────────

ALPACA_CLEANED_URL = (
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
    "main/alpaca_data_cleaned.json"
)

DOLLY_15K_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


# ── Download helpers ──────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "TitanAI/1.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 65536
        while True:
            data = response.read(chunk)
            if not data:
                break
            out.write(data)
            downloaded += len(data)
            if total:
                pct = downloaded / total * 100
                print(f"\r  Progress: {downloaded:,} / {total:,} bytes ({pct:.1f}%)", end="", flush=True)
    print()


# ── Alpaca Cleaned ────────────────────────────────────────────────────────────

def prepare_alpaca_cleaned(force: bool = False) -> int:
    """Download and convert Alpaca Cleaned to TitanAI JSONL format."""
    out_path = SFT_DIR / "alpaca_cleaned.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        print(f"  [SKIP] alpaca_cleaned.jsonl already exists ({count} examples)")
        return count

    raw_path = BASE / "data" / "raw_downloads" / "alpaca_cleaned.json"
    if not raw_path.exists() or force:
        download(ALPACA_CLEANED_URL, raw_path)

    with open(raw_path) as f:
        data = json.load(f)

    count = 0
    skipped = 0
    with open(out_path, "w") as out:
        for item in data:
            instruction = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            output = item.get("output", "").strip()

            # Quality filters
            if not instruction or not output:
                skipped += 1
                continue
            if len(output.split()) < 5:
                skipped += 1
                continue
            if len(instruction) > 1000 or len(output) > 2000:
                skipped += 1
                continue

            record = {
                "instruction": instruction,
                "input": inp,
                "response": output,
                "source": "alpaca_cleaned",
            }
            out.write(json.dumps(record) + "\n")
            count += 1

    print(f"  [OK] alpaca_cleaned.jsonl: {count} examples written, {skipped} skipped")
    return count


# ── Dolly 15K ─────────────────────────────────────────────────────────────────

DOLLY_CATEGORIES = {
    "open_qa", "closed_qa", "general_qa", "summarization",
    "information_extraction", "brainstorming", "classification",
}

def prepare_dolly_15k(force: bool = False) -> int:
    """Download and convert Dolly 15K to TitanAI JSONL format."""
    out_path = SFT_DIR / "dolly_15k.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        print(f"  [SKIP] dolly_15k.jsonl already exists ({count} examples)")
        return count

    raw_path = BASE / "data" / "raw_downloads" / "dolly_15k.jsonl"
    if not raw_path.exists() or force:
        download(DOLLY_15K_URL, raw_path)

    count = 0
    skipped = 0
    with open(raw_path) as inp_f, open(out_path, "w") as out:
        for line in inp_f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response = item.get("response", "").strip()
            category = item.get("category", "")

            if not instruction or not response:
                skipped += 1
                continue
            if len(response.split()) < 8:
                skipped += 1
                continue

            record = {
                "instruction": instruction,
                "input": context,
                "response": response,
                "source": f"dolly_{category}",
            }
            out.write(json.dumps(record) + "\n")
            count += 1

    print(f"  [OK] dolly_15k.jsonl: {count} examples written, {skipped} skipped")
    return count


# ── Validation ────────────────────────────────────────────────────────────────

def validate_file(path: Path) -> bool:
    """Spot-check a JSONL file for required fields."""
    errors = 0
    count = 0
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                assert "instruction" in record, "missing 'instruction'"
                assert "response" in record, "missing 'response'"
                count += 1
            except Exception as e:
                print(f"  ERROR line {i+1}: {e}")
                errors += 1
                if errors > 5:
                    print("  Too many errors — stopping validation.")
                    break
    print(f"  Validated {count} records, {errors} errors in {path.name}")
    return errors == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT v2 instruction datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing files")
    args = parser.parse_args()

    SFT_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        print("\n[Validate] Checking existing SFT v2 data files...\n")
        ok = True
        for name in ["alpaca_cleaned.jsonl", "dolly_15k.jsonl"]:
            path = SFT_DIR / name
            if not path.exists():
                print(f"  MISSING: {path}")
                ok = False
            else:
                ok = validate_file(path) and ok
        sys.exit(0 if ok else 1)

    print("\n[SFT v2 Data Prep] Downloading and formatting instruction datasets...\n")

    print("── Alpaca Cleaned ─────────────────────────────────────────────")
    alpaca_count = prepare_alpaca_cleaned(force=args.force)

    print("\n── Dolly 15K ──────────────────────────────────────────────────")
    dolly_count = prepare_dolly_15k(force=args.force)

    total = alpaca_count + dolly_count
    print(f"\n{'='*60}")
    print(f"  Total SFT v2 examples : {total:,}")
    print(f"  Alpaca Cleaned        : {alpaca_count:,}")
    print(f"  Dolly 15K             : {dolly_count:,}")
    print(f"  Output directory      : {SFT_DIR}")
    print(f"{'='*60}")
    print("\n  Next: run SFT v2 training:")
    print("  python scripts/run_sft_v2.py --config configs/titan_sft_v02.yaml \\")
    print("      --checkpoint checkpoints/crucible_v02/final.pt\n")


if __name__ == "__main__":
    main()
