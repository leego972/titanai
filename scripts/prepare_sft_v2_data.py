#!/usr/bin/env python3
"""
TitanAI — SFT v2 Data Preparation
====================================
Downloads and formats two open-source instruction datasets into TitanAI's
chat format ({"messages": [system, user, assistant]}) required by TitanSFTDataset.

Datasets:
  1. Alpaca Cleaned  — ~52K instruction-response pairs (cleaned version of
                       Stanford Alpaca, removing low-quality generations)
  2. Dolly 15K       — 15K human-written instruction pairs by Databricks

Output:
  data/sft/alpaca_cleaned.jsonl   (chat format — ready for TitanSFTDataset)
  data/sft/dolly_15k.jsonl        (chat format — ready for TitanSFTDataset)

NOTE: The previous version of this script wrote Alpaca format
({"instruction", "input", "response"}) which TitanSFTDataset silently ignores
(it requires the "messages" key). This version writes the correct chat format.

Usage:
  python scripts/prepare_sft_v2_data.py
  python scripts/prepare_sft_v2_data.py --validate-only
  python scripts/prepare_sft_v2_data.py --force
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent.parent
SFT_DIR = BASE / "data" / "sft"

ALPACA_CLEANED_URL = (
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
    "main/alpaca_data_cleaned.json"
)
DOLLY_15K_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)

# Generic Titan system prompt for all public-dataset examples.
# Domain-specific prompts (cybersecurity, cinema) live in their own SFT files.
TITAN_SYSTEM_PROMPT = (
    "You are Titan, a language model built from scratch by your user. "
    "Answer accurately, think step by step, and be genuinely helpful."
)


# ── Download helper ────────────────────────────────────────────────────────

def download(url: str, dest: Path) -> None:
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "TitanAI/1.0"})
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded:,} / {total:,} bytes ({downloaded/total*100:.1f}%)", end="", flush=True)
    print()


# ── Chat format builder ────────────────────────────────────────────────────

def to_chat(user_content: str, assistant_content: str) -> dict:
    """Return a TitanSFTDataset-compatible chat record."""
    return {
        "messages": [
            {"role": "system",    "content": TITAN_SYSTEM_PROMPT},
            {"role": "user",      "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]
    }


# ── Alpaca Cleaned ─────────────────────────────────────────────────────────

def prepare_alpaca_cleaned(force: bool = False) -> int:
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
            inp         = item.get("input",       "").strip()
            output      = item.get("output",      "").strip()

            if not instruction or not output:
                skipped += 1
                continue
            if len(output.split()) < 5:
                skipped += 1
                continue
            if len(instruction) > 1000 or len(output) > 2000:
                skipped += 1
                continue

            user_content = f"{instruction}\n\n{inp}" if inp else instruction
            out.write(json.dumps(to_chat(user_content, output)) + "\n")
            count += 1

    print(f"  [OK] alpaca_cleaned.jsonl: {count} examples written, {skipped} skipped")
    return count


# ── Dolly 15K ──────────────────────────────────────────────────────────────

def prepare_dolly_15k(force: bool = False) -> int:
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
            context     = item.get("context",     "").strip()
            response    = item.get("response",    "").strip()

            if not instruction or not response:
                skipped += 1
                continue
            if len(response.split()) < 8:
                skipped += 1
                continue

            user_content = f"{instruction}\n\nContext: {context}" if context else instruction
            out.write(json.dumps(to_chat(user_content, response)) + "\n")
            count += 1

    print(f"  [OK] dolly_15k.jsonl: {count} examples written, {skipped} skipped")
    return count


# ── Validation ─────────────────────────────────────────────────────────────

def validate_file(path: Path) -> bool:
    """Spot-check a JSONL file for the messages format required by TitanSFTDataset."""
    errors = 0
    count = 0
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                msgs = record.get("messages", [])
                assert isinstance(msgs, list) and len(msgs) >= 2, "messages must be list with >= 2 entries"
                roles = {m["role"] for m in msgs}
                assert "user"      in roles, "missing 'user' role"
                assert "assistant" in roles, "missing 'assistant' role"
                for m in msgs:
                    assert m.get("content", "").strip(), f"empty content in role '{m.get('role')}'"
                count += 1
            except Exception as e:
                print(f"  ERROR line {i+1}: {e}")
                errors += 1
                if errors > 5:
                    print("  Too many errors — stopping validation.")
                    break
    print(f"  Validated {count} records, {errors} errors in {path.name}")
    return errors == 0


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT v2 datasets in chat format")
    parser.add_argument("--force",         action="store_true", help="Re-download even if files exist")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing files")
    args = parser.parse_args()

    SFT_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        print("\n[Validate] Checking SFT v2 data files (chat format)...\n")
        ok = True
        for name in ["alpaca_cleaned.jsonl", "dolly_15k.jsonl"]:
            path = SFT_DIR / name
            if not path.exists():
                print(f"  MISSING: {path}")
                ok = False
            else:
                ok = validate_file(path) and ok
        sys.exit(0 if ok else 1)

    print("\n[SFT v2 Data Prep] Downloading and converting to chat format...\n")

    print("── Alpaca Cleaned ─────────────────────────────────────────────")
    alpaca_count = prepare_alpaca_cleaned(force=args.force)

    print("\n── Dolly 15K ──────────────────────────────────────────────────")
    dolly_count = prepare_dolly_15k(force=args.force)

    total = alpaca_count + dolly_count
    print(f"\n{'='*60}")
    print(f"  Total SFT v2 examples : {total:,}")
    print(f"  Alpaca Cleaned        : {alpaca_count:,}")
    print(f"  Dolly 15K             : {dolly_count:,}")
    print(f"  Format                : chat (TitanSFTDataset-compatible)")
    print(f"  Output directory      : {SFT_DIR}")
    print(f"{'='*60}")
    print("\n  Next: run SFT v2 training:")
    print("  python scripts/run_sft_v2.py --config configs/titan_sft_v02.yaml \\")
    print("      --checkpoint checkpoints/crucible_v02/final.pt\n")


if __name__ == "__main__":
    main()
