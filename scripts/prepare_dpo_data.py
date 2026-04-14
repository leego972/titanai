#!/usr/bin/env python3
"""
TitanAI — DPO Data Preparation
=================================
Builds a preference dataset (chosen / rejected pairs) for Direct Preference
Optimization (DPO) training.

Sources:
  1. Anthropic HH-RLHF  — Human preference data (helpful/harmless)
     ~170K pairs of (chosen, rejected) responses
     We use a filtered subset: 20K highest-quality pairs.
  2. Synthetic pairs from existing SFT data — automatically generates
     "bad" responses by sampling at high temperature as rejected examples.

Output:
  data/dpo/preference_pairs.jsonl

Each record:
  {
    "prompt":   "...",   # the instruction/question
    "chosen":   "...",   # preferred response
    "rejected": "..."    # dispreferred response
  }

Usage:
  python scripts/prepare_dpo_data.py
  python scripts/prepare_dpo_data.py --source hh_rlhf --limit 20000
  python scripts/prepare_dpo_data.py --source synthetic --checkpoint checkpoints/sft_v02/final.pt
"""

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent.parent
DPO_DIR = BASE / "data" / "dpo"

HH_RLHF_TRAIN_URL = (
    "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
    "helpful-base/train.jsonl.zst"
)
# Use the non-compressed version via datasets API alternative
HH_RLHF_HELPFUL_URL = (
    "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
    "helpful-base/train.jsonl"
)


# ── HH-RLHF Parser ───────────────────────────────────────────────────────────

def parse_hh_conversation(text: str) -> tuple:
    """
    Parse Anthropic HH-RLHF format into (prompt, response) tuple.
    Format: '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...'
    We extract the last Human turn as prompt and last Assistant turn as response.
    """
    parts = text.strip().split("\n\nAssistant:")
    if len(parts) < 2:
        return None, None

    response = parts[-1].strip()
    prompt_part = parts[-2]

    human_turns = prompt_part.split("\n\nHuman:")
    if not human_turns:
        return None, None

    last_human = human_turns[-1].strip()
    if last_human.startswith(":"):
        last_human = last_human[1:].strip()

    return last_human, response


def prepare_hh_rlhf(limit: int = 20000, force: bool = False) -> int:
    """Download and convert HH-RLHF to DPO preference pairs."""
    out_path = DPO_DIR / "preference_pairs.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        print(f"  [SKIP] preference_pairs.jsonl already exists ({count} pairs)")
        return count

    raw_path = BASE / "data" / "raw_downloads" / "hh_rlhf_helpful_train.jsonl"

    if not raw_path.exists() or force:
        print(f"  Downloading HH-RLHF helpful subset...")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(
                HH_RLHF_HELPFUL_URL,
                headers={"User-Agent": "TitanAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(raw_path, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while chunk := resp.read(65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r  {downloaded:,}/{total:,} bytes ({downloaded/total*100:.1f}%)", end="", flush=True)
            print()
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            print("  Falling back to synthetic data generation.")
            return prepare_synthetic_pairs(limit=min(limit, 5000))

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with open(raw_path) as inp, open(out_path, "w") as out:
        for line in inp:
            if count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                chosen_text = item.get("chosen", "")
                rejected_text = item.get("rejected", "")

                prompt, chosen = parse_hh_conversation(chosen_text)
                _, rejected = parse_hh_conversation(rejected_text)

                if not prompt or not chosen or not rejected:
                    skipped += 1
                    continue
                if len(chosen.split()) < 10 or len(rejected.split()) < 5:
                    skipped += 1
                    continue
                # Skip if chosen and rejected are too similar
                if chosen.strip()[:100] == rejected.strip()[:100]:
                    skipped += 1
                    continue

                record = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": "hh_rlhf_helpful",
                }
                out.write(json.dumps(record) + "\n")
                count += 1

            except Exception:
                skipped += 1
                continue

    print(f"  [OK] HH-RLHF: {count} pairs written, {skipped} skipped")
    return count


# ── Synthetic Fallback ────────────────────────────────────────────────────────

def prepare_synthetic_pairs(limit: int = 5000, force: bool = False) -> int:
    """
    Generate synthetic DPO pairs from existing SFT data.
    Uses SFT responses as 'chosen' and creates 'rejected' variants by
    truncating, shuffling sentences, or using lower-quality templates.
    This is a fallback when HH-RLHF is unavailable.
    """
    out_path = DPO_DIR / "preference_pairs.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        print(f"  [SKIP] preference_pairs.jsonl exists ({count} pairs)")
        return count

    sft_dir = BASE / "data" / "sft"
    sft_files = list(sft_dir.glob("*.jsonl"))
    if not sft_files:
        print("  [ERROR] No SFT data files found. Run prepare_sft_v2_data.py first.")
        sys.exit(1)

    examples = []
    for path in sft_files:
        with open(path) as f:
            for line in f:
                try:
                    examples.append(json.loads(line.strip()))
                except Exception:
                    pass

    random.shuffle(examples)
    examples = examples[:limit]

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_path, "w") as out:
        for item in examples:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            chosen = item.get("response", "")

            if not instruction or not chosen or len(chosen.split()) < 15:
                continue

            prompt = instruction
            if inp:
                prompt = f"{instruction}\n\n{inp}"

            # Create a clearly worse "rejected" response
            sentences = chosen.split(". ")
            if len(sentences) > 2:
                # Shuffle sentences to create incoherent response
                shuffled = sentences[:]
                random.shuffle(shuffled)
                rejected = ". ".join(shuffled[:max(1, len(shuffled)//2)])
            else:
                # Truncate and add vague filler
                words = chosen.split()
                rejected = " ".join(words[:max(5, len(words)//3)]) + " I'm not sure about the rest."

            record = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "synthetic_sft",
            }
            out.write(json.dumps(record) + "\n")
            count += 1

    print(f"  [OK] Synthetic pairs: {count} generated from SFT data")
    return count


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dpo_file(path: Path) -> bool:
    errors = 0
    count = 0
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                r = json.loads(line.strip())
                assert "prompt" in r and r["prompt"], "missing prompt"
                assert "chosen" in r and r["chosen"], "missing chosen"
                assert "rejected" in r and r["rejected"], "missing rejected"
                assert r["chosen"] != r["rejected"], "chosen == rejected"
                count += 1
            except Exception as e:
                print(f"  ERROR line {i+1}: {e}")
                errors += 1
                if errors > 5:
                    break
    print(f"  Validated {count} pairs, {errors} errors")
    return errors == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare DPO preference dataset")
    parser.add_argument(
        "--source", choices=["hh_rlhf", "synthetic", "auto"], default="auto",
        help="Data source: hh_rlhf (download), synthetic (from SFT data), auto (try hh_rlhf, fall back)"
    )
    parser.add_argument("--limit", type=int, default=20000, help="Max pairs to generate")
    parser.add_argument("--force", action="store_true", help="Re-generate even if file exists")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing file")
    args = parser.parse_args()

    DPO_DIR.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        path = DPO_DIR / "preference_pairs.jsonl"
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)
        ok = validate_dpo_file(path)
        sys.exit(0 if ok else 1)

    print("\n[DPO Data Prep] Building preference dataset...\n")

    if args.source == "hh_rlhf":
        count = prepare_hh_rlhf(limit=args.limit, force=args.force)
    elif args.source == "synthetic":
        count = prepare_synthetic_pairs(limit=args.limit, force=args.force)
    else:  # auto
        count = prepare_hh_rlhf(limit=args.limit, force=args.force)
        if count == 0:
            print("  HH-RLHF unavailable, falling back to synthetic pairs...")
            count = prepare_synthetic_pairs(limit=min(args.limit, 5000), force=args.force)

    out_path = DPO_DIR / "preference_pairs.jsonl"
    print(f"\n{'='*60}")
    print(f"  DPO pairs generated : {count:,}")
    print(f"  Output              : {out_path}")
    print(f"{'='*60}")
    print("\n  Validating output...")
    validate_dpo_file(out_path)
    print("\n  Next: run DPO training:")
    print("  python scripts/run_dpo.py --config configs/titan_dpo_v01.yaml \\")
    print("      --checkpoint checkpoints/sft_v02/final.pt\n")


if __name__ == "__main__":
    main()
