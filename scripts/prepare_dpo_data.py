#!/usr/bin/env python3
"""
TitanAI — DPO Data Preparation
=================================
Builds a preference dataset (chosen / rejected pairs) for Direct Preference
Optimization (DPO) training.

Sources:
  1. Anthropic HH-RLHF  — Human preference data (helpful/harmless)
     ~170K pairs of (chosen, rejected) responses via HuggingFace datasets lib.
  2. Synthetic pairs from existing SFT data — fallback when HH-RLHF is unavailable.

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
  python scripts/prepare_dpo_data.py --source synthetic
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
DPO_DIR = BASE / "data" / "dpo"


# ── HH-RLHF Parser ───────────────────────────────────────────────────────────

def parse_hh_conversation(text: str):
    """
    Parse Anthropic HH-RLHF format into (prompt, response).
    Format: '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...'
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
    """Download and convert HH-RLHF via HuggingFace datasets library."""
    out_path = DPO_DIR / "preference_pairs.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        if count > 0:
            print(f"  [SKIP] preference_pairs.jsonl already exists ({count} pairs)")
            return count

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] 'datasets' library not installed. pip install datasets")
        return 0

    print("  Loading HH-RLHF helpful-base split via HuggingFace datasets...")
    try:
        ds = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split="train",

        )
    except Exception as e:
        print(f"  [ERROR] Failed to load dataset: {e}")
        return 0

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0

    with open(out_path, "w") as out:
        for item in ds:
            if count >= limit:
                break
            try:
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

                if count % 1000 == 0:
                    print(f"  {count:,} pairs written...", flush=True)

            except Exception:
                skipped += 1
                continue

    print(f"  [OK] HH-RLHF: {count} pairs written, {skipped} skipped")
    return count


# ── Synthetic Fallback ────────────────────────────────────────────────────────

def prepare_synthetic_pairs(limit: int = 5000, force: bool = False) -> int:
    """
    Generate synthetic DPO pairs from existing SFT chat-format data.
    Uses assistant turns as 'chosen' and creates degraded 'rejected' variants.
    """
    out_path = DPO_DIR / "preference_pairs.jsonl"
    if out_path.exists() and not force:
        count = sum(1 for _ in open(out_path))
        if count > 0:
            print(f"  [SKIP] preference_pairs.jsonl exists ({count} pairs)")
            return count

    sft_dir = BASE / "data" / "sft"
    sft_files = list(sft_dir.glob("*.jsonl"))
    if not sft_files:
        print("  [ERROR] No SFT data files found. Run prepare_sft_v2_data.py first.")
        return 0

    examples = []
    for path in sft_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Support both chat format and legacy alpaca format
                    if "messages" in item:
                        messages = item["messages"]
                        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                        if user_msg and asst_msg:
                            examples.append({"prompt": user_msg, "chosen": asst_msg})
                    else:
                        instruction = item.get("instruction", "")
                        inp = item.get("input", "")
                        chosen = item.get("response", item.get("output", ""))
                        if instruction and chosen:
                            prompt = f"{instruction}\n\n{inp}" if inp else instruction
                            examples.append({"prompt": prompt, "chosen": chosen})
                except Exception:
                    pass

    random.shuffle(examples)
    examples = examples[:limit]

    DPO_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_path, "w") as out:
        for item in examples:
            prompt = item["prompt"]
            chosen = item["chosen"]

            if len(chosen.split()) < 15:
                continue

            # Create a degraded "rejected" response
            sentences = chosen.split(". ")
            if len(sentences) > 2:
                shuffled = sentences[:]
                random.shuffle(shuffled)
                rejected = ". ".join(shuffled[:max(1, len(shuffled) // 2)])
            else:
                words = chosen.split()
                rejected = " ".join(words[:max(5, len(words) // 3)]) + " I'm not entirely sure."

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
        help="Data source: hh_rlhf, synthetic, auto (try hh_rlhf then fallback)"
    )
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
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
            count = prepare_synthetic_pairs(limit=min(args.limit, 5000), force=True)

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
