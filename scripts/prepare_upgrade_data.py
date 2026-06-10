#!/usr/bin/env python3
"""
TitanAI — Upgrade Data Preparation
=====================================
Validates and converts upgrade JSONL files into the messages format
required by SFTDataset. Input files must contain either:

  a) Already-formatted messages records:
       {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

  b) Flat q/a records (legacy):
       {"q": "...", "a": "..."}

Output: one validated JSONL per upgrade in data/upgrades/.

Usage:
  python scripts/prepare_upgrade_data.py                  # all upgrades
  python scripts/prepare_upgrade_data.py --upgrade f      # single upgrade
  python scripts/prepare_upgrade_data.py --validate-only  # validate without writing
"""

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
UPGRADES_DIR = BASE / "data" / "upgrades"

# Source JSONL files (relative to repo root)
UPGRADE_SOURCES = {
    letter: BASE / f"data/upgrades/upgrade_{letter}.jsonl"
    for letter in "abcdefghijklmnopqrstuvwxyz"
}


# ---------------------------------------------------------------------------
# Core format function — MUST produce messages format for SFTDataset
# ---------------------------------------------------------------------------

def fmt(question: str, answer: str) -> dict:
    """Convert a Q/A pair to the messages format required by SFTDataset.

    SFTDataset expects:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    NOT the Alpaca format:
        {"instruction": "...", "input": "...", "response": "..."}  # WRONG
    """
    return {
        "messages": [
            {"role": "user",      "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# Record normalisation
# ---------------------------------------------------------------------------

def normalise(record: dict) -> dict | None:
    """Return a messages-format record, or None if the record is malformed."""
    # Already in messages format
    if "messages" in record:
        msgs = record["messages"]
        if (
            isinstance(msgs, list)
            and len(msgs) == 2
            and all("role" in m and "content" in m for m in msgs)
            and msgs[0]["role"] == "user"
            and msgs[1]["role"] == "assistant"
            and msgs[0]["content"].strip()
            and msgs[1]["content"].strip()
        ):
            return record
        return None  # malformed messages block

    # Legacy flat format: {"q": "...", "a": "..."}
    q = record.get("q", "").strip()
    a = record.get("a", "").strip()
    if q and a:
        return fmt(q, a)

    # Legacy instruction format: {"instruction": ..., "response": ...}
    instruction = record.get("instruction", "").strip()
    inp = record.get("input", "").strip()
    response = record.get("response", "").strip()
    if instruction and response:
        question = f"{instruction}\n\n{inp}".strip() if inp else instruction
        return fmt(question, response)

    return None


# ---------------------------------------------------------------------------
# Per-upgrade processing
# ---------------------------------------------------------------------------

def process_upgrade(letter: str, validate_only: bool = False) -> tuple[int, int]:
    """Process a single upgrade. Returns (written, skipped)."""
    src = UPGRADES_DIR / f"upgrade_{letter}.jsonl"
    dst = UPGRADES_DIR / f"upgrade_{letter}.jsonl"  # overwrite in-place

    if not src.exists():
        print(f"  [{letter.upper()}] MISSING: {src}")
        return 0, 0

    records = []
    skipped = 0
    with open(src) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [{letter.upper()}] JSON error on line {i}: {e}")
                skipped += 1
                continue
            record = normalise(raw)
            if record is None:
                print(f"  [{letter.upper()}] Skipped malformed record on line {i}")
                skipped += 1
                continue
            records.append(record)

    print(f"  [{letter.upper()}] {len(records)} valid records, {skipped} skipped")

    if not validate_only and records:
        UPGRADES_DIR.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  [{letter.upper()}] Written to {dst}")

    return len(records), skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare upgrade training data (messages format)")
    parser.add_argument("--upgrade",       type=str, help="Single upgrade letter (a-z)")
    parser.add_argument("--validate-only", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    UPGRADES_DIR.mkdir(parents=True, exist_ok=True)

    letters = [args.upgrade.lower()] if args.upgrade else list("abcdefghijklmnopqrstuvwxyz")

    total_written = 0
    total_skipped = 0

    print(f"\n[Upgrade Data Prep] {'Validating' if args.validate_only else 'Processing'} "
          f"{len(letters)} upgrade(s)...\n")

    for letter in letters:
        written, skipped = process_upgrade(letter, validate_only=args.validate_only)
        total_written += written
        total_skipped += skipped

    print(f"\n{'='*55}")
    print(f"  Total records : {total_written:,}")
    print(f"  Total skipped : {total_skipped:,}")
    print(f"  Format        : messages (SFTDataset-compatible)")
    print(f"{'='*55}\n")

    if total_skipped > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
