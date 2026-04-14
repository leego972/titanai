#!/usr/bin/env python3
"""
Titan AI — SFT Dataset Preparation and Validation
==================================================
Validates, analyses, and prepares the instruction fine-tuning datasets
in data/sft/ for use in the SFT trainer.

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --check          # Validate only, no output
    python scripts/prepare_sft_data.py --stats          # Print dataset statistics
"""

import json
import argparse
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent
SFT_DIR = BASE / "data" / "sft"
REQUIRED_ROLES = {"user", "assistant"}


def validate_example(example: dict, source: str, line_num: int) -> list:
    """Returns a list of error strings. Empty list means valid."""
    errors = []
    if "messages" not in example:
        errors.append(f"{source}:{line_num} — missing 'messages' key")
        return errors
    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append(f"{source}:{line_num} — 'messages' must be a non-empty list")
        return errors
    roles = {m.get("role") for m in messages}
    for required in REQUIRED_ROLES:
        if required not in roles:
            errors.append(f"{source}:{line_num} — missing required role: '{required}'")
    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"{source}:{line_num} message[{i}] — missing 'role'")
        if "content" not in msg:
            errors.append(f"{source}:{line_num} message[{i}] — missing 'content'")
        elif not isinstance(msg["content"], str) or not msg["content"].strip():
            errors.append(f"{source}:{line_num} message[{i}] — 'content' is empty or not a string")
    return errors


def load_and_validate(jsonl_path: Path, verbose: bool = True) -> tuple:
    """Load and validate a JSONL file. Returns (examples, error_count)."""
    examples = []
    error_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ERROR {jsonl_path.name}:{line_num} — JSON parse error: {e}")
                error_count += 1
                continue

            errors = validate_example(example, jsonl_path.name, line_num)
            if errors:
                for err in errors:
                    if verbose:
                        print(f"  ERROR {err}")
                error_count += 1
            else:
                examples.append(example)

    return examples, error_count


def print_stats(examples: list, source_name: str):
    """Print statistics about a dataset."""
    if not examples:
        print("  [empty]")
        return

    role_counts = Counter()
    content_lengths = []
    assistant_lengths = []

    for ex in examples:
        for msg in ex["messages"]:
            role_counts[msg["role"]] += 1
            content_lengths.append(len(msg["content"]))
            if msg["role"] == "assistant":
                assistant_lengths.append(len(msg["content"]))

    print(f"  Examples      : {len(examples)}")
    print(f"  Role counts   : {dict(role_counts)}")
    print(f"  Avg msg length: {sum(content_lengths) // len(content_lengths)} chars")
    if assistant_lengths:
        print(f"  Avg resp length: {sum(assistant_lengths) // len(assistant_lengths)} chars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate only, exit with error code if invalid")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if not SFT_DIR.exists():
        print(f"SFT directory not found: {SFT_DIR}")
        print("Run the training pipeline first to generate data/sft/*.jsonl files")
        return

    jsonl_files = sorted(SFT_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {SFT_DIR}")
        return

    total_examples = 0
    total_errors = 0

    for jsonl_path in jsonl_files:
        print(f"\n{'='*60}")
        print(f"File: {jsonl_path.name}")
        print(f"{'='*60}")

        examples, error_count = load_and_validate(jsonl_path)
        total_examples += len(examples)
        total_errors += error_count

        status = "✓ VALID" if error_count == 0 else f"✗ {error_count} ERRORS"
        print(f"Status: {status} | {len(examples)} valid examples")

        if args.stats or not args.check:
            print_stats(examples, jsonl_path.name)

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_examples} valid examples across {len(jsonl_files)} file(s)")
    if total_errors > 0:
        print(f"ERRORS: {total_errors} — fix before training")
        if args.check:
            raise SystemExit(1)
    else:
        print("All datasets valid. Ready for SFT training.")


if __name__ == "__main__":
    main()
