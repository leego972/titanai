#!/usr/bin/env python3
"""Audit TitanAI SFT inputs before training.

Checks:
- missing and empty files referenced by a YAML config
- evaluation/benchmark-looking files accidentally used for training
- invalid JSONL records
- usable chat/instruction record counts
- exact duplicate records within and across training files
- presence of source-grounding metadata where available

This is a data-quality gate, not a content-policy filter.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_WORDS = ("benchmark", "validation", "valid", "val.jsonl", "test.jsonl", "eval")
SOURCE_KEYS = {"source", "source_name", "source_url", "source_chunk_id"}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def usable_record(obj: dict) -> bool:
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        return any(isinstance(m, dict) and m.get("role") == "assistant" and str(m.get("content", "")).strip() for m in messages)
    instruction = str(obj.get("instruction") or obj.get("prompt") or "").strip()
    response = str(obj.get("response") or obj.get("output") or "").strip()
    return bool(instruction and response)


def canonical_hash(obj: dict) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def audit_file(path: Path, global_hashes: dict):
    result = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "rows": 0,
        "usable_rows": 0,
        "invalid_json": 0,
        "exact_duplicates_within": 0,
        "exact_duplicates_cross_file": 0,
        "rows_with_source_metadata": 0,
    }
    if not path.exists() or path.stat().st_size == 0:
        return result

    local = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result["rows"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                result["invalid_json"] += 1
                continue
            if usable_record(obj):
                result["usable_rows"] += 1
            if SOURCE_KEYS.intersection(obj.keys()):
                result["rows_with_source_metadata"] += 1
            h = canonical_hash(obj)
            local[h] += 1
            if h in global_hashes and global_hashes[h] != result["path"]:
                result["exact_duplicates_cross_file"] += 1
            else:
                global_hashes[h] = result["path"]

    result["exact_duplicates_within"] = sum(n - 1 for n in local.values() if n > 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/titan_1b_sft_all.yaml")
    ap.add_argument("--json", action="store_true", help="Print machine-readable report")
    args = ap.parse_args()

    config_path = resolve(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_paths = cfg.get("data", {}).get("sft_files", [])
    val_paths = cfg.get("data", {}).get("val_files", [])

    hashes = {}
    train_results = [audit_file(resolve(p), hashes) for p in train_paths]
    val_results = [audit_file(resolve(p), hashes) for p in val_paths]

    suspicious = [
        r["path"] for r in train_results
        if any(word in r["path"].lower() for word in EVAL_WORDS)
    ]
    missing = [r["path"] for r in train_results + val_results if not r["exists"]]
    empty = [r["path"] for r in train_results + val_results if r["exists"] and r["bytes"] == 0]

    summary = {
        "config": str(config_path.relative_to(ROOT)),
        "training_files": len(train_results),
        "validation_files": len(val_results),
        "training_rows": sum(r["rows"] for r in train_results),
        "training_usable_rows": sum(r["usable_rows"] for r in train_results),
        "validation_rows": sum(r["rows"] for r in val_results),
        "missing_files": missing,
        "empty_files": empty,
        "evaluation_like_files_in_training": suspicious,
        "invalid_json_rows": sum(r["invalid_json"] for r in train_results + val_results),
        "exact_duplicates_within_files": sum(r["exact_duplicates_within"] for r in train_results + val_results),
        "exact_duplicates_cross_file": sum(r["exact_duplicates_cross_file"] for r in train_results + val_results),
        "rows_with_source_metadata": sum(r["rows_with_source_metadata"] for r in train_results + val_results),
    }

    report = {"summary": summary, "training": train_results, "validation": val_results}
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("TitanAI SFT data audit")
    print("======================")
    for key, value in summary.items():
        print(f"{key}: {value}")

    hard_fail = bool(missing or empty or suspicious or summary["invalid_json_rows"])
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
