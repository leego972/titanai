#!/usr/bin/env python3
"""Fail-fast production preflight for a TitanAI 1B training run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]

FORBIDDEN_TRAIN_NAME_PARTS = (
    "benchmark",
    "/test.jsonl",
    "\\test.jsonl",
)
KNOWN_BAD = {
    "data/upgrades/upgrade_system_design.jsonl",
}


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else BASE / path


def fail(msg: str, errors: list[str]) -> None:
    errors.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def inspect_jsonl(path: Path, errors: list[str]) -> tuple[int, int]:
    rows = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except Exception as exc:
                bad += 1
                if bad <= 5:
                    fail(f"{path}: invalid JSON at line {lineno}: {exc}", errors)
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                bad += 1
                if bad <= 5:
                    fail(f"{path}: line {lineno} missing non-empty messages[]", errors)
                continue
            roles = [m.get("role") for m in msgs if isinstance(m, dict)]
            if "user" not in roles or "assistant" not in roles:
                bad += 1
                if bad <= 5:
                    fail(f"{path}: line {lineno} lacks user+assistant messages", errors)
    return rows, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-config", default="configs/titan_1b_sft_production.yaml")
    ap.add_argument("--model-config", default="titan_1b.yaml")
    ap.add_argument("--check-upgrades", action="store_true", default=True)
    args = ap.parse_args()

    errors: list[str] = []

    sft_cfg_path = resolve(args.sft_config)
    model_cfg_path = resolve(args.model_config)
    for p in (sft_cfg_path, model_cfg_path):
        if not p.exists():
            fail(f"Missing required config: {p}", errors)
    if errors:
        return 2

    sft = yaml.safe_load(sft_cfg_path.read_text())
    model = yaml.safe_load(model_cfg_path.read_text())

    # Architecture consistency.
    sm = sft.get("model", {})
    mm = model.get("model", {})
    keys = ("vocab_size", "d_model", "n_heads", "n_kv_heads", "n_layers", "d_ff", "max_seq_len")
    for key in keys:
        if sm.get(key) != mm.get(key):
            fail(f"Architecture mismatch for {key}: SFT={sm.get(key)} model={mm.get(key)}", errors)
    if not errors:
        ok("SFT architecture matches 1B model config")

    tok = resolve(sft["data"]["tokenizer_path"])
    if not tok.exists() or tok.stat().st_size == 0:
        fail(f"Tokenizer missing/empty: {tok}", errors)
    else:
        ok(f"Tokenizer present: {tok.relative_to(BASE)}")

    train_files = sft.get("data", {}).get("sft_files", [])
    val_files = sft.get("data", {}).get("val_files", [])
    if not train_files:
        fail("Production SFT config has no sft_files", errors)

    train_resolved = set()
    total_rows = 0
    for raw in train_files:
        norm = str(raw).replace("\\", "/")
        lower = norm.lower()
        if any(part in lower for part in FORBIDDEN_TRAIN_NAME_PARTS):
            fail(f"Evaluation/benchmark path present in training list: {raw}", errors)
        if norm in KNOWN_BAD:
            fail(f"Known-bad dataset present in production training: {raw}", errors)
        p = resolve(raw)
        train_resolved.add(p.resolve())
        if not p.exists():
            fail(f"Training dataset missing: {raw}", errors)
            continue
        if p.stat().st_size == 0:
            fail(f"Training dataset empty: {raw}", errors)
            continue
        rows, bad = inspect_jsonl(p, errors)
        total_rows += rows
        if rows == 0:
            fail(f"Training dataset has zero usable rows: {raw}", errors)
        elif bad == 0:
            ok(f"{raw}: {rows:,} rows")

    for raw in val_files:
        p = resolve(raw)
        if p.resolve() in train_resolved:
            fail(f"Validation file is also in training set: {raw}", errors)
        if not p.exists() or p.stat().st_size == 0:
            fail(f"Validation dataset missing/empty: {raw}", errors)
            continue
        rows, bad = inspect_jsonl(p, errors)
        if bad == 0:
            ok(f"Validation held out: {raw} ({rows:,} rows)")

    # Explicitly ensure the inference test set exists but is not trained on.
    test = resolve("data/sft/titan_inference_v2/test.jsonl")
    if not test.exists() or test.stat().st_size == 0:
        fail("Held-out inference test set missing/empty", errors)
    elif test.resolve() in train_resolved:
        fail("Inference test set is contaminated into training", errors)
    else:
        ok("Inference v2 test set remains held out")

    # Upgrade curriculum data preflight using the same preference rule as train_1b_pipeline.py.
    if args.check_upgrades:
        sys.path.insert(0, str(BASE))
        from scripts.train_1b_pipeline import UPGRADE_ORDER, select_dataset
        data_dir = BASE / "data" / "upgrades"
        for stage in UPGRADE_ORDER:
            try:
                selected = select_dataset(data_dir, stage)
                ok(f"upgrade {stage}: {selected.name}")
            except Exception as exc:
                fail(f"upgrade {stage}: {exc}", errors)

    print("\n=== TITAN 1B PREFLIGHT ===")
    print(f"SFT training rows scanned: {total_rows:,}")
    if errors:
        print(f"RESULT: NOT READY — {len(errors)} blocking issue(s)")
        return 1
    print("RESULT: READY FOR GPU TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
