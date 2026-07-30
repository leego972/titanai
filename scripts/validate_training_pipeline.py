#!/usr/bin/env python3
"""Validate TitanAI training configs and launcher compatibility before GPU work."""

import argparse
import inspect
import sys
from pathlib import Path

import yaml

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import TitanConfig, TitanLM


ARCH_KEYS = (
    "vocab_size", "d_model", "n_heads", "n_kv_heads", "n_layers",
    "d_ff", "max_seq_len", "rope_base", "tie_embeddings",
)


def load_yaml(name: str):
    path = BASE / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return path, yaml.safe_load(f)


def signature(cfg):
    model = cfg["model"]
    return tuple((key, model.get(key)) for key in ARCH_KEYS)


def validate_config(path: Path, cfg: dict):
    for section in ("model", "training", "data", "evaluation", "logging"):
        if section not in cfg:
            raise ValueError(f"{path}: missing section {section}")
    model_cfg = TitanConfig.from_dict(cfg)
    model = TitanLM(model_cfg)
    count = sum(p.numel() for p in model.parameters())
    del model
    tokenizer = cfg["data"].get("tokenizer_path") or cfg.get("tokenizer", {}).get("path")
    if not tokenizer:
        raise ValueError(f"{path}: tokenizer path missing")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("1b", "3b", "all"), default="all")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()

    groups = []
    if args.stage in ("1b", "all"):
        groups.append(("1B", "titan_1b.yaml", "titan_1b_instruct.yaml", "titan_1b_dpo.yaml"))
    if args.stage in ("3b", "all"):
        groups.append(("3B", "titan_3b.yaml", "titan_3b_instruct.yaml", "titan_3b_dpo.yaml"))

    failures = []
    for label, pretrain_name, sft_name, dpo_name in groups:
        loaded = [load_yaml(name) for name in (pretrain_name, sft_name, dpo_name)]
        signatures = [signature(cfg) for _, cfg in loaded]
        if not all(item == signatures[0] for item in signatures[1:]):
            failures.append(f"{label}: pretrain/SFT/DPO architecture mismatch")
            continue

        counts = []
        for path, cfg in loaded:
            try:
                counts.append(validate_config(path, cfg))
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")

            if args.require_data:
                data_files = cfg["data"].get("sft_files", []) + cfg["data"].get("dpo_files", [])
                for item in data_files:
                    target = Path(item) if Path(item).is_absolute() else BASE / item
                    if not target.exists():
                        failures.append(f"{path.name}: missing data file {target}")

        if counts:
            print(f"[OK] {label}: architecture consistent; {counts[0]:,} parameters")

    # Verify launcher functions still accept the arguments used by train_3b.sh.
    from scripts import run_sft_v2, run_dpo
    for module, name in ((run_sft_v2, "run_sft_v2"), (run_dpo, "run_dpo")):
        source = inspect.getsource(module.main)
        if "--out-dir" not in source:
            failures.append(f"{name}: --out-dir support missing")

    if failures:
        print("\nTraining pipeline validation FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Training pipeline validation PASSED")


if __name__ == "__main__":
    main()
