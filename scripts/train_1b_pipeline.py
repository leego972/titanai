#!/usr/bin/env python3
"""
TitanAI sequential domain-upgrade pipeline.

Production-readiness changes:
- fail fast on missing/empty datasets
- prefer upgraded dataset variants in this order: *_v2, *_expanded, base
- preserve ordered curriculum and resume semantics
- write the exact selected dataset for each stage into status.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

BASE = Path(__file__).parent.parent
CONFIGS_DIR = BASE / "configs"
TOKENIZER = "tokenizer/artifacts_v32k/tokenizer.json"

UPGRADE_ORDER = [
    "emo", "a", "b", "c", "d", "n", "m", "o", "ak", "ay", "h", "g", "f", "i",
    "bg", "bk", "be", "bx", "by", "bf", "bv", "bw", "bu", "bz", "z", "ar", "as",
    "bl", "bm", "bn", "x", "w", "bp", "ab", "k", "aa", "bo", "ba", "y", "j", "e",
    "l", "ap", "bb", "p", "ao", "v", "at", "br", "bs", "bt", "au", "aq", "aw",
    "bc", "cpp", "asm", "lnx", "cry", "cld", "stg", "ctx", "savant", "cor", "syn",
    "mtr", "rat", "evo", "stag", "win", "mal", "mob", "ctf", "an",
]

CYBER_BUILD_STAGES = {
    "a", "m", "n", "o", "ak", "ay", "f", "g", "h", "i", "be", "bf", "bg", "bk",
    "bu", "bv", "bw", "bx", "by", "bz", "win", "mal", "mob", "ctf", "ctx",
}
CORE_STAGES = {"b", "d", "ar", "as", "bl", "bm", "bn", "z", "c", "an"}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def max_steps_for(stage: str, scale: float) -> int:
    base = 10_000 if stage in CYBER_BUILD_STAGES else 8_000 if stage in CORE_STAGES else 1_500
    return max(1, int(base * scale))


def select_dataset(data_dir: Path, stage: str) -> Path:
    """Prefer the highest-quality available non-empty variant for a stage."""
    candidates = [
        data_dir / f"upgrade_{stage}_v2.jsonl",
        data_dir / f"upgrade_{stage}_expanded.jsonl",
        data_dir / f"upgrade_{stage}.jsonl",
    ]
    existing_empty = []
    for candidate in candidates:
        if candidate.exists():
            if candidate.stat().st_size > 0:
                return candidate
            existing_empty.append(str(candidate))
    detail = f" Empty candidates: {existing_empty}" if existing_empty else ""
    raise FileNotFoundError(
        f"No non-empty upgrade dataset found for stage '{stage}'. Checked: "
        + ", ".join(str(p) for p in candidates)
        + detail
    )


def main():
    parser = argparse.ArgumentParser(description="TitanAI domain-upgrade pipeline")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", default="titan_1b.yaml")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--data-dir", default="data/upgrades")
    args = parser.parse_args()

    base_checkpoint = resolve(args.base_checkpoint)
    output_dir = resolve(args.output_dir)
    model_config_path = resolve(args.model_config)
    data_dir = resolve(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Upgrade data directory not found: {data_dir}")

    # Preflight every stage before spending GPU time.
    selected = {stage: select_dataset(data_dir, stage) for stage in UPGRADE_ORDER}
    print("[upgrade] Dataset preflight passed for all stages")
    for stage in UPGRADE_ORDER:
        p = selected[stage]
        print(f"[upgrade] {stage:>6} -> {p.name} ({p.stat().st_size:,} bytes)")

    with model_config_path.open() as f:
        base_cfg = yaml.safe_load(f)
    model_cfg = dict(base_cfg["model"])
    model_name = base_cfg.get("project", {}).get("name", "titan")
    tokenizer_path = base_cfg.get("tokenizer", {}).get("path", TOKENIZER)

    status_file = output_dir / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
    else:
        status = {"completed_stages": [], "current_stage": None}
    status["selected_datasets"] = {k: str(v) for k, v in selected.items()}

    def save_status():
        status_file.write_text(json.dumps(status, indent=2))

    save_status()
    start_stage = args.resume_from or status.get("current_stage") or UPGRADE_ORDER[0]
    if start_stage not in UPGRADE_ORDER:
        raise ValueError(f"Unknown upgrade stage: {start_stage}")

    previous_checkpoint = base_checkpoint
    start_index = UPGRADE_ORDER.index(start_stage)
    if start_index > 0:
        prior = UPGRADE_ORDER[start_index - 1]
        prior_final = output_dir / f"upgrade_{prior}" / "final.pt"
        if prior_final.exists():
            previous_checkpoint = prior_final

    for stage in UPGRADE_ORDER[start_index:]:
        stage_dir = output_dir / f"upgrade_{stage}"
        final_checkpoint = stage_dir / "final.pt"
        if final_checkpoint.exists():
            print(f"[upgrade] {stage}: already complete")
            previous_checkpoint = final_checkpoint
            if stage not in status["completed_stages"]:
                status["completed_stages"].append(stage)
            continue

        data_file = selected[stage]
        steps = max_steps_for(stage, args.step_scale)
        config = {
            "project": {"name": f"{model_name}-upgrade-{stage}", "version": "1.1.0"},
            "model": model_cfg,
            "tokenizer": {"path": tokenizer_path, "vocab_size": model_cfg["vocab_size"]},
            "data": {
                "sft_files": [str(data_file)],
                "tokenizer_path": tokenizer_path,
                "val_split": 0.05,
                "split_seed": 1337,
                "num_workers": 2,
            },
            "training": {
                "batch_size": 1 if model_cfg["n_layers"] > 40 else 2,
                "gradient_accumulation_steps": 32 if model_cfg["n_layers"] > 40 else 16,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "max_steps": steps,
                "warmup_steps": min(400, max(1, steps // 20)),
                "lr_scheduler": "cosine",
                "lr_min_ratio": 0.1,
                "clip_grad_norm": 1.0,
                "log_interval": 50,
                "eval_interval": min(500, max(50, steps // 10)),
                "save_interval": min(1000, max(100, steps // 5)),
                "checkpoint_dir": str(stage_dir),
            },
            "evaluation": {"val_batch_size": 1, "num_eval_batches": 20},
            "logging": {
                "log_dir": str(stage_dir / "logs"),
                "experiment_name": f"{model_name}-upgrade-{stage}",
            },
        }

        generated_config = CONFIGS_DIR / f"generated_{model_name.replace('-', '_')}_upgrade_{stage}.yaml"
        generated_config.write_text(yaml.safe_dump(config, sort_keys=False))

        command = [
            sys.executable, "-u", str(BASE / "scripts" / "run_upgrade.py"),
            "--config", str(generated_config),
            "--checkpoint", str(previous_checkpoint),
        ]
        step_checkpoints = sorted(
            stage_dir.glob("step_*.pt"),
            key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)),
        ) if stage_dir.exists() else []
        if step_checkpoints:
            command += ["--resume", str(step_checkpoints[-1])]

        status["current_stage"] = stage
        status["current_dataset"] = str(data_file)
        save_status()
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })

        for attempt in range(1, 4):
            print(f"[upgrade] Starting {stage} with {data_file.name} (attempt {attempt}/3)")
            result = subprocess.run(command, cwd=str(BASE), env=env)
            if result.returncode == 0 and final_checkpoint.exists():
                previous_checkpoint = final_checkpoint
                if stage not in status["completed_stages"]:
                    status["completed_stages"].append(stage)
                save_status()
                break
            if attempt == 3:
                raise RuntimeError(f"Upgrade stage {stage} failed after three attempts")
            time.sleep(30)

    status["current_stage"] = None
    status["current_dataset"] = None
    status["final_checkpoint"] = str(previous_checkpoint)
    save_status()
    print(f"[upgrade] Complete. Final checkpoint: {previous_checkpoint}")


if __name__ == "__main__":
    main()
