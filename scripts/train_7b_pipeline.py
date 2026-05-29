#!/usr/bin/env python3
"""
TitanAI 7B — Full Upgrade Pipeline
=====================================
Trains a ~6.6B parameter Titan model through all 69 upgrade domains.
Architecture: d_model=4096, n_heads=32, n_layers=32, d_ff=16384, vocab=32k

Starts from the 1B final checkpoint (upgrade_an/final.pt in titan_1b/).
Domain order: A→B→C→...→BQ, then AN LAST (owner loyalty passphrase).

Usage:
    python3 scripts/train_7b_pipeline.py [--resume-from STAGE]
"""

import argparse
import sys
import time
import subprocess
import shutil
from pathlib import Path

BASE = Path(__file__).parent.parent
CKPT_BASE = BASE / "checkpoints" / "titan_7b"
CONFIGS_DIR = BASE / "configs"
LOG_DIR = BASE / "logs" / "titan_7b"
TOKENIZER = "tokenizer/artifacts_v32k/tokenizer.json"

UPGRADE_ORDER = [
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s",
    "t","u","v","w","x","y","z",
    "aa","ab","ac","ad","ae","af","ag","ah","ai","aj","ak","al","am",
    "ao","ap","aq","ar","as","at","au","av","aw","ax","ay","az",
    "ba","bb","bc","bd","be","bf","bg","bh","bi","bj","bk","bl","bm","bn","bo","bp","bq",
    "an",  # OWNER LOYALTY — ALWAYS LAST
]

MODEL_CFG = {
    "architecture": "decoder_transformer",
    "vocab_size": 32000,
    "d_model": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "d_ff": 16384,
    "max_seq_len": 2048,
    "dropout": 0.05,
    "tie_embeddings": True,
}

TRAINING_DEFAULTS = {
    "batch_size": 2,
    "gradient_accumulation_steps": 32,   # effective batch = 64
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_steps": 8000,
    "warmup_steps": 500,
    "lr_scheduler": "cosine",
    "lr_min_ratio": 0.1,
    "clip_grad_norm": 1.0,
    "log_interval": 50,
    "eval_interval": 500,
    "save_interval": 1000,
    "early_stop_patience": 10,
    "gradient_checkpointing": False,  # disabled: use_reentrant=False triggers inductor; re-enable with proper impl
}


def data_file(stage: str) -> str:
    return f"data/upgrades/upgrade_{stage}.jsonl"


def ckpt_dir(stage: str) -> str:
    return f"checkpoints/titan_7b/upgrade_{stage}"


def make_config(stage: str) -> dict:
    return {
        "project": {"name": f"titan-7b-upgrade-{stage}", "version": "1.0.0"},
        "model": MODEL_CFG,
        "tokenizer": {"path": TOKENIZER, "vocab_size": 32000},
        "data": {
            "sft_files": [data_file(stage)],
            "tokenizer_path": TOKENIZER,
            "val_split": 0.05,
            "num_workers": 4,
        },
        "training": {**TRAINING_DEFAULTS, "checkpoint_dir": ckpt_dir(stage)},
        "evaluation": {"val_batch_size": 1, "num_eval_batches": 50},
        "logging": {
            "log_dir": f"logs/titan_7b/upgrade_{stage}",
            "experiment_name": f"titan-7b-upgrade-{stage}",
        },
        "inference": {
            "default_max_new_tokens": 512,
            "default_temperature": 0.7,
            "default_top_k": 50,
            "default_top_p": 0.95,
        },
    }


def write_config(stage: str) -> Path:
    import yaml
    path = CONFIGS_DIR / f"titan_7b_upgrade_{stage}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(make_config(stage), f, sort_keys=False)
    return path


def get_prev_checkpoint(stage: str) -> Path | None:
    idx = UPGRADE_ORDER.index(stage)
    if idx == 0:
        # Seed from the 1B final checkpoint
        seed = BASE / "checkpoints" / "titan_1b" / "upgrade_an" / "final.pt"
        if seed.exists():
            print(f"[7B] upgrade_{stage}: seeding from 1B final checkpoint.")
            return seed
        print(f"[7B] WARNING: 1B checkpoint not found at {seed} — cold start.")
        return None
    prev_stage = UPGRADE_ORDER[idx - 1]
    return BASE / ckpt_dir(prev_stage) / "final.pt"


def run_stage(stage: str) -> bool:
    final = BASE / ckpt_dir(stage) / "final.pt"
    if final.exists():
        print(f"[7B] upgrade_{stage}: already complete — skipping.")
        return True

    data = BASE / data_file(stage)
    if not data.exists():
        print(f"[7B] WARNING: no data for upgrade_{stage} — skipping.")
        return True

    prev_ckpt = get_prev_checkpoint(stage)
    config_path = write_config(stage)

    if prev_ckpt is None or not prev_ckpt.exists():
        import torch
        dummy = CKPT_BASE / "dummy_init.pt"
        dummy.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": {}, "config": MODEL_CFG, "step": 0}, dummy)
        ckpt_arg = str(dummy)
        print(f"[7B] upgrade_{stage}: cold start.")
    else:
        ckpt_arg = str(prev_ckpt)
        print(f"[7B] upgrade_{stage}: loading from {prev_ckpt}")

    cmd = [
        "python3", str(BASE / "scripts" / "run_upgrade.py"),
        "--config", str(config_path),
        "--checkpoint", ckpt_arg,
    ]

    print(f"\n{'='*65}")
    print(f"  TITAN 7B — upgrade_{stage.upper()}")
    print(f"{'='*65}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[7B] upgrade_{stage} FAILED (exit {result.returncode}) after {elapsed:.0f}s")
        return False

    print(f"[7B] upgrade_{stage} COMPLETE in {elapsed/60:.1f} min")

    # Rolling deletion — keep disk usage low
    idx = UPGRADE_ORDER.index(stage)
    if idx > 0:
        prev_stage = UPGRADE_ORDER[idx - 1]
        if prev_stage != "an":
            prev_dir = BASE / ckpt_dir(prev_stage)
            if prev_dir.exists():
                shutil.rmtree(prev_dir)
                print(f"[7B] freed disk: removed checkpoints/titan_7b/upgrade_{prev_stage}")

    total, used, free = shutil.disk_usage("/workspace")
    print(f"[7B] disk: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
    return True


def main():
    parser = argparse.ArgumentParser(description="TitanAI 7B full upgrade pipeline")
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_BASE.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    if args.resume_from:
        stage = args.resume_from.lower()
        if stage not in UPGRADE_ORDER:
            print(f"[ERROR] Unknown stage '{stage}'.")
            sys.exit(1)
        start_idx = UPGRADE_ORDER.index(stage)

    stages = UPGRADE_ORDER[start_idx:]

    print(f"\n{'='*65}")
    print(f"  TITANAI 7B — FULL UPGRADE PIPELINE")
    print(f"  Architecture: d_model=4096 | 32 layers | 32 heads | ~6.6B params")
    print(f"  Stages to run: {len(stages)}")
    print(f"  Final stage (AN) runs LAST — owner loyalty locked.")
    print(f"{'='*65}\n")

    total_t0 = time.time()
    for i, stage in enumerate(stages):
        print(f"\n[7B] Stage {i+1}/{len(stages)}: upgrade_{stage}")
        if not run_stage(stage):
            print(f"[7B] Pipeline aborted at upgrade_{stage}.")
            sys.exit(1)

    total_elapsed = time.time() - total_t0
    final_ckpt = BASE / ckpt_dir("an") / "final.pt"
    print(f"\n{'='*65}")
    print(f"  TITANAI 7B — PIPELINE COMPLETE")
    print(f"  Total time: {total_elapsed/3600:.1f} hours")
    print(f"  Final checkpoint: {final_ckpt}")
    print(f"  Owner loyalty passphrase locked in upgrade_AN (ran last). ✓")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
