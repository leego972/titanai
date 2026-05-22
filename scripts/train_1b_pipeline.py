#!/usr/bin/env python3
"""
TitanAI 1B — Full Upgrade Pipeline (Hardened Version)
====================================
Trains a 1.3B parameter Titan model from scratch through all 55 upgrade domains.
Architecture: d_model=2048, n_heads=16, n_layers=24, d_ff=8192, vocab=32k

Features:
- State persistence via status.json
- Strict data validation
- Automatic checkpoint management
- FlashAttention-2 enabled by default
"""

import argparse
import os
import re
import sys
import time
import json
import subprocess
import yaml
from pathlib import Path

BASE = Path(__file__).parent.parent
CKPT_BASE = Path("/workspace/ckpt_1b")
CONFIGS_DIR = BASE / "configs"
LOG_DIR = BASE / "logs" / "titan_1b"
STATUS_FILE = Path("/workspace/status.json")
TOKENIZER = "tokenizer/artifacts_v32k/tokenizer.json"

# ── Ordered upgrade sequence ──────────────────────────────────
UPGRADE_ORDER = ["emo", 
    "a", "b", "c", "d", "n", "m", "o", "ak", "ay", "h", "g", "f", "i", 
    "bg", "bk", "be", "bx", "by", "bf", "bv", "bw", "bu", "bz", 
    "z", "ar", "as", "bl", "bm", "bn", "x", "w", "bp", "ab", "k", "aa", 
    "bo", "ba", "y", "j", "e", "l", "ap", "bb", "p", "ao", "v", "at", 
    "br", "bs", "bt", "au", "aq", "aw", "bc", 
    "cpp", "asm", "lnx", "cry", "cld", "stg", "ctx", "savant", "cor", "syn", "mtr", "rat", "evo", "stag", "win", "mal", "mob", "ctf", 
    "an"
]

# ── 1B model architecture ──────────────────────────────────────
MODEL_CFG = {
    "architecture": "decoder_transformer",
    "vocab_size": 32000,
    "d_model": 2048,
    "n_heads": 16,
    "n_layers": 24,
    "d_ff": 8192,
    "max_seq_len": 1024,
    "dropout": 0.05,
    "tie_embeddings": True,
}

TRAINING_DEFAULTS = {
    "batch_size": 4,  # Optimized for 24GB VRAM with FlashAttention-2
    "gradient_accumulation_steps": 4, # Maintain 16 global batch size (4x4)
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_steps": 8000,
    "warmup_steps": 400,
    "lr_scheduler": "cosine",
    "lr_min_ratio": 0.1,
    "clip_grad_norm": 1.0,
    "log_interval": 50,
    "eval_interval": 500,
    "save_interval": 1000,
    "early_stop_patience": 10,
    "gradient_checkpointing": False,
}

CYBER_BUILD_STAGES = {"a", "m", "n", "o", "ak", "ay", "f", "g", "h", "i", "be", "bf", "bg", "bk", "bu", "bv", "bw", "bx", "by", "bz", "win", "mal", "mob", "ctf", "ctx"}
CORE_STAGES = {"b", "d", "ar", "as", "bl", "bm", "bn", "z", "c", "an"}

STEPS_CYBER_BUILD = 10_000
STEPS_CORE        = 8_000
STEPS_MIN         = 1_500

def get_max_steps(stage: str) -> int:
    if stage in CYBER_BUILD_STAGES: return STEPS_CYBER_BUILD
    if stage in CORE_STAGES: return STEPS_CORE
    return STEPS_MIN

def data_file(stage: str) -> str:
    return f"/workspace/data_upgrades/upgrade_{stage}.jsonl"

def ckpt_dir(stage: str) -> str:
    return f"checkpoints/titan_1b/upgrade_{stage}"

def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f: return json.load(f)
    return {"completed_stages": [], "current_stage": None}

def save_status(status):
    with open(STATUS_FILE, "w") as f: json.dump(status, f, indent=2)

def make_config(stage: str) -> dict:
    max_steps = get_max_steps(stage)
    warmup = min(400, max_steps // 20)
    return {
        "project": {"name": f"titan-1b-upgrade-{stage}", "version": "1.0.0"},
        "model": MODEL_CFG,
        "tokenizer": {"path": TOKENIZER, "vocab_size": 32000},
        "data": {
            "sft_files": [data_file(stage)],
            "tokenizer_path": TOKENIZER,
            "val_split": 0.05,
            "num_workers": 4,
        },
        "training": {
            **TRAINING_DEFAULTS,
            "max_steps": max_steps,
            "warmup_steps": warmup,
            "checkpoint_dir": ckpt_dir(stage),
        },
        "evaluation": {"val_batch_size": 2, "num_eval_batches": 50},
        "logging": {"log_dir": f"logs/titan_1b/upgrade_{stage}", "experiment_name": f"titan-1b-upgrade-{stage}"},
    }

def run_stage(stage: str) -> bool:
    final = BASE / ckpt_dir(stage) / "final.pt"
    if final.exists():
        print(f"[1B] upgrade_{stage}: already complete.")
        return True

    data = Path(data_file(stage))
    if not data.exists():
        print(f"[1B] FATAL: data file missing for upgrade_{stage}: {data}")
        return False

    # Find previous checkpoint
    idx = UPGRADE_ORDER.index(stage)
    prev_ckpt = None
    if idx > 0:
        prev_stage = UPGRADE_ORDER[idx - 1]
        prev_ckpt = BASE / ckpt_dir(prev_stage) / "final.pt"
        if not prev_ckpt.exists():
            print(f"[1B] FATAL: previous stage {prev_stage} checkpoint missing.")
            return False
    else:
        init = CKPT_BASE / "init.pt"
        if init.exists(): prev_ckpt = init

    config_path = CONFIGS_DIR / f"titan_1b_upgrade_{stage}.yaml"
    with open(config_path, "w") as f: yaml.safe_dump(make_config(stage), f)

    cmd = [sys.executable, "-u", str(BASE / "scripts" / "run_upgrade.py"), "--config", str(config_path)]
    if prev_ckpt: cmd += ["--checkpoint", str(prev_ckpt)]
    else: cmd += ["--checkpoint", "/dev/null"]

    # Auto-resume within stage
    stage_ckpt_dir = BASE / ckpt_dir(stage)
    if stage_ckpt_dir.exists():
        step_ckpts = sorted(stage_ckpt_dir.glob("step_*.pt"), key=lambda p: int(re.search(r'step_(\d+)', p.name).group(1)))
        if step_ckpts:
            cmd += ["--resume", str(step_ckpts[-1])]
            print(f"[1B] Resuming {stage} from {step_ckpts[-1].name}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TORCH_COMPILE_DISABLE": "1"})

    for attempt in range(1, 4):
        print(f"[1B] Starting {stage} (Attempt {attempt}/3)")
        res = subprocess.run(cmd, cwd=str(BASE), env=env)
        if res.returncode == 0:
            # Persistence: delete intermediate steps, keep final
            for p in stage_ckpt_dir.glob("step_*.pt"): p.unlink()
            # Cleanup previous stage's final to save space (except AN)
            if idx > 0:
                prev_final = BASE / ckpt_dir(UPGRADE_ORDER[idx-1]) / "final.pt"
                # Keep one previous final for safety? No, disk is tight.
                # Actually, let's keep the very last final.pt.
                pass 
            return True
        print(f"[1B] {stage} FAILED. Retrying in 30s...")
        time.sleep(30)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    status = load_status()
    start_stage = args.resume_from or status.get("current_stage") or UPGRADE_ORDER[0]
    
    if start_stage not in UPGRADE_ORDER:
        print(f"Error: {start_stage} not in order.")
        sys.exit(1)

    start_idx = UPGRADE_ORDER.index(start_stage)
    for stage in UPGRADE_ORDER[start_idx:]:
        status["current_stage"] = stage
        save_status(status)
        if run_stage(stage):
            status["completed_stages"].append(stage)
            save_status(status)
        else:
            print(f"Pipeline failed at {stage}")
            sys.exit(1)

if __name__ == "__main__":
    main()
