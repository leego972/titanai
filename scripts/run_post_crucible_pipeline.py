#!/usr/bin/env python3
"""
Titan AI — Post-Crucible Agentic Training Pipeline
Sequentially executes:
  1. SFT (Instruction Tuning)     → sft_train.py + titan_sft_v01.yaml
  2. Tool Use Fine-Tuning         → run_tool_tuning.py + titan_tool_v01.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent

def run_stage(script_name, config_path, extra_args, stage_name):
    print(f"\n{'='*60}")
    print(f"STARTING STAGE: {stage_name.upper()}")
    print(f"{'='*60}\n")

    script_path = BASE / "scripts" / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)

    config_file = BASE / "configs" / config_path
    if not config_file.exists():
        print(f"[ERROR] Config not found: {config_file}")
        sys.exit(1)

    cmd = ["python3", str(script_path), "--config", str(config_file)] + extra_args
    print(f"[CMD] {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[COMPLETE] {stage_name.upper()}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {stage_name.upper()} (exit code: {e.returncode})\n")
        sys.exit(e.returncode)

def find_crucible_checkpoint():
    """Find the latest crucible checkpoint automatically."""
    candidates = [
        BASE / "checkpoints/crucible_v02/final.pt",
        BASE / "checkpoints/probe_v015/final.pt",
        BASE / "checkpoints/crucible_v02",
    ]
    for c in candidates:
        if c.exists():
            if c.is_dir():
                # Find latest .pt in dir
                pts = sorted(c.glob("*.pt"), key=lambda x: x.stat().st_mtime)
                if pts:
                    return str(pts[-1])
            else:
                return str(c)
    return None

def main():
    parser = argparse.ArgumentParser(description="Titan post-Crucible pipeline")
    parser.add_argument("--start-from", choices=["sft", "tool"], default="sft")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Base checkpoint to start SFT from (auto-detected if not set)")
    args = parser.parse_args()

    # Auto-detect checkpoint if not provided
    base_ckpt = args.checkpoint or find_crucible_checkpoint()
    if not base_ckpt:
        print("[ERROR] No base checkpoint found. Pass --checkpoint path/to/final.pt")
        sys.exit(1)
    print(f"[INFO] Using base checkpoint: {base_ckpt}")

    stages = [
        ("sft",  "sft_train.py",       "titan_sft_v01.yaml", ["--checkpoint", base_ckpt], "Instruction Tuning (SFT)"),
        ("tool", "run_tool_tuning.py",  "titan_tool_v01.yaml", [],                          "Tool Use Fine-Tuning"),
    ]

    start_idx = 0
    for i, (sid, *_) in enumerate(stages):
        if sid == args.start_from:
            start_idx = i
            break

    for stage_id, script_name, config_path, extra_args, display_name in stages[start_idx:]:
        run_stage(script_name, config_path, extra_args, display_name)

    print("\n" + "="*60)
    print("ALL POST-CRUCIBLE TRAINING STAGES COMPLETE!")
    print("Titan is now instruction-following and tool-capable.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
