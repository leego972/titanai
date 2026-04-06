"""
Titan Training Entry Point
==========================
Run this script to train TitanLM from scratch or resume from a checkpoint.

Usage:
    # From titan-model/ directory:
    python scripts/train.py

    # With custom config:
    python scripts/train.py --config configs/titan_config.yaml

    # Resume from checkpoint:
    python scripts/train.py --resume checkpoints/step_500.pt

    # Auto-resume from latest checkpoint:
    python scripts/train.py --auto-resume
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.trainer import train
from training.checkpoint import get_latest_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train TitanLM from scratch")
    parser.add_argument("--config", default="configs/titan_config.yaml",
                        help="Path to titan_config.yaml")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory of the titan-model project")
    parser.add_argument("--resume", default=None,
                        help="Path to a specific checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from the latest checkpoint if one exists")
    args = parser.parse_args()

    config_path = os.path.join(args.base_dir, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    resume_from = args.resume
    if args.auto_resume and resume_from is None:
        checkpoint_dir = os.path.join(args.base_dir, config["training"]["checkpoint_dir"])
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            print(f"[Train] Auto-resuming from: {latest}")
            resume_from = latest
        else:
            print("[Train] No existing checkpoint found. Starting from scratch.")

    train(config, resume_from=resume_from, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
