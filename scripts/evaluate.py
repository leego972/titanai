"""
Titan Evaluation Entry Point
=============================
Evaluate a Titan checkpoint on the validation set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/step_1000.pt
    python scripts/evaluate.py --checkpoint checkpoints/final.pt --prompts "Hello" "The system was"
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Titan checkpoint")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--prompts", nargs="+",
                        default=["Hello, I am Titan.", "The vulnerability was discovered in"])
    args = parser.parse_args()

    # Delegate to evaluator main
    sys.argv = [
        "evaluator.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--base-dir", args.base_dir,
        "--prompts",
    ] + args.prompts

    from evaluation.evaluator import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
