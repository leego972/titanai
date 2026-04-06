"""
Titan Inference Entry Point
============================
Run interactive or single-shot inference with a trained Titan checkpoint.

Usage (interactive):
    python scripts/infer.py --checkpoint checkpoints/final.pt

Usage (single prompt):
    python scripts/infer.py --checkpoint checkpoints/final.pt --prompt "Hello, Titan."
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run Titan inference")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    sys.argv = [
        "infer.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--base-dir", args.base_dir,
    ]
    if args.prompt:
        sys.argv += ["--prompt", args.prompt]
    if args.max_tokens:
        sys.argv += ["--max-tokens", str(args.max_tokens)]
    if args.temperature:
        sys.argv += ["--temperature", str(args.temperature)]
    if args.top_k:
        sys.argv += ["--top-k", str(args.top_k)]

    from inference.infer import main as infer_main
    infer_main()


if __name__ == "__main__":
    main()
