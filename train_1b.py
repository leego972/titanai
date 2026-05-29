"""
train_1b.py — Launch continued pretraining for the TitanAI 1B model.

This script is the training entrypoint used by titanai_phase2_1b.sh.
It locates the main training script (scripts/train.py) within the repository
and delegates to it via subprocess, so it works regardless of whether it is
called from:

  - The Vast.ai server after git-clone (scripts/train_1b.py at /workspace/titanai)
  - The local Replit workspace (training/scripts/train_1b.py)

Usage:
    python3 training/scripts/train_1b.py \\
        --config    training/configs/titan_1b.yaml \\
        --checkpoint checkpoints/titan_1b_pretrain/init.pt

Requirements:
    - Full git clone of leego972/titanai (scripts/train.py must exist)
    - PyTorch, yaml, and the model/data/training packages from that clone
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """
    Locate the repository root from this script's own path.

    This file lives in a directory named 'scripts':
      - Local workspace : <repo_root>/training/scripts/train_1b.py
      - Server (cloned) : <repo_root>/scripts/train_1b.py

    If the parent of the scripts directory is named 'training', we are one
    level deeper than the cloned layout, so we go up one extra level.
    """
    scripts_dir = Path(__file__).resolve().parent   # …/scripts/
    parent = scripts_dir.parent                      # …/training/ or repo root
    if parent.name == "training":
        return parent.parent   # local layout: repo root is two levels up
    return parent              # server layout: repo root is one level up


def _find_train_py(repo_root: Path) -> Path:
    """
    Find the main training entrypoint (train.py) in the repository.
    Checks the standard server location first, then the local workspace copy.
    """
    candidates = [
        repo_root / "scripts" / "train.py",               # server / git-clone layout
        repo_root / "training" / "scripts" / "train.py",  # future local copy
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]   # return first for the error message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch 1B continued pretraining for TitanAI"
    )
    parser.add_argument("--config", default=None,
                        help="Path to titan_1b.yaml (auto-located if not set)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the init checkpoint from upscale_to_1b.py")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from the latest checkpoint")
    args = parser.parse_args()

    repo_root = _repo_root()
    print(f"[train_1b] Repo root : {repo_root}")

    # ── locate config ─────────────────────────────────────────────────────────
    if args.config:
        config_path = Path(args.config)
    elif (repo_root / "configs" / "titan_1b.yaml").exists():
        config_path = repo_root / "configs" / "titan_1b.yaml"
    else:
        config_path = repo_root / "training" / "configs" / "titan_1b.yaml"

    if not config_path.exists():
        print(f"[train_1b] ERROR: config not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[train_1b] Config    : {config_path}")

    # ── validate checkpoint ───────────────────────────────────────────────────
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"[train_1b] ERROR: checkpoint not found at {ckpt}", file=sys.stderr)
            sys.exit(1)
        print(f"[train_1b] Checkpoint: {ckpt}")
    else:
        ckpt = None

    # ── locate train.py ───────────────────────────────────────────────────────
    train_py = _find_train_py(repo_root)
    if not train_py.exists():
        print(
            f"[train_1b] ERROR: training entrypoint not found at {train_py}\n"
            f"[train_1b]   This script requires a full git clone of leego972/titanai.\n"
            f"[train_1b]   On the Vast.ai server:\n"
            f"[train_1b]     git clone https://github.com/leego972/titanai /workspace/titanai\n"
            f"[train_1b]     cd /workspace/titanai && bash scripts/titanai_phase2_1b.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"[train_1b] Train py  : {train_py}")

    # ── build command ─────────────────────────────────────────────────────────
    cmd = [sys.executable, str(train_py), "--config", str(config_path)]
    if ckpt is not None:
        cmd += ["--resume", str(ckpt)]
    if args.auto_resume:
        cmd.append("--auto-resume")

    print(f"[train_1b] Running   : {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(repo_root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
