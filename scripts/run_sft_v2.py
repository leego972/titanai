#!/usr/bin/env python3
"""
TitanAI — SFT v2 Training Entry Point
=======================================
Improved instruction fine-tuning using Alpaca Cleaned + Dolly 15K.

Usage:
  python scripts/run_sft_v2.py \\
      --config configs/titan_sft_v02.yaml \\
      --checkpoint checkpoints/crucible_v02/final.pt

  # Resume interrupted run:
  python scripts/run_sft_v2.py \\
      --config configs/titan_sft_v02.yaml \\
      --checkpoint checkpoints/crucible_v02/final.pt \\
      --resume checkpoints/sft_v02/step_2000.pt
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import random_split
from tokenizers import Tokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import TitanLM, TitanConfig
from training.sft_trainer import train_sft
from training.checkpoint import build_model
from data.sft_dataset import TitanSFTDataset


def main():
    parser = argparse.ArgumentParser(description="TitanAI SFT v2 Training")
    parser.add_argument("--config", required=True, help="Path to titan_sft_v02.yaml")
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint (crucible output)")
    parser.add_argument("--resume", default=None, help="Resume from SFT checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Validate data files exist
    sft_files = cfg["data"]["sft_files"]
    missing = [p for p in sft_files if not Path(BASE / p).exists()]
    if missing:
        print(f"\n[ERROR] Missing SFT data files:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun first: python scripts/prepare_sft_v2_data.py\n")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT v2] Device: {device}")

    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])
    print(f"[SFT v2] Tokenizer: {tokenizer.get_vocab_size()} tokens")

    # Build and load model
    model_config = TitanConfig.from_dict(cfg)
    model = build_model(model_config).to(device)

    print(f"[SFT v2] Loading base checkpoint: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("Ensure crucible training has completed.")
        sys.exit(1)

    base_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_state = base_state.get("model_state_dict", base_state)
    missing_keys, unexpected = model.load_state_dict(model_state, strict=False)
    if missing_keys:
        print(f"[SFT v2] WARNING: {len(missing_keys)} missing keys")
    print(f"[SFT v2] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load dataset
    full_dataset = TitanSFTDataset(
        jsonl_paths=[str(BASE / p) for p in sft_files],
        tokenizer=tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        verbose=True,
    )
    stats = full_dataset.get_stats()
    print(f"[SFT v2] Dataset: {json.dumps(stats)}")

    if len(full_dataset) == 0:
        print("[ERROR] No training examples loaded.")
        sys.exit(1)

    val_size = max(10, int(len(full_dataset) * cfg["data"].get("val_split", 0.05)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[SFT v2] Train: {train_size} | Val: {val_size}")

    # Run training (reuse existing sft_trainer)
    train_sft(cfg, model, train_dataset, val_dataset, device, args.resume)

    print(f"\n[SFT v2] Done. Checkpoint: checkpoints/sft_v02/final.pt")
    print("[SFT v2] Next: python scripts/run_dpo.py --config configs/titan_dpo_v01.yaml \\")
    print("              --checkpoint checkpoints/sft_v02/final.pt")


if __name__ == "__main__":
    main()
