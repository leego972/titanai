#!/usr/bin/env python3
"""
TitanAI — DPO Training Entry Point
=====================================
Runs the Direct Preference Optimization alignment pass.

Usage:
  python scripts/run_dpo.py \
      --config configs/titan_dpo_v01.yaml \
      --checkpoint checkpoints/sft_v02/final.pt

  # Resume interrupted DPO run:
  python scripts/run_dpo.py \
      --config configs/titan_dpo_v01.yaml \
      --checkpoint checkpoints/sft_v02/final.pt \
      --resume checkpoints/dpo_v01/step_1500.pt
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import random_split
from tokenizers import Tokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import TitanLM, TitanConfig, build_model
from training.checkpoint import load_checkpoint
from training.dpo_trainer import train_dpo
from data.dpo_dataset import TitanDPODataset


def main():
    parser = argparse.ArgumentParser(description="TitanAI DPO Alignment Training")
    parser.add_argument("--config", required=True, help="Path to titan_dpo_v01.yaml")
    parser.add_argument("--checkpoint", required=True, help="SFT model checkpoint (policy init + reference)")
    parser.add_argument("--resume", default=None, help="Resume from DPO checkpoint")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Validate DPO data exists
    dpo_files = cfg["data"]["dpo_files"]
    missing = [p for p in dpo_files if not Path(BASE / p).exists()]
    if missing:
        print(f"\n[ERROR] Missing DPO data files:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun first: python scripts/prepare_dpo_data.py\n")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DPO] Device: {device}")

    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Build policy model from config dict
    policy = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_state = state.get("model_state_dict", state)
    policy.load_state_dict(model_state, strict=False)
    print(f"[DPO] Policy model loaded: {sum(p.numel() for p in policy.parameters()):,} params")

    # Build frozen reference model (same arch + weights, no grad)
    reference = build_model(cfg).to(device)
    reference.load_state_dict(model_state, strict=False)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    print(f"[DPO] Reference model loaded (frozen)")

    # Resume policy from DPO checkpoint if provided
    if args.resume and Path(args.resume).exists():
        print(f"[DPO] Resuming policy from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load DPO dataset — prepend BASE so paths work from any cwd
    full_dataset = TitanDPODataset(
        jsonl_paths=[str(BASE / p) for p in dpo_files],
        tokenizer=tokenizer,
        max_seq_len=cfg["data"]["max_seq_len"],
        verbose=True,
    )
    stats = full_dataset.get_stats()
    print(f"[DPO] Dataset stats: {stats}")

    if len(full_dataset) == 0:
        print("[ERROR] No DPO pairs loaded. Check data/dpo/preference_pairs.jsonl")
        sys.exit(1)

    val_size = max(10, int(len(full_dataset) * cfg["data"].get("val_split", 0.05)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"[DPO] Train pairs: {train_size} | Val pairs: {val_size}")

    train_dpo(cfg, policy, reference, train_dataset, val_dataset, device)

    print(f"\n[DPO] Done. Final checkpoint: checkpoints/dpo_v01/final.pt")
    print("[DPO] Titan is now instruction-tuned and preference-aligned.")
    print("[DPO] Next: deploy the model via the API server.")


if __name__ == "__main__":
    main()
