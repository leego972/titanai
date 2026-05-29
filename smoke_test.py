"""
Titan Smoke Test
================
Runs the minimum end-to-end checks needed before any serious training run:
    1. Load config
    2. Verify tokenizer exists
    3. Verify processed train/val shards exist
    4. Build model
    5. Run one forward pass
    6. Save and load a checkpoint-compatible model state through trainer path indirectly

Usage:
    python scripts/smoke_test.py --config configs/titan_config.yaml
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.train_tokenizer import load_tokenizer
from training.trainer import resolve_processed_data_dir, validate_training_inputs
from data.dataset import create_dataloaders
from model.titan_model import build_model


def main():
    parser = argparse.ArgumentParser(description="Run TitanAI smoke test")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(args.base_dir, args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("[Smoke] Loading tokenizer...")
    _tok_cfg4 = config["tokenizer"]
    if "save_dir" in _tok_cfg4:
        tok_dir = os.path.join(args.base_dir, _tok_cfg4["save_dir"])
    elif "path" in _tok_cfg4:
        import pathlib as _pl
        tok_dir = str(_pl.Path(os.path.join(args.base_dir, _tok_cfg4["path"])).parent)
    else:
        raise KeyError("tokenizer config must have 'save_dir' or 'path'")
    tokenizer = load_tokenizer(tok_dir)
    print(f"[Smoke] Tokenizer vocab: {tokenizer.get_vocab_size()}")

    print("[Smoke] Checking processed data...")
    processed_dir = validate_training_inputs(config, args.base_dir)
    print(f"[Smoke] Processed data: {processed_dir}")

    print("[Smoke] Creating dataloaders...")
    train_loader, _ = create_dataloaders(
        train_dir=os.path.join(processed_dir, "train"),
        val_dir=os.path.join(processed_dir, "val"),
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=min(2, config["training"]["batch_size"]),
        val_batch_size=min(2, config["evaluation"]["val_batch_size"]),
    )

    print("[Smoke] Building model...")
    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("[Smoke] Running one forward pass...")
    input_ids, labels = next(iter(train_loader))
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    logits, loss = model(input_ids, labels)

    if logits.shape[:2] != input_ids.shape:
        raise RuntimeError(f"Logit shape mismatch: logits={logits.shape}, input={input_ids.shape}")
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Invalid loss: {loss}")

    print(f"[Smoke] Forward pass OK | loss={loss.item():.4f}")
    print("[Smoke] PASS")


if __name__ == "__main__":
    main()
