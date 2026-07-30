#!/usr/bin/env python3
"""TitanAI DPO training entry point for any configured Titan architecture."""

import argparse
import copy
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import random_split
from tokenizers import Tokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import build_model
from training.dpo_trainer import train_dpo
from data.dpo_dataset import TitanDPODataset


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def main():
    parser = argparse.ArgumentParser(description="TitanAI DPO training")
    parser.add_argument("--config", required=True, help="Path to DPO YAML config")
    parser.add_argument("--checkpoint", required=True, help="SFT checkpoint")
    parser.add_argument("--resume", default=None, help="Resume policy from DPO checkpoint")
    parser.add_argument("--out-dir", default=None, help="Override training.checkpoint_dir")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    with config_path.open() as f:
        cfg = copy.deepcopy(yaml.safe_load(f))

    required = ("model", "training", "data", "evaluation", "logging")
    missing_sections = [section for section in required if section not in cfg]
    if missing_sections:
        raise KeyError(f"Config missing required sections: {missing_sections}")

    if args.out_dir:
        cfg["training"]["checkpoint_dir"] = str(_resolve(args.out_dir))
        cfg["logging"]["log_dir"] = str(_resolve(args.out_dir) / "logs")

    tokenizer_path = cfg["data"].get("tokenizer_path") or cfg.get("tokenizer", {}).get("path")
    if not tokenizer_path:
        raise KeyError("DPO config must define data.tokenizer_path or tokenizer.path")
    tokenizer_path = _resolve(tokenizer_path)
    cfg["data"]["tokenizer_path"] = str(tokenizer_path)

    dpo_files = [str(_resolve(p)) for p in cfg["data"]["dpo_files"]]
    missing = [p for p in dpo_files if not Path(p).exists()]
    if missing:
        print("\n[ERROR] Missing DPO data files:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    cfg["data"]["dpo_files"] = dpo_files

    checkpoint = _resolve(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SFT checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DPO] Device: {device}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    expected_vocab = cfg["model"]["vocab_size"]
    if tokenizer.get_vocab_size() != expected_vocab:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer={tokenizer.get_vocab_size()} "
            f"model={expected_vocab}"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model_state = state.get("model_state_dict", state)
    ckpt_dtype = next(iter(model_state.values())).dtype

    policy = build_model(cfg)
    reference = build_model(cfg)
    try:
        policy.load_state_dict(model_state, strict=True)
        reference.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the supplied DPO config. "
            f"checkpoint={checkpoint} config={config_path}\n{exc}"
        ) from exc

    policy = policy.to(device).to(ckpt_dtype)
    reference = reference.to(device).to(ckpt_dtype)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    print(f"[DPO] Policy/reference architecture: {sum(p.numel() for p in policy.parameters()):,} params")

    if args.resume:
        resume_path = _resolve(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_state = torch.load(resume_path, map_location=device, weights_only=True)
        policy.load_state_dict(resume_state["model_state_dict"], strict=True)

    full_dataset = TitanDPODataset(
        jsonl_paths=dpo_files,
        tokenizer=tokenizer,
        max_seq_len=cfg["data"]["max_seq_len"],
        verbose=True,
    )
    if len(full_dataset) < 2:
        raise ValueError("DPO dataset must contain at least two usable preference pairs")

    val_fraction = float(cfg["data"].get("val_split", 0.05))
    val_size = min(len(full_dataset) - 1, max(1, int(len(full_dataset) * val_fraction)))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(int(cfg["data"].get("split_seed", 1337)))
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"[DPO] Train pairs: {train_size} | Val pairs: {val_size}")

    train_dpo(cfg, policy, reference, train_dataset, val_dataset, device)
    print(f"\n[DPO] Done. Checkpoints: {cfg['training']['checkpoint_dir']}")


if __name__ == "__main__":
    main()
