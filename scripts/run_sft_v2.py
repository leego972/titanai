#!/usr/bin/env python3
"""
TitanAI — SFT v2 Training Entry Point
=======================================
Instruction fine-tuning for any Titan architecture described by the supplied
config. Supports explicit output-directory overrides for 1B and 3B pipelines.

If data.val_files is provided, validation is loaded from those files and is
never sampled from training data. If data.val_files is absent, the legacy
random train/validation split remains available for backward compatibility.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import random_split
from tokenizers import Tokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import build_model
from training.sft_trainer import train_sft
from data.sft_dataset import TitanSFTDataset


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE / p


def _load_model_state(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    return state.get("model_state_dict", state)


def _resolve_required_files(paths, label):
    resolved = [str(_resolve(p)) for p in paths]
    missing = [p for p in resolved if not Path(p).exists()]
    if missing:
        print(f"\n[ERROR] Missing {label} data files:")
        for item in missing:
            print(f"  - {item}")
        sys.exit(1)
    return resolved


def main():
    parser = argparse.ArgumentParser(description="TitanAI SFT v2 Training")
    parser.add_argument("--config", required=True, help="Path to SFT YAML config")
    parser.add_argument("--checkpoint", required=True, help="Base model checkpoint")
    parser.add_argument("--resume", default=None, help="Resume from SFT checkpoint")
    parser.add_argument("--out-dir", default=None, help="Override training.checkpoint_dir")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)

    required = ("model", "training", "data", "evaluation", "logging")
    missing_sections = [section for section in required if section not in cfg]
    if missing_sections:
        raise KeyError(f"Config missing required sections: {missing_sections}")

    if args.out_dir:
        cfg["training"]["checkpoint_dir"] = str(_resolve(args.out_dir))
        cfg["logging"]["log_dir"] = str(_resolve(args.out_dir) / "logs")

    tokenizer_path = cfg["data"].get("tokenizer_path") or cfg.get("tokenizer", {}).get("path")
    if not tokenizer_path:
        raise KeyError("SFT config must define data.tokenizer_path or tokenizer.path")
    tokenizer_path = _resolve(tokenizer_path)
    cfg["data"]["tokenizer_path"] = str(tokenizer_path)

    sft_files = _resolve_required_files(cfg["data"]["sft_files"], "SFT training")
    cfg["data"]["sft_files"] = sft_files

    val_files_cfg = cfg["data"].get("val_files") or []
    val_files = _resolve_required_files(val_files_cfg, "SFT validation") if val_files_cfg else []
    if val_files:
        cfg["data"]["val_files"] = val_files

    checkpoint = _resolve(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT v2] Device: {device}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    expected_vocab = cfg["model"]["vocab_size"]
    if tokenizer.get_vocab_size() != expected_vocab:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer={tokenizer.get_vocab_size()} "
            f"model={expected_vocab}"
        )

    model = build_model(cfg).to(device)
    model_state = _load_model_state(checkpoint, device)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture does not match the supplied SFT config. "
            f"checkpoint={checkpoint} config={config_path}\n{exc}"
        ) from exc

    print(f"[SFT v2] Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    train_source = TitanSFTDataset(
        jsonl_paths=sft_files,
        tokenizer=tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        verbose=True,
    )
    train_stats = train_source.get_stats()
    print(f"[SFT v2] Training dataset: {json.dumps(train_stats)}")
    if len(train_source) < 2:
        raise ValueError("SFT training dataset must contain at least two usable examples")

    if val_files:
        train_dataset = train_source
        val_dataset = TitanSFTDataset(
            jsonl_paths=val_files,
            tokenizer=tokenizer,
            max_seq_len=cfg["model"]["max_seq_len"],
            verbose=True,
        )
        if len(val_dataset) == 0:
            raise ValueError("Explicit validation dataset contains no usable examples")
        print(
            f"[SFT v2] Explicit split — Train: {len(train_dataset)} | "
            f"Val: {len(val_dataset)}"
        )
    else:
        val_fraction = float(cfg["data"].get("val_split", 0.05))
        val_size = min(len(train_source) - 1, max(1, int(len(train_source) * val_fraction)))
        train_size = len(train_source) - val_size
        generator = torch.Generator().manual_seed(int(cfg["data"].get("split_seed", 1337)))
        train_dataset, val_dataset = random_split(
            train_source, [train_size, val_size], generator=generator
        )
        print(f"[SFT v2] Random split — Train: {train_size} | Val: {val_size}")

    resume = str(_resolve(args.resume)) if args.resume else None
    train_sft(cfg, model, train_dataset, val_dataset, device, resume)

    final_dir = cfg["training"]["checkpoint_dir"]
    print(f"\n[SFT v2] Done. Checkpoints: {final_dir}")


if __name__ == "__main__":
    main()
