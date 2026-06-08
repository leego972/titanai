#!/usr/bin/env python3
"""
prep_local_corpus.py — Fast data prep for existing raw corpus files.

Tokenises all *.txt files under data/raw/ using the 32k SentencePiece
tokeniser and writes flat uint16 .bin shards (nanoGPT format) to:

  data/processed/train/shard_0000.bin
  data/processed/val/shard_0000.bin

Use this instead of prepare_data.py when the raw text corpus is already
present on the Vast.ai instance (no need for the multi-hour HF download).

Usage:
  cd /workspace/titanai
  python3 scripts/prep_local_corpus.py
  python3 scripts/prep_local_corpus.py --val_frac 0.03  # smaller val set
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np

REPO    = Path(__file__).parent.parent
RAW_DIR = REPO / "data" / "raw"
OUT_DIR = REPO / "data" / "processed"
TOK_PATH = REPO / "tokenizer" / "artifacts_v32k" / "tokenizer.json"


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOK_PATH))
        print(f"[prep] Tokenizer loaded from {TOK_PATH}")
        print(f"[prep] Vocab size: {tok.get_vocab_size():,}")
        return tok
    except ImportError:
        print("[prep] tokenizers library not found — trying transformers...")
    try:
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=str(TOK_PATH))
        print(f"[prep] Tokenizer loaded via transformers")
        return tok
    except Exception as e:
        print(f"[prep] ERROR loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)


def encode(tok, text: str) -> list[int]:
    """Encode text → list of token IDs regardless of tokenizer type."""
    try:
        return tok.encode(text).ids          # tokenizers.Tokenizer
    except AttributeError:
        return tok.encode(text)              # transformers.PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_frac", type=float, default=0.05,
                        help="Fraction of tokens used for validation (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tok = load_tokenizer()

    # Collect all .txt files under data/raw/, skip README / tiny files
    txt_files = sorted(glob.glob(str(RAW_DIR / "**" / "*.txt"), recursive=True))
    txt_files = [f for f in txt_files if Path(f).stat().st_size > 512]
    if not txt_files:
        print(f"[prep] No .txt files found under {RAW_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[prep] Found {len(txt_files):,} text files under {RAW_DIR}")

    random.seed(args.seed)
    random.shuffle(txt_files)

    all_ids: list[int] = []
    for i, fp in enumerate(txt_files, 1):
        try:
            text = open(fp, errors="replace").read().strip()
            if not text:
                continue
            ids = encode(tok, text)
            all_ids.extend(ids)
            if i % 500 == 0 or i == len(txt_files):
                print(f"  [{i:5d}/{len(txt_files)}] tokens so far: {len(all_ids):,}")
        except Exception as e:
            print(f"  skip {fp}: {e}")

    total = len(all_ids)
    print(f"\n[prep] Total tokens: {total:,}")
    if total < 4096:
        print("[prep] ERROR: fewer than 4 096 tokens — not enough to train", file=sys.stderr)
        sys.exit(1)

    # Train / val split
    n_val   = max(2048, int(total * args.val_frac))
    n_train = total - n_val

    val_ids   = np.array(all_ids[:n_val],   dtype=np.uint16)
    train_ids = np.array(all_ids[n_val:],   dtype=np.uint16)

    # Save flat uint16 .bin shards (nanoGPT format)
    train_dir = OUT_DIR / "train"
    val_dir   = OUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    (val_dir   / "shard_0000.bin").write_bytes(val_ids.tobytes())
    (train_dir / "shard_0000.bin").write_bytes(train_ids.tobytes())

    print(f"[prep] Saved {n_train:,} train tokens  → {train_dir}/shard_0000.bin")
    print(f"[prep] Saved {n_val:,}   val tokens    → {val_dir}/shard_0000.bin")
    print("[prep] Done — run training with: bash scripts/titanai_phase2_1b.sh --auto-resume")


if __name__ == "__main__":
    main()
