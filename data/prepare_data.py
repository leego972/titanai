"""
Titan Data Pipeline
===================
Ingests raw text corpora, cleans and normalizes them, deduplicates,
splits into train/val sets, tokenizes, and shards into binary files
ready for the training loop.

Accepted input formats:
    - Plain text files (.txt) in data/raw/
    - One document per line, OR free-form multi-line text
    - UTF-8 encoding required

Usage:
    python data/prepare_data.py --config configs/titan_config.yaml

To add new domain-specific data later:
    1. Place new .txt files in data/raw/
    2. Re-run this script — it will re-process everything cleanly.
    3. Retrain the tokenizer if vocabulary needs updating.
"""

import os
import re
import sys
import json
import argparse
import unicodedata
from pathlib import Path
from typing import Iterator

import yaml
import numpy as np

# Add parent dir to path so we can import tokenizer module
sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.train_tokenizer import load_tokenizer


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Apply corpus preprocessing rules:
    - Normalize unicode to NFC
    - Remove non-printable / control characters (except newlines and tabs)
    - Collapse excessive whitespace
    - Normalize line endings
    """
    # Unicode normalization
    text = unicodedata.normalize("NFC", text)
    # Remove non-printable control characters (keep \n, \t, \r)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]", " ", text)
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse excessive spaces (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def is_low_quality(text: str, min_chars: int = 50) -> bool:
    """Filter out very short or near-empty documents."""
    return len(text.strip()) < min_chars


def deduplicate(documents: list[str]) -> list[str]:
    """Simple exact-match deduplication on stripped documents."""
    seen = set()
    unique = []
    for doc in documents:
        key = doc.strip()
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


# ─── Document Ingestion ───────────────────────────────────────────────────────

def load_documents_from_dir(raw_dir: str) -> list[str]:
    """
    Load all .txt files from raw_dir.
    Each file is treated as one or more documents separated by blank lines.
    """
    raw_path = Path(raw_dir)
    all_docs = []
    txt_files = sorted(raw_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    for fpath in txt_files:
        print(f"[Data] Loading {fpath.name}...")
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Split on double newlines to get paragraph-level documents
        paragraphs = re.split(r"\n\s*\n", content)
        for para in paragraphs:
            cleaned = normalize_text(para)
            if not is_low_quality(cleaned):
                all_docs.append(cleaned)

    print(f"[Data] Loaded {len(all_docs)} raw documents from {len(txt_files)} file(s)")
    return all_docs


# ─── Tokenization & Sharding ─────────────────────────────────────────────────

def tokenize_documents(
    documents: list[str],
    tokenizer,
    max_seq_len: int,
    stride: int,
) -> list[list[int]]:
    """
    Tokenize all documents using a sliding window approach.
    Returns a list of token ID sequences, each of length <= max_seq_len.
    """
    all_sequences = []
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    for doc in documents:
        # Encode without adding special tokens (we handle them manually)
        encoded = tokenizer.encode(doc)
        ids = encoded.ids

        # Slide a window over the token IDs
        start = 0
        while start < len(ids):
            chunk = ids[start : start + max_seq_len - 2]  # -2 for BOS/EOS
            if len(chunk) < 4:  # Skip very short chunks
                break
            sequence = [bos_id] + chunk + [eos_id]
            all_sequences.append(sequence)
            start += stride
            if start + stride > len(ids) and start < len(ids):
                break  # Avoid tiny trailing chunks

    return all_sequences


def save_shards(
    sequences: list[list[int]],
    output_dir: str,
    split_name: str,
    shard_size: int,
    max_seq_len: int,
):
    """
    Save tokenized sequences as numpy binary shards.
    Each shard is a numpy array of shape (N, max_seq_len) with int32 dtype.
    Shorter sequences are padded with the pad token ID (0).
    """
    os.makedirs(output_dir, exist_ok=True)
    pad_id = 0  # <pad> is always token 0

    shard_idx = 0
    for i in range(0, len(sequences), shard_size):
        batch = sequences[i : i + shard_size]
        # Pad all sequences to max_seq_len
        padded = np.full((len(batch), max_seq_len), pad_id, dtype=np.int32)
        for j, seq in enumerate(batch):
            length = min(len(seq), max_seq_len)
            padded[j, :length] = seq[:length]

        shard_path = os.path.join(output_dir, f"{split_name}_shard_{shard_idx:04d}.npy")
        np.save(shard_path, padded)
        print(f"[Data] Saved {shard_path} ({len(batch)} sequences)")
        shard_idx += 1

    return shard_idx


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Titan data preparation pipeline")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]
    raw_dir = os.path.join(args.base_dir, data_cfg["raw_dir"])
    processed_dir = os.path.join(args.base_dir, data_cfg["processed_dir"])
    tok_save_dir = os.path.join(args.base_dir, tok_cfg["save_dir"])
    max_seq_len = data_cfg["max_seq_len"]
    stride = data_cfg["stride"]
    shard_size = data_cfg["shard_size"]
    train_split = data_cfg["train_split"]

    # Load tokenizer
    print("[Data] Loading tokenizer...")
    tokenizer = load_tokenizer(tok_save_dir)
    print(f"[Data] Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    # Load and clean documents
    documents = load_documents_from_dir(raw_dir)
    documents = deduplicate(documents)
    print(f"[Data] After deduplication: {len(documents)} documents")

    # Train/val split
    split_idx = int(len(documents) * train_split)
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]
    print(f"[Data] Train: {len(train_docs)} docs | Val: {len(val_docs)} docs")

    # Tokenize
    print("[Data] Tokenizing train set...")
    train_seqs = tokenize_documents(train_docs, tokenizer, max_seq_len, stride)
    print(f"[Data] Train sequences: {len(train_seqs)}")

    print("[Data] Tokenizing val set...")
    val_seqs = tokenize_documents(val_docs, tokenizer, max_seq_len, stride)
    print(f"[Data] Val sequences: {len(val_seqs)}")

    # Save shards
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")
    n_train_shards = save_shards(train_seqs, train_dir, "train", shard_size, max_seq_len)
    n_val_shards = save_shards(val_seqs, val_dir, "val", shard_size, max_seq_len)

    # Save metadata
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "max_seq_len": max_seq_len,
        "stride": stride,
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "n_train_shards": n_train_shards,
        "n_val_shards": n_val_shards,
        "train_dir": train_dir,
        "val_dir": val_dir,
    }
    meta_path = os.path.join(processed_dir, "meta.json")
    os.makedirs(processed_dir, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Data] Metadata saved to {meta_path}")
    print("[Data] Pipeline complete.")


if __name__ == "__main__":
    main()
