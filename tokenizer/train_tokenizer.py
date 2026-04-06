"""
Titan Tokenizer Pipeline
========================
Trains a BPE (Byte-Pair Encoding) tokenizer on a given text corpus.
Saves all artifacts to the configured output directory.

Usage:
    python tokenizer/train_tokenizer.py --config configs/titan_config.yaml

To add new data later:
    Append new text files to data/raw/ and re-run this script.
    The tokenizer will retrain on the full combined corpus.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders, normalizers
from tokenizers.implementations import ByteLevelBPETokenizer


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_corpus_files(raw_dir: str) -> list[str]:
    """Collect all .txt files from the raw data directory."""
    raw_path = Path(raw_dir)
    files = list(raw_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {raw_dir}. "
            "Place your training corpus text files there."
        )
    return [str(f) for f in files]


def train_tokenizer(config: dict, base_dir: str) -> Tokenizer:
    """
    Train a BPE tokenizer on the corpus defined in config.
    Returns the trained Tokenizer object.
    """
    tok_cfg = config["tokenizer"]
    raw_dir = os.path.join(base_dir, config["data"]["raw_dir"])
    save_dir = os.path.join(base_dir, tok_cfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    corpus_files = get_corpus_files(raw_dir)
    print(f"[Tokenizer] Training on {len(corpus_files)} file(s): {corpus_files}")

    special_tokens = tok_cfg["special_tokens"]
    vocab_size = tok_cfg["vocab_size"]
    min_frequency = tok_cfg["min_frequency"]

    # Build tokenizer using HuggingFace tokenizers library
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: NFD unicode normalization + lowercase optional
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
    ])

    # Pre-tokenizer: split on whitespace and punctuation (ByteLevel)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train
    tokenizer.train(files=corpus_files, trainer=trainer)

    # Post-processor: add BOS/EOS tokens automatically
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <sep> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
            ("<sep>", tokenizer.token_to_id("<sep>")),
        ],
    )

    # Save artifacts
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[Tokenizer] Saved tokenizer to {tokenizer_path}")
    print(f"[Tokenizer] Vocabulary size: {tokenizer.get_vocab_size()}")

    # Save vocab and merges separately for inspection
    tokenizer.model.save(save_dir)
    print(f"[Tokenizer] Saved vocab.json and merges.txt to {save_dir}")

    # Save special token IDs for reference
    special_ids = {tok: tokenizer.token_to_id(tok) for tok in special_tokens}
    import json
    with open(os.path.join(save_dir, "special_tokens.json"), "w") as f:
        json.dump(special_ids, f, indent=2)
    print(f"[Tokenizer] Special token IDs: {special_ids}")

    return tokenizer


def load_tokenizer(save_dir: str) -> Tokenizer:
    """Load a previously trained Titan tokenizer from disk."""
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run train_tokenizer.py first."
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def test_tokenizer(tokenizer: Tokenizer):
    """Quick sanity check on the trained tokenizer."""
    test_sentences = [
        "Hello, Titan. How are you?",
        "def train(model, optimizer, data):",
        "The attacker exploited a buffer overflow vulnerability.",
    ]
    print("\n[Tokenizer] Sanity check:")
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Input:   {sentence}")
        print(f"  Tokens:  {encoded.tokens[:15]}{'...' if len(encoded.tokens) > 15 else ''}")
        print(f"  IDs:     {encoded.ids[:15]}{'...' if len(encoded.ids) > 15 else ''}")
        print(f"  Decoded: {decoded}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Train Titan BPE tokenizer")
    parser.add_argument(
        "--config",
        default="configs/titan_config.yaml",
        help="Path to titan_config.yaml",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory of the titan-model project",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = train_tokenizer(config, args.base_dir)
    test_tokenizer(tokenizer)
    print("[Tokenizer] Training complete.")


if __name__ == "__main__":
    main()
