"""
Titan Tokenizer Pipeline
========================
Trains a BPE tokenizer on the configured raw text corpus.
Saves tokenizer.json, vocab.json, merges.txt, and special token IDs.

Usage:
    python tokenizer/train_tokenizer.py --config configs/titan_config.yaml
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders, normalizers


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_corpus_files(raw_dir: str, corpus_buckets: list[str] | None = None) -> list[str]:
    """Collect .txt/.md files recursively from data/raw and known corpus buckets."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files: list[Path] = []
    if corpus_buckets:
        for bucket in corpus_buckets:
            bucket_path = raw_path / bucket
            if bucket_path.exists():
                files.extend(bucket_path.glob("**/*.txt"))
                files.extend(bucket_path.glob("**/*.md"))

    # Compatibility path: include loose files directly under data/raw.
    files.extend(raw_path.glob("*.txt"))
    files.extend(raw_path.glob("*.md"))

    # Fallback: fully recursive scan if the structured buckets are not populated.
    if not files:
        files.extend(raw_path.glob("**/*.txt"))
        files.extend(raw_path.glob("**/*.md"))

    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No .txt/.md files found in {raw_dir}. "
            "Place training corpus files under data/raw/corpus_A_general or another corpus bucket."
        )
    return [str(f) for f in files]


def train_tokenizer(config: dict, base_dir: str) -> Tokenizer:
    tok_cfg = config["tokenizer"]
    raw_dir = os.path.join(base_dir, config["data"]["raw_dir"])
    save_dir = os.path.join(base_dir, tok_cfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    corpus_buckets = config.get("data", {}).get("corpus_buckets")
    corpus_files = get_corpus_files(raw_dir, corpus_buckets)
    print(f"[Tokenizer] Training on {len(corpus_files)} file(s)")
    for fpath in corpus_files[:20]:
        print(f"  - {fpath}")
    if len(corpus_files) > 20:
        print(f"  ... plus {len(corpus_files) - 20} more")

    special_tokens = tok_cfg["special_tokens"]
    vocab_size = tok_cfg["vocab_size"]
    min_frequency = tok_cfg["min_frequency"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train(files=corpus_files, trainer=trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    sep_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <sep> $B:1 <eos>:1",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id), ("<sep>", sep_id)],
    )

    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[Tokenizer] Saved tokenizer to {tokenizer_path}")
    print(f"[Tokenizer] Vocabulary size: {tokenizer.get_vocab_size()}")

    tokenizer.model.save(save_dir)
    print(f"[Tokenizer] Saved vocab.json and merges.txt to {save_dir}")

    special_ids = {tok: tokenizer.token_to_id(tok) for tok in special_tokens}
    with open(os.path.join(save_dir, "special_tokens.json"), "w") as f:
        json.dump(special_ids, f, indent=2)
    print(f"[Tokenizer] Special token IDs: {special_ids}")

    return tokenizer


def load_tokenizer(save_dir: str) -> Tokenizer:
    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run tokenizer/train_tokenizer.py first."
        )
    return Tokenizer.from_file(tokenizer_path)


def test_tokenizer(tokenizer: Tokenizer):
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
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(args.base_dir, args.config)
    config = load_config(config_path)
    tokenizer = train_tokenizer(config, args.base_dir)
    test_tokenizer(tokenizer)
    print("[Tokenizer] Training complete.")


if __name__ == "__main__":
    main()
