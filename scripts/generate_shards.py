"""
TitanAI Dataset Shard Generator
=================================
Tokenizes the full corpus with weighted sampling to enforce approved ratios,
then writes binary .npy shards and a manifest.json.

Approved ratios (Probe v0.1.5 target: 1B tokens):
  A: General Language       35%
  B: Reasoning / Planning   15%
  C: Technical / Systems    20%
  D: Cybersecurity          20%
  E: Film / Cinema          10%

The pipeline oversamples underweight buckets and undersamples overweight ones
to hit these ratios regardless of raw document counts.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

import argparse as _ap
  import os as _os

  _ap_parser = _ap.ArgumentParser(description="TitanAI shard generator")
  _ap_parser.add_argument("--config",  default=None,
                           help="YAML config to read max_seq_len and tokenizer path from")
  _ap_parser.add_argument("--tokens",  type=int, default=None,
                           help="Target token count (overrides hardcoded 100M default)")
  _ap_parser.add_argument("--shard-size", type=int, default=None,
                           help="Tokens per shard file (default: 10M)")
  _ap_args, _ = _ap_parser.parse_known_args()

  # Read YAML config if provided
  _yaml_cfg = {}
  if _ap_args.config:
      import yaml as _yaml
      with open(_ap_args.config) as _f: _yaml_cfg = _yaml.safe_load(_f)

  
BASE      = Path(__file__).parent.parent
RAW       = BASE / "data" / "raw"
SHARDS_DIR = BASE / "data" / "shards"
SHARDS_DIR.mkdir(parents=True, exist_ok=True)

_yaml_tok = _yaml_cfg.get("data", {}).get("tokenizer_path")
  TOK_PATH  = (BASE / _yaml_tok) if _yaml_tok else (BASE / "tokenizer" / "titan_32k" / "tokenizer.json")
MANIFEST  = BASE / "data" / "manifest.json"

# Approved target ratios
BUCKET_RATIOS = {
    "corpus_A_general":   0.35,
    "corpus_B_reasoning": 0.15,
    "corpus_C_technical": 0.20,
    "corpus_D_cyber":     0.20,
    "corpus_E_cinema":    0.10,
}

# Probe target: 100M tokens (representative sample for sandbox run)
# Full Probe (1B) requires GPU — this generates a proportional sample
# TARGET_TOKENS: use --tokens CLI arg, or YAML training.max_steps * eff_batch_size, or 100M default
  _yaml_tr = _yaml_cfg.get("training", {})
  _yaml_m  = _yaml_cfg.get("model", {})
  if _ap_args.tokens:
      TARGET_TOKENS = _ap_args.tokens
  elif _yaml_tr.get("max_steps") and _yaml_tr.get("batch_size") and _yaml_tr.get("gradient_accumulation_steps"):
      _eff_batch = _yaml_tr["batch_size"] * _yaml_tr["gradient_accumulation_steps"] * _yaml_m.get("max_seq_len", 2048)
      TARGET_TOKENS = _yaml_tr["max_steps"] * _eff_batch
      print(f"[Shards] Token target from config: {TARGET_TOKENS:,} "
            f"({_yaml_tr['max_steps']:,} steps × {_eff_batch:,} tokens/step)")
  else:
      TARGET_TOKENS = 100_000_000   # 100M default (sandbox/test only)
      print("[Shards] WARNING: using 100M default. Pass --tokens 10000000000 for 10B token run.")
SHARD_SIZE      = _ap_args.shard_size or 10_000_000  # 10M tokens per shard default
MAX_SEQ_LEN     = _yaml_m.get("max_seq_len", 2048)  # reads from --config if provided
RANDOM_SEED     = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_tokenizer():
    print(f"[Shards] Loading tokenizer from {TOK_PATH}")
    tok = Tokenizer.from_file(str(TOK_PATH))
    print(f"[Shards] Tokenizer vocab size: {tok.get_vocab_size()}")
    return tok


def collect_files_by_bucket():
    files_by_bucket = {}
    for bucket_name in BUCKET_RATIOS:
        bucket_dir = RAW / bucket_name
        files = sorted(bucket_dir.glob("*.txt"))
        files_by_bucket[bucket_name] = files
        print(f"[Shards]   {bucket_name}: {len(files)} files")
    return files_by_bucket


def compute_target_tokens_per_bucket(total_tokens):
    targets = {}
    for bucket, ratio in BUCKET_RATIOS.items():
        targets[bucket] = int(total_tokens * ratio)
    return targets


def tokenize_file(tok, filepath, max_len=MAX_SEQ_LEN):
    """Tokenize a single file, return list of token ID sequences."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if not text or len(text) < 50:
            return []
        enc = tok.encode(text)
        ids = enc.ids
        # Split into max_len chunks
        chunks = []
        for i in range(0, len(ids), max_len):
            chunk = ids[i:i + max_len]
            if len(chunk) >= 32:  # discard very short chunks
                chunks.append(chunk)
        return chunks
    except Exception:
        return []


def generate_shards():
    tok = load_tokenizer()
    files_by_bucket = collect_files_by_bucket()
    targets = compute_target_tokens_per_bucket(TARGET_TOKENS)

    print(f"\n[Shards] Target: {TARGET_TOKENS:,} tokens total")
    for b, t in targets.items():
        print(f"[Shards]   {b}: {t:,} tokens ({BUCKET_RATIOS[b]:.0%})")

    # Build a weighted file list: for each bucket, cycle through files
    # until we hit the token target for that bucket
    all_token_sequences = []
    bucket_stats = {}

    for bucket_name, target_tok in targets.items():
        files = list(files_by_bucket[bucket_name])
        if not files:
            print(f"[Shards] WARNING: {bucket_name} is empty — skipping")
            bucket_stats[bucket_name] = {"files_used": 0, "tokens": 0}
            continue

        random.shuffle(files)
        tokens_collected = 0
        files_used = 0
        file_idx = 0

        print(f"\n[Shards] Processing {bucket_name} (target: {target_tok:,} tokens)...")

        while tokens_collected < target_tok:
            f = files[file_idx % len(files)]
            file_idx += 1
            chunks = tokenize_file(tok, f)
            for chunk in chunks:
                all_token_sequences.append(chunk)
                tokens_collected += len(chunk)
                if tokens_collected >= target_tok:
                    break
            files_used += 1
            if file_idx > len(files) * 20:  # safety: max 20x oversampling
                print(f"[Shards]   WARNING: {bucket_name} hit oversampling limit at {tokens_collected:,} tokens")
                break

        bucket_stats[bucket_name] = {
            "files_used": files_used,
            "tokens": tokens_collected,
            "actual_ratio": tokens_collected / TARGET_TOKENS if TARGET_TOKENS > 0 else 0
        }
        print(f"[Shards]   {bucket_name}: {tokens_collected:,} tokens from {files_used} files")

    # Shuffle all sequences
    print(f"\n[Shards] Shuffling {len(all_token_sequences):,} sequences...")
    random.shuffle(all_token_sequences)

    # Write shards
    total_tokens_written = 0
    shard_idx = 0
    current_shard = []
    current_shard_tokens = 0
    shard_files = []

    for seq in all_token_sequences:
        current_shard.extend(seq)
        current_shard_tokens += len(seq)
        total_tokens_written += len(seq)

        if current_shard_tokens >= SHARD_SIZE:
            shard_path = SHARDS_DIR / f"shard_{shard_idx:04d}.npy"
            arr = np.array(current_shard, dtype=np.uint16)
            np.save(str(shard_path), arr)
            shard_files.append({
                "path": str(shard_path),
                "tokens": current_shard_tokens,
                "shard_idx": shard_idx
            })
            print(f"[Shards] Wrote shard {shard_idx:04d}: {current_shard_tokens:,} tokens → {shard_path.name}")
            shard_idx += 1
            current_shard = []
            current_shard_tokens = 0

    # Write final partial shard
    if current_shard:
        shard_path = SHARDS_DIR / f"shard_{shard_idx:04d}.npy"
        arr = np.array(current_shard, dtype=np.uint16)
        np.save(str(shard_path), arr)
        shard_files.append({
            "path": str(shard_path),
            "tokens": current_shard_tokens,
            "shard_idx": shard_idx
        })
        print(f"[Shards] Wrote shard {shard_idx:04d}: {current_shard_tokens:,} tokens → {shard_path.name}")

    # Write manifest
    manifest = {
        "version": "1.0",
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
        "tokenizer": str(TOK_PATH),
        "vocab_size": tok.get_vocab_size(),
        "target_tokens": TARGET_TOKENS,
        "total_tokens_written": total_tokens_written,
        "shard_size": SHARD_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "random_seed": RANDOM_SEED,
        "approved_ratios": BUCKET_RATIOS,
        "bucket_stats": bucket_stats,
        "shards": shard_files,
        "n_shards": len(shard_files),
        "status": "complete"
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Shards] ===== SHARD GENERATION COMPLETE =====")
    print(f"[Shards] Total tokens written: {total_tokens_written:,}")
    print(f"[Shards] Total shards: {len(shard_files)}")
    print(f"[Shards] Manifest saved: {MANIFEST}")
    print(f"\n[Shards] Actual ratios achieved:")
    for b, stats in bucket_stats.items():
        actual = stats.get("actual_ratio", 0)
        target = BUCKET_RATIOS[b]
        diff = abs(actual - target)
        status = "OK" if diff <= 0.03 else "WARN"
        print(f"  {b:30s}: {actual:.1%} (target {target:.0%}) [{status}]")

    return manifest


if __name__ == "__main__":
    generate_shards()
