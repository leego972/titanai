"""
Titan Data Preparation Pipeline — v2
======================================
Implements all preprocessing rules approved in the TitanAI Advancement Package v2.

Rules enforced (in order):
    1.  Normalization       — UTF-8 NFC, strip control chars, collapse whitespace
    2.  Length filtering    — drop documents < MIN_WORDS words
    3.  Symbol ratio filter — drop documents where symbols > MAX_SYMBOL_RATIO of chars
    4.  Boilerplate filter  — drop documents containing known boilerplate phrases
    5.  Exact deduplication — SHA-256 hash per document, drop duplicates
    6.  MinHash/LSH         — near-duplicate removal (Jaccard threshold 0.8)
    7.  Metadata headers    — prepend [Source: X | Corpus: Y] to tech/cyber docs
    8.  Chunking            — respect paragraph/natural boundaries, with overlap
    9.  Deterministic split — 98/2 train/val split seeded before tokenization
    10. Manifest generation — version, token counts, ratios, source hashes
    11. Rejected data log   — every dropped document logged with reason

Usage:
    python data/prepare_data.py --config configs/titan_config.yaml [--corpus-version v1.0.0]
"""

import os
import re
import sys
import json
import hashlib
import random
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timezone

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizer.train_tokenizer import load_tokenizer

# ─── MinHash availability ─────────────────────────────────────────────────────
try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    print("[WARNING] datasketch not installed — MinHash/LSH deduplication disabled.")

# ─── Constants ────────────────────────────────────────────────────────────────

BOILERPLATE_PHRASES = [
    "terms of service", "privacy policy", "accept cookies",
    "all rights reserved", "copyright notice", "subscribe to our newsletter",
    "click here to", "follow us on", "share this article",
    "this page uses cookies", "gdpr", "cookie consent",
]

# Corpora that receive metadata headers (technical and cybersecurity)
METADATA_HEADER_CORPORA = {"corpus_C_technical", "corpus_D_cyber"}

CORPUS_HEADER = {
    "corpus_A_general":   "[Source: General | Corpus: A-General]",
    "corpus_B_reasoning": "[Source: Reasoning | Corpus: B-Reasoning]",
    "corpus_C_technical": "[Source: Technical | Corpus: C-Technical]",
    "corpus_D_cyber":     "[Source: Cybersecurity | Corpus: D-Cyber]",
    "corpus_E_cinema":    "[Source: Cinema | Corpus: E-Cinema]",
    "corpus_F_instruct":  "[Source: Instruction | Corpus: F-Instruct]",
}

MINHASH_NUM_PERM   = 128
MINHASH_THRESHOLD  = 0.80    # Jaccard similarity threshold
MIN_WORDS          = 50
MAX_SYMBOL_RATIO   = 0.30
TRAIN_RATIO        = 0.98
RANDOM_SEED        = 42
CHUNK_OVERLAP_TOKS = 64      # token overlap between chunks
SHARD_SIZE         = 10_000  # sequences per .npy shard


# ─── Rule 1: Normalization ────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    # Strip control chars, keep standard whitespace
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t "
    )
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ─── Rules 2–4: Filters ───────────────────────────────────────────────────────

def passes_length_filter(text: str) -> bool:
    return len(text.split()) >= MIN_WORDS


def passes_symbol_ratio_filter(text: str) -> bool:
    if not text:
        return False
    alpha = sum(1 for c in text if c.isalnum() or c.isspace())
    symbol = len(text) - alpha
    return (symbol / len(text)) <= MAX_SYMBOL_RATIO


def passes_boilerplate_filter(text: str) -> bool:
    lower = text.lower()
    return not any(phrase in lower for phrase in BOILERPLATE_PHRASES)


# ─── Rules 5–6: Deduplication ─────────────────────────────────────────────────

def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_minhash(text: str) -> Optional[object]:
    if not MINHASH_AVAILABLE:
        return None
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    words = text.lower().split()
    for i in range(max(1, len(words) - 4)):
        shingle = " ".join(words[i:i + 5])
        m.update(shingle.encode("utf-8"))
    return m


# ─── Rule 7: Metadata headers ─────────────────────────────────────────────────

def add_metadata_header(text: str, corpus_name: str) -> str:
    if corpus_name in METADATA_HEADER_CORPORA:
        header = CORPUS_HEADER.get(corpus_name, "")
        if header:
            return f"{header}\n{text}"
    return text


# ─── Rule 8: Chunking ─────────────────────────────────────────────────────────

def chunk_document(text: str, max_tokens: int, tokenizer) -> List[str]:
    """
    Chunk a document into max_tokens-length pieces, respecting paragraph
    boundaries. Applies CHUNK_OVERLAP_TOKS token overlap between chunks.
    """
    tokens = tokenizer.encode(text).ids
    if len(tokens) <= max_tokens:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current_paras: List[str] = []
    current_tok_count = 0

    for para in paragraphs:
        para_toks = len(tokenizer.encode(para).ids)
        if current_tok_count + para_toks > max_tokens and current_paras:
            chunks.append("\n\n".join(current_paras))
            # Carry over overlap
            overlap_paras: List[str] = []
            overlap_count = 0
            for p in reversed(current_paras):
                pt = len(tokenizer.encode(p).ids)
                if overlap_count + pt <= CHUNK_OVERLAP_TOKS:
                    overlap_paras.insert(0, p)
                    overlap_count += pt
                else:
                    break
            current_paras = overlap_paras
            current_tok_count = overlap_count
        current_paras.append(para)
        current_tok_count += para_toks

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks if chunks else [text]


# ─── Per-corpus processing ────────────────────────────────────────────────────

def process_corpus(
    raw_dir: Path,
    corpus_name: str,
    tokenizer,
    max_seq_len: int,
    rejected_log: List[Dict],
    exact_seen: set,
    lsh: Optional[object],
) -> List[List[int]]:
    corpus_path = raw_dir / corpus_name
    if not corpus_path.exists():
        print(f"  [SKIP] {corpus_name} — directory not found at {corpus_path}")
        return []

    files = sorted(
        list(corpus_path.glob("**/*.txt")) +
        list(corpus_path.glob("**/*.md"))
    )
    print(f"  [{corpus_name}] {len(files)} files")

    bos_id = tokenizer.token_to_id("<bos>") or 1
    eos_id = tokenizer.token_to_id("<eos>") or 2

    all_seqs: List[List[int]] = []
    rejected_count = 0

    for fpath in files:
        rel = str(fpath)
        try:
            raw = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            rejected_log.append({"file": rel, "reason": f"read_error:{e}"})
            rejected_count += 1
            continue

        text = normalize_text(raw)

        if not passes_length_filter(text):
            rejected_log.append({"file": rel, "reason": "too_short"})
            rejected_count += 1
            continue
        if not passes_symbol_ratio_filter(text):
            rejected_log.append({"file": rel, "reason": "high_symbol_ratio"})
            rejected_count += 1
            continue
        if not passes_boilerplate_filter(text):
            rejected_log.append({"file": rel, "reason": "boilerplate"})
            rejected_count += 1
            continue

        h = document_hash(text)
        if h in exact_seen:
            rejected_log.append({"file": rel, "reason": "exact_duplicate"})
            rejected_count += 1
            continue
        exact_seen.add(h)

        if MINHASH_AVAILABLE and lsh is not None:
            mh = build_minhash(text)
            if mh is not None:
                if lsh.query(mh):
                    rejected_log.append({"file": rel, "reason": "near_duplicate"})
                    rejected_count += 1
                    continue
                lsh.insert(h, mh)

        text = add_metadata_header(text, corpus_name)
        chunks = chunk_document(text, max_seq_len - 2, tokenizer)

        for chunk in chunks:
            ids = tokenizer.encode(chunk).ids
            if len(ids) < 4:
                continue
            seq = [bos_id] + ids[: max_seq_len - 2] + [eos_id]
            all_seqs.append(seq)

    print(f"  [{corpus_name}] {len(all_seqs)} sequences kept, {rejected_count} rejected")
    return all_seqs


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(config_path: str, corpus_version: str = "v1.0.0") -> Dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_dir    = Path(config_path).parent.parent
    raw_dir     = base_dir / "data" / "raw"
    proc_dir    = base_dir / "data" / "processed" / corpus_version
    rejected_dir = base_dir / "data" / "rejected"
    tok_dir     = base_dir / cfg["tokenizer"]["save_dir"]
    max_seq_len = cfg["model"]["max_seq_len"]

    for d in [proc_dir / "train", proc_dir / "val", rejected_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[Pipeline] TitanAI data preparation — version {corpus_version}")
    print(f"[Pipeline] Raw dir : {raw_dir}")
    print(f"[Pipeline] Output  : {proc_dir}")

    tokenizer = load_tokenizer(str(tok_dir))
    print(f"[Pipeline] Tokenizer vocab size: {tokenizer.get_vocab_size()}")

    exact_seen: set = set()
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_NUM_PERM) if MINHASH_AVAILABLE else None
    rejected_log: List[Dict] = []

    corpus_buckets = [
        "corpus_A_general",
        "corpus_B_reasoning",
        "corpus_C_technical",
        "corpus_D_cyber",
        "corpus_E_cinema",
    ]

    all_seqs: List[List[int]] = []
    bucket_token_counts: Dict[str, int] = {}

    for bucket in corpus_buckets:
        seqs = process_corpus(raw_dir, bucket, tokenizer, max_seq_len,
                              rejected_log, exact_seen, lsh)
        tok_count = sum(len(s) for s in seqs)
        bucket_token_counts[bucket] = tok_count
        all_seqs.extend(seqs)

    total_tokens = sum(bucket_token_counts.values())
    print(f"\n[Pipeline] Total sequences : {len(all_seqs):,}")
    print(f"[Pipeline] Total tokens    : {total_tokens:,}")

    # Rule 9: Deterministic train/val split
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(all_seqs)))
    rng.shuffle(indices)
    split_idx = int(len(indices) * TRAIN_RATIO)
    train_idx = indices[:split_idx]
    val_idx   = indices[split_idx:]
    print(f"[Pipeline] Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    def save_shards(idx_list: List[int], split: str) -> int:
        tok_total = 0
        shard_num = 0
        for start in range(0, len(idx_list), SHARD_SIZE):
            batch = idx_list[start:start + SHARD_SIZE]
            arr = np.zeros((len(batch), max_seq_len), dtype=np.int32)
            for i, si in enumerate(batch):
                seq = all_seqs[si][:max_seq_len]
                arr[i, :len(seq)] = seq
                tok_total += len(seq)
            path = proc_dir / split / f"shard_{shard_num:04d}.npy"
            np.save(str(path), arr)
            shard_num += 1
        print(f"[Pipeline] {split} shards: {shard_num}")
        return tok_total

    train_tokens = save_shards(train_idx, "train")
    val_tokens   = save_shards(val_idx,   "val")

    # Source hashes for manifest
    source_hashes: Dict[str, str] = {}
    for bucket in corpus_buckets:
        bp = raw_dir / bucket
        if bp.exists():
            files = sorted(list(bp.glob("**/*.txt")) + list(bp.glob("**/*.md")))
            h = hashlib.sha256()
            for f in files:
                h.update(f.read_bytes())
            source_hashes[bucket] = h.hexdigest()

    # Rule 10: Manifest
    manifest = {
        "version": corpus_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing_rules": {
            "min_words": MIN_WORDS,
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "minhash_threshold": MINHASH_THRESHOLD,
            "minhash_active": MINHASH_AVAILABLE,
            "train_ratio": TRAIN_RATIO,
            "random_seed": RANDOM_SEED,
            "chunk_overlap_tokens": CHUNK_OVERLAP_TOKS,
        },
        "token_counts": {
            "total": total_tokens,
            "train": train_tokens,
            "val": val_tokens,
            "by_bucket": bucket_token_counts,
        },
        "token_ratios": {
            b: round(c / total_tokens, 4) if total_tokens > 0 else 0
            for b, c in bucket_token_counts.items()
        },
        "sequence_counts": {
            "total": len(all_seqs),
            "train": len(train_idx),
            "val": len(val_idx),
        },
        "source_hashes": source_hashes,
    }

    manifest_path = proc_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Pipeline] Manifest: {manifest_path}")

    # Rule 11: Rejected log
    rej_path = rejected_dir / f"{corpus_version}_rejected.jsonl"
    with open(rej_path, "w") as f:
        for entry in rejected_log:
            f.write(json.dumps(entry) + "\n")
    print(f"[Pipeline] Rejected log: {rej_path} ({len(rejected_log)} entries)")

    # Token ratio summary
    print("\n[Pipeline] Token ratio summary:")
    for b, c in bucket_token_counts.items():
        r = c / total_tokens if total_tokens > 0 else 0
        print(f"  {b:<30} {c:>12,} tokens  ({r:.1%})")

    print(f"\n[Pipeline] Done. Version {corpus_version} ready.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titan data preparation pipeline")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--corpus-version", default="v1.0.0")
    args = parser.parse_args()
    run_pipeline(args.config, args.corpus_version)
