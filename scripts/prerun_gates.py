"""
TitanAI Pre-Run Approval Gates
================================
Implements the 6 formal stop/go gates defined in the Advancement Package v2.
ALL gates must pass before any training run is approved to launch.

Gates:
    0-A  Model architecture: dummy forward/backward pass
    0-B  Corpus quality sampling: manual review flag check
    0-C  Manifest: generated, valid, ratios within tolerance
    0-D  Training config: generated and parseable
    0-E  Run budget: explicitly approved via flag or env var
    0-F  Tokenizer + sharding: complete with no corruption

Usage:
    python scripts/prerun_gates.py --config configs/titan_config.yaml \\
                                   --corpus-version v1.0.0 \\
                                   --budget-approved \\
                                   --quality-sampled

Exit codes:
    0  All gates passed — training approved
    1  One or more gates failed — training BLOCKED
"""

import sys
import os
import json
import math
import argparse
import hashlib
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Gate results ─────────────────────────────────────────────────────────────

GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_WARN = "WARN"


def gate_result(gate_id: str, status: str, message: str) -> Dict:
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(status, "?")
    print(f"  [{icon}] Gate {gate_id}: {status} — {message}")
    return {"gate": gate_id, "status": status, "message": message}


# ─── Gate 0-A: Model architecture ─────────────────────────────────────────────

def gate_0a_model_architecture(cfg: Dict) -> Dict:
    """Run a dummy forward/backward pass to confirm model builds and gradients flow."""
    try:
        import torch
        from model.titan_model import build_model

        device = torch.device("cpu")
        model = build_model(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        # Dummy input
        batch_size = 2
        seq_len = min(16, cfg["model"]["max_seq_len"])
        vocab_size = cfg["model"]["vocab_size"]
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels    = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        logits, loss = model(input_ids, labels)
        loss.backward()

        assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch"
        assert not math.isnan(loss.item()), "Loss is NaN"
        assert not math.isinf(loss.item()), "Loss is Inf"

        # Check gradients flow
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, "No gradients computed"
        assert all(not math.isnan(g) for g in grad_norms), "NaN gradients detected"

        return gate_result("0-A", GATE_PASS,
                           f"Model OK — {n_params:,} params, loss={loss.item():.4f}, "
                           f"grad layers={len(grad_norms)}")
    except Exception as e:
        return gate_result("0-A", GATE_FAIL, f"Model forward/backward failed: {e}")


# ─── Gate 0-B: Corpus quality sampling ────────────────────────────────────────

def gate_0b_corpus_quality(quality_sampled: bool, raw_dir: Path) -> Dict:
    """
    Checks that corpus quality sampling has been manually performed.
    Requires --quality-sampled flag OR a quality_review.json file in raw_dir.
    """
    review_file = raw_dir / "quality_review.json"

    if quality_sampled:
        return gate_result("0-B", GATE_PASS,
                           "--quality-sampled flag provided. Manual review confirmed.")

    if review_file.exists():
        try:
            with open(review_file) as f:
                review = json.load(f)
            reviewed_buckets = review.get("reviewed_buckets", [])
            required = ["corpus_A_general", "corpus_B_reasoning",
                        "corpus_C_technical", "corpus_D_cyber", "corpus_E_cinema"]
            missing = [b for b in required if b not in reviewed_buckets]
            if missing:
                return gate_result("0-B", GATE_FAIL,
                                   f"quality_review.json missing buckets: {missing}")
            return gate_result("0-B", GATE_PASS,
                               f"quality_review.json found, all buckets reviewed.")
        except Exception as e:
            return gate_result("0-B", GATE_FAIL, f"quality_review.json parse error: {e}")

    return gate_result("0-B", GATE_FAIL,
                       "No quality review confirmed. Pass --quality-sampled or create "
                       f"data/raw/quality_review.json. See MANIFEST_SCHEMA.md for format.")


# ─── Gate 0-C: Manifest ───────────────────────────────────────────────────────

APPROVED_RATIOS = {
    "corpus_A_general":   0.35,
    "corpus_B_reasoning": 0.15,
    "corpus_C_technical": 0.20,
    "corpus_D_cyber":     0.20,
    "corpus_E_cinema":    0.10,
}
RATIO_TOLERANCE = 0.02  # ±2%


def gate_0c_manifest(proc_dir: Path, corpus_version: str) -> Dict:
    """Validate manifest.json exists and token ratios are within approved tolerances."""
    manifest_path = proc_dir / corpus_version / "manifest.json"

    if not manifest_path.exists():
        return gate_result("0-C", GATE_FAIL,
                           f"manifest.json not found at {manifest_path}. "
                           "Run data/prepare_data.py first.")

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        return gate_result("0-C", GATE_FAIL, f"manifest.json parse error: {e}")

    issues = []

    # Check random seed
    rules = manifest.get("preprocessing_rules", {})
    if rules.get("random_seed") != 42:
        issues.append(f"random_seed={rules.get('random_seed')} (expected 42)")
    if abs(rules.get("train_ratio", 0) - 0.98) > 0.001:
        issues.append(f"train_ratio={rules.get('train_ratio')} (expected 0.98)")

    # Check token ratios
    ratios = manifest.get("token_ratios", {})
    for bucket, target in APPROVED_RATIOS.items():
        actual = ratios.get(bucket, 0.0)
        if abs(actual - target) > RATIO_TOLERANCE:
            issues.append(f"{bucket}: ratio={actual:.3f} (target={target}±{RATIO_TOLERANCE})")

    # Check source hashes populated
    source_hashes = manifest.get("source_hashes", {})
    empty_hashes = [b for b, h in source_hashes.items() if not h]
    if empty_hashes:
        issues.append(f"Empty source hashes: {empty_hashes}")

    # Check val split non-empty
    seq_counts = manifest.get("sequence_counts", {})
    if seq_counts.get("val", 0) == 0:
        issues.append("val sequence count is 0")

    # Check no bucket has 0 tokens
    tok_by_bucket = manifest.get("token_counts", {}).get("by_bucket", {})
    zero_buckets = [b for b, c in tok_by_bucket.items() if c == 0]
    if zero_buckets:
        issues.append(f"Zero-token buckets: {zero_buckets}")

    if issues:
        return gate_result("0-C", GATE_FAIL, "Manifest issues: " + "; ".join(issues))

    total_tokens = manifest.get("token_counts", {}).get("total", 0)
    return gate_result("0-C", GATE_PASS,
                       f"Manifest valid — {total_tokens:,} tokens, "
                       f"version={manifest.get('version')}")


# ─── Gate 0-D: Training config ────────────────────────────────────────────────

REQUIRED_CONFIG_KEYS = [
    ("model", "vocab_size"),
    ("model", "n_layers"),
    ("model", "n_heads"),
    ("model", "d_model"),
    ("model", "max_seq_len"),
    ("training", "learning_rate"),
    ("training", "batch_size"),
    ("training", "max_steps"),
    ("training", "warmup_steps"),
    ("training", "grad_clip"),
    ("tokenizer", "vocab_size"),
    ("data", "raw_dir"),
    ("data", "processed_dir"),
]


def gate_0d_training_config(cfg: Dict, config_path: str) -> Dict:
    """Validate that the training config is complete and parseable."""
    missing = []
    for section, key in REQUIRED_CONFIG_KEYS:
        if section not in cfg or key not in cfg[section]:
            missing.append(f"{section}.{key}")

    if missing:
        return gate_result("0-D", GATE_FAIL,
                           f"Config missing keys: {missing}")

    # Sanity checks
    issues = []
    m = cfg["model"]
    t = cfg["training"]

    if m["d_model"] % m["n_heads"] != 0:
        issues.append(f"d_model ({m['d_model']}) not divisible by n_heads ({m['n_heads']})")
    if t["learning_rate"] <= 0 or t["learning_rate"] > 0.1:
        issues.append(f"learning_rate={t['learning_rate']} looks wrong")
    if t["warmup_steps"] >= t["max_steps"]:
        issues.append(f"warmup_steps >= max_steps")
    if t.get("grad_clip", 0) <= 0:
        issues.append(f"grad_clip must be > 0")

    if issues:
        return gate_result("0-D", GATE_WARN,
                           "Config loaded but has warnings: " + "; ".join(issues))

    return gate_result("0-D", GATE_PASS,
                       f"Config valid — {config_path}")


# ─── Gate 0-E: Budget approval ────────────────────────────────────────────────

def gate_0e_budget_approval(budget_approved: bool) -> Dict:
    """
    Requires explicit budget approval before any training run.
    Pass --budget-approved flag or set TITAN_BUDGET_APPROVED=1 env var.
    """
    env_approved = os.environ.get("TITAN_BUDGET_APPROVED", "").strip() in ("1", "true", "yes")

    if budget_approved or env_approved:
        return gate_result("0-E", GATE_PASS,
                           "Run budget explicitly approved.")

    return gate_result("0-E", GATE_FAIL,
                       "Run budget not approved. Pass --budget-approved flag or set "
                       "TITAN_BUDGET_APPROVED=1 environment variable. "
                       "Review compute/cost estimates in Advancement Package v2 Section 6/7 "
                       "before approving.")


# ─── Gate 0-F: Tokenizer + sharding integrity ─────────────────────────────────

def gate_0f_tokenizer_shards(cfg: Dict, proc_dir: Path, corpus_version: str, base_dir: Path) -> Dict:
    """
    Verify tokenizer loads successfully and at least one shard exists
    with non-zero, non-NaN values.
    """
    issues = []

    # Check tokenizer
    tok_dir = base_dir / cfg["tokenizer"]["save_dir"]
    try:
        from tokenizer.train_tokenizer import load_tokenizer
        tokenizer = load_tokenizer(str(tok_dir))
        vocab_size = tokenizer.get_vocab_size()
        expected_vocab = cfg["tokenizer"]["vocab_size"]
        if vocab_size != expected_vocab:
            issues.append(f"Tokenizer vocab size {vocab_size} != config {expected_vocab}")
        # Quick encode test
        test_enc = tokenizer.encode("hello world")
        if len(test_enc.ids) == 0:
            issues.append("Tokenizer encode returned empty IDs")
    except Exception as e:
        issues.append(f"Tokenizer load failed: {e}")

    # Check shards
    train_shard_dir = proc_dir / corpus_version / "train"
    val_shard_dir   = proc_dir / corpus_version / "val"

    for split, shard_dir in [("train", train_shard_dir), ("val", val_shard_dir)]:
        shards = sorted(shard_dir.glob("shard_*.npy")) if shard_dir.exists() else []
        if not shards:
            issues.append(f"No shards found in {shard_dir}")
            continue
        # Check first shard
        try:
            arr = np.load(str(shards[0]))
            if arr.ndim != 2:
                issues.append(f"{split} shard has wrong shape: {arr.shape}")
            if arr.size == 0:
                issues.append(f"{split} shard is empty")
            if np.any(np.isnan(arr.astype(float))):
                issues.append(f"{split} shard contains NaN values")
            if np.all(arr == 0):
                issues.append(f"{split} shard is all zeros (likely corruption)")
        except Exception as e:
            issues.append(f"{split} shard load failed: {e}")

    if issues:
        return gate_result("0-F", GATE_FAIL, "; ".join(issues))

    n_train = len(sorted(train_shard_dir.glob("shard_*.npy")))
    n_val   = len(sorted(val_shard_dir.glob("shard_*.npy")))
    return gate_result("0-F", GATE_PASS,
                       f"Tokenizer OK (vocab={vocab_size}), "
                       f"shards: train={n_train}, val={n_val}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TitanAI pre-run approval gates")
    parser.add_argument("--config",          default="configs/titan_config.yaml")
    parser.add_argument("--corpus-version",  default="v1.0.0")
    parser.add_argument("--budget-approved", action="store_true",
                        help="Explicitly approve the run budget")
    parser.add_argument("--quality-sampled", action="store_true",
                        help="Confirm manual corpus quality sampling has been done")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  TitanAI Pre-Run Approval Gates")
    print("="*60)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_dir  = Path(args.config).parent.parent
    raw_dir   = base_dir / "data" / "raw"
    proc_dir  = base_dir / "data" / "processed"

    print(f"\n  Config       : {args.config}")
    print(f"  Corpus ver.  : {args.corpus_version}")
    print(f"  Base dir     : {base_dir}\n")

    results: List[Dict] = []

    results.append(gate_0a_model_architecture(cfg))
    results.append(gate_0b_corpus_quality(args.quality_sampled, raw_dir))
    results.append(gate_0c_manifest(proc_dir, args.corpus_version))
    results.append(gate_0d_training_config(cfg, args.config))
    results.append(gate_0e_budget_approval(args.budget_approved))
    results.append(gate_0f_tokenizer_shards(cfg, proc_dir, args.corpus_version, base_dir))

    # Summary
    print("\n" + "="*60)
    failed = [r for r in results if r["status"] == GATE_FAIL]
    warned = [r for r in results if r["status"] == GATE_WARN]
    passed = [r for r in results if r["status"] == GATE_PASS]

    print(f"  PASSED : {len(passed)}/6")
    print(f"  WARNED : {len(warned)}/6")
    print(f"  FAILED : {len(failed)}/6")

    if failed:
        print("\n  *** TRAINING BLOCKED ***")
        print("  The following gates must pass before training can begin:")
        for r in failed:
            print(f"    Gate {r['gate']}: {r['message']}")
        print("="*60 + "\n")
        sys.exit(1)
    else:
        print("\n  *** ALL GATES PASSED — TRAINING APPROVED ***")
        print("="*60 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
