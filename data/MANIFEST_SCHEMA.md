# TitanAI Manifest Schema

Every processed corpus version produces a `manifest.json` in `data/processed/{version}/`.
This file is the authoritative record of what was processed and how.

## Schema

```json
{
  "version": "v1.0.0",
  "created_at": "2026-04-06T00:00:00+00:00",
  "preprocessing_rules": {
    "min_words": 50,
    "max_symbol_ratio": 0.30,
    "minhash_threshold": 0.80,
    "minhash_active": true,
    "train_ratio": 0.98,
    "random_seed": 42,
    "chunk_overlap_tokens": 64
  },
  "token_counts": {
    "total": 5000000000,
    "train": 4900000000,
    "val": 100000000,
    "by_bucket": {
      "corpus_A_general":   1750000000,
      "corpus_B_reasoning":  750000000,
      "corpus_C_technical": 1000000000,
      "corpus_D_cyber":     1000000000,
      "corpus_E_cinema":     500000000
    }
  },
  "token_ratios": {
    "corpus_A_general":   0.35,
    "corpus_B_reasoning": 0.15,
    "corpus_C_technical": 0.20,
    "corpus_D_cyber":     0.20,
    "corpus_E_cinema":    0.10
  },
  "sequence_counts": {
    "total": 2441406,
    "train": 2392578,
    "val":     48828
  },
  "source_hashes": {
    "corpus_A_general":   "sha256hex...",
    "corpus_B_reasoning": "sha256hex...",
    "corpus_C_technical": "sha256hex...",
    "corpus_D_cyber":     "sha256hex...",
    "corpus_E_cinema":    "sha256hex..."
  }
}
```

## Gate 0-C Verification Checklist

Before approving a manifest for training:

- [ ] `token_ratios` match the approved targets (A:35%, B:15%, C:20%, D:20%, E:10%) within ±2%
- [ ] `preprocessing_rules.random_seed` is 42 (deterministic split)
- [ ] `preprocessing_rules.train_ratio` is 0.98
- [ ] `source_hashes` are populated for all five buckets
- [ ] `minhash_active` is true (or explicitly approved if false)
- [ ] `sequence_counts.val` > 0 (validation split is non-empty)
- [ ] No bucket has 0 tokens (all buckets contributed data)
