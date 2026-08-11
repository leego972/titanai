# TitanAI Training Monitor

Updated: 2026-08-11 UTC

## Reasoning dataset checkpoint

A new professional reasoning SFT pack is now on `main`:

- `data/sft/titan_complex_commonsense_v1.jsonl` — 68 curated complex-commonsense examples.
- `data/sft/titan_longitudinal_reflection_v1.jsonl` — 24 curated persistent-context, memory, reflection, belief-revision, and outcome-learning examples.
- `configs/titan_reasoning_sft_v1.yaml` — dedicated conservative SFT configuration.
- `scripts/validate_reasoning_datasets.py` — strict structural, uniqueness, minimum-quality, and category-diversity checks.
- CI runs the reasoning dataset validator before the miniature end-to-end training smoke test.

The new records use TitanSFTDataset-compatible chat format (`messages`: system/user/assistant) and are original curated examples rather than scraped conversational content.

## Previous monitor state

Updated: 2026-06-08 12:56:16 UTC

Monitor restarted — watching existing instance.

Found running instance 40075840 (RTX 4090). Monitoring resumed.
