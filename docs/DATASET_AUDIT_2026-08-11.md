# TitanAI Dataset Audit — 2026-08-11

## Objective

Bring TitanAI's SFT corpus up to the professional/source-grounded quality standard defined in the TitanAI PhD-Level Source Library plan, while preserving the current model's intended capabilities.

This audit is about data correctness, training hygiene, coverage and evaluation separation. It does not add a behavioral-policy/refusal/capability restriction layer.

## Verified P0 issues

### 1. Benchmark leakage in the all-SFT training config

`configs/titan_1b_sft_all.yaml` currently lists `data/upgrades/hackersignal_benchmark.jsonl` as a training file. A benchmark must remain held out or its score becomes unreliable.

Action: remove benchmark/evaluation files from the production training mix and reserve them for evaluation only.

### 2. Broken paths in the all-SFT training config

The config references:

- `data/new_data/hackersignal_10k.jsonl`
- `data/new_data/hackersignal_benchmark.jsonl`

The current repository does not contain `data/new_data/`.

Action: delete stale references rather than silently relying on `TitanSFTDataset` to skip missing files.

### 3. Empty files referenced as training datasets

Verified examples include:

- `data/upgrades/upgrade_a.jsonl` — 0 bytes
- `data/upgrades/upgrade_ac.jsonl` — 0 bytes

Action: remove empty inputs from active training configs. Empty placeholder files may remain in the repository only if clearly documented as placeholders.

### 4. Existing loader silently skips missing data

`TitanSFTDataset` warns and continues when a listed path is absent. This is convenient during development but dangerous for a production training recipe because an intended domain can disappear without making the run fail.

`run_sft_v2.py` already performs a preflight missing-file check; production SFT should use this entry point.

## Verified P1 quality issue

### System-design dataset requires replacement

`data/upgrades/upgrade_system_design.jsonl` is not a serious distributed-systems/system-design corpus. Sample rows contain malformed questions such as corporate-history fragments about Cadence Design Systems and then answer with company-history material. This does not train architecture, scalability, reliability or distributed-system reasoning at the required level.

Status: **REPLACE**, not merely expand.

Replacement coverage should include:

- requirements and workload estimation
- consistency/availability tradeoffs
- replication and partitioning
- queues and event-driven systems
- caching and invalidation
- storage/index design
- rate limiting
- failure modes and graceful degradation
- observability and SLOs
- multi-region architecture
- capacity planning
- migration and rollback design
- incident/root-cause reasoning

Source preference: Google SRE, AWS/Azure/GCP architecture references, PostgreSQL, Redis, Kafka, RabbitMQ, OpenTelemetry, Prometheus, OpenAPI and other primary/official documentation.

## Existing programming data: useful but below final provenance standard

`data/upgrades/upgrade_programming_expanded.jsonl` contains materially useful advanced topics including memory safety, asyncio, Rust ownership, distributed rate limiting, SQL optimization, zero-downtime deployment and Linux scheduling.

However, the sampled rows do not carry source provenance fields and some answers bundle claims that should be independently verified before being treated as gold-standard training data.

Status: **KEEP AS LEGACY / REBUILD AS SOURCE-GROUNDED V2**.

## Titan Inference SFT v2

Current manifest:

- 51,500 total examples
- 46,350 train
- 2,575 validation
- 2,575 test
- 20 reasoning categories
- no behavioral-policy/refusal/capability restriction layer

Strengths:

- broad inference coverage
- explicit category/difficulty/domain metadata
- clean train/validation/test separation
- deterministic generation and checksums

Limitation:

A substantial portion is synthetic/template-generated. It is appropriate as a controlled reasoning SFT/evaluation corpus, but it should not automatically dominate the final production mixture until benchmark gains are demonstrated and the highest-value categories are rebuilt with harder source-grounded cases.

Status: **USE CONTROLLED; UPGRADE TO SOURCE-GROUNDED V3**.

## Training-pipeline correction added in this checkpoint

`run_sft_v2.py` now supports `data.val_files`. When supplied, validation is loaded explicitly instead of being randomly sampled from training data.

New controlled configuration:

`configs/titan_1b_sft_inference_v2.yaml`

It uses only the 10 inference-v2 training shards for gradient updates, uses `validation.jsonl` as explicit validation, and leaves `test.jsonl` untouched for final evaluation.

## Priority replacement / expansion roadmap

### P0 — Training hygiene

1. Remove benchmark/eval files from training.
2. Remove stale/missing paths.
3. Remove empty active inputs.
4. Run `scripts/audit_sft_data.py` before every SFT run.
5. Keep explicit validation/test sets isolated.

### P1 — Rebuild weak core intelligence datasets

1. System design / distributed systems — replace current dataset.
2. Software debugging and code repair — source-grounded professional cases.
3. Git/GitHub/GitHub Actions — real repo workflows, CI diagnosis and recovery.
4. Linux/networking/databases — primary documentation and realistic diagnostics.
5. AI/ML engineering — model architecture, training, evaluation, RAG, MLOps and inference systems.
6. Mathematical reasoning — proof, probability, statistics, optimization and numerical reasoning.
7. Evidence synthesis / root cause — incomplete, conflicting and noisy evidence.

### P2 — Domain depth

1. Cybersecurity — authoritative-source detection, remediation, incident reasoning, secure architecture and authorized-lab validation.
2. Cinema/Virelle — production planning, camera, sound, edit, VFX, budgeting, scheduling, distribution and failure recovery.
3. Business — unit economics, pricing, funnels, operations, forecasting and decision-making under uncertainty.
4. Electronics/GPU/Vast.ai — CUDA, GPU memory, distributed training, containers, instance selection and run diagnosis.
5. Science and broad academic knowledge — math, physics, chemistry, biology, economics, law and cognition from high-quality sources.

## Dataset acceptance gate

A replacement dataset should not be marked production-ready merely because a file exists or reaches a row target. It should pass:

- valid JSONL / schema checks
- no empty rows
- exact + near-duplicate checks
- source/provenance tracking where applicable
- factual verification of sampled rows
- code execution/static verification where applicable
- professional depth review
- held-out benchmark improvement
- no regression toward vague, shallow or fabricated answers

## Commands

Audit the current all-SFT recipe:

```bash
python scripts/audit_sft_data.py --config configs/titan_1b_sft_all.yaml
```

Audit the controlled inference-v2 recipe:

```bash
python scripts/audit_sft_data.py --config configs/titan_1b_sft_inference_v2.yaml
```

Run controlled inference-v2 SFT:

```bash
python scripts/run_sft_v2.py \
  --config configs/titan_1b_sft_inference_v2.yaml \
  --checkpoint checkpoints/final.pt
```

Do not promote the resulting checkpoint solely on training loss. Compare it against the untouched inference-v2 test set and Titan's broader acceptance benchmarks first.
