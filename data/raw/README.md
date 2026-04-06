# TitanAI Corpus Structure

This directory holds the raw text corpora for TitanAI pretraining.
Each subdirectory is a corpus bucket with a defined role and token ratio target.

## Bucket Definitions

| Bucket | Role | Target Ratio | Approved Sources |
|---|---|---|---|
| `corpus_A_general` | General language, conversation, world knowledge | 35% | Books3, OpenWebText, Wikipedia, CC-News |
| `corpus_B_reasoning` | Reasoning, planning, summarization, structured output | 15% | OpenHermes reasoning traces, Chain-of-thought datasets, arXiv abstracts |
| `corpus_C_technical` | Software, systems, infrastructure, code | 20% | GitHub (high-star repos), official docs, StackOverflow Q&A |
| `corpus_D_cyber` | Cybersecurity, security architecture | 20% | CVE databases, MITRE ATT&CK, CIS/NIST benchmarks, security whitepapers, defensive playbooks |
| `corpus_E_cinema` | Film and cinema production | 10% | Screenplays, production manuals, cinematography guides |
| `corpus_F_instruct` | Instruction-tuning data (reserved) | Post-pretraining only | Alpaca, ShareGPT, curated Q&A — NOT used in pretraining |

## Intake Rules

1. All files must be `.txt` or `.md`, UTF-8 encoded.
2. One document per file, OR multiple documents separated by blank lines.
3. Do NOT place `corpus_F_instruct` data in any pretraining run. It is reserved for SFT only.
4. After adding files, run `data/prepare_data.py` to regenerate shards and manifest.
5. Each corpus version is immutable once `manifest.json` is generated. Create a new version for new data.

## Versioning

Processed data lives in `data/processed/{version}/` with a `manifest.json` recording:
- Exact token counts and ratios per bucket
- Source file hashes (reproducibility)
- Preprocessing parameters used
- Train/val split counts

Never modify a processed version directory. Always create a new version.
