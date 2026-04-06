# TitanAI — Base Model Foundation

TitanAI is an owned, from-scratch language model stack. It is not a wrapper over OpenAI, Anthropic, or any third-party API. Every component — the tokenizer, the model architecture, the training loop, the checkpoints — is built and owned in this project.

This repository contains the complete base-model factory for TitanAI, ready for training, evaluation, inference, and future domain specialization in cybersecurity and secure system design.

---

## Architecture

TitanAI uses a **decoder-only Transformer** (GPT-style) implemented from scratch in PyTorch.

| Component | Choice | Reason |
|---|---|---|
| Tokenizer | BPE (Byte-Pair Encoding) | Proven, efficient, fully owned vocabulary |
| Architecture | Decoder-only Transformer | Stable, well-understood, ideal for causal LM |
| Positional Encoding | Learned absolute embeddings | Simple and effective for initial training |
| Activation | GELU | Standard for modern language models |
| Normalization | Pre-LayerNorm | Better training stability than post-norm |
| Training Objective | Causal Language Modeling (next-token prediction) | Standard for generative base models |
| LR Schedule | Cosine decay with linear warmup | Smooth convergence, cost-efficient |

**Current base model size: ~5.3M parameters** (pipeline validation size). Designed to scale to 1.5B+ parameters by updating `configs/titan_config.yaml`.

---

## Project Structure

```
titan-model/
├── configs/
│   └── titan_config.yaml        # All hyperparameters — edit here, not in code
├── tokenizer/
│   ├── train_tokenizer.py       # Train and save Titan's BPE tokenizer
│   └── artifacts/               # Saved tokenizer files (after training)
├── data/
│   ├── prepare_data.py          # Clean, tokenize, and shard the corpus
│   ├── dataset.py               # PyTorch Dataset loader
│   └── raw/                     # Place .txt corpus files here
├── model/
│   └── titan_model.py           # TitanLM architecture (pure PyTorch)
├── training/
│   ├── trainer.py               # Training loop with grad accumulation + LR schedule
│   └── checkpoint.py            # Save/load full training state
├── evaluation/
│   └── evaluator.py             # Validation loss, perplexity, sample generation
├── inference/
│   └── infer.py                 # Interactive CLI and programmatic inference
├── checkpoints/                 # Saved model checkpoints
├── logs/                        # Training and evaluation logs (CSV)
├── scripts/
│   ├── bootstrap.py             # Full end-to-end pipeline validation
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   └── infer.py                 # Inference entry point
└── docs/
    ├── build_plan.md            # Architecture decisions and build plan
    └── NEXT_TRAINING_PLAN.md    # Roadmap for domain training and scaling
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch tokenizers numpy pyyaml
```

### 2. Run the full bootstrap (downloads seed corpus, trains tokenizer, trains model, evaluates)
```bash
cd titan-model/
python scripts/bootstrap.py
```

### 3. Train on your own data
Place `.txt` files in `data/raw/`, then:
```bash
python tokenizer/train_tokenizer.py --config configs/titan_config.yaml
python data/prepare_data.py --config configs/titan_config.yaml
python scripts/train.py --config configs/titan_config.yaml
```

### 4. Resume an interrupted training run
```bash
python scripts/train.py --auto-resume
# or specify a checkpoint:
python scripts/train.py --resume checkpoints/step_1000.pt
```

### 5. Evaluate a checkpoint
```bash
python scripts/evaluate.py --checkpoint checkpoints/final.pt
```

### 6. Run inference
```bash
# Interactive mode:
python scripts/infer.py --checkpoint checkpoints/final.pt

# Single prompt:
python scripts/infer.py --checkpoint checkpoints/final.pt --prompt "Hello, Titan."
```

---

## Configuration

All model and training parameters are controlled by `configs/titan_config.yaml`. No code changes are needed to adjust model size, context length, batch size, or training steps.

To scale the model, update the `model:` section:
```yaml
model:
  d_model: 2048
  n_heads: 16
  n_layers: 24
  d_ff: 8192
  vocab_size: 32000
  max_seq_len: 2048
```

---

## Checkpoint System

Checkpoints are saved automatically every `save_interval` steps (default: 500) to `checkpoints/`. Each checkpoint contains the full model weights, optimizer state, scheduler state, and training step — enabling exact resume from any point.

```
checkpoints/
├── step_500.pt
├── step_1000.pt
├── step_1500.pt
└── final.pt
```

---

## Roadmap to Domain Specialization

See `docs/NEXT_TRAINING_PLAN.md` for the full roadmap. The high-level path is:

1. **Scale the corpus** — Add Wikipedia, technical documentation, and cybersecurity data to `data/raw/`
2. **Scale the model** — Update `titan_config.yaml` to ~1.5B parameters
3. **Full pre-training run** — Train on the full corpus with GPU infrastructure
4. **Instruction fine-tuning** — Fine-tune on curated prompt-response pairs
5. **Cybersecurity specialization** — Fine-tune on CVE databases, exploit write-ups, and defensive architecture data

---

## Non-Negotiables

- TitanAI does **not** call OpenAI, Anthropic, or any third-party model API
- TitanAI's tokenizer is trained on its own corpus
- TitanAI's weights are initialized from scratch and owned by this project
- Every training run produces real checkpoints that can be resumed and inspected

---

## License

Proprietary — Archibald Titan AI. All rights reserved.
