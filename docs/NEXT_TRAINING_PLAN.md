# Titan Base Model: Next Training Plan

This document outlines the roadmap for taking the Titan base model from its current pipeline-validation state to a fully trained, domain-specialized language model. The foundation built in Phase 1 provides a complete, modular, and cost-aware stack. The next steps focus on scaling the data, scaling the model, and applying domain-specific fine-tuning.

## 1. What Was Built in Phase 1

The current `titan-model` directory contains a fully functional, end-to-end language model training factory. This is not a wrapper over an external API; it is a custom PyTorch implementation of a decoder-only Transformer.

The following components are fully operational:
- **Tokenizer Pipeline**: A custom Byte-Pair Encoding (BPE) tokenizer that trains directly on the provided corpus, ensuring Titan owns its vocabulary.
- **Data Pipeline**: An ingestion system that cleans, normalizes, deduplicates, tokenizes, and shards raw text into binary `.npy` files for efficient training.
- **Model Architecture**: A stable, Pre-LayerNorm decoder-only Transformer (`TitanLM`) with causal self-attention and GELU activations.
- **Training Loop**: A robust training system featuring gradient accumulation, cosine learning rate scheduling with linear warmup, gradient clipping, and automatic checkpointing.
- **Evaluation Harness**: Scripts to measure validation loss and perplexity, alongside qualitative sample generation to track model coherence.
- **Inference Runtime**: A lightweight serving layer for interactive CLI generation or programmatic API integration.

## 2. How to Operate the Pipeline

The pipeline is designed to be executed sequentially. All configurations are managed centrally in `configs/titan_config.yaml`.

### Training the Tokenizer
To build the vocabulary from the raw text corpus:
```bash
python scripts/train_tokenizer.py --config configs/titan_config.yaml
```

### Preparing the Dataset
To process the raw text into tokenized shards:
```bash
python scripts/prepare_data.py --config configs/titan_config.yaml
```

### Training the Model
To start training from scratch:
```bash
python scripts/train.py --config configs/titan_config.yaml
```
To resume an interrupted run from the latest checkpoint:
```bash
python scripts/train.py --auto-resume
```

### Evaluating Checkpoints
To evaluate a specific checkpoint against the validation set:
```bash
python scripts/evaluate.py --checkpoint checkpoints/step_1000.pt
```

### Running Inference
To interact with the trained model via the CLI:
```bash
python scripts/infer.py --checkpoint checkpoints/final.pt
```

## 3. What is Needed for the Next Phase of Domain Training

The current model is sized for pipeline validation (~15M parameters). To achieve the ultimate goal of a general conversational AI with deep cybersecurity specialization, the following steps must be taken.

### Step 1: Curate the Pre-training Corpus
Before scaling the model, a high-quality, diverse pre-training corpus must be assembled. This should include:
- **General Knowledge**: Wikipedia, Books3, or a filtered subset of CommonCrawl to teach the model basic grammar, reasoning, and world knowledge.
- **Technical Data**: High-quality code repositories (GitHub), StackOverflow discussions, and technical documentation to build software engineering capabilities.
- **Cybersecurity Data**: CVE databases, exploit write-ups, security architecture whitepapers, and defensive playbooks to seed the domain specialization.

*Action Item*: Gather at least 10GB to 50GB of clean text data and place it in `data/raw/`.

### Step 2: Scale the Architecture
Once the data is ready, the model architecture must be scaled up to handle the increased complexity. Update `configs/titan_config.yaml` with the following targets:
- **Vocab Size**: Increase from 8,000 to 32,000 or 50,000.
- **Model Size**: Scale from ~15M parameters to ~1.5B parameters (e.g., `d_model=2048`, `n_heads=16`, `n_layers=24`).
- **Context Length**: Increase from 256 to 2048 or 4096 tokens.

### Step 3: Distributed Training Infrastructure
Training a 1.5B parameter model requires significant compute. The current training loop supports single-GPU execution.
*Action Item*: Upgrade `training/trainer.py` to support PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP) across multiple GPUs.

### Step 4: Instruction Fine-Tuning
The base model will learn to predict the next token, but it will not naturally act as an assistant.
*Action Item*: Implement an instruction fine-tuning phase using a curated dataset of prompt-response pairs (e.g., OpenAssistant or a custom cybersecurity QA dataset). This requires modifying the data pipeline to support prompt masking (where loss is only calculated on the response tokens).

## 4. What Should Be Improved Before Scaling

Before committing significant financial resources to a large-scale training run, the following improvements should be made to the foundation:

- **Flash Attention**: Integrate `FlashAttention-2` into `model/titan_model.py` to significantly reduce memory usage and increase training speed for longer context windows.
- **Rotary Positional Embeddings (RoPE)**: Replace the current absolute positional embeddings with RoPE to improve length extrapolation and training stability.
- **Data Streaming**: The current `TitanShardDataset` loads all shards into memory. For a 50GB corpus, this must be updated to stream shards lazily from disk or cloud storage.
- **Experiment Tracking**: Integrate Weights & Biases (W&B) or MLflow into `training/trainer.py` for real-time monitoring of loss curves and hardware utilization.

By following this plan, Titan will evolve from a validated pipeline into a production-ready, domain-specialized intelligence.
