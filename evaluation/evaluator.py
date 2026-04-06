"""
Titan Evaluation Harness
========================
Evaluates TitanLM on the validation set.
Metrics:
    - Validation loss (cross-entropy)
    - Perplexity (exp(val_loss))
    - Sample text generations
    - Checkpoint comparison

Usage (standalone):
    python scripts/evaluate.py --config configs/titan_config.yaml --checkpoint checkpoints/step_1000.pt

What "better" means at this stage:
    - Lower validation loss = better
    - Lower perplexity = better
    - Generated text should be coherent (not random noise) after sufficient training
    - A perplexity below 50 on a small corpus indicates the model is learning
"""

import os
import sys
import math
import json
import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.titan_model import TitanLM, TitanConfig, build_model
from training.checkpoint import load_checkpoint
from tokenizer.train_tokenizer import load_tokenizer


# ─── Core Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: TitanLM,
    val_loader,
    device: torch.device,
    num_batches: int = 20,
) -> tuple[float, float]:
    """
    Compute validation loss and perplexity over num_batches batches.
    Returns (val_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        _, loss = model(input_ids, labels)
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_batches
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return avg_loss, perplexity


# ─── Sample Generation ────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    model: TitanLM,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = None,
) -> list[dict]:
    """
    Generate text samples from a list of prompts.
    Returns a list of dicts with 'prompt' and 'generated' keys.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>")
    results = []

    for prompt in prompts:
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([[bos_id] + encoded.ids], dtype=torch.long, device=device)

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )

        # Decode only the newly generated tokens
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        # Remove EOS if present
        if eos_id in new_ids:
            new_ids = new_ids[:new_ids.index(eos_id)]
        generated_text = tokenizer.decode(new_ids)

        results.append({
            "prompt": prompt,
            "generated": generated_text,
        })

    return results


# ─── Checkpoint Comparison ────────────────────────────────────────────────────

def compare_checkpoints(
    checkpoint_paths: list[str],
    config: dict,
    val_loader,
    device: torch.device,
    num_batches: int = 20,
) -> list[dict]:
    """
    Evaluate multiple checkpoints and compare their validation metrics.
    Returns a list of dicts sorted by val_loss ascending.
    """
    results = []
    for ckpt_path in checkpoint_paths:
        print(f"[Eval] Evaluating checkpoint: {ckpt_path}")
        model = build_model(config).to(device)
        step = load_checkpoint(ckpt_path, model, device=device)
        val_loss, val_ppl = evaluate(model, val_loader, device, num_batches)
        results.append({
            "checkpoint": ckpt_path,
            "step": step,
            "val_loss": val_loss,
            "perplexity": val_ppl,
        })
        del model

    results.sort(key=lambda x: x["val_loss"])
    return results


# ─── Standalone Evaluation Script ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Titan checkpoint")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--prompts", nargs="+",
                        default=["Hello, I am Titan.", "The security vulnerability was"],
                        help="Prompts for sample generation")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    # Load model
    model = build_model(config).to(device)
    step = load_checkpoint(args.checkpoint, model, device=device)
    print(f"[Eval] Loaded checkpoint from step {step}")

    # Load tokenizer
    tok_save_dir = os.path.join(args.base_dir, config["tokenizer"]["save_dir"])
    tokenizer = load_tokenizer(tok_save_dir)

    # Load val data
    from data.dataset import create_dataloaders
    processed_dir = os.path.join(args.base_dir, config["data"]["processed_dir"])
    _, val_loader = create_dataloaders(
        train_dir=os.path.join(processed_dir, "train"),
        val_dir=os.path.join(processed_dir, "val"),
        max_seq_len=config["data"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        val_batch_size=config["evaluation"]["val_batch_size"],
    )

    # Evaluate
    val_loss, val_ppl = evaluate(model, val_loader, device,
                                  config["evaluation"]["num_eval_batches"])
    print(f"\n[Eval] Results at step {step}:")
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Perplexity:      {val_ppl:.2f}")

    # Generate samples
    print("\n[Eval] Sample generations:")
    samples = generate_samples(
        model, tokenizer, args.prompts,
        max_new_tokens=config["evaluation"]["sample_max_new_tokens"],
        temperature=config["evaluation"]["sample_temperature"],
        top_k=config["evaluation"]["sample_top_k"],
        device=device,
    )
    for s in samples:
        print(f"\n  Prompt:    {s['prompt']}")
        print(f"  Generated: {s['generated']}")

    # Save results
    results_path = os.path.join(args.base_dir, "logs", f"eval_step_{step}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"step": step, "val_loss": val_loss, "perplexity": val_ppl,
                   "samples": samples}, f, indent=2)
    print(f"\n[Eval] Results saved to {results_path}")


if __name__ == "__main__":
    main()
