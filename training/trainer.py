"""
Titan Training Loop
===================
Trains TitanLM from scratch on the prepared dataset.
Supports:
    - Gradient accumulation for larger effective batch sizes
    - Cosine LR schedule with linear warmup
    - Gradient clipping
    - Checkpoint saving and resuming
    - Validation loss evaluation at configurable intervals
    - Clean failure and resume from any checkpoint

Usage:
    python scripts/train.py --config configs/titan_config.yaml

Resume from checkpoint:
    python scripts/train.py --config configs/titan_config.yaml --resume checkpoints/step_1000.pt
"""

import os
import sys
import json
import math
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.titan_model import TitanLM, TitanConfig, build_model
from data.dataset import create_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint
from evaluation.evaluator import evaluate


# ─── LR Schedule ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.
    LR goes from 0 → max_lr over warmup_steps,
    then cosine decays to min_lr_ratio * max_lr over remaining steps.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ─── Logger ───────────────────────────────────────────────────────────────────

class TrainingLogger:
    """Simple CSV + console logger for training metrics."""

    def __init__(self, log_dir: str, experiment_name: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{experiment_name}_train.csv")
        self.val_log_path = os.path.join(log_dir, f"{experiment_name}_val.csv")

        # Write headers if files don't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("step,loss,lr,tokens_per_sec,elapsed_sec\n")
        if not os.path.exists(self.val_log_path):
            with open(self.val_log_path, "w") as f:
                f.write("step,val_loss,val_perplexity\n")

    def log_train(self, step: int, loss: float, lr: float, tps: float, elapsed: float):
        line = f"{step},{loss:.6f},{lr:.8f},{tps:.1f},{elapsed:.1f}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        print(f"[Train] step={step:6d} | loss={loss:.4f} | lr={lr:.2e} | tok/s={tps:.0f}")

    def log_val(self, step: int, val_loss: float, val_ppl: float):
        line = f"{step},{val_loss:.6f},{val_ppl:.4f}"
        with open(self.val_log_path, "a") as f:
            f.write(line + "\n")
        print(f"[Val]   step={step:6d} | val_loss={val_loss:.4f} | perplexity={val_ppl:.2f}")


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(config: dict, resume_from: str = None, base_dir: str = "."):
    """
    Main training function.
    Args:
        config: Full config dict loaded from titan_config.yaml
        resume_from: Path to checkpoint file to resume from (or None)
        base_dir: Base directory of the titan-model project
    """
    train_cfg = config["training"]
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    log_cfg = config["logging"]

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    processed_dir = os.path.join(base_dir, data_cfg["processed_dir"])
    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        val_batch_size=eval_cfg["val_batch_size"],
    )
    train_iter = iter(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config)
    model = model.to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Separate weight decay: apply only to weight matrices, not biases/norms
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if resume_from:
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        print(f"[Train] Resumed from step {start_step}")
    elif train_cfg.get("resume_from"):
        ckpt_path = os.path.join(base_dir, train_cfg["resume_from"])
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        print(f"[Train] Resumed from step {start_step}")

    # ── Logger ───────────────────────────────────────────────────────────────
    log_dir = os.path.join(base_dir, log_cfg["log_dir"])
    logger = TrainingLogger(log_dir, log_cfg["experiment_name"])

    checkpoint_dir = os.path.join(base_dir, train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Training Loop ────────────────────────────────────────────────────────
    grad_accum_steps = train_cfg["gradient_accumulation_steps"]
    max_steps = train_cfg["max_steps"]
    log_interval = train_cfg["log_interval"]
    eval_interval = train_cfg["eval_interval"]
    save_interval = train_cfg["save_interval"]
    clip_grad = train_cfg["clip_grad_norm"]

    model.train()
    accum_loss = 0.0
    accum_tokens = 0
    t0 = time.time()
    optimizer.zero_grad()

    print(f"[Train] Starting training from step {start_step} to {max_steps}")

    for step in range(start_step, max_steps):
        # ── Fetch batch ─────────────────────────────────────────────────────
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # ── Forward pass ────────────────────────────────────────────────────
        _, loss = model(input_ids, labels)
        loss = loss / grad_accum_steps
        loss.backward()

        accum_loss += loss.item()
        accum_tokens += input_ids.numel()

        # ── Gradient accumulation step ──────────────────────────────────────
        if (step + 1) % grad_accum_steps == 0:
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ─────────────────────────────────────────────────────────
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            tps = accum_tokens / elapsed
            current_lr = scheduler.get_last_lr()[0]
            logger.log_train(step + 1, accum_loss * grad_accum_steps / log_interval,
                             current_lr, tps, elapsed)
            accum_loss = 0.0
            accum_tokens = 0
            t0 = time.time()

        # ── Evaluation ──────────────────────────────────────────────────────
        if (step + 1) % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device,
                                         eval_cfg["num_eval_batches"])
            logger.log_val(step + 1, val_loss, val_ppl)
            model.train()

        # ── Checkpoint ──────────────────────────────────────────────────────
        if (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step + 1}.pt")
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, config)
            print(f"[Train] Checkpoint saved: {ckpt_path}")

    # ── Final checkpoint ─────────────────────────────────────────────────────
    final_path = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, max_steps, config)
    print(f"[Train] Training complete. Final checkpoint: {final_path}")

    return model
