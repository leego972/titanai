"""
Titan SFT Trainer
=================
Supervised Fine-Tuning (instruction tuning) trainer for TitanLM.

Loads a pre-trained TitanLM checkpoint and fine-tunes it on instruction
data using prompt masking — loss is computed only on assistant response tokens.

Usage:
    python scripts/sft_train.py --config configs/titan_sft_v01.yaml \
        --checkpoint checkpoints/probe_v015/final.pt

Resume:
    python scripts/sft_train.py --config configs/titan_sft_v01.yaml \
        --checkpoint checkpoints/probe_v015/final.pt \
        --resume checkpoints/sft_v01/step_500.pt

Key differences from CLM trainer:
    - Loads a pre-trained checkpoint (not random init)
    - Uses TitanSFTDataset with prompt masking
    - Lower learning rate (fine-tuning, not pretraining)
    - Shorter run (epochs over SFT data, not billions of tokens)
    - Loss is computed only on assistant response tokens
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
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.titan_model import TitanConfig, build_model
from data.sft_dataset import TitanSFTDataset, IGNORE_INDEX
from training.checkpoint import save_checkpoint
from tokenizers import Tokenizer


# ── LR schedule (same cosine warmup as CLM trainer) ──────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


# ── Logger ────────────────────────────────────────────────────────────────────

class SFTLogger:
    def __init__(self, log_dir, experiment_name):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{experiment_name}_sft.csv")
        self.val_log_path = os.path.join(log_dir, f"{experiment_name}_sft_val.csv")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("step,loss,lr,elapsed_sec\n")
        if not os.path.exists(self.val_log_path):
            with open(self.val_log_path, "w") as f:
                f.write("step,val_loss,val_perplexity\n")

    def log_train(self, step, loss, lr, elapsed):
        line = f"{step},{loss:.6f},{lr:.8f},{elapsed:.1f}"
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        print(f"[SFT Train] step={step:5d} | loss={loss:.4f} | lr={lr:.2e} | elapsed={elapsed:.0f}s")

    def log_val(self, step, val_loss, val_ppl):
        line = f"{step},{val_loss:.6f},{val_ppl:.4f}"
        with open(self.val_log_path, "a") as f:
            f.write(line + "\n")
        print(f"[SFT Val]   step={step:5d} | val_loss={val_loss:.4f} | perplexity={val_ppl:.2f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sft(model, val_loader, device, num_batches=20):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_run = 0
    _use_bf16 = device.type == "cuda"

    for batch in val_loader:
        if batches_run >= num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Shift for causal LM: predict token[i+1] from token[i] (bf16 autocast for flash-attn)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_bf16):
            logits, _ = model(input_ids[:, :-1])
            shift_labels = labels[:, 1:]

            # Flatten and compute loss (only on non-masked positions)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                    label_smoothing=0.1,
            )
        # Count unmasked tokens for perplexity calculation
        unmasked = (shift_labels != IGNORE_INDEX).sum().item()
        if unmasked > 0:
            total_loss += loss.item() * unmasked
            total_tokens += unmasked
        batches_run += 1

    model.train()
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Titan SFT Trainer")
    parser.add_argument("--config", required=True, help="Path to SFT YAML config")
    parser.add_argument("--checkpoint", required=True, help="Pre-trained base model checkpoint to fine-tune from")
    parser.add_argument("--resume", default=None, help="Resume SFT from a checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SFT] Device: {device}")

    # Load tokenizer
    tokenizer_path = data_cfg["tokenizer_path"]
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"[SFT] Tokenizer loaded from {tokenizer_path} | vocab_size={tokenizer.get_vocab_size()}")

    # Build model from config
    model = build_model(cfg)
    model = model.to(device)

    # Load pre-trained base checkpoint
    print(f"[SFT] Loading base checkpoint: {args.checkpoint}")
    base_state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_state = base_state.get("model_state_dict", base_state)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[SFT] WARNING: {len(missing)} missing keys in checkpoint")
    if unexpected:
        print(f"[SFT] WARNING: {len(unexpected)} unexpected keys in checkpoint")
    print(f"[SFT] Base model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    # Validate clip_grad_norm
    clip_grad = train_cfg["clip_grad_norm"]
    if clip_grad <= 0:
        raise ValueError(f"clip_grad_norm must be > 0 (got {clip_grad})")

    # Load SFT dataset
    jsonl_paths = data_cfg["sft_files"]
    full_dataset = TitanSFTDataset(
        jsonl_paths=jsonl_paths,
        tokenizer=tokenizer,
        max_seq_len=model_cfg["max_seq_len"],
        verbose=True,
    )
    stats = full_dataset.get_stats()
    print(f"[SFT] Dataset stats: {json.dumps(stats, indent=2)}")

    if len(full_dataset) == 0:
        raise ValueError("No training examples loaded. Check sft_files paths in config.")

    # Train / val split
    val_size = max(2, int(len(full_dataset) * data_cfg.get("val_split", 0.1)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg.get("val_batch_size", 8),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
    )

    # Optimizer — SFT uses lower LR than pretraining
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    max_steps = train_cfg["max_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=max_steps,
        min_lr_ratio=train_cfg.get("lr_min_ratio", 0.1),
    )

    # Checkpoint directory
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = SFTLogger(log_cfg["log_dir"], log_cfg["experiment_name"])

    # Resume from SFT checkpoint if requested
    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"[SFT] Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"[SFT] Resumed at step {start_step}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    step = start_step
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    log_interval = train_cfg.get("log_interval", 10)
    eval_interval = train_cfg.get("eval_interval", 100)
    save_interval = train_cfg.get("save_interval", 200)
    start_time = time.time()

    print(f"[SFT] Starting fine-tuning: {max_steps} steps, {train_size} train / {val_size} val examples")

    optimizer.zero_grad()
    accum_loss = 0.0
    accum_tokens = 0
    micro_step = 0  # counts every micro-batch; optimizer fires every grad_accum_steps

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = train_cfg.get("early_stop_patience", 8)
    _stop_training = False

    while step < max_steps and not _stop_training:
        for batch in train_loader:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass — shift inputs for causal LM (bf16 autocast for flash-attn)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model(input_ids[:, :-1])
                shift_labels = labels[:, 1:]

                # Loss — only on unmasked (assistant) tokens
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=0.1,
                )

            # Scale for gradient accumulation
            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_tokens += (shift_labels != IGNORE_INDEX).sum().item()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                # NaN/Inf check
                if not math.isfinite(accum_loss):
                    print(f"[SFT] FATAL: Loss diverged at step {step} (loss={accum_loss:.4f}) — saving emergency checkpoint")
                    save_checkpoint(str(checkpoint_dir / f"emergency_step_{step}.pt"), model, optimizer, scheduler, step, cfg)
                    raise RuntimeError(f"Loss is {accum_loss} at step {step}. Training halted.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    logger.log_train(step, accum_loss / grad_accum_steps, lr, elapsed)

                if step % eval_interval == 0:
                    val_loss, val_ppl = evaluate_sft(model, val_loader, device, eval_cfg.get("num_eval_batches", 10))
                    logger.log_val(step, val_loss, val_ppl)
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stop_patience:
                            print(f"[SFT] Early stopping at step {step} (patience={early_stop_patience})")
                            _stop_training = True
                            break

                if step % save_interval == 0:
                    save_checkpoint(str(checkpoint_dir / f"step_{step}.pt"), model, optimizer, scheduler, step, cfg)
                    print(f"[SFT] Checkpoint saved: step_{step}.pt")

                accum_loss = 0.0
                accum_tokens = 0

    # Final save
    final_path = checkpoint_dir / "final.pt"
    save_checkpoint(str(final_path), model, optimizer, scheduler, step, cfg)
    print(f"[SFT] Training complete. Final checkpoint: {final_path}")

    # Final evaluation
    val_loss, val_ppl = evaluate_sft(model, val_loader, device)
    logger.log_val(step, val_loss, val_ppl)
    print(f"[SFT] Final validation — loss={val_loss:.4f} | perplexity={val_ppl:.2f}")



if __name__ == "__main__":
    main()


def train_sft(cfg, model, train_dataset, val_dataset, device, resume=None):
    """
    Called by run_upgrade.py with pre-built model and datasets.
    Signature: train_sft(cfg, model, train_dataset, val_dataset, device, resume)
    """
    import math, time
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    train_cfg  = cfg["training"]
    eval_cfg   = cfg.get("evaluation", {})
    log_cfg    = cfg.get("logging", {"log_dir": "logs/upgrade", "experiment_name": "upgrade"})

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg.get("val_batch_size", 8),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
    )

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
    )
    max_steps = train_cfg["max_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=max_steps,
        min_lr_ratio=train_cfg.get("lr_min_ratio", 0.1),
    )

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = SFTLogger(log_cfg["log_dir"], log_cfg["experiment_name"])
    clip_grad = train_cfg["clip_grad_norm"]

    # --- Resume ---
    start_step = 0
    if resume and Path(resume).exists():
        print(f"[SFT] Resuming from: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"[SFT] Resumed at step {start_step}")

    # --- Training loop ---
    grad_accum  = train_cfg.get("gradient_accumulation_steps", 1)
    log_every   = train_cfg.get("log_interval", 25)
    eval_every  = train_cfg.get("eval_interval", 500)
    save_every  = train_cfg.get("save_interval", 1000)

    model.train()
    step       = start_step
    micro_step = 0
    accum_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    print(f"[SFT] Training {max_steps} steps | {len(train_dataset)} train / {len(val_dataset)} val examples")

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, _ = model(input_ids[:, :-1])
                shift_labels = labels[:, 1:]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=0.1,
                )

            (loss / grad_accum).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if not math.isfinite(accum_loss):
                    save_checkpoint(str(checkpoint_dir / f"emergency_{step}.pt"), model, optimizer, scheduler, step, cfg)
                    raise RuntimeError(f"Loss diverged at step {step}: {accum_loss}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_every == 0:
                    logger.log_train(step, accum_loss / grad_accum, scheduler.get_last_lr()[0], time.time() - start_time)
                if step % eval_every == 0:
                    vl, vp = evaluate_sft(model, val_loader, device, eval_cfg.get("num_eval_batches", 50))
                    logger.log_val(step, vl, vp)
                if step % save_every == 0:
                    save_checkpoint(str(checkpoint_dir / f"step_{step}.pt"), model, optimizer, scheduler, step, cfg)

                accum_loss = 0.0

    # Final save + eval
    final = checkpoint_dir / "final.pt"
    save_checkpoint(str(final), model, optimizer, scheduler, step, cfg)
    vl, vp = evaluate_sft(model, val_loader, device)
    logger.log_val(step, vl, vp)
    print(f"[SFT] Done. Final: {final} | val_loss={vl:.4f} | ppl={vp:.2f}")
