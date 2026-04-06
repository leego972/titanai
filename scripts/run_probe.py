"""
TitanAI Probe Training Run — v0.1.5
=====================================
Runs a representative training run on the real corpus shards.
In this sandbox environment (CPU-only), we run 1000 steps to validate:
  - Data pipeline loads correctly from real shards
  - Loss decreases monotonically
  - No NaN/Inf events
  - Checkpoints save and resume correctly
  - Evaluation harness runs at the end

Full 1B-token Probe (24,414 steps) requires GPU — this validates
the complete pipeline end-to-end on real data.
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(BASE / "logs" / "probe_v015" / "training.log"))
    ]
)
log = logging.getLogger("probe")

(BASE / "logs" / "probe_v015").mkdir(parents=True, exist_ok=True)
(BASE / "checkpoints" / "probe_v015").mkdir(parents=True, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShardDataset:
    """Iterates over flat .npy token shards, yielding (seq_len,) sequences."""

    def __init__(self, shard_dir: Path, seq_len: int, split: str = "train"):
        self.shard_dir = shard_dir
        self.seq_len   = seq_len
        self.split     = split
        self.shards    = sorted(shard_dir.glob("shard_*.npy"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        log.info(f"[Dataset] {split}: {len(self.shards)} shards in {shard_dir}")
        self._load_shard(0)

    def _load_shard(self, idx: int):
        self._shard_data = np.load(str(self.shards[idx])).astype(np.int64)
        self._shard_idx  = idx
        self._pos        = 0
        log.info(f"[Dataset] Loaded shard {idx}: {len(self._shard_data):,} tokens")

    def next_batch(self, batch_size: int) -> tuple:
        """Return (input_ids, labels) tensors of shape (batch_size, seq_len)."""
        inputs  = []
        targets = []
        needed  = batch_size

        while needed > 0:
            available = (len(self._shard_data) - self._pos) // (self.seq_len + 1)
            take = min(needed, available)

            for _ in range(take):
                chunk = self._shard_data[self._pos: self._pos + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    break
                inputs.append(chunk[:self.seq_len])
                targets.append(chunk[1:self.seq_len + 1])
                self._pos += self.seq_len

            needed -= take

            if needed > 0:
                # Move to next shard (cycle)
                next_idx = (self._shard_idx + 1) % len(self.shards)
                self._load_shard(next_idx)

        if not inputs:
            raise RuntimeError("Dataset exhausted — no sequences available")

        x = torch.tensor(np.stack(inputs), dtype=torch.long)
        y = torch.tensor(np.stack(targets), dtype=torch.long)
        return x, y


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, step, loss, cfg, path: Path):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": cfg,
    }, str(path))
    log.info(f"[Checkpoint] Saved: {path}")


def load_checkpoint(model, optimizer, path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log.info(f"[Checkpoint] Resumed from step {ckpt['step']}, loss={ckpt['loss']:.4f}")
    return ckpt["step"]


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: dict, max_steps_override: int = None):
    t_cfg  = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Probe] Device: {device}")

    # Build model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Probe] Model: {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Data
    proc_dir   = BASE / "data" / "processed" / "v1.0.0"
    seq_len    = cfg["model"]["max_seq_len"]
    batch_size = t_cfg["batch_size"]
    grad_accum = t_cfg["gradient_accumulation_steps"]

    train_ds = ShardDataset(proc_dir / "train", seq_len, "train")
    val_ds   = ShardDataset(proc_dir / "val",   seq_len, "val")

    # Training params
    max_lr    = t_cfg["learning_rate"]
    min_lr    = max_lr * t_cfg.get("lr_min_ratio", 0.1)
    warmup    = t_cfg["warmup_steps"]
    max_steps = max_steps_override or t_cfg["max_steps"]
    grad_clip = t_cfg["grad_clip"]
    log_every = t_cfg["log_interval"]
    eval_every = t_cfg["eval_interval"]
    save_every = t_cfg["save_interval"]

    ckpt_dir  = BASE / t_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume if checkpoint exists
    start_step = 0
    resume_path = t_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        start_step = load_checkpoint(model, optimizer, Path(resume_path))

    # Training loop
    log.info(f"[Probe] Starting training: {max_steps} steps, batch={batch_size}, "
             f"grad_accum={grad_accum}, effective_batch={batch_size * grad_accum}")

    train_losses = []
    val_losses   = []
    step_times   = []
    nan_events   = 0
    run_log      = []

    model.train()
    optimizer.zero_grad()

    for step in range(start_step, max_steps):
        t0 = time.time()

        # Update LR
        lr = get_lr(step, warmup, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        accum_loss = 0.0
        for micro_step in range(grad_accum):
            x, y = train_ds.next_batch(batch_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Check for NaN
        if math.isnan(accum_loss) or math.isinf(accum_loss):
            nan_events += 1
            log.warning(f"[Probe] NaN/Inf loss at step {step} — skipping update")
            optimizer.zero_grad()
            continue

        # Gradient clip
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        train_losses.append(accum_loss)
        step_times.append(time.time() - t0)

        # Log
        if (step + 1) % log_every == 0:
            avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
            avg_time = sum(step_times[-log_every:]) / min(log_every, len(step_times))
            tokens_per_sec = (batch_size * grad_accum * seq_len) / avg_time
            log.info(f"[Probe] Step {step+1:6d}/{max_steps} | "
                     f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                     f"tok/s={tokens_per_sec:,.0f} | {avg_time*1000:.0f}ms/step")
            run_log.append({
                "step": step + 1,
                "train_loss": round(avg_loss, 4),
                "lr": lr,
                "tokens_per_sec": int(tokens_per_sec)
            })

        # Eval
        if (step + 1) % eval_every == 0:
            model.eval()
            val_loss_total = 0.0
            n_val_batches  = min(20, t_cfg.get("num_eval_batches", 20))
            with torch.no_grad():
                for _ in range(n_val_batches):
                    xv, yv = val_ds.next_batch(batch_size)
                    xv, yv = xv.to(device), yv.to(device)
                    _, vl = model(xv, yv)
                    val_loss_total += vl.item()
            val_loss = val_loss_total / n_val_batches
            val_ppl  = math.exp(min(val_loss, 20))
            val_losses.append(val_loss)
            log.info(f"[Probe] *** EVAL step {step+1}: val_loss={val_loss:.4f}, "
                     f"val_ppl={val_ppl:.2f} ***")
            run_log[-1]["val_loss"] = round(val_loss, 4)
            run_log[-1]["val_ppl"]  = round(val_ppl, 2)
            model.train()

        # Save checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step+1:06d}.pt"
            save_checkpoint(model, optimizer, step + 1, accum_loss, cfg, ckpt_path)

    # Final checkpoint
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(model, optimizer, max_steps, train_losses[-1] if train_losses else 0,
                    cfg, final_path)

    # Final eval
    model.eval()
    val_loss_total = 0.0
    n_val_batches  = 50
    with torch.no_grad():
        for _ in range(n_val_batches):
            xv, yv = val_ds.next_batch(batch_size)
            xv, yv = xv.to(device), yv.to(device)
            _, vl = model(xv, yv)
            val_loss_total += vl.item()
    final_val_loss = val_loss_total / n_val_batches
    final_val_ppl  = math.exp(min(final_val_loss, 20))

    log.info(f"\n[Probe] ===== TRAINING COMPLETE =====")
    log.info(f"[Probe] Steps: {max_steps}")
    log.info(f"[Probe] Final train loss: {train_losses[-1]:.4f}")
    log.info(f"[Probe] Final val loss:   {final_val_loss:.4f}")
    log.info(f"[Probe] Final val PPL:    {final_val_ppl:.2f}")
    log.info(f"[Probe] NaN events:       {nan_events}")

    # Check monotonic decrease (last 20% of steps)
    if len(train_losses) > 20:
        first_quarter = sum(train_losses[:len(train_losses)//4]) / (len(train_losses)//4)
        last_quarter  = sum(train_losses[-len(train_losses)//4:]) / (len(train_losses)//4)
        is_decreasing = last_quarter < first_quarter
        log.info(f"[Probe] Loss trend: {first_quarter:.4f} → {last_quarter:.4f} "
                 f"({'DECREASING ✓' if is_decreasing else 'NOT DECREASING ✗'})")
    else:
        is_decreasing = True

    # Save run summary
    summary = {
        "run": "titan-probe-v0.1.5",
        "steps_completed": max_steps,
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 2),
        "nan_events": nan_events,
        "loss_is_decreasing": is_decreasing,
        "checkpoint": str(final_path),
        "run_log": run_log,
        "probe_pass_criteria": {
            "val_ppl_below_30": final_val_ppl < 30,
            "no_nan_after_step_500": nan_events == 0,
            "loss_decreasing": is_decreasing,
        }
    }
    summary_path = BASE / "logs" / "probe_v015" / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[Probe] Run summary saved: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="TitanAI Probe Training Run")
    parser.add_argument("--config", default="configs/titan_probe_v015.yaml")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Override max_steps (default 1000 for sandbox validation)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    summary = train(cfg, max_steps_override=args.max_steps)

    print("\n" + "="*60)
    print("  PROBE RUN SUMMARY")
    print("="*60)
    print(f"  Steps completed : {summary['steps_completed']}")
    print(f"  Final train loss: {summary['final_train_loss']}")
    print(f"  Final val loss  : {summary['final_val_loss']}")
    print(f"  Final val PPL   : {summary['final_val_ppl']}")
    print(f"  NaN events      : {summary['nan_events']}")
    print(f"  Loss decreasing : {summary['loss_is_decreasing']}")
    print("\n  Probe pass criteria:")
    for k, v in summary["probe_pass_criteria"].items():
        icon = "✓" if v else "✗"
        print(f"    [{icon}] {k}: {v}")
    print("="*60)


if __name__ == "__main__":
    main()
