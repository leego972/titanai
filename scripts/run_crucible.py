"""
TitanAI Crucible Training Run — v0.2.0
=======================================
Full pretraining run on RTX 4090 (49GB VRAM).
125M parameter model, trained on balanced 5-bucket corpus.

Architecture: 12 layers, 12 heads, d_model=768, d_ff=3072, vocab=32000
Batch: 16 sequences × 2048 tokens × grad_accum=8 = effective batch 131,072 tokens
Steps: 76,294 (approx 5B tokens at effective batch size)
"""
import os
import sys
import json
import time
import math
import logging
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import yaml

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import build_model

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = BASE / "logs" / "crucible_v02"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(BASE / "checkpoints" / "crucible_v02").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "training.log"))
    ]
)
log = logging.getLogger("crucible")

# ── Dataset ───────────────────────────────────────────────────────────────────
class ShardDataset:
    """Iterates over .npy token shards, yielding (batch, seq_len) sequences."""
    def __init__(self, shard_dir: Path, seq_len: int, split: str = "train"):
        self.shard_dir = shard_dir
        self.seq_len   = seq_len
        self.split     = split
        # Support both flat npy arrays and pre-chunked 2D arrays
        self.shards = sorted(shard_dir.glob(f"{split}_*.npy"))
        if not self.shards:
            # Try without split prefix
            self.shards = sorted(shard_dir.glob("*.npy"))
            self.shards = [s for s in self.shards if split in s.name]
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards found in {shard_dir}")
        log.info(f"[Dataset] {split}: {len(self.shards)} shards in {shard_dir}")
        self._load_shard(0)

    def _load_shard(self, idx: int):
        raw = np.load(str(self.shards[idx]))
        # If 2D (pre-chunked sequences), flatten to 1D token stream
        if raw.ndim == 2:
            raw = raw.reshape(-1)
        self._shard_data = raw.astype(np.int64)
        self._shard_idx  = idx
        self._pos        = 0
        log.info(f"[Dataset] Loaded shard {idx}: {len(self._shard_data):,} tokens")

    def next_batch(self, batch_size: int):
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
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
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
    log.info(f"[Checkpoint] Saved: {path} (step={step}, loss={loss:.4f})")

def load_checkpoint(model, optimizer, path: Path):
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    log.info(f"[Checkpoint] Resumed from step {ckpt['step']}, loss={ckpt['loss']:.4f}")
    return ckpt["step"]

# ── Training ──────────────────────────────────────────────────────────────────
def train(cfg: dict):
    t_cfg  = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[Crucible] Device: {device}")
    if device.type == "cuda":
        log.info(f"[Crucible] GPU: {torch.cuda.get_device_name(0)}, "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    # Build model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Crucible] Model: {n_params:,} parameters")
    log.info(f"[Crucible] Architecture: {cfg['model']['n_layers']}L x "
             f"{cfg['model']['n_heads']}H x d={cfg['model']['d_model']}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )

    # Data — shards are in data/shards/
    shard_dir  = BASE / "data" / "shards"
    seq_len    = cfg["model"]["max_seq_len"]
    # Use smaller batch for RTX 4090 memory safety (effective batch maintained via grad_accum)
    batch_size = 8   # 8 × 2048 × 8 accum = 131,072 effective tokens
    grad_accum = 16  # Effective batch = 131,072 tokens ≈ same as config's 512 × 2048

    train_ds = ShardDataset(shard_dir, seq_len, "train")
    val_ds   = ShardDataset(shard_dir, seq_len, "val")

    # Training params
    max_lr    = t_cfg["learning_rate"]
    min_lr    = max_lr * t_cfg.get("lr_min_ratio", 0.1)
    warmup    = t_cfg["warmup_steps"]
    max_steps = t_cfg["max_steps"]
    grad_clip = t_cfg["grad_clip"]
    log_every  = t_cfg["log_interval"]
    eval_every = t_cfg["eval_interval"]
    save_every = t_cfg["save_interval"]
    ckpt_dir   = BASE / t_cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume if checkpoint exists
    start_step = 0
    resume_path = t_cfg.get("resume_from")
    if resume_path and Path(resume_path).exists():
        start_step = load_checkpoint(model, optimizer, Path(resume_path))
    else:
        # Check for latest checkpoint
        existing = sorted(ckpt_dir.glob("step_*.pt"))
        if existing:
            start_step = load_checkpoint(model, optimizer, existing[-1])

    log.info(f"[Crucible] Training: {max_steps} steps, batch={batch_size}, "
             f"grad_accum={grad_accum}, effective_batch={batch_size * grad_accum * seq_len:,} tokens")
    log.info(f"[Crucible] LR: {max_lr:.2e} → {min_lr:.2e}, warmup={warmup} steps")

    train_losses = []
    val_losses   = []
    nan_events   = 0
    run_log      = []

    model.train()
    optimizer.zero_grad()
    accum_loss   = 0.0
    accum_tokens = 0
    step_times   = []

    for step in range(start_step, max_steps):
        t0 = time.time()

        # Set LR
        lr = get_lr(step, warmup, max_steps, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Micro-steps for gradient accumulation
        micro_loss = 0.0
        for micro in range(grad_accum):
            x, y = train_ds.next_batch(batch_size)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            # NaN/Inf detection (PRE-FLIGHT FIX)
            raw_loss = loss.item()
            if not math.isfinite(raw_loss):
                nan_events += 1
                log.error(f"[Crucible] NaN/Inf loss at step {step+1}, micro {micro}: {raw_loss}")
                if nan_events >= 3:
                    # Emergency checkpoint
                    emerg_path = ckpt_dir / f"emergency_step_{step+1}.pt"
                    save_checkpoint(model, optimizer, step + 1, float('nan'), cfg, emerg_path)
                    raise RuntimeError(
                        f"Training diverged: {nan_events} NaN events. "
                        f"Emergency checkpoint: {emerg_path}"
                    )
                continue

            (loss / grad_accum).backward()
            micro_loss += raw_loss
            accum_tokens += x.numel()

        # Gradient clipping (enforced)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if grad_norm > grad_clip * 5:
            log.warning(f"[Crucible] Gradient spike at step {step+1}: "
                       f"norm={grad_norm:.2f} (clip={grad_clip})")

        optimizer.step()
        optimizer.zero_grad()

        step_time = time.time() - t0
        step_times.append(step_time)
        avg_loss = micro_loss / grad_accum
        accum_loss += avg_loss
        train_losses.append(avg_loss)

        # Logging
        if (step + 1) % log_every == 0:
            avg_t = sum(step_times[-log_every:]) / min(len(step_times), log_every)
            tps = accum_tokens / sum(step_times[-log_every:]) if step_times else 0
            log.info(f"[Crucible] step {step+1:6d}/{max_steps} | "
                     f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                     f"tok/s={tps:,.0f} | {avg_t*1000:.0f}ms/step | "
                     f"grad_norm={grad_norm:.3f}")
            run_log.append({
                "step": step + 1,
                "train_loss": round(avg_loss, 4),
                "lr": lr,
                "tokens_per_sec": int(tps),
                "grad_norm": round(float(grad_norm), 4),
            })
            accum_loss   = 0.0
            accum_tokens = 0

        # Evaluation
        if (step + 1) % eval_every == 0:
            model.eval()
            val_loss_total = 0.0
            n_val = 50
            with torch.no_grad():
                for _ in range(n_val):
                    xv, yv = val_ds.next_batch(batch_size)
                    xv, yv = xv.to(device), yv.to(device)
                    _, vl = model(xv, yv)
                    val_loss_total += vl.item()
            val_loss = val_loss_total / n_val
            val_ppl  = math.exp(min(val_loss, 20))
            val_losses.append(val_loss)
            log.info(f"[Crucible] *** EVAL step {step+1}: "
                     f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f} ***")
            if run_log:
                run_log[-1]["val_loss"] = round(val_loss, 4)
                run_log[-1]["val_ppl"]  = round(val_ppl, 2)
            model.train()

        # Checkpoint
        if (step + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step+1:06d}.pt"
            save_checkpoint(model, optimizer, step + 1, avg_loss, cfg, ckpt_path)

        # Save run log periodically
        if (step + 1) % 1000 == 0:
            with open(LOG_DIR / "run_log.json", "w") as f:
                json.dump(run_log, f, indent=2)

    # Final checkpoint
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(model, optimizer, max_steps,
                    train_losses[-1] if train_losses else 0, cfg, final_path)

    # Final eval
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for _ in range(100):
            xv, yv = val_ds.next_batch(batch_size)
            xv, yv = xv.to(device), yv.to(device)
            _, vl = model(xv, yv)
            val_loss_total += vl.item()
    final_val_loss = val_loss_total / 100
    final_val_ppl  = math.exp(min(final_val_loss, 20))

    log.info(f"\n[Crucible] ===== TRAINING COMPLETE =====")
    log.info(f"[Crucible] Steps: {max_steps}")
    log.info(f"[Crucible] Final train loss: {train_losses[-1]:.4f}")
    log.info(f"[Crucible] Final val loss:   {final_val_loss:.4f}")
    log.info(f"[Crucible] Final val PPL:    {final_val_ppl:.2f}")
    log.info(f"[Crucible] NaN events:       {nan_events}")

    # Loss trend
    if len(train_losses) > 20:
        q = len(train_losses) // 4
        first_q = sum(train_losses[:q]) / q
        last_q  = sum(train_losses[-q:]) / q
        is_dec  = last_q < first_q
        log.info(f"[Crucible] Loss trend: {first_q:.4f} → {last_q:.4f} "
                 f"({'DECREASING ✓' if is_dec else 'NOT DECREASING ✗'})")
    else:
        is_dec = True

    summary = {
        "run": "titan-crucible-v0.2.0",
        "steps_completed": max_steps,
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 2),
        "nan_events": nan_events,
        "loss_is_decreasing": is_dec,
        "checkpoint": str(final_path),
        "run_log": run_log,
        "crucible_pass_criteria": {
            "val_ppl_below_15": final_val_ppl < 15.0,
            "no_nan_events": nan_events == 0,
            "loss_decreasing": is_dec,
        }
    }
    with open(LOG_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[Crucible] Run summary saved.")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TitanAI Crucible Training Run v0.2.0")
    parser.add_argument("--config", default="configs/titan_crucible_v02.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["training"]["resume_from"] = args.resume

    summary = train(cfg)

    print("\n" + "="*60)
    print("  CRUCIBLE RUN SUMMARY")
    print("="*60)
    print(f"  Steps completed : {summary['steps_completed']}")
    print(f"  Final train loss: {summary['final_train_loss']}")
    print(f"  Final val loss  : {summary['final_val_loss']}")
    print(f"  Final val PPL   : {summary['final_val_ppl']}")
    print(f"  NaN events      : {summary['nan_events']}")
    print(f"  Loss decreasing : {summary['loss_is_decreasing']}")
    print("\n  Crucible pass criteria:")
    for k, v in summary["crucible_pass_criteria"].items():
        icon = "✓" if v else "✗"
        print(f"    [{icon}] {k}: {v}")
    print("="*60)


if __name__ == "__main__":
    main()
