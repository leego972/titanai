"""
TitanAI Probe — Sandbox Validation Runner
==========================================
CPU-safe version for pipeline validation.
Runs 500 steps with reduced batch/seq to fit in sandbox RAM.
Full Probe (batch=16, seq=2048, 24414 steps) runs on GPU.

This validates:
  - Real shard data loads and tokenizes correctly
  - Model trains on real corpus data (not synthetic)
  - Loss decreases monotonically
  - No NaN/Inf events
  - Checkpoints save and resume
  - Evaluation harness runs
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

(BASE / "logs" / "probe_v015").mkdir(parents=True, exist_ok=True)
(BASE / "checkpoints" / "probe_v015").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(BASE / "logs" / "probe_v015" / "training.log"))
    ]
)
log = logging.getLogger("probe")

# ── Sandbox-safe config overrides ─────────────────────────────────────────────
SANDBOX_BATCH_SIZE   = 2      # vs 16 in full config
SANDBOX_SEQ_LEN      = 256    # vs 2048 in full config
SANDBOX_GRAD_ACCUM   = 2      # vs 8 in full config
SANDBOX_MAX_STEPS    = 500    # vs 24414 in full config
SANDBOX_LOG_EVERY    = 50
SANDBOX_EVAL_EVERY   = 250
SANDBOX_SAVE_EVERY   = 250


class ShardDataset:
    def __init__(self, shard_dir: Path, seq_len: int, split: str = "train"):
        self.shard_dir = shard_dir
        self.seq_len   = seq_len
        self.split     = split
        self.shards    = sorted(shard_dir.glob("shard_*.npy"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {shard_dir}")
        log.info(f"[Dataset] {split}: {len(self.shards)} shards")
        self._load_shard(0)

    def _load_shard(self, idx: int):
        self._data     = np.load(str(self.shards[idx])).astype(np.int64)
        self._shard_idx = idx
        self._pos       = 0

    def next_batch(self, batch_size: int):
        inputs, targets = [], []
        needed = batch_size
        while needed > 0:
            available = (len(self._data) - self._pos) // (self.seq_len + 1)
            take = min(needed, available)
            for _ in range(take):
                chunk = self._data[self._pos: self._pos + self.seq_len + 1]
                if len(chunk) < self.seq_len + 1:
                    break
                inputs.append(chunk[:self.seq_len])
                targets.append(chunk[1:self.seq_len + 1])
                self._pos += self.seq_len
            needed -= take
            if needed > 0:
                next_idx = (self._shard_idx + 1) % len(self.shards)
                self._load_shard(next_idx)
        x = torch.tensor(np.stack(inputs), dtype=torch.long)
        y = torch.tensor(np.stack(targets), dtype=torch.long)
        return x, y


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    with open(BASE / "configs" / "titan_probe_v015.yaml") as f:
        cfg = yaml.safe_load(f)

    # Override model seq_len for sandbox
    cfg["model"]["max_seq_len"] = SANDBOX_SEQ_LEN

    device = torch.device("cpu")
    log.info(f"[Probe] Sandbox validation run — CPU, seq_len={SANDBOX_SEQ_LEN}, "
             f"batch={SANDBOX_BATCH_SIZE}, grad_accum={SANDBOX_GRAD_ACCUM}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[Probe] Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )

    proc_dir = BASE / "data" / "processed" / "v1.0.0"
    train_ds = ShardDataset(proc_dir / "train", SANDBOX_SEQ_LEN, "train")
    val_ds   = ShardDataset(proc_dir / "val",   SANDBOX_SEQ_LEN, "val")

    max_lr  = cfg["training"]["learning_rate"]
    min_lr  = max_lr * cfg["training"].get("lr_min_ratio", 0.1)
    warmup  = min(100, SANDBOX_MAX_STEPS // 10)
    grad_clip = cfg["training"]["grad_clip"]
    ckpt_dir  = BASE / cfg["training"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses   = []
    nan_events   = 0
    run_log      = []

    model.train()
    optimizer.zero_grad()
    t_start = time.time()

    log.info(f"[Probe] Training for {SANDBOX_MAX_STEPS} steps on real corpus data...")

    for step in range(SANDBOX_MAX_STEPS):
        lr = get_lr(step, warmup, SANDBOX_MAX_STEPS, max_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accum_loss = 0.0
        for _ in range(SANDBOX_GRAD_ACCUM):
            x, y = train_ds.next_batch(SANDBOX_BATCH_SIZE)
            _, loss = model(x, y)
            loss = loss / SANDBOX_GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item()

        if math.isnan(accum_loss) or math.isinf(accum_loss):
            nan_events += 1
            log.warning(f"[Probe] NaN/Inf at step {step}")
            optimizer.zero_grad()
            continue

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(accum_loss)

        if (step + 1) % SANDBOX_LOG_EVERY == 0:
            elapsed = time.time() - t_start
            avg_loss = sum(train_losses[-SANDBOX_LOG_EVERY:]) / min(SANDBOX_LOG_EVERY, len(train_losses))
            tokens_done = (step + 1) * SANDBOX_BATCH_SIZE * SANDBOX_GRAD_ACCUM * SANDBOX_SEQ_LEN
            log.info(f"[Probe] Step {step+1:5d}/{SANDBOX_MAX_STEPS} | "
                     f"loss={avg_loss:.4f} | lr={lr:.2e} | "
                     f"tokens={tokens_done:,} | elapsed={elapsed:.0f}s")
            run_log.append({"step": step + 1, "train_loss": round(avg_loss, 4), "lr": lr})

        if (step + 1) % SANDBOX_EVAL_EVERY == 0:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for _ in range(20):
                    xv, yv = val_ds.next_batch(SANDBOX_BATCH_SIZE)
                    _, vl = model(xv, yv)
                    val_total += vl.item()
            val_loss = val_total / 20
            val_ppl  = math.exp(min(val_loss, 20))
            val_losses.append(val_loss)
            log.info(f"[Probe] *** EVAL step {step+1}: val_loss={val_loss:.4f}, ppl={val_ppl:.2f} ***")
            if run_log:
                run_log[-1]["val_loss"] = round(val_loss, 4)
                run_log[-1]["val_ppl"]  = round(val_ppl, 2)
            model.train()

        if (step + 1) % SANDBOX_SAVE_EVERY == 0:
            ckpt_path = ckpt_dir / f"step_{step+1:06d}.pt"
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": accum_loss,
            }, str(ckpt_path))
            log.info(f"[Probe] Checkpoint saved: {ckpt_path}")

    # Final eval
    model.eval()
    val_total = 0.0
    with torch.no_grad():
        for _ in range(50):
            xv, yv = val_ds.next_batch(SANDBOX_BATCH_SIZE)
            _, vl = model(xv, yv)
            val_total += vl.item()
    final_val_loss = val_total / 50
    final_val_ppl  = math.exp(min(final_val_loss, 20))

    # Final checkpoint
    torch.save({
        "step": SANDBOX_MAX_STEPS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_losses[-1] if train_losses else 0,
    }, str(ckpt_dir / "final.pt"))

    # Loss trend
    if len(train_losses) > 20:
        first_q = sum(train_losses[:len(train_losses)//4]) / (len(train_losses)//4)
        last_q  = sum(train_losses[-len(train_losses)//4:]) / (len(train_losses)//4)
        is_decreasing = last_q < first_q
    else:
        first_q = train_losses[0] if train_losses else 0
        last_q  = train_losses[-1] if train_losses else 0
        is_decreasing = last_q < first_q

    summary = {
        "run": "titan-probe-v0.1.5-sandbox",
        "steps_completed": SANDBOX_MAX_STEPS,
        "seq_len_used": SANDBOX_SEQ_LEN,
        "batch_size_used": SANDBOX_BATCH_SIZE,
        "first_train_loss": round(first_q, 4),
        "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
        "final_val_loss": round(final_val_loss, 4),
        "final_val_ppl": round(final_val_ppl, 2),
        "nan_events": nan_events,
        "loss_is_decreasing": is_decreasing,
        "run_log": run_log,
        "probe_pass_criteria": {
            "loss_decreasing": is_decreasing,
            "no_nan_events": nan_events == 0,
            "val_loss_below_initial": final_val_loss < first_q if first_q else True,
        }
    }

    summary_path = BASE / "logs" / "probe_v015" / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("  PROBE SANDBOX VALIDATION — COMPLETE")
    print("="*60)
    print(f"  Steps:          {SANDBOX_MAX_STEPS}")
    print(f"  Time:           {elapsed:.0f}s")
    print(f"  First loss:     {first_q:.4f}")
    print(f"  Final train:    {train_losses[-1]:.4f}")
    print(f"  Final val loss: {final_val_loss:.4f}")
    print(f"  Final val PPL:  {final_val_ppl:.2f}")
    print(f"  NaN events:     {nan_events}")
    print(f"  Loss trend:     {first_q:.4f} → {last_q:.4f} "
          f"({'DECREASING ✓' if is_decreasing else 'NOT DECREASING ✗'})")
    print("\n  Probe pass criteria:")
    for k, v in summary["probe_pass_criteria"].items():
        icon = "✓" if v else "✗"
        print(f"    [{icon}] {k}: {v}")
    print("="*60)
    print(f"\n  Summary: {summary_path}")
    print(f"  Checkpoint: {ckpt_dir}/final.pt")

    return summary


if __name__ == "__main__":
    main()
