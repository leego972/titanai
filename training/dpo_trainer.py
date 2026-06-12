"""
TitanAI — DPO Trainer
======================
Direct Preference Optimization training loop.

DPO Loss (Rafailov et al., 2023):
  L_DPO = -E[ log σ( β * (log π(y_w|x) - log π_ref(y_w|x))
                       - β * (log π(y_l|x) - log π_ref(y_l|x)) ) ]

Where:
  π       = policy model (being trained)
  π_ref   = frozen reference model (SFT checkpoint)
  y_w     = chosen (preferred) response
  y_l     = rejected (dispreferred) response
  β       = temperature controlling divergence from reference (default 0.1)

The reference model runs in no_grad mode and its logprobs are used as a
baseline. The policy model learns to increase the relative logprob of chosen
over rejected vs. the reference, without needing an explicit reward model.
"""

import copy
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

IGNORE_INDEX = -100


# ── LR Schedule (reused from SFT trainer) ────────────────────────────────────

class CosineScheduleWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if s <= self.warmup_steps:
                lr = base_lr * (s / max(1, self.warmup_steps))
            else:
                progress = (s - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"_step": self._step, "base_lrs": self.base_lrs}

    def load_state_dict(self, d):
        self._step = d["_step"]
        self.base_lrs = d["base_lrs"]


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


# ── Log-prob computation ──────────────────────────────────────────────────────

def compute_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sequence sum of log-probabilities for response tokens.

    Args:
        model     : TitanLM (policy or reference)
        input_ids : (B, T)
        labels    : (B, T) — IGNORE_INDEX on prompt tokens

    Returns:
        logprobs  : (B,) — sum of log-probs over response tokens per sequence
    """
    logits, _ = model(input_ids)  # (B, T, V)
    # Shift: predict token t+1 from position t
    shift_logits = logits[:, :-1, :].contiguous()     # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()          # (B, T-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)    # (B, T-1, V)

    # Gather logprob of the actual token at each position
    B, T_minus1, V = log_probs.shape
    token_logprobs = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                       # (B, T-1)

    # Mask out prompt positions (IGNORE_INDEX)
    mask = (shift_labels != IGNORE_INDEX).to(token_logprobs.dtype)
    return (token_logprobs * mask).sum(dim=-1)          # (B,)


# ── DPO Loss ──────────────────────────────────────────────────────────────────

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO loss and diagnostic metrics.

    Returns:
        loss           : scalar
        chosen_rewards : (B,) — chosen implicit rewards
        rejected_rewards: (B,) — rejected implicit rewards
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ── DPO Logger ────────────────────────────────────────────────────────────────

class DPOLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = open(self.log_dir / f"{experiment_name}_train.csv", "w", newline="")
        self.val_log = open(self.log_dir / f"{experiment_name}_val.csv", "w", newline="")
        self.train_writer = csv.writer(self.train_log)
        self.val_writer = csv.writer(self.val_log)
        self.train_writer.writerow(["step", "loss", "reward_margin", "chosen_reward", "rejected_reward", "lr", "elapsed_s"])
        self.val_writer.writerow(["step", "val_loss", "val_reward_margin", "val_accuracy"])

    def log_train(self, step, loss, reward_margin, chosen_r, rejected_r, lr, elapsed):
        self.train_writer.writerow([step, f"{loss:.6f}", f"{reward_margin:.4f}", f"{chosen_r:.4f}", f"{rejected_r:.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
        self.train_log.flush()

    def log_val(self, step, val_loss, val_margin, val_acc):
        self.val_writer.writerow([step, f"{val_loss:.6f}", f"{val_margin:.4f}", f"{val_acc:.4f}"])
        self.val_log.flush()
        print(f"  [DPO Val] step={step} | loss={val_loss:.4f} | reward_margin={val_margin:.4f} | accuracy={val_acc:.2%}")

    def close(self):
        self.train_log.close()
        self.val_log.close()


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dpo(policy, reference, val_loader, device, beta, num_batches=30):
    policy.eval()
    total_loss = 0.0
    total_margin = 0.0
    total_acc = 0.0
    count = 0

    for i, (c_ids, c_labels, r_ids, r_labels) in enumerate(val_loader):
        if i >= num_batches:
            break
        c_ids, c_labels = c_ids.to(device), c_labels.to(device)
        r_ids, r_labels = r_ids.to(device), r_labels.to(device)

        pol_c = compute_logprobs(policy, c_ids, c_labels)
        pol_r = compute_logprobs(policy, r_ids, r_labels)
        ref_c = compute_logprobs(reference, c_ids, c_labels)
        ref_r = compute_logprobs(reference, r_ids, r_labels)

        loss, chosen_rew, rejected_rew = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        margin = (chosen_rew - rejected_rew).mean().item()
        acc = (chosen_rew > rejected_rew).float().mean().item()

        total_loss += loss.item()
        total_margin += margin
        total_acc += acc
        count += 1

    if count == 0:
        return float("inf"), 0.0, 0.0
    policy.train()
    return total_loss / count, total_margin / count, total_acc / count


# ── Main Training Loop ────────────────────────────────────────────────────────

def train_dpo(cfg: Dict, model, reference_model, train_dataset, val_dataset, device):
    """
    Full DPO training loop.

    Args:
        cfg              : full config dict (parsed YAML)
        model            : TitanLM policy (will be trained)
        reference_model  : TitanLM reference (frozen SFT model)
        train_dataset    : TitanDPODataset
        val_dataset      : TitanDPODataset
        device           : torch.device
    """
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]

    beta = train_cfg.get("beta", 0.1)
    max_steps = train_cfg["max_steps"]
    grad_accum_steps = train_cfg["gradient_accumulation_steps"]
    clip_grad = train_cfg["clip_grad_norm"]
    log_interval = train_cfg["log_interval"]
    eval_interval = train_cfg["eval_interval"]
    save_interval = train_cfg["save_interval"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_cfg.get("val_batch_size", 2),
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        betas=(0.9, 0.95),
    )
    scheduler = CosineScheduleWithWarmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=max_steps,
        min_lr_ratio=train_cfg.get("lr_min_ratio", 0.1),
    )

    logger = DPOLogger(log_cfg["log_dir"], log_cfg["experiment_name"])
    reference_model.eval()

    print(f"[DPO] Starting training | beta={beta} | max_steps={max_steps} | lr={train_cfg['learning_rate']:.2e}")
    print(f"[DPO] Training pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    model.train()
    step = 0
    micro_step = 0
    start_time = time.time()
    accum_loss = 0.0
    accum_margin = 0.0
    accum_chosen_r = 0.0
    accum_rejected_r = 0.0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            c_ids, c_labels, r_ids, r_labels = [t.to(device) for t in batch]

            # Policy logprobs
            pol_chosen = compute_logprobs(model, c_ids, c_labels)
            pol_rejected = compute_logprobs(model, r_ids, r_labels)

            # Reference logprobs (no grad)
            with torch.no_grad():
                ref_chosen = compute_logprobs(reference_model, c_ids, c_labels)
                ref_rejected = compute_logprobs(reference_model, r_ids, r_labels)

            loss, chosen_rew, rejected_rew = dpo_loss(
                pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta
            )

            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_margin += (chosen_rew - rejected_rew).mean().item()
            accum_chosen_r += chosen_rew.mean().item()
            accum_rejected_r += rejected_rew.mean().item()
            micro_step += 1

            # Optimizer step every grad_accum_steps micro-batches
            if micro_step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % log_interval == 0:
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = accum_loss / log_interval
                    avg_margin = accum_margin / log_interval
                    avg_chosen = accum_chosen_r / log_interval
                    avg_rejected = accum_rejected_r / log_interval
                    print(
                        f"  step={step:5d} | loss={avg_loss:.4f} | "
                        f"margin={avg_margin:+.3f} | lr={lr:.2e} | t={elapsed:.0f}s"
                    )
                    logger.log_train(step, avg_loss, avg_margin, avg_chosen, avg_rejected, lr, elapsed)
                    accum_loss = accum_margin = accum_chosen_r = accum_rejected_r = 0.0

                if step % eval_interval == 0:
                    val_loss, val_margin, val_acc = evaluate_dpo(
                        model, reference_model, val_loader, device, beta,
                        eval_cfg.get("num_eval_batches", 30)
                    )
                    logger.log_val(step, val_loss, val_margin, val_acc)

                if step % save_interval == 0:
                    ckpt_path = checkpoint_dir / f"step_{step}.pt"
                    save_checkpoint(model, optimizer, scheduler, step, ckpt_path)
                    print(f"  [DPO] Checkpoint saved: {ckpt_path.name}")

                if step >= max_steps:
                    break

    # Final save and eval
    final_path = checkpoint_dir / "final.pt"
    save_checkpoint(model, optimizer, scheduler, step, final_path)
    print(f"\n[DPO] Training complete. Final checkpoint: {final_path}")

    val_loss, val_margin, val_acc = evaluate_dpo(
        model, reference_model, val_loader, device, beta
    )
    logger.log_val(step, val_loss, val_margin, val_acc)
    logger.close()

    print(f"[DPO] Final — loss={val_loss:.4f} | reward_margin={val_margin:+.3f} | accuracy={val_acc:.2%}")
    if val_acc >= 0.7:
        print("[DPO] GATE PASSED: accuracy >= 70% — model successfully aligned.")
    else:
        print("[DPO] WARNING: accuracy < 70% — consider more data or additional epochs.")
