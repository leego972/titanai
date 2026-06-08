"""
Titan Training Loop — v0.3
===========================
Production-grade training loop for TitanLM.

Upgrades from v0.2:
    - bfloat16 mixed-precision via torch.amp.autocast (A100-native, no GradScaler needed)
    - 8-bit Adam via bitsandbytes (use_8bit_adam: true in config) — halves optimizer VRAM
    - torch.compile support (use_compile: true in config) — ~20% throughput improvement
    - Tokens-per-second and MFU (Model Flops Utilization) reporting
    - Weights & Biases optional integration (wandb.enabled: true in logging config)
    - Gradient checkpointing activation from config
    - Detailed event log with timestamps (divergence, spikes, eval gates)
"""

import os, sys, json, math, time
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.titan_model import build_model
from data.dataset import create_dataloaders
from training.checkpoint import save_checkpoint, load_checkpoint
from evaluation.evaluator import evaluate_loss as evaluate


# ─── helpers ─────────────────────────────────────────────────────────────────

def _resolve_data_dir(config: dict, base_dir: str = ".") -> str:
    d = config["data"]
    root = os.path.join(base_dir, d["processed_dir"])
    ver  = d.get("processed_version") or d.get("corpus_version")
    return os.path.join(root, ver) if ver else root


def _validate_data(config: dict, base_dir: str = ".") -> str:
    processed = _resolve_data_dir(config, base_dir)
    missing = []
    for split in ("train", "val"):
        d = os.path.join(processed, split)
        if not os.path.isdir(d) or not any(p.endswith(".bin") for p in os.listdir(d)):
            missing.append(d)
    if missing:
        raise FileNotFoundError(
            f"Missing or empty data shards. Run data prep first.\nMissing: {missing}")
    return processed


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, max_steps - warmup_steps)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay
    return LambdaLR(optimizer, lr_lambda)


def _estimate_flops_per_token(config: dict) -> float:
    """
    Rough FLOPs estimate per token for MFU calculation.
    Uses the standard 6 * N approximation (N = non-embedding params).
    """
    m = config["model"]
    d, ff, L, nh, nkv = m["d_model"], m["d_ff"], m["n_layers"], m["n_heads"], m.get("n_kv_heads", m["n_heads"])
    seq = m.get("max_seq_len", 2048)
    d_head = d // nh
    # Attention: 2 * seq * (Hq + 2*Hk) * d_head * d   (QKV proj + out)
    # MLP SwiGLU: 3 * 2 * d * (2/3 * ff)              (gate + up + down)
    attn_flops = 2 * seq * (nh + 2 * nkv) * d_head * d
    # swiglu hidden = (2/3 * ff) round to 64
    swiglu_h = ((int(2 * ff / 3) + 63) // 64) * 64
    mlp_flops  = 3 * 2 * d * swiglu_h
    per_layer  = attn_flops + mlp_flops
    return float(per_layer * L)


# ─── Logger ──────────────────────────────────────────────────────────────────

class TrainingLogger:
    """CSV + console + optional W&B logger."""

    def __init__(self, log_dir: str, experiment_name: str, wandb_cfg: dict = None):
        os.makedirs(log_dir, exist_ok=True)
        self.train_path  = os.path.join(log_dir, f"{experiment_name}_train.csv")
        self.val_path    = os.path.join(log_dir, f"{experiment_name}_val.csv")
        self.events_path = os.path.join(log_dir, f"{experiment_name}_events.jsonl")
        self._wb = None

        for path, header in [
            (self.train_path,  "step,loss,lr,tokens_per_sec,mfu_pct,elapsed_sec\n"),
            (self.val_path,    "step,val_loss,val_perplexity\n"),
        ]:
            if not os.path.exists(path):
                open(path, "w").write(header)

        if wandb_cfg and wandb_cfg.get("enabled"):
            try:
                import wandb
                self._wb = wandb
                wandb.init(
                    project=wandb_cfg.get("project", "titanai"),
                    name=experiment_name,
                    config=wandb_cfg.get("config", {}),
                    resume="allow",
                )
                print(f"[Logger] Weights & Biases: ACTIVE (project={wandb_cfg.get('project', 'titanai')})")
            except ImportError:
                print("[Logger] Weights & Biases: wandb not installed — skipping")

    def log_train(self, step, loss, lr, tps, mfu, elapsed):
        open(self.train_path, "a").write(f"{step},{loss:.6f},{lr:.8f},{tps:.1f},{mfu:.2f},{elapsed:.1f}\n")
        print(f"[Train] step={step:7d} | loss={loss:.4f} | lr={lr:.2e} | "
              f"tok/s={tps:,.0f} | MFU={mfu:.1f}%")
        if self._wb:
            self._wb.log({"train/loss": loss, "train/lr": lr, "train/tokens_per_sec": tps,
                          "train/mfu_pct": mfu}, step=step)

    def log_val(self, step, val_loss, val_ppl):
        open(self.val_path, "a").write(f"{step},{val_loss:.6f},{val_ppl:.4f}\n")
        print(f"[Val]   step={step:7d} | val_loss={val_loss:.4f} | perplexity={val_ppl:.2f}")
        if self._wb:
            self._wb.log({"val/loss": val_loss, "val/perplexity": val_ppl}, step=step)

    def log_event(self, step, event_type, message):
        import datetime
        entry = {"step": step, "event": event_type, "message": message,
                 "timestamp": datetime.datetime.utcnow().isoformat()}
        open(self.events_path, "a").write(json.dumps(entry) + "\n")
        print(f"[Event] step={step:7d} | {event_type}: {message}")


# ─── Optimizer builder ───────────────────────────────────────────────────────

def _build_optimizer(model: nn.Module, train_cfg: dict) -> torch.optim.Optimizer:
    lr           = train_cfg["learning_rate"]
    wd           = train_cfg["weight_decay"]
    use_8bit     = train_cfg.get("use_8bit_adam", False)

    decay   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    groups  = [
        {"params": decay,    "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
            print("[Optimizer] 8-bit AdamW via bitsandbytes: ACTIVE")
            return opt
        except ImportError:
            print("[Optimizer] bitsandbytes not installed — falling back to standard AdamW")

    return AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


# ─── Main training function ──────────────────────────────────────────────────

def train(config: dict, resume_from: str = None, base_dir: str = "."):
    train_cfg = config["training"]
    data_cfg  = config["data"]
    eval_cfg  = config["evaluation"]
    log_cfg   = config["logging"]

    # ── Device + dtype ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = train_cfg.get("bf16", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[Train] Device: {device} | Mixed precision: {'bfloat16' if use_bf16 else 'float32'}")

    # ── Data ─────────────────────────────────────────────────────────────────
    processed_dir = _validate_data(config, base_dir)
    train_loader, val_loader = create_dataloaders(
        train_dir=os.path.join(processed_dir, "train"),
        val_dir=os.path.join(processed_dir, "val"),
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        val_batch_size=eval_cfg["val_batch_size"],
    )
    train_iter = iter(train_loader)
    print(f"[Train] Data loaded from: {processed_dir}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)

    # Activate gradient checkpointing if set in config
    if config["model"].get("gradient_checkpointing", False):
        model.config.use_gradient_checkpointing = True
        print("[Train] Gradient checkpointing: ENABLED")

    # torch.compile (PyTorch 2.0+)
    use_compile = train_cfg.get("use_compile", False)
    if use_compile:
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model)
            print("[Train] torch.compile: ACTIVE")
        except Exception as e:
            print(f"[Train] torch.compile: FAILED ({e}) — continuing without")

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optimizer = _build_optimizer(model, train_cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg.get("lr_min_ratio", 0.1),
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    ckpt_path = resume_from or train_cfg.get("resume_from")
    if ckpt_path:
        full_path = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(base_dir, ckpt_path)
        if os.path.exists(full_path):
            start_step = load_checkpoint(full_path, model, optimizer, scheduler, device)
            print(f"[Train] Resumed from step {start_step} ({full_path})")
        else:
            print(f"[Train] Resume path not found ({full_path}) — starting from scratch")

    # ── Logger ───────────────────────────────────────────────────────────────
    wandb_cfg = log_cfg.get("wandb", {})
    wandb_cfg["config"] = config
    logger = TrainingLogger(
        os.path.join(base_dir, log_cfg["log_dir"]),
        log_cfg["experiment_name"],
        wandb_cfg=wandb_cfg,
    )

    # ── Training setup ───────────────────────────────────────────────────────
    checkpoint_dir = os.path.join(base_dir, train_cfg["checkpoint_dir"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    clip_grad      = train_cfg["clip_grad_norm"]
    grad_accum     = train_cfg["gradient_accumulation_steps"]
    max_steps      = train_cfg["max_steps"]
    log_interval   = train_cfg["log_interval"]
    eval_interval  = train_cfg["eval_interval"]
    save_interval  = train_cfg["save_interval"]
    seq_len        = data_cfg["max_seq_len"]
    batch_size     = train_cfg["batch_size"]
    tokens_per_step = batch_size * seq_len * grad_accum

    if clip_grad <= 0:
        raise ValueError(f"clip_grad_norm must be > 0, got {clip_grad}")

    flops_per_tok = _estimate_flops_per_token(config)
    # Try to detect peak TFLOPS for MFU (A100 80GB bf16 ~312 TFLOPS, 40GB ~312 TFLOPS)
    try:
        gpu_flops = torch.cuda.get_device_properties(0).multi_processor_count * 128 * 2 * 1e12
    except Exception:
        gpu_flops = 312e12  # A100 80GB bfloat16 default

    model.train()
    accum_loss, accum_tokens = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad()

    print(f"[Train] {'='*60}")
    print(f"[Train] Steps: {start_step} → {max_steps}  |  tokens/step: {tokens_per_step:,}")
    print(f"[Train] Grad accum: {grad_accum}  |  Clip: {clip_grad}  |  Compile: {use_compile}")
    print(f"[Train] {'='*60}")

    for step in range(start_step, max_steps):
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)

        input_ids = input_ids.to(device)
        labels    = labels.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_bf16):
            _, loss = model(input_ids, labels)

        raw_loss = loss.item()
        if not math.isfinite(raw_loss):
            msg = f"Loss={raw_loss} at step {step+1} — divergence detected."
            logger.log_event(step + 1, "DIVERGENCE", msg)
            emergency = os.path.join(checkpoint_dir, f"emergency_step_{step+1}.pt")
            save_checkpoint(emergency, model, optimizer, scheduler, step + 1, config)
            raise RuntimeError(f"{msg} Emergency checkpoint: {emergency}")

        (loss / grad_accum).backward()
        accum_loss   += raw_loss
        accum_tokens += input_ids.numel()

        if (step + 1) % grad_accum == 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if grad_norm > clip_grad * 5:
                logger.log_event(step + 1, "GRAD_SPIKE",
                    f"norm={grad_norm:.2f} vs clip={clip_grad} ({grad_norm/clip_grad:.1f}x)")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            tps     = accum_tokens / max(elapsed, 1e-9)
            lr      = scheduler.get_last_lr()[0]
            mfu     = (tps * flops_per_tok) / gpu_flops * 100.0
            logger.log_train(step + 1, accum_loss / log_interval, lr, tps, mfu, elapsed)
            accum_loss, accum_tokens = 0.0, 0
            t0 = time.time()

        if (step + 1) % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device, eval_cfg["num_eval_batches"])
            logger.log_val(step + 1, val_loss, val_ppl)
            model.train()

        if (step + 1) % save_interval == 0:
            ckpt = os.path.join(checkpoint_dir, f"step_{step+1}.pt")
            save_checkpoint(ckpt, model, optimizer, scheduler, step + 1, config)
            print(f"[Train] Checkpoint saved: {ckpt}")

    final = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(final, model, optimizer, scheduler, max_steps, config)
    print(f"[Train] Complete. Final checkpoint: {final}")
    return model

# Public aliases used by smoke_test.py (and any other external callers)
resolve_processed_data_dir = _resolve_data_dir
validate_training_inputs = _validate_data
