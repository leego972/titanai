"""
Streaming pretraining for upscaled Titan-330M (depth-upscaled from 109M).

Streams FineWeb-Edu directly from HuggingFace (no terabytes pre-downloaded),
tokenizes on-the-fly with the existing 32k tokenizer, packs into 2048-token
sequences, and trains with full safety rails:
  - bf16 mixed precision
  - gradient clipping
  - NaN/Inf detection → emergency checkpoint and halt
  - cosine LR schedule with warmup
  - eval every N steps on a held-out FineWeb slice
  - save best-by-val-loss + periodic step checkpoints

Designed for: RTX 4090, 24 GB VRAM, ~30k tokens/sec on Titan-330M.
"""
import argparse, json, math, os, sys, time, signal
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, "/workspace/titanai")
from model.titan_model import build_model
from tokenizers import Tokenizer
from datasets import load_dataset

# ─── CLI ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--init-from", required=True, help="upscaled init checkpoint .pt")
ap.add_argument("--resume", default=None, help="resume from step_N.pt (overrides --init-from for weights/optim/sched)")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--tokenizer", default="/workspace/titanai/tokenizer/titan_32k/tokenizer.json")
ap.add_argument("--seq-len", type=int, default=2048)
ap.add_argument("--batch-size", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lr-min-ratio", type=float, default=0.1)
ap.add_argument("--warmup-steps", type=int, default=500)
ap.add_argument("--max-steps", type=int, default=100000)  # ~6.5B tokens at eff_batch=32 × seq=2048
ap.add_argument("--weight-decay", type=float, default=0.1)
ap.add_argument("--clip-grad", type=float, default=1.0)
ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
ap.add_argument("--dataset-config", default="sample-10BT")
ap.add_argument("--val-hash-mod", type=int, default=64, help="val examples = id_hash %% mod == 0 (~1.5%%)")
ap.add_argument("--divergence-patience", type=int, default=3, help="halt if val_loss rises this many evals in a row")
ap.add_argument("--force-dropout", type=float, default=0.0, help="override dropout from checkpoint config")
ap.add_argument("--seed", type=int, default=20260417)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(args.seed)
device = torch.device("cuda")

# ─── Load tokenizer ─────────────────────────────────────────────────────────
print(f"[pretrain] loading tokenizer {args.tokenizer}")
tok = Tokenizer.from_file(args.tokenizer)
VOCAB = tok.get_vocab_size()
EOS = tok.token_to_id("</s>") or tok.token_to_id("<eos>") or tok.token_to_id("<|endoftext|>") or 2
print(f"[pretrain] vocab={VOCAB} eos={EOS}")

# ─── Build model from upscaled init ─────────────────────────────────────────
print(f"[pretrain] loading init {args.init_from}")
ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
raw_cfg = ck["config"]
mcfg = raw_cfg.get("model", raw_cfg)
model_cfg = {
    "architecture": mcfg.get("architecture", "decoder_transformer"),
    "vocab_size": mcfg["vocab_size"],
    "d_model": mcfg["d_model"],
    "n_heads": mcfg["n_heads"],
    "n_layers": mcfg["n_layers"],
    "d_ff": mcfg["d_ff"],
    "max_seq_len": mcfg["max_seq_len"],
    "dropout": args.force_dropout,  # 0.0 by default for pretrain stability
    "tie_embeddings": mcfg.get("tie_embeddings", True),
}
model = build_model({"model": model_cfg})
missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
print(f"[pretrain] loaded weights — missing={len(missing)} unexpected={len(unexpected)}")
model = model.to(device)
n_real = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[pretrain] trainable params: {n_real/1e6:.2f}M")

# ─── Optimizer + scheduler ──────────────────────────────────────────────────
no_decay = ["bias", "ln", "norm", "embedding"]
decay_params, nodecay_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if any(nd in n.lower() for nd in no_decay):
        nodecay_params.append(p)
    else:
        decay_params.append(p)

optim = AdamW(
    [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=True,
)

def lr_lambda(step):
    if step < args.warmup_steps:
        return float(step) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cos

sched = LambdaLR(optim, lr_lambda=lr_lambda)

# ─── Streaming data loader (on-the-fly tokenization + packing) ──────────────
import hashlib
def _is_val(example_id):
    h = int(hashlib.md5(str(example_id).encode()).hexdigest()[:8], 16)
    return (h % args.val_hash_mod) == 0

def stream_iter(want_val: bool):
    """Infinite stream of packed (seq_len+1)-length token sequences.
    Deterministic train/val split by hash(id) — guarantees zero overlap."""
    epoch = 0
    while True:
        ds = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
        ds = ds.shuffle(seed=args.seed + epoch, buffer_size=10000)
        buf = []
        for ex in ds:
            ex_id = ex.get("id") or ex.get("url") or ""
            if _is_val(ex_id) != want_val:
                continue
            text = ex.get("text", "")
            if not text:
                continue
            ids = tok.encode(text).ids + [EOS]
            buf.extend(ids)
            while len(buf) >= args.seq_len + 1:
                chunk = buf[: args.seq_len + 1]
                buf = buf[args.seq_len:]
                yield torch.tensor(chunk, dtype=torch.long)
        epoch += 1

def batch_iter(it, batch_size):
    while True:
        batch = [next(it) for _ in range(batch_size)]
        x = torch.stack(batch, dim=0)
        yield x[:, :-1].to(device, non_blocking=True), x[:, 1:].to(device, non_blocking=True)

print("[pretrain] starting data streams…")
train_stream = batch_iter(stream_iter(want_val=False), args.batch_size)
val_stream   = batch_iter(stream_iter(want_val=True),  args.batch_size)

# ─── Optional resume ────────────────────────────────────────────────────────
start_step = 0
if args.resume:
    print(f"[pretrain] RESUMING from {args.resume}")
    rk = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(rk["model_state_dict"])
    if "optimizer_state_dict" in rk:
        optim.load_state_dict(rk["optimizer_state_dict"])
    if "scheduler_state_dict" in rk:
        sched.load_state_dict(rk["scheduler_state_dict"])
    start_step = int(rk.get("step", 0))
    print(f"[pretrain] resumed at step {start_step}")

# warm-up: pull one batch to make sure pipeline works
print("[pretrain] warming up data pipeline (one batch)…")
xb, yb = next(train_stream)
print(f"[pretrain] first batch shape: {tuple(xb.shape)} (input), {tuple(yb.shape)} (labels)")

# ─── Training loop ──────────────────────────────────────────────────────────
log_path = Path(args.out_dir) / "train.jsonl"
event_path = Path(args.out_dir) / "_events.jsonl"
def log_event(d):
    with event_path.open("a") as f:
        f.write(json.dumps(d) + "\n")
def log_step(d):
    with log_path.open("a") as f:
        f.write(json.dumps(d) + "\n")

@torch.no_grad()
def evaluate(n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = next(val_stream)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

best_val = float("inf")
ema_loss = None
loss_window = deque(maxlen=50)
val_history = deque(maxlen=args.divergence_patience + 1)
t_start = time.time()
tokens_seen = 0
model.train()
print(f"[pretrain] training loop start — max_steps={args.max_steps} eff_batch={args.batch_size*args.grad_accum} resume_from_step={start_step}")

for step in range(start_step + 1, args.max_steps + 1):
    optim.zero_grad(set_to_none=True)
    accum_loss = 0.0
    for micro in range(args.grad_accum):
        x, y = next(train_stream)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        (loss / args.grad_accum).backward()
        accum_loss += loss.item() / args.grad_accum
        tokens_seen += x.numel()

    # NaN/Inf check
    if not math.isfinite(accum_loss):
        print(f"[pretrain] !!! non-finite loss at step {step}: {accum_loss}")
        emerg = Path(args.out_dir) / f"emergency_step_{step}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": {"model": model_cfg}}, emerg)
        log_event({"event": "halt_nonfinite", "step": step, "loss": accum_loss, "ckpt": str(emerg)})
        sys.exit(2)

    # gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if not math.isfinite(grad_norm.item()):
        print(f"[pretrain] !!! non-finite grad norm at step {step}: {grad_norm.item()}")
        log_event({"event": "halt_grad_nan", "step": step, "grad_norm": grad_norm.item()})
        sys.exit(2)

    optim.step()
    sched.step()

    loss_window.append(accum_loss)
    ema_loss = accum_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * accum_loss

    if step % args.log_every == 0:
        elapsed = time.time() - t_start
        toks_per_sec = tokens_seen / max(1e-9, elapsed)
        cur_lr = sched.get_last_lr()[0]
        msg = (f"[pretrain] step={step}/{args.max_steps} "
               f"loss={accum_loss:.4f} ema={ema_loss:.4f} "
               f"grad_norm={grad_norm.item():.3f} lr={cur_lr:.2e} "
               f"tok/s={toks_per_sec:.0f} seen={tokens_seen/1e6:.1f}M")
        print(msg, flush=True)
        log_step({"step": step, "loss": accum_loss, "ema": ema_loss,
                  "grad_norm": grad_norm.item(), "lr": cur_lr,
                  "tokens_seen": tokens_seen, "tok_per_sec": toks_per_sec})

    if step % args.eval_every == 0:
        val_loss = evaluate()
        ppl = math.exp(min(20, val_loss))
        cur_lr = sched.get_last_lr()[0]
        print(f"[pretrain] *** EVAL step={step} val_loss={val_loss:.4f} ppl={ppl:.2f} ***", flush=True)
        log_step({"step": step, "val_loss": val_loss, "val_ppl": ppl, "lr": cur_lr, "eval": True})
        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(args.out_dir) / "best.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "val_loss": val_loss,
                        "config": {"model": model_cfg}}, best_path)
            print(f"[pretrain] *** new best val_loss={val_loss:.4f} → {best_path} ***", flush=True)

        # divergence guard: halt if val rises N consecutive evals after warmup is done
        val_history.append(val_loss)
        if step > args.warmup_steps * 4 and len(val_history) == val_history.maxlen:
            strictly_rising = all(val_history[i] < val_history[i+1] for i in range(len(val_history)-1))
            if strictly_rising:
                print(f"[pretrain] !!! VAL DIVERGENCE — last {len(val_history)} evals strictly rising: "
                      f"{[round(v,4) for v in val_history]}. Halting + saving emergency.", flush=True)
                emerg = Path(args.out_dir) / f"emergency_diverge_step_{step}.pt"
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": sched.state_dict(),
                            "val_loss": val_loss,
                            "val_history": list(val_history),
                            "config": {"model": model_cfg}}, emerg)
                log_event({"event": "halt_val_divergence", "step": step,
                           "val_history": list(val_history), "ckpt": str(emerg)})
                sys.exit(3)

    if step % args.save_every == 0:
        ckpt_path = Path(args.out_dir) / f"step_{step}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": {"model": model_cfg}}, ckpt_path)
        print(f"[pretrain] saved {ckpt_path}", flush=True)
        # rotate: keep only last 3 + best
        all_steps = sorted(Path(args.out_dir).glob("step_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
        for old in all_steps[:-3]:
            old.unlink(missing_ok=True)

# final
final_path = Path(args.out_dir) / "final.pt"
torch.save({"step": args.max_steps, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "config": {"model": model_cfg}}, final_path)
print(f"[pretrain] DONE — final saved to {final_path}", flush=True)
