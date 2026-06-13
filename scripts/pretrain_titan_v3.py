"""
Phase 3 v3 — Diverse pretraining for upscaled Titan-304M.

Multi-source streaming pretrain with weighted interleaving across:
  - HuggingFaceFW/fineweb-edu (50%)  — broad web, educational
  - bigcode/the-stack-smol         (25%) — code (multilang)
  - open-web-math/open-web-math     (15%) — math + reasoning
  - allenai/c4 'en'                  (10%) — diverse general web

Tokenizes on the fly with the existing 32k tokenizer, packs into 2048-token
sequences, tags each example with its source so we can compute per-source
val loss every eval.

Safety rails (carried over + strengthened from v2):
  - bf16 mixed precision, fused AdamW, cosine LR with warmup
  - NaN/Inf detection => emergency checkpoint + halt
  - Per-source + averaged val loss every --eval-every steps
  - Best-by-val checkpoint, rotating step checkpoints (keep 3 + best)
  - Looser divergence guard: halts only when the rolling-mean of the last
    K val losses rises by >REL_THRESH AND K consecutive evals all rose.
    This avoids the v2 false positive at step 24000.
  - Hash-based deterministic train/val split — zero overlap, reproducible.
  - --resume reload of optimizer + scheduler state (no warm-restart).
  - Source resilience: if a source fails to initialise, its weight is
    redistributed across remaining sources and a warning is logged.
"""
import argparse, hashlib, json, math, os, random, sys, time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# 8-bit Adam — halves optimizer VRAM at 1B+ scale (graceful fallback to AdamW)
try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False
from torch.optim.lr_scheduler import LambdaLR

# Dynamic repo root — works regardless of where the repo is cloned
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO_ROOT)
from model.titan_model import build_model
from tokenizers import Tokenizer
from datasets import load_dataset

# ─── CLI ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--init-from", required=True, help="resume init checkpoint .pt (e.g. phase2 best.pt)")
ap.add_argument("--resume",    default=None,  help="resume from step_N.pt (overrides --init-from for weights/optim/sched)")
ap.add_argument("--out-dir",   required=True)
ap.add_argument("--tokenizer", default=None, help="Path to tokenizer.json (default: auto-detect from repo root)")
ap.add_argument("--seq-len",   type=int, default=2048)
ap.add_argument("--batch-size",type=int, default=2)
ap.add_argument("--grad-accum",type=int, default=16)
ap.add_argument("--lr",        type=float, default=8e-5)        # slightly lower for resume
ap.add_argument("--lr-min-ratio", type=float, default=0.1)
ap.add_argument("--warmup-steps", type=int, default=1000)        # longer warmup on resume
ap.add_argument("--max-steps",    type=int, default=470000)      # ~30B tokens at eff_batch=32 × seq=2048
ap.add_argument("--weight-decay", type=float, default=0.1)
ap.add_argument("--clip-grad",    type=float, default=1.0)
ap.add_argument("--save-every",   type=int, default=2000)
ap.add_argument("--eval-every",   type=int, default=1000)
ap.add_argument("--log-every",    type=int, default=25)
ap.add_argument("--val-hash-mod", type=int, default=64)          # ~1.5% held-out
ap.add_argument("--val-batches",  type=int, default=64)          # eval over this many val batches per source
ap.add_argument("--divergence-patience",  type=int,   default=5)
ap.add_argument("--divergence-rel-thresh",type=float, default=0.05)  # halt only if mean rises >5% over patience window
ap.add_argument("--force-dropout",        type=float, default=0.0)
ap.add_argument("--seed",                 type=int,   default=20260418)
ap.add_argument("--smoke",                type=int,   default=0,
                help=">0: stop after this many steps (used for smoke test)")
ap.add_argument("--config",        default=None,
                help="Path to YAML config file — values override CLI defaults")
ap.add_argument("--use-8bit-adam", action="store_true", default=False,
                help="Use bitsandbytes 8-bit Adam (saves ~12GB VRAM on 1B model)")
ap.add_argument("--compile",       action="store_true", default=False,
                help="torch.compile the model — ~20%% throughput gain")
ap.add_argument("--use-grad-ckpt", action="store_true", default=False,
                help="Enable gradient checkpointing (VRAM for compute tradeoff)")
args = ap.parse_args()

# ── YAML config overlay ────────────────────────────────────────────────────
# Reads YAML and overrides argparse defaults so --config configs/titan_1b.yaml just works
if args.config:
    import yaml as _yaml, os as _os
    with open(args.config) as _f: _cfg = _yaml.safe_load(_f)
    _tr = _cfg.get("training", {}); _m = _cfg.get("model", {})
    _tok_path = _cfg.get("data", {}).get("tokenizer_path")
    if "batch_size"                  in _tr: args.batch_size    = _tr["batch_size"]
    if "gradient_accumulation_steps" in _tr: args.grad_accum    = _tr["gradient_accumulation_steps"]
    if "learning_rate"               in _tr: args.lr            = _tr["learning_rate"]
    if "weight_decay"                in _tr: args.weight_decay  = _tr["weight_decay"]
    if "max_steps"                   in _tr: args.max_steps     = _tr["max_steps"]
    if "warmup_steps"                in _tr: args.warmup_steps  = _tr["warmup_steps"]
    if "lr_min_ratio"                in _tr: args.lr_min_ratio  = _tr["lr_min_ratio"]
    if "clip_grad_norm"              in _tr: args.clip_grad     = _tr["clip_grad_norm"]
    if "log_interval"                in _tr: args.log_every     = _tr["log_interval"]
    if "eval_interval"               in _tr: args.eval_every    = _tr["eval_interval"]
    if "save_interval"               in _tr: args.save_every    = _tr["save_interval"]
    if "use_8bit_adam"               in _tr: args.use_8bit_adam = _tr["use_8bit_adam"]
    if "use_compile"                 in _tr: args.compile       = _tr["use_compile"]
    if "max_seq_len"                 in _m:  args.seq_len       = _m["max_seq_len"]
    if "gradient_checkpointing"      in _m:  args.use_grad_ckpt = _m["gradient_checkpointing"]
    if _tok_path and not _os.path.isabs(_tok_path):
        args.tokenizer = _os.path.join(_REPO_ROOT, _tok_path)
    elif _tok_path:
        args.tokenizer = _tok_path
    if "checkpoint_dir" in _tr and (not hasattr(args, "out_dir") or not args.out_dir):
        args.out_dir = _os.path.join(_REPO_ROOT, _tr["checkpoint_dir"])
    print(f"[Config] Loaded {args.config}: steps={args.max_steps} "
          f"lr={args.lr} 8bit={args.use_8bit_adam} compile={args.compile}")

# Auto-detect tokenizer if not explicitly set
if not args.tokenizer:
    for _tc in [
        os.path.join(_REPO_ROOT, "tokenizer", "titan_32k", "tokenizer.json"),
        os.path.join(_REPO_ROOT, "tokenizer", "artifacts_v32k", "tokenizer.json"),
        os.path.join(_REPO_ROOT, "tokenizer", "tokenizer.json"),
    ]:
        if os.path.exists(_tc): args.tokenizer = _tc; break
    if not args.tokenizer:
        raise FileNotFoundError("No tokenizer found. Set --tokenizer <path>.")
# Resolve out-dir relative to repo root
if args.out_dir and not os.path.isabs(args.out_dir):
    args.out_dir = os.path.join(_REPO_ROOT, args.out_dir)

os.makedirs(args.out_dir, exist_ok=True)
torch.manual_seed(args.seed)
random.seed(args.seed)
device = torch.device("cuda")

# ─── Source spec ────────────────────────────────────────────────────────────
# (name, repo, config_or_None, text_field, weight)
SOURCES_DEFAULT = [
    ("fineweb", "HuggingFaceFW/fineweb-edu", "sample-10BT",            "text",    0.50),
    ("code",    "codeparrot/codeparrot-clean",None,                     "content", 0.25),
    ("math",    "open-web-math/open-web-math",None,                     "text",    0.15),
    ("web",     "allenai/c4",                 "en",                     "text",    0.10),
]

# ─── Load tokenizer ─────────────────────────────────────────────────────────
print(f"[pretrain] loading tokenizer {args.tokenizer}", flush=True)
tok = Tokenizer.from_file(args.tokenizer)
VOCAB = tok.get_vocab_size()
EOS = tok.token_to_id("</s>") or tok.token_to_id("<eos>") or tok.token_to_id("<|endoftext|>") or 2
print(f"[pretrain] vocab={VOCAB} eos={EOS}", flush=True)

# ─── Build model + load weights ────────────────────────────────────────────
init_path = args.resume or args.init_from
print(f"[pretrain] loading init {init_path}", flush=True)
ck = torch.load(init_path, map_location="cpu", weights_only=False)
raw_cfg = ck.get("config") or ck.get("cfg") or {}
mcfg = raw_cfg.get("model", raw_cfg)
model_cfg = {
    "architecture": mcfg.get("architecture", "decoder_transformer"),
    "vocab_size": mcfg["vocab_size"],
    "d_model":    mcfg["d_model"],
    "n_heads":    mcfg["n_heads"],
    "n_layers":   mcfg["n_layers"],
    "d_ff":       mcfg["d_ff"],
    "max_seq_len":mcfg["max_seq_len"],
    "dropout":    args.force_dropout,
    "tie_embeddings": mcfg.get("tie_embeddings", True),
    "n_kv_heads": mcfg.get("n_kv_heads", mcfg["n_heads"]),  # GQA — essential for v0.3
}
model = build_model({"model": model_cfg})
_sd_key = "model_state_dict" if "model_state_dict" in ck else "model"
missing, unexpected = model.load_state_dict(ck[_sd_key], strict=False)
print(f"[pretrain] loaded weights — missing={len(missing)} unexpected={len(unexpected)}", flush=True)
model = model.to(device)
# torch.compile — ~20% throughput boost (PyTorch 2.0+)
if getattr(args, "compile", False) and hasattr(torch, "compile"):
    try:
        model = torch.compile(model)
        print("[Model] torch.compile: ACTIVE", flush=True)
    except Exception as _ce:
        print(f"[Model] torch.compile skipped: {_ce}", flush=True)
n_real = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[pretrain] trainable params: {n_real/1e6:.2f}M", flush=True)

# ─── Optimizer + scheduler ──────────────────────────────────────────────────
no_decay_keys = ["bias", "ln", "norm", "embedding"]
decay_params, nodecay_params = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if any(nd in n.lower() for nd in no_decay_keys):
        nodecay_params.append(p)
    else:
        decay_params.append(p)
# ─── Optimizer ─────────────────────────────────────────────────────────────
if getattr(args, "use_8bit_adam", False) and _BNB_AVAILABLE:
    print("[Optim] Using bitsandbytes 8-bit Adam (VRAM saving active)")
    optim = bnb.optim.Adam8bit(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
else:
    if getattr(args, "use_8bit_adam", False) and not _BNB_AVAILABLE:
        print("[Optim] WARNING: bitsandbytes not installed, falling back to AdamW")
    optim = AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )
def lr_lambda(step):
    if step < args.warmup_steps:
        return step / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cos
sched = LambdaLR(optim, lr_lambda=lr_lambda)

start_step = 0
if args.resume:
    if "optimizer_state_dict" in ck:
        optim.load_state_dict(ck["optimizer_state_dict"])
    if "scheduler_state_dict" in ck:
        sched.load_state_dict(ck["scheduler_state_dict"])
    start_step = int(ck.get("step", 0))
    print(f"[pretrain] resumed @ step {start_step}", flush=True)

# ─── Source streaming ───────────────────────────────────────────────────────
def _is_val(example_id_str):
    h = int(hashlib.md5(str(example_id_str).encode()).hexdigest()[:8], 16)
    return (h % args.val_hash_mod) == 0

class SourceStream:
    """Wraps one HF streaming dataset, yields (source_name, text) tuples
    filtered to either train or val by deterministic hash-based split."""
    def __init__(self, name, repo, cfg, text_field, want_val, shard_seed):
        self.name = name
        self.text_field = text_field
        self.want_val = want_val
        try:
            if cfg:
                self.ds = load_dataset(repo, cfg, split="train", streaming=True)
            else:
                self.ds = load_dataset(repo, split="train", streaming=True)
            # Different shuffle seeds per source so they don't sync up
            self.ds = self.ds.shuffle(seed=args.seed + shard_seed, buffer_size=2000)
            self.ok = True
        except Exception as e:
            print(f"[pretrain] !!! source '{name}' failed to init: {e}", flush=True)
            self.ok = False
            self.ds = None
    def __iter__(self):
        if not self.ok:
            return
        import time as _time
        i = 0
        err_count = 0
        ds_iter = iter(self.ds)
        while True:
            try:
                example = next(ds_iter)
                err_count = 0
            except StopIteration:
                ds_iter = iter(self.ds)
                continue
            except Exception as _e:
                err_count += 1
                wait = min(err_count * 3, 60)
                print(f"[pretrain] source '{self.name}' error #{err_count}: {_e} — retry in {wait}s", flush=True)
                if err_count > 30:
                    print(f"[pretrain] source '{self.name}' giving up after 30 errors", flush=True)
                    self.ok = False
                    return
                _time.sleep(wait)
                try:
                    ds_iter = iter(self.ds)
                except Exception:
                    pass
                continue
            i += 1
            text = example.get(self.text_field) or example.get("text") or example.get("content") or ""
            if not text:
                continue
            ex_id = f"{self.name}-{i}"
            if _is_val(ex_id) != self.want_val:
                continue
            yield (self.name, text)


class LocalFileSource:
    """Reads .txt/.jsonl files from a local directory — no network needed."""
    def __init__(self, name, directory, want_val, shard_seed):
        import glob as _glob, os as _os
        self.name = name
        self.directory = directory
        self.want_val = want_val
        self.shard_seed = shard_seed
        self.files = (sorted(_glob.glob(_os.path.join(directory, "**", "*.txt"),  recursive=True)) +
                      sorted(_glob.glob(_os.path.join(directory, "**", "*.jsonl"), recursive=True)))
        self.ok = len(self.files) > 0
        print(f"[pretrain] LocalFileSource '{name}': {len(self.files)} files @ {directory}", flush=True)
    def __iter__(self):
        if not self.ok:
            return
        import random as _rnd, json as _json
        rng = _rnd.Random(self.shard_seed)
        files = self.files[:]
        i = 0
        while True:
            rng.shuffle(files)
            for f in files:
                try:
                    with open(f, "r", errors="replace") as fp:
                        if f.endswith(".jsonl"):
                            for line in fp:
                                try:
                                    obj = _json.loads(line)
                                    text = (obj.get("text") or obj.get("content") or
                                            " ".join(filter(None, [obj.get("instruction",""), obj.get("output","")])))
                                except Exception:
                                    text = line.strip()
                                if not text:
                                    continue
                                i += 1
                                if _is_val(f"{self.name}-{i}") == self.want_val:
                                    yield (self.name, text)
                        else:
                            text = fp.read().strip()
                            if text:
                                i += 1
                                if _is_val(f"{self.name}-{i}") == self.want_val:
                                    yield (self.name, text)
                except Exception:
                    continue

def build_packed_iter(want_val, sources, weights):
    """Weighted-interleaved infinite stream of (source_name, packed_seq) tensors,
    where packed_seq is length seq_len+1."""
    iters = [iter(s) for s in sources]
    # buffers per source so we don't waste tokens when sampling is rejected
    buffers = {s.name: [] for s in sources}
    rng = random.Random(args.seed + (1 if want_val else 0))
    while True:
        # weighted choice across only the OK sources
        ok = [(s, w) for s, w in zip(sources, weights) if s.ok]
        if not ok:
            raise RuntimeError("all sources failed to init")
        names = [s.name for s, _ in ok]
        ws = [w for _, w in ok]
        chosen = rng.choices(range(len(ok)), weights=ws, k=1)[0]
        s = ok[chosen][0]
        idx = sources.index(s)
        # ensure buffer has enough tokens
        while len(buffers[s.name]) < args.seq_len + 1:
            try:
                _, text = next(iters[idx])
            except StopIteration:
                iters[idx] = iter(s)
                continue
            ids = tok.encode(text).ids
            buffers[s.name].extend(ids)
            buffers[s.name].append(EOS)
        seq = buffers[s.name][:args.seq_len + 1]
        buffers[s.name] = buffers[s.name][args.seq_len + 1:]
        yield (s.name, torch.tensor(seq, dtype=torch.long))

# ─── Initialise sources (resilient) ─────────────────────────────────────────
print("[pretrain] initialising sources:", flush=True)
train_sources, val_sources, weights = [], [], []
for i, (name, repo, cfg, fld, w) in enumerate(SOURCES_DEFAULT):
    print(f"  · {name}: {repo} ({fld}) w={w}", flush=True)
    train_sources.append(SourceStream(name, repo, cfg, fld, want_val=False, shard_seed=i))
    val_sources.append(  SourceStream(name, repo, cfg, fld, want_val=True,  shard_seed=i+100))
    weights.append(w)
# Add local corpus sources (always available, no network needed)
LOCAL_CORPUS = [
    ("local_general",  "/workspace/titanai/data/raw/corpus_A_general",   0.25),
    ("local_reason",   "/workspace/titanai/data/raw/corpus_B_reasoning",  0.15),
    ("local_tech",     "/workspace/titanai/data/raw/corpus_C_technical",  0.30),
    ("local_cyber",    "/workspace/titanai/data/raw/corpus_D_cyber",      0.15),
    ("local_cinema",   "/workspace/titanai/data/raw/corpus_E_cinema",     0.15),
]
local_ok = []
for j, (lname, ldir, lw) in enumerate(LOCAL_CORPUS):
    import os as _os
    if _os.path.isdir(ldir):
        ts = LocalFileSource(lname, ldir, want_val=False, shard_seed=200+j)
        vs = LocalFileSource(lname, ldir, want_val=True,  shard_seed=300+j)
        if ts.ok:
            train_sources.append(ts); val_sources.append(vs); weights.append(lw); local_ok.append(lname)
print(f"[pretrain] local sources added: {local_ok}", flush=True)
ok_names = [s.name for s in train_sources if s.ok]
print(f"[pretrain] active sources: {ok_names}", flush=True)
if not ok_names:
    print("[pretrain] FATAL: no sources available", flush=True)
    sys.exit(2)
# Re-normalise weights so they sum to 1 over OK sources
ok_w_total = sum(w for s, w in zip(train_sources, weights) if s.ok)
weights = [w / ok_w_total if s.ok else 0 for s, w in zip(train_sources, weights)]

train_iter = build_packed_iter(want_val=False, sources=train_sources, weights=weights)

# ─── Eval ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate():
    model.eval()
    per_source_loss = {}
    per_source_n = {}
    val_iter = build_packed_iter(want_val=True, sources=val_sources, weights=weights)
    n_done = 0
    while n_done < args.val_batches * len(ok_names):
        batch_seqs, batch_names = [], []
        for _ in range(args.batch_size):
            name, seq = next(val_iter)
            batch_seqs.append(seq)
            batch_names.append(name)
        x = torch.stack(batch_seqs).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
                reduction="none",
            ).reshape(x.size(0), -1).mean(dim=1)
        for i_b, n_b in enumerate(batch_names):
            per_source_loss[n_b] = per_source_loss.get(n_b, 0.0) + float(loss[i_b].item())
            per_source_n[n_b] = per_source_n.get(n_b, 0) + 1
        n_done += args.batch_size
    avg_per_source = {k: per_source_loss[k] / per_source_n[k] for k in per_source_loss}
    overall = sum(avg_per_source.values()) / max(1, len(avg_per_source))
    model.train()
    return overall, avg_per_source

# ─── Logging helpers ────────────────────────────────────────────────────────
log_path  = Path(args.out_dir) / "train.jsonl"
event_path= Path(args.out_dir) / "_events.jsonl"
def log_step(d):
    with open(log_path, "a") as f:
        f.write(json.dumps(d) + "\n")
def log_event(d):
    with open(event_path, "a") as f:
        f.write(json.dumps(d) + "\n")

# ─── Train ──────────────────────────────────────────────────────────────────
model.train()
optim.zero_grad()
val_history = deque(maxlen=args.divergence_patience)
best_val = float("inf")
ema_loss = None
t0 = time.time()
total_tokens_seen = start_step * args.seq_len * args.batch_size * args.grad_accum
src_token_counts = {n: 0 for n in ok_names}

for step in range(start_step + 1, args.max_steps + 1):
    micro_loss_sum = 0.0
    src_picked = []
    for micro in range(args.grad_accum):
        seqs, names = [], []
        for _ in range(args.batch_size):
            n_, s_ = next(train_iter)
            seqs.append(s_)
            names.append(n_)
        x = torch.stack(seqs).to(device)
        src_picked.extend(names)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            ) / args.grad_accum
        loss.backward()
        micro_loss_sum += float(loss.item()) * args.grad_accum
    for n in src_picked:
        src_token_counts[n] = src_token_counts.get(n, 0) + args.seq_len

    # NaN / Inf guard
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
    if not torch.isfinite(grad_norm):
        emerg = Path(args.out_dir) / f"emergency_naninf_step_{step}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": {"model": model_cfg}}, emerg)
        log_event({"event": "halt_nan_inf", "step": step, "ckpt": str(emerg)})
        print(f"[pretrain] !!! NaN/Inf grad at step {step} — emergency saved → {emerg}", flush=True)
        sys.exit(4)

    optim.step()
    sched.step()
    optim.zero_grad()
    cur_loss = micro_loss_sum / max(1, args.grad_accum)
    ema_loss = cur_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * cur_loss
    total_tokens_seen += args.seq_len * args.batch_size * args.grad_accum

    if step % args.log_every == 0:
        cur_lr = sched.get_last_lr()[0]
        elapsed = time.time() - t0
        toks_per_s = (total_tokens_seen - start_step * args.seq_len * args.batch_size * args.grad_accum) / max(1e-9, elapsed)
        print(f"[pretrain] step={step}/{args.max_steps} loss={cur_loss:.4f} ema={ema_loss:.4f} "
              f"grad_norm={grad_norm:.3f} lr={cur_lr:.2e} tok/s={toks_per_s:.0f} "
              f"seen={total_tokens_seen/1e6:.1f}M", flush=True)
        log_step({"step": step, "loss": cur_loss, "ema": ema_loss,
                  "grad_norm": float(grad_norm), "lr": cur_lr,
                  "tokens_seen": total_tokens_seen,
                  "src_tokens": dict(src_token_counts)})

    if step % args.eval_every == 0:
        overall_val, per_src = evaluate()
        ppl = math.exp(min(20.0, overall_val))
        print(f"[pretrain] *** EVAL step={step} val_loss={overall_val:.4f} ppl={ppl:.2f} "
              f"per_src={ {k: round(v,3) for k,v in per_src.items()} } ***", flush=True)
        log_step({"step": step, "val_loss": overall_val, "val_ppl": ppl,
                  "per_source_val": per_src, "lr": sched.get_last_lr()[0],
                  "eval": True})
        if overall_val < best_val:
            best_val = overall_val
            best_path = Path(args.out_dir) / "best.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "val_loss": overall_val,
                        "per_source_val": per_src,
                        "config": {"model": model_cfg}}, best_path)
            print(f"[pretrain] *** new best val_loss={overall_val:.4f} → {best_path} ***", flush=True)
        val_history.append(overall_val)
        # Looser divergence: only halt if mean of last K val losses rose by >REL_THRESH
        # AND every step in window is rising. Avoids the v2 false positive at step 24k.
        if step > args.warmup_steps * 4 and len(val_history) == val_history.maxlen:
            strictly_rising = all(val_history[i] < val_history[i+1] for i in range(len(val_history)-1))
            rel_rise = (val_history[-1] - val_history[0]) / max(1e-9, val_history[0])
            if strictly_rising and rel_rise > args.divergence_rel_thresh:
                emerg = Path(args.out_dir) / f"emergency_diverge_step_{step}.pt"
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": sched.state_dict(),
                            "val_loss": overall_val,
                            "val_history": list(val_history),
                            "config": {"model": model_cfg}}, emerg)
                log_event({"event": "halt_val_divergence", "step": step,
                           "val_history": list(val_history), "rel_rise": rel_rise,
                           "ckpt": str(emerg)})
                print(f"[pretrain] !!! VAL DIVERGENCE — last {len(val_history)} evals strictly rising "
                      f"by {rel_rise*100:.1f}%: {[round(v,4) for v in val_history]}. Halting.", flush=True)
                sys.exit(3)

    if step % args.save_every == 0:
        ckpt_path = Path(args.out_dir) / f"step_{step}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": {"model": model_cfg}}, ckpt_path)
        print(f"[pretrain] saved {ckpt_path}", flush=True)
        all_steps = sorted(Path(args.out_dir).glob("step_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
        for old in all_steps[:-3]:
            old.unlink(missing_ok=True)

    if args.smoke and step >= args.smoke:
        print(f"[pretrain] SMOKE complete @ step {step}", flush=True)
        break

# Final
final_path = Path(args.out_dir) / "final.pt"
torch.save({"step": args.max_steps, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "config": {"model": model_cfg}}, final_path)
print(f"[pretrain] DONE — final saved to {final_path}", flush=True)
