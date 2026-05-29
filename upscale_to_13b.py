"""
upscale_to_13b.py — Inflate a 7B TitanAI checkpoint to 13B architecture.

Usage:
    python scripts/upscale_to_13b.py \
        --src_checkpoint checkpoints/titan_7b_pretrain/best_model.pt \
        --src_config    configs/titan_7b.yaml \
        --dst_config    configs/titan_13b.yaml \
        --dst_checkpoint checkpoints/titan_13b_pretrain/init.pt

Strategy:
    - Embedding & lm_head: copy rows/cols that exist; zero-pad new dimensions.
    - LayerNorm (ln1, ln2, ln_final): zero-pad weight and bias to new d_model.
    - Attention qkv_proj / out_proj: copy source weights into top-left block of
      the expanded weight matrix; initialise new rows/cols with small Gaussian
      noise scaled to 0.02 / sqrt(2 * n_layers_dst) for stability.
    - MLP fc1 / fc2: same block-copy + noise-pad strategy.
    - Extra layers (when n_layers_dst > n_layers_src): cycle existing layers
      round-robin so every source layer's knowledge is reused evenly.

All new parameters are initialised with small noise so they do not create
gradient dead-zones while still being close to zero.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml


# ── helpers ───────────────────────────────────────────────────────────────────

def _noise(shape, std):
    return torch.randn(shape) * std


def expand_1d(src: torch.Tensor, new_size: int, noise_std: float) -> torch.Tensor:
    """Expand a 1-D tensor (bias / LayerNorm weight) from src.size(0) to new_size."""
    out = _noise((new_size,), noise_std)
    out[: src.size(0)] = src
    return out


def expand_2d(src: torch.Tensor, new_rows: int, new_cols: int, noise_std: float) -> torch.Tensor:
    """
    Expand a 2-D weight matrix from (src_rows, src_cols) to (new_rows, new_cols).
    The source weights occupy the top-left block; the rest is small noise.
    """
    out = _noise((new_rows, new_cols), noise_std)
    out[: src.size(0), : src.size(1)] = src
    return out


def layer_key(layer_idx: int) -> str:
    return f"blocks.{layer_idx}"


# ── main ──────────────────────────────────────────────────────────────────────

def upscale(
    src_checkpoint: str,
    src_config_path: str,
    dst_config_path: str,
    dst_checkpoint: str,
):
    print(f"[upscale_to_13b] Loading source config  : {src_config_path}")
    with open(src_config_path) as f:
        src_cfg = yaml.safe_load(f)
    print(f"[upscale_to_13b] Loading target config  : {dst_config_path}")
    with open(dst_config_path) as f:
        dst_cfg = yaml.safe_load(f)

    src_m = src_cfg["model"]
    dst_m = dst_cfg["model"]

    src_d  = src_m["d_model"];  dst_d  = dst_m["d_model"]
    src_ff = src_m["d_ff"];     dst_ff = dst_m["d_ff"]
    src_nl = src_m["n_layers"]; dst_nl = dst_m["n_layers"]
    vocab  = dst_m["vocab_size"]

    print(f"[upscale_to_13b] Source: d_model={src_d}, d_ff={src_ff}, n_layers={src_nl}")
    print(f"[upscale_to_13b] Target: d_model={dst_d}, d_ff={dst_ff}, n_layers={dst_nl}")

    noise_std = 0.02 / math.sqrt(2 * dst_nl)

    print(f"[upscale_to_13b] Loading source checkpoint: {src_checkpoint}")
    ckpt = torch.load(src_checkpoint, map_location="cpu", weights_only=False)
    src_sd = ckpt.get("model_state_dict", ckpt)

    dst_sd = {}

    # ── token embedding ───────────────────────────────────────────────────────
    src_emb = src_sd["token_embedding.weight"]   # (vocab, src_d)
    dst_sd["token_embedding.weight"] = expand_2d(src_emb, vocab, dst_d, noise_std)
    print(f"[upscale_to_13b]  token_embedding: {tuple(src_emb.shape)} → {tuple(dst_sd['token_embedding.weight'].shape)}")

    # ── ln_final ──────────────────────────────────────────────────────────────
    dst_sd["ln_final.weight"] = expand_1d(src_sd["ln_final.weight"], dst_d, noise_std)
    dst_sd["ln_final.bias"]   = expand_1d(src_sd["ln_final.bias"],   dst_d, noise_std)

    # ── lm_head (may be tied — store anyway; loader will handle tie) ──────────
    if "lm_head.weight" in src_sd:
        src_lm = src_sd["lm_head.weight"]   # (vocab, src_d)
        dst_sd["lm_head.weight"] = expand_2d(src_lm, vocab, dst_d, noise_std)

    # ── transformer blocks ───────────────────────────────────────────────────
    for dst_idx in range(dst_nl):
        src_idx = dst_idx % src_nl          # cycle source layers
        sp = f"blocks.{src_idx}"
        dp = f"blocks.{dst_idx}"

        # LayerNorm weights + biases
        dst_sd[f"{dp}.ln1.weight"] = expand_1d(src_sd[f"{sp}.ln1.weight"], dst_d, noise_std)
        dst_sd[f"{dp}.ln1.bias"]   = expand_1d(src_sd[f"{sp}.ln1.bias"],   dst_d, noise_std)
        dst_sd[f"{dp}.ln2.weight"] = expand_1d(src_sd[f"{sp}.ln2.weight"], dst_d, noise_std)
        dst_sd[f"{dp}.ln2.bias"]   = expand_1d(src_sd[f"{sp}.ln2.bias"],   dst_d, noise_std)

        # qkv_proj: (3*src_d, src_d) → (3*dst_d, dst_d)
        qkv_src = src_sd[f"{sp}.attn.qkv_proj.weight"]
        dst_sd[f"{dp}.attn.qkv_proj.weight"] = expand_2d(qkv_src, 3 * dst_d, dst_d, noise_std)

        # out_proj: (src_d, src_d) → (dst_d, dst_d)
        op_src = src_sd[f"{sp}.attn.out_proj.weight"]
        dst_sd[f"{dp}.attn.out_proj.weight"] = expand_2d(op_src, dst_d, dst_d, noise_std)

        # fc1: (src_ff, src_d) → (dst_ff, dst_d)
        fc1_src = src_sd[f"{sp}.mlp.fc1.weight"]
        dst_sd[f"{dp}.mlp.fc1.weight"] = expand_2d(fc1_src, dst_ff, dst_d, noise_std)

        # fc2: (src_d, src_ff) → (dst_d, dst_ff)
        fc2_src = src_sd[f"{sp}.mlp.fc2.weight"]
        dst_sd[f"{dp}.mlp.fc2.weight"] = expand_2d(fc2_src, dst_d, dst_ff, noise_std)

        # causal_mask — persistent buffer, same shape for both models (max_seq_len unchanged).
        # Must be copied so load_state_dict(strict=True) succeeds on the target model.
        mask_key = f"{sp}.attn.causal_mask"
        if mask_key in src_sd:
            dst_sd[f"{dp}.attn.causal_mask"] = src_sd[mask_key].clone()
        else:
            # Reconstruct if not present in source checkpoint (persistent=True default)
            max_seq = dst_m.get("max_seq_len", 2048)
            dst_sd[f"{dp}.attn.causal_mask"] = (
                torch.tril(torch.ones(max_seq, max_seq))
                .view(1, 1, max_seq, max_seq)
            )

    print(f"[upscale_to_13b] Inflated {src_nl} → {dst_nl} transformer blocks")

    # ── count parameters ──────────────────────────────────────────────────────
    total = sum(t.numel() for t in dst_sd.values())
    print(f"[upscale_to_13b] Total parameters in new checkpoint: {total:,}  (~{total/1e9:.2f}B)")

    # ── validation: instantiate target model and dry-run load_state_dict ──────
    print("[upscale_to_13b] Validating checkpoint loads cleanly into target architecture...")
    try:
        import sys as _sys
        _repo = Path(__file__).resolve().parent
        if _repo.parent.name == "training":
            _repo = _repo.parent.parent
        else:
            _repo = _repo.parent
        if str(_repo) not in _sys.path:
            _sys.path.insert(0, str(_repo))
        from model.titan_model import TitanConfig, TitanLM
        dst_config_obj = TitanConfig(
            vocab_size=dst_m["vocab_size"],
            d_model=dst_m["d_model"],
            n_heads=dst_m["n_heads"],
            n_layers=dst_m["n_layers"],
            d_ff=dst_m["d_ff"],
            max_seq_len=dst_m.get("max_seq_len", 2048),
            dropout=dst_m.get("dropout", 0.05),
            tie_embeddings=dst_m.get("tie_embeddings", True),
        )
        target_model = TitanLM(dst_config_obj)
        missing, unexpected = target_model.load_state_dict(dst_sd, strict=True)
        if missing:
            print(f"[upscale_to_13b] WARNING: missing keys in checkpoint: {missing}", flush=True)
        if unexpected:
            print(f"[upscale_to_13b] WARNING: unexpected keys in checkpoint: {unexpected}", flush=True)
        if not missing and not unexpected:
            print("[upscale_to_13b] Validation PASSED — state dict loads with strict=True")
        del target_model
    except ImportError:
        print("[upscale_to_13b] Skipping model validation (model module not importable here)")
    except Exception as ve:
        raise RuntimeError(f"[upscale_to_13b] Validation FAILED — checkpoint would not load: {ve}") from ve

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = Path(dst_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "model_state_dict": dst_sd,
        "config": dst_cfg,
        "upscaled_from": src_checkpoint,
        "step": 0,
        "best_val_loss": float("inf"),
    }
    torch.save(save_obj, out_path)
    print(f"[upscale_to_13b] Saved 13B checkpoint → {out_path}")


def _repo_root() -> Path:
    """
    Locate the repository root from this script's own path.

    This file is always placed in a 'scripts' directory:
      - Local workspace : <repo_root>/training/scripts/upscale_to_13b.py
      - Server (cloned) : <repo_root>/scripts/upscale_to_13b.py

    We detect which by inspecting the grandparent directory name:
    if it is 'training', we are one level deeper than the server layout.
    """
    scripts_dir = Path(__file__).resolve().parent   # …/scripts/
    parent = scripts_dir.parent                      # …/training/ or repo root
    if parent.name == "training":
        # local layout: training/scripts/ → repo root is two levels up
        return parent.parent
    # server layout: scripts/ → repo root is one level up
    return parent


def parse_args():
    root = _repo_root()
    # Config is at training/configs/ locally; at configs/ on the cloned server.
    if (root / "configs" / "titan_13b.yaml").exists():
        default_dst_config = str(root / "configs" / "titan_13b.yaml")
    else:
        default_dst_config = str(root / "training" / "configs" / "titan_13b.yaml")

    if (root / "configs" / "titan_7b.yaml").exists():
        default_src_config = str(root / "configs" / "titan_7b.yaml")
    else:
        default_src_config = str(root / "training" / "configs" / "titan_7b.yaml")

    p = argparse.ArgumentParser(description="Inflate a 7B TitanAI checkpoint to 13B")
    p.add_argument("--src_checkpoint", required=True,
                   help="Path to the source 7B checkpoint (.pt)")
    p.add_argument("--src_config", default=default_src_config,
                   help="Path to the source model YAML config (default: auto-detected titan_7b.yaml)")
    p.add_argument("--dst_config", default=default_dst_config,
                   help="Path to the target 13B YAML config (default: auto-detected)")
    p.add_argument("--dst_checkpoint",
                   default=str(root / "checkpoints" / "titan_13b_pretrain" / "init.pt"),
                   help="Where to write the inflated 13B checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upscale(
        src_checkpoint=args.src_checkpoint,
        src_config_path=args.src_config,
        dst_config_path=args.dst_config,
        dst_checkpoint=args.dst_checkpoint,
    )
