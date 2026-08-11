"""
upscale_to_3b.py — Inflate a TitanAI 1B v0.3 checkpoint to 3B architecture.

Architecture differences handled (v0.3 → v0.3, same arch family):
    - RMSNorm : no .bias key — expand weight only
    - SwiGLU  : gate_proj, up_proj, down_proj  (was fc1, fc2)
    - GQA     : q_proj, k_proj, v_proj, out_proj (was qkv_proj, out_proj)
              - q_proj : (n_heads × d_head_src, d_src) → (n_heads × d_head_dst, d_dst)
              - k_proj / v_proj : (n_kv_heads × d_head_src, d_src) → (n_kv_heads × d_head_dst, d_dst)

Strategy:
    - All weight tensors: copy source into top-left block of target shape,
      pad remainder with small Gaussian noise (std = 0.02 / sqrt(2 × n_layers_dst))
    - Extra layers (dst_nl > src_nl): cycle source layers round-robin
    - Validates by loading state dict into a real TitanLM instance before saving

Usage:
    python scripts/upscale_to_3b.py \
        --src_checkpoint checkpoints/titan_1b_pretrain/final.pt \
        --src_config     configs/titan_1b.yaml \
        --dst_config     configs/titan_3b.yaml \
        --dst_checkpoint checkpoints/titan_3b_pretrain/init.pt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml


def _noise(shape, std):
    return torch.randn(shape) * std


def expand_1d(src: torch.Tensor, new_size: int, std: float) -> torch.Tensor:
    """Expand a 1-D tensor (RMSNorm weight) from src.size(0) to new_size."""
    out = _noise((new_size,), std)
    out[: src.size(0)] = src
    return out


def expand_2d(src: torch.Tensor, new_rows: int, new_cols: int, std: float) -> torch.Tensor:
    """Expand a 2-D weight from (src_rows, src_cols) to (new_rows, new_cols)."""
    out = _noise((new_rows, new_cols), std)
    out[: src.size(0), : src.size(1)] = src
    return out


def _swiglu_hidden(d_ff: int) -> int:
    """Match the SwiGLU hidden dim computation in TitanConfig.__post_init__."""
    return ((int(2 * d_ff / 3) + 63) // 64) * 64


def _repo_root() -> Path:
    scripts_dir = Path(__file__).resolve().parent
    parent = scripts_dir.parent
    return parent.parent if parent.name == "training" else parent


def upscale(src_checkpoint: str, src_config_path: str, dst_config_path: str, dst_checkpoint: str):
    print(f"[upscale_to_3b] Source config : {src_config_path}")
    print(f"[upscale_to_3b] Target config : {dst_config_path}")

    with open(src_config_path) as f:
        src_cfg = yaml.safe_load(f)
    with open(dst_config_path) as f:
        dst_cfg = yaml.safe_load(f)

    sm, dm = src_cfg["model"], dst_cfg["model"]

    src_d = sm["d_model"]
    dst_d = dm["d_model"]
    src_ff = sm["d_ff"]
    dst_ff = dm["d_ff"]
    src_nl = sm["n_layers"]
    dst_nl = dm["n_layers"]
    src_nh = sm["n_heads"]
    dst_nh = dm["n_heads"]
    src_nkv = sm.get("n_kv_heads", src_nh)
    dst_nkv = dm.get("n_kv_heads", dst_nh)
    vocab = dm["vocab_size"]

    src_dh = src_d // src_nh
    dst_dh = dst_d // dst_nh
    src_swi = _swiglu_hidden(src_ff)
    dst_swi = _swiglu_hidden(dst_ff)

    print(
        f"[upscale_to_3b] Src: d={src_d}, nh={src_nh}, nkv={src_nkv}, "
        f"nl={src_nl}, ff={src_ff}, swi={src_swi}"
    )
    print(
        f"[upscale_to_3b] Dst: d={dst_d}, nh={dst_nh}, nkv={dst_nkv}, "
        f"nl={dst_nl}, ff={dst_ff}, swi={dst_swi}"
    )

    std = 0.02 / math.sqrt(2 * dst_nl)

    print(f"[upscale_to_3b] Loading: {src_checkpoint}")
    raw = torch.load(src_checkpoint, map_location="cpu")
    src_sd = raw.get("model_state_dict", raw)
    dst_sd = {}

    src_emb = src_sd["token_embedding.weight"]
    dst_sd["token_embedding.weight"] = expand_2d(src_emb, vocab, dst_d, std)
    print(
        f"[upscale_to_3b]  token_embedding: {tuple(src_emb.shape)} → "
        f"{tuple(dst_sd['token_embedding.weight'].shape)}"
    )

    if "lm_head.weight" in src_sd:
        dst_sd["lm_head.weight"] = expand_2d(src_sd["lm_head.weight"], vocab, dst_d, std)

    dst_sd["ln_final.weight"] = expand_1d(src_sd["ln_final.weight"], dst_d, std)

    for dst_idx in range(dst_nl):
        src_idx = dst_idx % src_nl
        sp = f"blocks.{src_idx}"
        dp = f"blocks.{dst_idx}"

        dst_sd[f"{dp}.norm1.weight"] = expand_1d(src_sd[f"{sp}.norm1.weight"], dst_d, std)
        dst_sd[f"{dp}.norm2.weight"] = expand_1d(src_sd[f"{sp}.norm2.weight"], dst_d, std)

        q_src = src_sd[f"{sp}.attn.q_proj.weight"]
        dst_sd[f"{dp}.attn.q_proj.weight"] = expand_2d(q_src, dst_nh * dst_dh, dst_d, std)

        k_src = src_sd[f"{sp}.attn.k_proj.weight"]
        dst_sd[f"{dp}.attn.k_proj.weight"] = expand_2d(k_src, dst_nkv * dst_dh, dst_d, std)

        v_src = src_sd[f"{sp}.attn.v_proj.weight"]
        dst_sd[f"{dp}.attn.v_proj.weight"] = expand_2d(v_src, dst_nkv * dst_dh, dst_d, std)

        o_src = src_sd[f"{sp}.attn.out_proj.weight"]
        dst_sd[f"{dp}.attn.out_proj.weight"] = expand_2d(o_src, dst_d, dst_d, std)

        gate_src = src_sd[f"{sp}.mlp.gate_proj.weight"]
        dst_sd[f"{dp}.mlp.gate_proj.weight"] = expand_2d(gate_src, dst_swi, dst_d, std)

        up_src = src_sd[f"{sp}.mlp.up_proj.weight"]
        dst_sd[f"{dp}.mlp.up_proj.weight"] = expand_2d(up_src, dst_swi, dst_d, std)

        down_src = src_sd[f"{sp}.mlp.down_proj.weight"]
        dst_sd[f"{dp}.mlp.down_proj.weight"] = expand_2d(down_src, dst_d, dst_swi, std)

    print(f"[upscale_to_3b] Inflated {src_nl} → {dst_nl} transformer blocks")

    total = sum(t.numel() for t in dst_sd.values())
    print(f"[upscale_to_3b] Total parameters : {total:,}  (~{total / 1e9:.3f}B)")

    print("[upscale_to_3b] Validating state dict loads into target architecture...")
    try:
        repo = _repo_root()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from model.titan_model import TitanConfig, TitanLM

        dst_model_cfg = TitanConfig(
            vocab_size=dm["vocab_size"],
            d_model=dm["d_model"],
            n_heads=dm["n_heads"],
            n_kv_heads=dm.get("n_kv_heads", dm["n_heads"]),
            n_layers=dm["n_layers"],
            d_ff=dm["d_ff"],
            max_seq_len=dm.get("max_seq_len", 2048),
            dropout=dm.get("dropout", 0.05),
            tie_embeddings=dm.get("tie_embeddings", True),
            use_gradient_checkpointing=dm.get("gradient_checkpointing", False),
        )
        target = TitanLM(dst_model_cfg)
        missing, unexpected = target.load_state_dict(dst_sd, strict=True)
        if missing:
            print(f"[upscale_to_3b] WARNING missing keys: {missing}")
        if unexpected:
            print(f"[upscale_to_3b] WARNING unexpected keys: {unexpected}")
        if not missing and not unexpected:
            print("[upscale_to_3b] Validation PASSED — strict load OK")
        del target
    except ImportError:
        print("[upscale_to_3b] Skipping model validation (model module unavailable)")
    except Exception as e:
        raise RuntimeError(f"[upscale_to_3b] Validation FAILED: {e}") from e

    out = Path(dst_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": dst_sd,
            "config": dst_cfg,
            "upscaled_from": src_checkpoint,
            "arch_version": "0.3",
            "step": 0,
            "best_val_loss": float("inf"),
        },
        out,
    )
    print(f"[upscale_to_3b] Saved 3B init checkpoint → {out}")


def parse_args():
    root = _repo_root()
    p = argparse.ArgumentParser(description="Inflate a 1B TitanAI v0.3 checkpoint to 3B")
    p.add_argument("--src_checkpoint", required=True)
    p.add_argument("--src_config", default=str(root / "configs" / "titan_1b.yaml"))
    p.add_argument("--dst_config", default=str(root / "configs" / "titan_3b.yaml"))
    p.add_argument(
        "--dst_checkpoint",
        default=str(root / "checkpoints" / "titan_3b_pretrain" / "init.pt"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upscale(args.src_checkpoint, args.src_config, args.dst_config, args.dst_checkpoint)
