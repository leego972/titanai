"""
upscale_to_3b.py — Expand a trained 1B TitanAI checkpoint to 3B.

Strategy: DEPTH SCALING — keep d_model/d_ff/head dims identical to 1B,
expand n_layers 24 → 66.  New layers initialised by CYCLING 1B weights:
  3B layer i  gets weights from  1B layer (i % 24)

Why cycling: each block learns a residual refinement step. Cycling gives the
3B a strong starting point — extra layers contribute meaningful signal from
step 0, not pure noise. All weight shapes are IDENTICAL to 1B so no
dimension expansion is needed at all.

Usage:
    python upscale_to_3b.py \
        --src_checkpoint checkpoints/titan_1b_dpo/final.pt \
        --dst_checkpoint checkpoints/titan_3b_pretrain/init.pt \
        --src_config     titan_1b.yaml \
        --dst_config     titan_3b.yaml
"""
import argparse, sys
from pathlib import Path
import torch
import yaml


def _repo_root():
    p = Path(__file__).resolve().parent
    return p.parent if p.parent.name == "training" else p


def upscale(src_checkpoint, src_config_path, dst_config_path, dst_checkpoint):
    print(f"[upscale_to_3b] Source : {src_checkpoint}")
    print(f"[upscale_to_3b] Target : {dst_checkpoint}")

    with open(src_config_path) as f: src_cfg = yaml.safe_load(f)
    with open(dst_config_path) as f: dst_cfg = yaml.safe_load(f)

    src_nl = src_cfg["model"]["n_layers"]   # 24
    dst_nl = dst_cfg["model"]["n_layers"]   # 66
    src_d  = src_cfg["model"]["d_model"]
    dst_d  = dst_cfg["model"]["d_model"]

    if src_d != dst_d:
        raise ValueError(f"d_model mismatch ({src_d} vs {dst_d}). "
                         "upscale_to_3b uses depth-only scaling — d_model must stay the same.")

    print(f"[upscale_to_3b] Layers : {src_nl} → {dst_nl}  (cycle period {src_nl})")

    raw    = torch.load(src_checkpoint, map_location="cpu", weights_only=False)
    src_sd = raw.get("model_state_dict", raw.get("model", raw))

    dst_sd = {}

    # Non-layer weights: copy directly
    for key in list(src_sd.keys()):
        if not key.startswith("blocks."):
            dst_sd[key] = src_sd[key].clone()
            print(f"[upscale_to_3b]   copied: {key}  {tuple(src_sd[key].shape)}")

    # Transformer blocks: cycle 1B layers across 66 slots
    for dst_idx in range(dst_nl):
        src_idx    = dst_idx % src_nl
        src_prefix = f"blocks.{src_idx}."
        dst_prefix = f"blocks.{dst_idx}."
        layer_keys = [k for k in src_sd if k.startswith(src_prefix)]
        if not layer_keys:
            raise KeyError(f"No keys found for source layer {src_idx}")
        for sk in layer_keys:
            dst_sd[dst_prefix + sk[len(src_prefix):]] = src_sd[sk].clone()

    total = sum(t.numel() for t in dst_sd.values())
    print(f"[upscale_to_3b] Parameters: {total:,}  ({total/1e9:.3f}B)")

    # Validate
    print("[upscale_to_3b] Validating against 3B architecture...")
    try:
        repo = _repo_root()
        if str(repo) not in sys.path: sys.path.insert(0, str(repo))
        from model.titan_model import TitanConfig, TitanLM
        dm = dst_cfg["model"]
        cfg_obj = TitanConfig(
            vocab_size=dm["vocab_size"], d_model=dm["d_model"],
            n_heads=dm["n_heads"], n_kv_heads=dm.get("n_kv_heads", dm["n_heads"]),
            n_layers=dm["n_layers"], d_ff=dm["d_ff"],
            max_seq_len=dm.get("max_seq_len", 2048), dropout=dm.get("dropout", 0.05),
            tie_embeddings=dm.get("tie_embeddings", True),
            use_gradient_checkpointing=dm.get("gradient_checkpointing", False),
        )
        model = TitanLM(cfg_obj)
        missing, unexpected = model.load_state_dict(dst_sd, strict=True)
        if not missing and not unexpected:
            print("[upscale_to_3b] Validation PASSED")
        else:
            if missing:    print(f"[upscale_to_3b] WARNING missing:    {missing[:5]}")
            if unexpected: print(f"[upscale_to_3b] WARNING unexpected: {unexpected[:5]}")
        del model
    except ImportError:
        print("[upscale_to_3b] Skipping live validation (model module unavailable)")
    except Exception as e:
        raise RuntimeError(f"[upscale_to_3b] Validation FAILED: {e}") from e

    # Save
    out = Path(dst_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": dst_sd,
        "config": dst_cfg,
        "upscaled_from": str(src_checkpoint),
        "arch_version": "0.3",
        "step": 0,
        "best_val_loss": float("inf"),
    }, out)
    print(f"[upscale_to_3b] Saved → {out}  ({out.stat().st_size/1e9:.2f} GB)")


def parse_args():
    p = argparse.ArgumentParser(description="Depth-scale trained 1B → 3B TitanAI")
    p.add_argument("--src_checkpoint", required=True)
    p.add_argument("--src_config",     default="titan_1b.yaml")
    p.add_argument("--dst_config",     default="titan_3b.yaml")
    p.add_argument("--dst_checkpoint", default="checkpoints/titan_3b_pretrain/init.pt")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    upscale(args.src_checkpoint, args.src_config, args.dst_config, args.dst_checkpoint)
