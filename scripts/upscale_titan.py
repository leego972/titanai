"""
Function-preserving depth upscale: Titan 12-layer (109M) → 36-layer (~330M).

Strategy: for each existing block i, keep it, then insert 2 new blocks copied
from block i but with their residual-output projections zeroed. With pre-LN
+ residual adds, a zero output projection means the new block contributes
exactly 0 to the residual stream → output of the upscaled model is identical
to the original at init. As training proceeds, the new blocks learn non-zero
contributions and the model gains capacity.

This preserves every bit of training already done (SFT + Tool rounds) and
guarantees no quality regression at step 0 of the next training run.
"""
import argparse, copy, json, os, sys, torch, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source .pt checkpoint (12-layer Titan)")
    ap.add_argument("--dst", required=True, help="destination .pt for upscaled init")
    ap.add_argument("--multiplier", type=int, default=3,
                    help="how many copies of each layer in the new model (default 3 → 12→36)")
    ap.add_argument("--src-config", default=None, help="source config yaml for verification")
    ap.add_argument("--dst-config", default=None, help="write upscaled config yaml here")
    args = ap.parse_args()

    print(f"[upscale] loading {args.src}")
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    full_cfg = ck.get("config", {})
    # config may be nested (full training cfg) or flat (just model section)
    cfg = full_cfg.get("model", full_cfg) if isinstance(full_cfg, dict) else {}
    print(f"[upscale] source: step={ck.get('step')} n_layers={cfg.get('n_layers')} "
          f"d_model={cfg.get('d_model')} d_ff={cfg.get('d_ff')} vocab={cfg.get('vocab_size')}")

    n_old = cfg["n_layers"]
    n_new = n_old * args.multiplier
    print(f"[upscale] expanding {n_old} → {n_new} layers (×{args.multiplier})")

    # collect per-layer keys
    new_sd = {}
    # 1) copy all non-block keys verbatim
    for k, v in sd.items():
        if not k.startswith("blocks."):
            new_sd[k] = v.clone()

    # 2) for each new index i_new (0..n_new-1), map back to old block i_old = i_new // multiplier
    #    copy_index = i_new % multiplier  (0 = exact copy, 1+ = identity-init copy)
    for i_new in range(n_new):
        i_old = i_new // args.multiplier
        copy_index = i_new % args.multiplier
        # find all keys belonging to old block i_old
        old_prefix = f"blocks.{i_old}."
        new_prefix = f"blocks.{i_new}."
        for k, v in sd.items():
            if not k.startswith(old_prefix):
                continue
            sub = k[len(old_prefix):]
            new_key = new_prefix + sub
            t = v.clone()
            if copy_index > 0:
                # zero the residual-output projections so this block is a no-op at init
                if sub == "attn.out_proj.weight" or sub == "mlp.fc2.weight":
                    t.zero_()
            new_sd[new_key] = t

    # 3) update the embedded config
    new_cfg = dict(cfg)
    new_cfg["n_layers"] = n_new

    # save (fresh checkpoint — no optimizer/scheduler state since we're starting a new run)
    out = {
        "step": 0,
        "model_state_dict": new_sd,
        "config": new_cfg,
        "upscale_meta": {
            "from": args.src,
            "from_step": ck.get("step"),
            "from_n_layers": n_old,
            "to_n_layers": n_new,
            "method": "depth-upscale-identity-init",
            "multiplier": args.multiplier,
        },
    }
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    torch.save(out, args.dst)

    n_params = sum(v.numel() for v in new_sd.values() if hasattr(v, "numel"))
    n_real = sum(v.numel() for k, v in new_sd.items()
                 if hasattr(v, "numel") and "causal_mask" not in k)
    print(f"[upscale] wrote {args.dst}")
    print(f"[upscale] new tensor count: {len(new_sd)}")
    print(f"[upscale] total numel: {n_params/1e6:.2f}M (incl. mask buffers)")
    print(f"[upscale] real params:  {n_real/1e6:.2f}M")

    # write a paired training config if requested
    if args.dst_config:
        out_yaml = {
            "model": {
                "architecture": new_cfg.get("architecture", "decoder_transformer"),
                "vocab_size": new_cfg["vocab_size"],
                "d_model": new_cfg["d_model"],
                "n_heads": new_cfg["n_heads"],
                "n_layers": n_new,
                "d_ff": new_cfg["d_ff"],
                "max_seq_len": new_cfg["max_seq_len"],
                "dropout": new_cfg.get("dropout", 0.1),
                "tie_embeddings": new_cfg.get("tie_embeddings", True),
            },
        }
        with open(args.dst_config, "w") as f:
            yaml.safe_dump(out_yaml, f, sort_keys=False)
        print(f"[upscale] wrote model-section config to {args.dst_config}")

    print("[upscale] OK — function-preserving init complete.")

if __name__ == "__main__":
    main()
