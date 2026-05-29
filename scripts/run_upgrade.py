import argparse
import sys
from pathlib import Path
import yaml
import torch
import math

try:
    import torch._inductor.config as _ind_cfg
    _ind_cfg.compile_threads = 1
except Exception:
    pass

from torch.utils.data import random_split
from tokenizers import Tokenizer

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from training.sft_trainer import train_sft
from model.titan_model import TitanConfig, build_model, load_state_dict_compat
from data.sft_dataset import TitanSFTDataset

def main():
    parser = argparse.ArgumentParser(description="TitanAI Upgrade Training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stage_name = cfg["project"]["name"]
    ckpt_dir = cfg["training"]["checkpoint_dir"]

    # Skip if already complete
    final_ckpt = BASE / ckpt_dir / "final.pt"
    if final_ckpt.exists():
        print(f"[{stage_name}] Already complete — skipping.")
        return

    sft_files = cfg["data"]["sft_files"]
    missing = [p for p in sft_files if not Path(BASE / p).exists()]
    if missing:
        print(f"\n[ERROR] Missing data for {stage_name}:")
        for m in missing: print(f"  - {m}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{stage_name}] Device: {device}")
    tokenizer = Tokenizer.from_file(cfg["data"]["tokenizer_path"])

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = build_model(cfg)
    _raw_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = _raw_ckpt.get("model_state_dict", _raw_ckpt)

    # Handle GQA Mismatch
    new_state = {}
    model_keys = set(model.state_dict().keys())
    for k, v in state_dict.items():
        if k.startswith("module."): k = k[7:]
        if k in model_keys:
            target_shape = model.state_dict()[k].shape
            if v.shape != target_shape:
                if "attn.k_proj.weight" in k or "attn.v_proj.weight" in k:
                    # v shape is (old_kv_heads * d_head, d_model)
                    # target shape is (new_kv_heads * d_head, d_model)
                    d_model = v.shape[1]
                    old_kv_total = v.shape[0]
                    new_kv_total = target_shape[0]
                    
                    # Calculate d_head from model config
                    d_head = target_shape[0] // cfg['model'].get('n_kv_heads', cfg['model']['n_heads'])
                    
                    old_kv_heads = old_kv_total // d_head
                    new_kv_heads = new_kv_total // d_head
                    
                    if old_kv_heads > new_kv_heads:
                        # Downsampling heads (e.g. 16 -> 4)
                        print(f"[GQA Fix] Downsampling {k} from {old_kv_heads} to {new_kv_heads} heads")
                        v = v.view(old_kv_heads, d_head, d_model)
                        v = v[:new_kv_heads, :, :]
                        v = v.reshape(new_kv_total, d_model)
                    else:
                        # Upsampling heads (e.g. 4 -> 16)
                        print(f"[GQA Fix] Expanding {k} from {old_kv_heads} to {new_kv_heads} heads")
                        n_repeats = new_kv_heads // old_kv_heads
                        v = v.view(old_kv_heads, d_head, d_model)
                        v = v.repeat_interleave(n_repeats, dim=0)
                        v = v.reshape(new_kv_total, d_model)
        new_state[k] = v

    load_state_dict_compat(model, new_state)
    del _raw_ckpt, state_dict, new_state
    torch.cuda.empty_cache()

    model = model.to(device=device, dtype=torch.bfloat16)
    print(f"[{stage_name}] Loaded: {sum(p.numel() for p in model.parameters()):,} params")

    full_dataset = TitanSFTDataset(
        jsonl_paths=[str(BASE / p) for p in sft_files],
        tokenizer=tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        verbose=True,
    )

    if len(full_dataset) == 0:
        print(f"[ERROR] No data loaded for {stage_name}.")
        sys.exit(1)

    val_size = max(10, int(len(full_dataset) * cfg["data"].get("val_split", 0.05)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_sft(cfg, model, train_dataset, val_dataset, device, args.resume)
    print(f"\n[{stage_name}] COMPLETE -> {ckpt_dir}/final.pt")

if __name__ == "__main__":
    main()
