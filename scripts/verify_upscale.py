"""
Smoke test: load both the original 109M Titan and the upscaled 330M Titan,
run the same prompt through both, and verify outputs match (function-preserving).
"""
import sys, torch, yaml
sys.path.insert(0, "/workspace/titanai")
from model.titan_model import build_model

OLD_CKPT = "/workspace/titanai/checkpoints/tool_v01/final.pt"
NEW_CKPT = "/workspace/titanai/checkpoints/titan_v2_init/initial.pt"

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = ck["config"]
    cfg = raw.get("model", raw)
    model_cfg = {
        "architecture": cfg.get("architecture", "decoder_transformer"),
        "vocab_size": cfg["vocab_size"],
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "d_ff": cfg["d_ff"],
        "max_seq_len": cfg["max_seq_len"],
        "dropout": 0.0,
        "tie_embeddings": cfg.get("tie_embeddings", True),
    }
    model = build_model({"model": model_cfg})
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    return model

with torch.no_grad():
    old = load(OLD_CKPT)
    new = load(NEW_CKPT)
    print(f"old layers: {len(old.blocks)}  new layers: {len(new.blocks)}")
    ids = torch.randint(0, 32000, (1, 16))
    o1, _ = old(ids)
    o2, _ = new(ids)
    diff = (o1 - o2).abs().max().item()
    print(f"max |old - new| logit diff: {diff:.6e}")
    if diff < 1e-3:
        print("PASS: upscaled model is function-equivalent to original")
    else:
        print("FAIL: outputs differ — identity init broken")
        sys.exit(1)
