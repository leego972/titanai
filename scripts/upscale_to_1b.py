"""
  upscale_to_1b.py — Inflate a 109M TitanAI checkpoint to 1B v0.3 architecture.

  Handles the transition from v0.2 (LayerNorm + GELU + MHA) to v0.3 architecture:
      - LayerNorm (ln1/ln2/ln_final) → RMSNorm (norm1/norm2/ln_final, weight only, no bias)
      - GELU MLP (fc1, fc2) → SwiGLU (gate_proj, up_proj, down_proj)
      - MHA (qkv_proj) → GQA (q_proj, k_proj, v_proj, out_proj)

  If your source 109M checkpoint uses the v0.3 architecture already, this script
  still works — it detects the key names and handles both cases.

  Usage:
      python scripts/upscale_to_1b.py \
          --src_checkpoint checkpoints/titan_109m_pretrain/best_model.pt \
          --src_config     configs/titan_109m.yaml \
          --dst_config     configs/titan_1b.yaml \
          --dst_checkpoint checkpoints/titan_1b_pretrain/init.pt
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

  def expand_1d(src, new_size, std):
      out = _noise((new_size,), std)
      out[:src.size(0)] = src
      return out

  def expand_2d(src, new_rows, new_cols, std):
      out = _noise((new_rows, new_cols), std)
      out[:src.size(0), :src.size(1)] = src
      return out

  def _swiglu_hidden(d_ff):
      return ((int(2 * d_ff / 3) + 63) // 64) * 64

  def _repo_root():
      scripts_dir = Path(__file__).resolve().parent
      parent = scripts_dir.parent
      return parent.parent if parent.name == "training" else parent

  def _detect_arch(src_sd: dict) -> str:
      """Detect whether the source checkpoint uses v0.2 (old) or v0.3 (new) arch."""
      keys = list(src_sd.keys())
      if any("qkv_proj" in k for k in keys):
          return "v0.2"
      if any("q_proj" in k for k in keys):
          return "v0.3"
      return "unknown"


  # ── main ──────────────────────────────────────────────────────────────────────

  def upscale(src_checkpoint, src_config_path, dst_config_path, dst_checkpoint):
      print(f"[upscale_to_1b] Source config : {src_config_path}")
      print(f"[upscale_to_1b] Target config : {dst_config_path}")

      with open(src_config_path) as f: src_cfg = yaml.safe_load(f)
      with open(dst_config_path) as f: dst_cfg = yaml.safe_load(f)

      sm, dm = src_cfg["model"], dst_cfg["model"]

      src_d   = sm["d_model"];  dst_d   = dm["d_model"]
      src_ff  = sm["d_ff"];     dst_ff  = dm["d_ff"]
      src_nl  = sm["n_layers"]; dst_nl  = dm["n_layers"]
      src_nh  = sm["n_heads"];  dst_nh  = dm["n_heads"]
      src_nkv = sm.get("n_kv_heads", src_nh)
      dst_nkv = dm.get("n_kv_heads", dst_nh)
      vocab   = dm["vocab_size"]

      src_dh  = src_d // src_nh
      dst_dh  = dst_d // dst_nh
      dst_swi = _swiglu_hidden(dst_ff)

      std = 0.02 / math.sqrt(2 * dst_nl)

      print(f"[upscale_to_1b] Loading: {src_checkpoint}")
      raw = torch.load(src_checkpoint, map_location="cpu")
      src_sd = raw.get("model_state_dict", raw)

      arch = _detect_arch(src_sd)
      print(f"[upscale_to_1b] Detected source arch: {arch}")

      dst_sd = {}

      # ── token embedding ───────────────────────────────────────────────────────
      dst_sd["token_embedding.weight"] = expand_2d(
          src_sd["token_embedding.weight"], vocab, dst_d, std)

      if "lm_head.weight" in src_sd:
          dst_sd["lm_head.weight"] = expand_2d(
              src_sd["lm_head.weight"], vocab, dst_d, std)

      # ── ln_final → RMSNorm weight only ───────────────────────────────────────
      # Source may use LayerNorm (has bias) or RMSNorm (weight only)
      ln_w_key = "ln_final.weight"
      dst_sd["ln_final.weight"] = expand_1d(src_sd[ln_w_key], dst_d, std)
      # Do NOT copy ln_final.bias — v0.3 uses RMSNorm (no bias)

      # ── transformer blocks ────────────────────────────────────────────────────
      for dst_idx in range(dst_nl):
          src_idx = dst_idx % src_nl
          sp = f"blocks.{src_idx}"
          dp = f"blocks.{dst_idx}"

          # Norm layers — always write norm1/norm2 (v0.3 RMSNorm, weight only)
          for norm_key in ("norm1", "norm2"):
              # Handle old arch key names (ln1, ln2) or new (norm1, norm2)
              src_norm_key = norm_key if f"{sp}.{norm_key}.weight" in src_sd else                              ("ln1" if norm_key == "norm1" else "ln2")
              dst_sd[f"{dp}.{norm_key}.weight"] = expand_1d(
                  src_sd[f"{sp}.{src_norm_key}.weight"], dst_d, std)
              # Do NOT copy norm bias — v0.3 RMSNorm has no bias

          # ── Attention ─────────────────────────────────────────────────────────
          if arch == "v0.2":
              # Source has qkv_proj (combined) — split into q, k, v for v0.3 GQA
              qkv = src_sd[f"{sp}.attn.qkv_proj.weight"]  # (3*src_d, src_d)
              # Split into Q / K / V along dim 0
              q_s, k_s, v_s = qkv.chunk(3, dim=0)   # each (src_d, src_d)
              dst_sd[f"{dp}.attn.q_proj.weight"] = expand_2d(
                  q_s, dst_nh * dst_dh, dst_d, std)
              # KV: source MHA (src_nh × src_dh each) → GQA (dst_nkv × dst_dh)
              # Average-pool source KV heads down to dst_nkv for better init
              src_nh_kv = src_nh  # source is full MHA
              if dst_nkv < src_nh_kv:
                  # Reshape and mean across groups of (src_nh / dst_nkv) heads
                  group = src_nh_kv // dst_nkv
                  k_s_r = k_s.view(dst_nkv, group, src_dh, src_d).mean(dim=1)  # (dst_nkv, src_dh, src_d)
                  v_s_r = v_s.view(dst_nkv, group, src_dh, src_d).mean(dim=1)
                  k_s_r = k_s_r.reshape(dst_nkv * src_dh, src_d)
                  v_s_r = v_s_r.reshape(dst_nkv * src_dh, src_d)
                  dst_sd[f"{dp}.attn.k_proj.weight"] = expand_2d(
                      k_s_r, dst_nkv * dst_dh, dst_d, std)
                  dst_sd[f"{dp}.attn.v_proj.weight"] = expand_2d(
                      v_s_r, dst_nkv * dst_dh, dst_d, std)
              else:
                  dst_sd[f"{dp}.attn.k_proj.weight"] = expand_2d(
                      k_s, dst_nkv * dst_dh, dst_d, std)
                  dst_sd[f"{dp}.attn.v_proj.weight"] = expand_2d(
                      v_s, dst_nkv * dst_dh, dst_d, std)
          else:
              # Source already has v0.3 GQA keys
              dst_sd[f"{dp}.attn.q_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.attn.q_proj.weight"], dst_nh * dst_dh, dst_d, std)
              dst_sd[f"{dp}.attn.k_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.attn.k_proj.weight"], dst_nkv * dst_dh, dst_d, std)
              dst_sd[f"{dp}.attn.v_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.attn.v_proj.weight"], dst_nkv * dst_dh, dst_d, std)

          dst_sd[f"{dp}.attn.out_proj.weight"] = expand_2d(
              src_sd[f"{sp}.attn.out_proj.weight"], dst_d, dst_d, std)

          # ── MLP ───────────────────────────────────────────────────────────────
          if arch == "v0.2":
              # Source has fc1 / fc2 (GELU) → map to gate_proj / down_proj
              # up_proj gets noise init (no source equivalent in GELU MLP)
              fc1 = src_sd[f"{sp}.mlp.fc1.weight"]   # (src_ff, src_d)
              fc2 = src_sd[f"{sp}.mlp.fc2.weight"]   # (src_d, src_ff)
              dst_sd[f"{dp}.mlp.gate_proj.weight"] = expand_2d(fc1, dst_swi, dst_d, std)
              dst_sd[f"{dp}.mlp.up_proj.weight"]   = _noise((dst_swi, dst_d), std)
              dst_sd[f"{dp}.mlp.down_proj.weight"] = expand_2d(fc2, dst_d, dst_swi, std)
          else:
              # Source has SwiGLU keys
              dst_sd[f"{dp}.mlp.gate_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.mlp.gate_proj.weight"], dst_swi, dst_d, std)
              dst_sd[f"{dp}.mlp.up_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.mlp.up_proj.weight"], dst_swi, dst_d, std)
              dst_sd[f"{dp}.mlp.down_proj.weight"] = expand_2d(
                  src_sd[f"{sp}.mlp.down_proj.weight"], dst_d, dst_swi, std)

      print(f"[upscale_to_1b] Inflated {src_nl} → {dst_nl} blocks")

      total = sum(t.numel() for t in dst_sd.values())
      print(f"[upscale_to_1b] Parameters: {total:,}  (~{total/1e9:.3f}B)")

      # ── Validate ──────────────────────────────────────────────────────────────
      print("[upscale_to_1b] Validating checkpoint loads into target architecture...")
      try:
          repo = _repo_root()
          if str(repo) not in sys.path:
              sys.path.insert(0, str(repo))
          from model.titan_model import TitanConfig, TitanLM
          cfg_obj = TitanConfig(
              vocab_size=dm["vocab_size"], d_model=dm["d_model"],
              n_heads=dm["n_heads"], n_kv_heads=dm.get("n_kv_heads", dm["n_heads"]),
              n_layers=dm["n_layers"], d_ff=dm["d_ff"],
              max_seq_len=dm.get("max_seq_len", 2048), dropout=dm.get("dropout", 0.05),
              tie_embeddings=dm.get("tie_embeddings", True),
              use_gradient_checkpointing=dm.get("gradient_checkpointing", False),
          )
          tgt = TitanLM(cfg_obj)
          missing, unexpected = tgt.load_state_dict(dst_sd, strict=True)
          if not missing and not unexpected:
              print("[upscale_to_1b] Validation PASSED")
          else:
              if missing:    print(f"[upscale_to_1b] WARNING missing: {missing}")
              if unexpected: print(f"[upscale_to_1b] WARNING unexpected: {unexpected}")
          del tgt
      except ImportError:
          print("[upscale_to_1b] Skipping validation (model module unavailable)")
      except Exception as e:
          raise RuntimeError(f"[upscale_to_1b] Validation FAILED: {e}") from e

      # ── Save ──────────────────────────────────────────────────────────────────
      out = Path(dst_checkpoint)
      out.parent.mkdir(parents=True, exist_ok=True)
      torch.save({
          "model_state_dict": dst_sd,
          "config": dst_cfg,
          "upscaled_from": src_checkpoint,
          "arch_version": "0.3",
          "step": 0,
          "best_val_loss": float("inf"),
      }, out)
      print(f"[upscale_to_1b] Saved 1B init checkpoint → {out}")


  def parse_args():
      root = _repo_root()
      p = argparse.ArgumentParser(description="Upscale 109M → 1B TitanAI v0.3")
      p.add_argument("--src_checkpoint", required=True)
      p.add_argument("--src_config",
          default=str(root / "configs" / "titan_109m.yaml"))
      p.add_argument("--dst_config",
          default=str(root / "configs" / "titan_1b.yaml"))
      p.add_argument("--dst_checkpoint",
          default=str(root / "checkpoints" / "titan_1b_pretrain" / "init.pt"))
      return p.parse_args()


  if __name__ == "__main__":
      args = parse_args()
      upscale(args.src_checkpoint, args.src_config, args.dst_config, args.dst_checkpoint)
  