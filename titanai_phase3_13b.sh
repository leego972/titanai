#!/usr/bin/env bash
# titanai_phase3_13b.sh — Continued pretraining at 13B scale
#
# Phase 4 of the TitanAI training chain: 7B → 13B
#
# Run AFTER upscale_to_13b.py has produced the inflated init checkpoint.
#
# This script can be invoked from any working directory:
#
#   bash training/scripts/titanai_phase3_13b.sh             # local workspace
#   bash scripts/titanai_phase3_13b.sh                       # Vast.ai server
#
# The repo root is derived from this script's own location (two levels up when
# in training/scripts/, one level up when in scripts/).
#
# Optional flags:
#   --checkpoint <path>   path to init checkpoint
#   --config     <path>   path to titan_13b.yaml config
#   --log_dir    <path>   directory for pretrain.log

set -euo pipefail

# ── locate repo root from script's own path ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script can live in:
#   <root>/training/scripts/  (local Replit workspace)
#   <root>/scripts/           (server after git clone)
# Detect which by checking the immediate parent's name.
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_NAME="$(basename "$PARENT_DIR")"
if [ "$PARENT_NAME" = "training" ]; then
  # Local layout: training/scripts/ → repo root is two levels up
  REPO_ROOT="$(cd "$PARENT_DIR/.." && pwd)"
else
  # Server layout: scripts/ → repo root is one level up
  REPO_ROOT="$PARENT_DIR"
fi

# ── default paths (all relative to REPO_ROOT) ─────────────────────────────────
# Config: pushed to configs/titan_13b.yaml on GitHub; also at training/configs/ locally
if [ -f "$REPO_ROOT/configs/titan_13b.yaml" ]; then
  DEFAULT_CONFIG="$REPO_ROOT/configs/titan_13b.yaml"
else
  DEFAULT_CONFIG="$REPO_ROOT/training/configs/titan_13b.yaml"
fi

CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/checkpoints/titan_13b_pretrain/init.pt}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/titan_13b}"
LOG_FILE="${LOG_DIR}/pretrain.log"
MIN_VRAM_GB=80

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --config)     CONFIG="$2";     shift 2 ;;
    --log_dir)    LOG_DIR="$2";    LOG_FILE="${LOG_DIR}/pretrain.log"; shift 2 ;;
    *) echo "[13b-pretrain] Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── locate training entrypoint ────────────────────────────────────────────────
# train_13b.py lives alongside this script and handles entrypoint detection.
TRAIN_PY="$SCRIPT_DIR/train_13b.py"
if [ ! -f "$TRAIN_PY" ]; then
  echo "[13b-pretrain] ERROR: training launcher not found at $TRAIN_PY" >&2
  exit 1
fi

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "$CONFIG" ]; then
  echo "[13b-pretrain] ERROR: config not found at $CONFIG" >&2
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "[13b-pretrain] ERROR: init checkpoint not found at $CHECKPOINT" >&2
  echo "[13b-pretrain]   Run upscale_to_13b.py first:" >&2
  echo "[13b-pretrain]     python3 $SCRIPT_DIR/upscale_to_13b.py \\" >&2
  echo "[13b-pretrain]       --src_checkpoint <7b_best_model.pt> \\" >&2
  echo "[13b-pretrain]       --src_config <titan_7b.yaml> \\" >&2
  echo "[13b-pretrain]       --dst_config $CONFIG \\" >&2
  echo "[13b-pretrain]       --dst_checkpoint $CHECKPOINT" >&2
  exit 1
fi

# ── VRAM check ────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  FREE_MiB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  FREE_GB=$(( FREE_MiB / 1024 ))
  echo "[13b-pretrain] GPU free VRAM: ${FREE_GB} GB"
  if (( FREE_GB < MIN_VRAM_GB )); then
    echo "[13b-pretrain] WARNING: only ${FREE_GB} GB free — 13B training needs ${MIN_VRAM_GB}+ GB." >&2
  fi
else
  echo "[13b-pretrain] nvidia-smi not found — skipping VRAM check"
fi

# ── environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$LOG_DIR"

# ── launch training ───────────────────────────────────────────────────────────
echo "[13b-pretrain] =================================================="
echo "[13b-pretrain] Starting 13B continued pretraining"
echo "[13b-pretrain]   Repo root  : $REPO_ROOT"
echo "[13b-pretrain]   Train py   : $TRAIN_PY"
echo "[13b-pretrain]   Config     : $CONFIG"
echo "[13b-pretrain]   Checkpoint : $CHECKPOINT"
echo "[13b-pretrain]   Log file   : $LOG_FILE"
echo "[13b-pretrain] =================================================="

python3 "$TRAIN_PY" \
  --config     "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -eq 0 ]; then
  BEST_CKPT="$REPO_ROOT/checkpoints/titan_13b_pretrain/best_model.pt"
  echo ""
  echo "[13b-pretrain] Pretraining complete."
  echo "[13b-pretrain]   Best checkpoint: $BEST_CKPT"
  echo "[13b-pretrain]   Next step: run the full A-Z upgrade pipeline on $BEST_CKPT"
else
  echo "[13b-pretrain] Training exited with code $EXIT_CODE — check $LOG_FILE" >&2
  exit "$EXIT_CODE"
fi
