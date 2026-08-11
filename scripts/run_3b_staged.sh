#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-}"
if [[ ! "$STAGE" =~ ^[0-6]$ ]]; then
  echo "Usage: bash scripts/run_3b_staged.sh <stage 0..6>"
  echo "Optional env: TITAN_1B_CKPT=/path/to/1b/final.pt"
  exit 2
fi

CONFIGS=(
  "configs/3b_staged/stage0_stabilize_100m.yaml"
  "configs/3b_staged/stage1_500m.yaml"
  "configs/3b_staged/stage2_2b.yaml"
  "configs/3b_staged/stage3_5b.yaml"
  "configs/3b_staged/stage4_10b.yaml"
  "configs/3b_staged/stage5_20b.yaml"
  "configs/3b_staged/stage6_40b.yaml"
)
OUTS=(
  "checkpoints/titan_3b_stage0"
  "checkpoints/titan_3b_stage1"
  "checkpoints/titan_3b_stage2"
  "checkpoints/titan_3b_stage3"
  "checkpoints/titan_3b_stage4"
  "checkpoints/titan_3b_stage5"
  "checkpoints/titan_3b_stage6"
)

CONFIG="${CONFIGS[$STAGE]}"
OUT="${OUTS[$STAGE]}"
INIT3B="checkpoints/titan_3b_pretrain/init.pt"

mkdir -p "$OUT" checkpoints/titan_3b_pretrain

# Cheap safety checks before consuming paid GPU time.
python scripts/validate_reasoning_datasets.py
python -m py_compile scripts/pretrain_titan_v3.py scripts/upscale_to_3b.py model/titan_model.py

if [[ "$STAGE" == "0" ]]; then
  if [[ ! -f "$INIT3B" ]]; then
    SRC="${TITAN_1B_CKPT:-checkpoints/titan_1b_pretrain/final.pt}"
    if [[ ! -f "$SRC" ]]; then
      echo "ERROR: 1B source checkpoint not found: $SRC"
      echo "Set TITAN_1B_CKPT=/absolute/path/to/your/1B/final.pt and rerun."
      exit 3
    fi
    echo "[Titan3B] Creating validated 3B init from: $SRC"
    python scripts/upscale_to_3b.py \
      --src_checkpoint "$SRC" \
      --src_config configs/titan_1b.yaml \
      --dst_config configs/titan_3b.yaml \
      --dst_checkpoint "$INIT3B"
  fi
  INIT="$INIT3B"
  RESUME_ARGS=()
else
  PREV=$((STAGE-1))
  PREV_OUT="${OUTS[$PREV]}"
  if [[ -f "$PREV_OUT/final.pt" ]]; then
    RESUME="$PREV_OUT/final.pt"
  elif [[ -f "$PREV_OUT/best.pt" ]]; then
    RESUME="$PREV_OUT/best.pt"
  else
    echo "ERROR: Stage $PREV checkpoint missing in $PREV_OUT"
    echo "Complete the previous stage first."
    exit 4
  fi
  INIT="$RESUME"
  RESUME_ARGS=(--resume "$RESUME")
fi

# If this stage was interrupted, resume the newest step checkpoint automatically.
LATEST_STEP="$(find "$OUT" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "$LATEST_STEP" ]]; then
  echo "[Titan3B] Resuming interrupted Stage $STAGE from $LATEST_STEP"
  INIT="$LATEST_STEP"
  RESUME_ARGS=(--resume "$LATEST_STEP")
fi

echo "============================================================"
echo " TitanAI 3B staged training"
echo " Stage : $STAGE"
echo " Config: $CONFIG"
echo " Init  : $INIT"
echo " Out   : $OUT"
echo "============================================================"

python scripts/pretrain_titan_v3.py \
  --init-from "$INIT" \
  --out-dir "$OUT" \
  --config "$CONFIG" \
  "${RESUME_ARGS[@]}"

echo "[Titan3B] Stage $STAGE completed. Final checkpoint: $OUT/final.pt"
