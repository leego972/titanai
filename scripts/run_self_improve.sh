#!/bin/bash
# TitanAI Self-Improvement Launcher
# Runs one self-improvement cycle and logs output.
# Can be triggered manually or via cron.
#
# Cron example (run nightly at 2AM):
#   0 2 * * * /workspace/titanai/scripts/run_self_improve.sh >> /workspace/titanai/logs/self_improve/cron.log 2>&1

set -e
cd /workspace/titanai

LOG_DIR="logs/self_improve"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cycle_${TIMESTAMP}.log"

echo "========================================"
echo "  TitanAI Self-Improvement Cycle"
echo "  $(date)"
echo "========================================"

# ── Ensure a production checkpoint exists ────────────────────────────────────
PROD_CKPT="checkpoints/self_improve/production.pt"
if [ ! -f "$PROD_CKPT" ]; then
  echo "[Setup] No production checkpoint found — finding latest upgrade checkpoint..."
  mkdir -p checkpoints/self_improve

  LATEST=$(ls -t checkpoints/upgrade_*/final.pt 2>/dev/null | head -1)
  if [ -z "$LATEST" ]; then
    echo "[ERROR] No trained upgrade checkpoints found. Run the upgrade pipeline first."
    exit 1
  fi
  cp "$LATEST" "$PROD_CKPT"
  echo "[Setup] Initialised production checkpoint from: $LATEST"
fi

# ── Run self-improvement cycle ───────────────────────────────────────────────
echo "[Run] Starting self-improvement cycle | log: $LOG_FILE"
python3 scripts/self_improve.py \
  --config configs/self_improve.yaml \
  2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "[Done] Self-improvement cycle completed successfully."
  echo "       Log: $LOG_FILE"
  echo "       Cycle log: $LOG_DIR/cycles.jsonl"
else
  echo ""
  echo "[FAIL] Self-improvement cycle failed (exit=$EXIT_CODE)."
  echo "       Check log: $LOG_FILE"
fi

exit $EXIT_CODE
