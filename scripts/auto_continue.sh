#!/bin/bash
# TitanAI — Auto-Continue Pipeline
# ===================================
# Run this ONCE while crucible is still training.
# It watches for the crucible checkpoint, then automatically runs
# the full post-crucible pipeline (data prep → SFT v2 → tool → DPO).
#
# Usage (run this one command from the Vast.ai web terminal):
#   nohup bash scripts/auto_continue.sh > logs/auto_continue.log 2>&1 &

WORKSPACE="/workspace/titanai"
CRUCIBLE_CHECKPOINT="$WORKSPACE/checkpoints/crucible_v02/final.pt"
POLL_INTERVAL=300
LOG_DIR="$WORKSPACE/logs"
LOCK_FILE="/tmp/titan_auto_continue.lock"

mkdir -p "$LOG_DIR"

echo "======================================"
echo "  TitanAI Auto-Continue Pipeline"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Watching for: $CRUCIBLE_CHECKPOINT"
echo "======================================"

if [ -f "$LOCK_FILE" ]; then
    existing_pid=$(cat "$LOCK_FILE")
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "[WARN] Already running (PID $existing_pid). Exiting."
        exit 0
    fi
fi
echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

export TITAN_SKIP_SHUTDOWN=true
echo "[INFO] Auto-shutdown suppressed — will only shut down after pipeline completes."

echo ""
echo "[WAIT] Checking every ${POLL_INTERVAL}s for crucible completion..."

while [ ! -f "$CRUCIBLE_CHECKPOINT" ]; do
    echo "  [$(date -u '+%H:%M UTC')] Crucible still running... next check in ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
done

echo ""
echo "======================================"
echo "  CRUCIBLE COMPLETE"
echo "  Checkpoint: $CRUCIBLE_CHECKPOINT"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================"

sleep 30

echo "[GIT] Pulling latest scripts..."
cd "$WORKSPACE"
git pull --ff-only origin main 2>&1 || echo "[WARN] git pull failed — using local scripts"

echo ""
echo "======================================"
echo "  Starting Post-Crucible Pipeline v2"
echo "  data_prep → sft_v2 → tool → dpo"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================"

python3 scripts/run_full_pipeline_v2.py \
    --checkpoint checkpoints/crucible_v02/final.pt \
    2>&1 | tee "$LOG_DIR/post_crucible_pipeline.log"

PIPELINE_EXIT=${PIPESTATUS[0]}

if [ $PIPELINE_EXIT -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "  ALL DONE — TITAN IS READY"
    echo "  Final model: checkpoints/dpo_v01/final.pt"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================"
    if [ -n "${TITAN_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "${TITAN_WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"event\":\"pipeline_complete\",\"model\":\"dpo_v01\",\"timestamp\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"}" \
            --max-time 10 || true
    fi
else
    echo ""
    echo "[FAILED] Pipeline exit code: $PIPELINE_EXIT"
    echo "Check: $LOG_DIR/post_crucible_pipeline.log"
fi

echo "[Shutdown] Shutting down in 120 seconds..."
sleep 120
shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
