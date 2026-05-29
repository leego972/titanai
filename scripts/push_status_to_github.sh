#!/bin/bash
# push_status_to_github.sh — auto-run every 30 min via cron
# Pushes live training status, corpus progress, log tails to GitHub
set -euo pipefail
REPO=/workspace/titanai
cd "$REPO"

STEP=$(grep -oP "step=\s*\K[0-9]+" logs/titan_1b/upgrade_bm/training.log 2>/dev/null | tail -1 || echo "unknown")
LOSS=$(grep -oP "loss=\K[0-9.]+" logs/titan_1b/upgrade_bm/training.log 2>/dev/null | tail -1 || echo "unknown")
STAGE=$(cat /workspace/status.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('current_stage','?'))" 2>/dev/null || echo "unknown")
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "unknown")
CORPUS_C=$(find data/raw/corpus_C_technical -name "*.txt" 2>/dev/null | wc -l || echo 0)
LOADER=$(ps aux | grep load_corpus | grep -v grep | awk '{print $11}' | xargs -I{} basename {} .py 2>/dev/null | head -1 || echo "none")

python3 -c "
import json
data = {
  \"updated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
  \"training\": { \"stage\": \"$STAGE\", \"step\": \"$STEP\", \"loss\": \"$LOSS\", \"gpu\": \"$GPU\" },
  \"corpus_injection\": { \"corpus_C_docs\": $CORPUS_C, \"active_loader\": \"$LOADER\" }
}
json.dump(data, open('data/live_status.json', 'w'), indent=2)
"

tail -50 logs/titan_1b/upgrade_bm/training.log > data/training_log_latest.txt 2>/dev/null || true
tail -20 data/corpus_master.log > data/corpus_progress_latest.txt 2>/dev/null || true

git config user.email "titanai-bot@vast.ai"
git config user.name  "TitanAI Status Bot"
git add data/live_status.json data/training_log_latest.txt data/corpus_progress_latest.txt
if ! git diff --cached --quiet; then
  MSG="status: stage=$STAGE step=$STEP loss=$LOSS corpus_C=${CORPUS_C}docs"
  git commit -m "$MSG"
  git push origin main -q
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushed to GitHub OK"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No changes"
fi
