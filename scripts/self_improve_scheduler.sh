#!/bin/bash
# Loop-based scheduler (runs inside a screen session — no cron needed)
# Runs a self-improvement cycle every 12 hours.
cd /workspace/titanai
while true; do
  echo ""
  echo "=== Self-Improve Cycle @ $(date) ==="
  python3 scripts/self_improve.py --config configs/self_improve.yaml     >> logs/self_improve/cron.log 2>&1
  echo "[Scheduler] Next cycle in 12 hours..."
  sleep 43200
done
