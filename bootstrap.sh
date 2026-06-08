#!/bin/bash
# TitanAI bootstrap — runs on Vast.ai cold boot
set -e
export DEBIAN_FRONTEND=noninteractive

LOG=/workspace/logs/titanai_full
mkdir -p "$LOG"

echo "[bootstrap] $(date) Starting TitanAI bootstrap" | tee -a "$LOG/bootstrap.log"

# Clone or update repo
REPO_DIR=/workspace/titanai
if [ -d "$REPO_DIR/.git" ]; then
  echo "[bootstrap] Updating repo..." | tee -a "$LOG/bootstrap.log"
  cd "$REPO_DIR" && git pull origin main 2>&1 | tee -a "$LOG/bootstrap.log"
else
  echo "[bootstrap] Cloning repo..." | tee -a "$LOG/bootstrap.log"
  cd /workspace && git clone "https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git" titanai 2>&1 | tee -a "$LOG/bootstrap.log"
  cd "$REPO_DIR"
fi

echo "[bootstrap] Launching train_all.sh..." | tee -a "$LOG/bootstrap.log"
nohup bash /workspace/titanai/train_all.sh > "$LOG/master.log" 2>&1 &
echo "[bootstrap] train_all.sh PID=$! launched" | tee -a "$LOG/bootstrap.log"