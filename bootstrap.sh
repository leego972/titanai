#!/bin/bash
# TitanAI bootstrap — Vast.ai cold boot (repo is PUBLIC)
set -e
LOG=/workspace/logs/titanai_full
mkdir -p "$LOG"
echo "[boot] $(date) starting" | tee "$LOG/bootstrap.log"

# Clone public repo (no token needed)
if [ -d /workspace/titanai/.git ]; then
  git -C /workspace/titanai pull origin main 2>&1 | tee -a "$LOG/bootstrap.log"
else
  git clone https://github.com/leego972/titanai.git /workspace/titanai 2>&1 | tee -a "$LOG/bootstrap.log"
fi

echo "[boot] Launching training" | tee -a "$LOG/bootstrap.log"
cd /workspace/titanai
nohup bash train_all.sh >> "$LOG/master.log" 2>&1 &
echo "[boot] train_all.sh PID=$! launched" | tee -a "$LOG/bootstrap.log"
