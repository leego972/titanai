#!/bin/bash
# TitanAI — Vast.ai cold-boot launcher
# Calls train_all.sh directly (no auth gates, public repo)

LOG=/workspace/logs/titanai_full
mkdir -p "$LOG"
echo "[boot] $(date -u '+%Y-%m-%d %H:%M:%S UTC') — bootstrap started" | tee "$LOG/bootstrap.log"

# Clone or update (public repo — no token needed for read)
REPO=/workspace/titanai
if [ -d "$REPO/.git" ]; then
    echo "[boot] Updating repo..." | tee -a "$LOG/bootstrap.log"
    git -C "$REPO" pull origin main 2>&1 | tee -a "$LOG/bootstrap.log" || true
else
    echo "[boot] Cloning repo..." | tee -a "$LOG/bootstrap.log"
    git clone https://github.com/leego972/titanai.git "$REPO" 2>&1 | tee -a "$LOG/bootstrap.log"
fi

echo "[boot] Launching train_all.sh..." | tee -a "$LOG/bootstrap.log"
cd "$REPO"
nohup bash train_all.sh >> "$LOG/master.log" 2>&1 &
TRAIN_PID=$!
echo "[boot] train_all.sh PID=$TRAIN_PID — training running" | tee -a "$LOG/bootstrap.log"
