#!/bin/bash
# TitanAI — Vast.ai launcher with watchdog + checkpoint rotation

LOG=/workspace/logs/titanai_full
mkdir -p "$LOG"
exec >> "$LOG/bootstrap.log" 2>&1
echo "[boot] $(date -u) — TitanAI bootstrap started"

# Free disk first
pip cache purge 2>/dev/null || true
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /root/.cache/huggingface 2>/dev/null || true
echo "[boot] Disk after cleanup: $(df -h / | tail -1)"

# Clone or update repo
REPO=/workspace/titanai
if [ -d "$REPO/.git" ]; then
    echo "[boot] Pulling latest code..."
    git -C "$REPO" pull origin main 2>&1 || true
else
    echo "[boot] Cloning repo..."
    git clone https://github.com/leego972/titanai.git "$REPO" 2>&1
fi
cd "$REPO"
echo "[boot] Repo: $(git rev-parse --short HEAD)"

# Install ALL dependencies
echo "[boot] Installing dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3
pip install flash-attn --no-build-isolation -q 2>&1 | tail -2 || echo "[WARN] FA2 unavailable — SDPA fallback"
pip cache purge 2>/dev/null || true

# Prepare data if missing
if [ ! -d "$REPO/data/processed" ] || [ -z "$(ls -A $REPO/data/processed 2>/dev/null)" ]; then
    echo "[boot] Preparing training data..."
    python3 prepare_data.py 2>&1 | tee "$LOG/data_prep.log"
fi

# Checkpoint rotation — keeps only 2 newest per phase, runs every 5 min
(while true; do
    sleep 300
    for CKDIR in checkpoints/titan_1b_pretrain checkpoints/titan_1b checkpoints/titan_1b_instruct checkpoints/titan_1b_dpo; do
        mapfile -t CKPTS < <(ls -t "$REPO/$CKDIR/"*.pt 2>/dev/null)
        if [ "${#CKPTS[@]}" -gt 2 ]; then
            for old in "${CKPTS[@]:2}"; do
                echo "[cleanup] $(date -u) Removing old checkpoint: $(basename $old)"
                rm -f "$old"
            done
            echo "[cleanup] Disk now: $(df -h / | tail -1)"
        fi
    done
done) &
echo "[boot] Checkpoint rotation daemon running (keeps 2 newest)"

start_training() {
    LATEST=$(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && export TITAN_RESUME="$LATEST" && echo "[train] Resuming from: $(basename $LATEST)"
    echo "[train] $(date -u) Starting train_all.sh..."
    bash "$REPO/train_all.sh" >> "$LOG/master.log" 2>&1
    echo "[train] $(date -u) train_all.sh exited with code $?"
}

# Watchdog — auto-restarts training if it crashes, every 60s check
(while true; do
    sleep 60
    if ! pgrep -f "train\.py|trainer\.py|train_all" > /dev/null 2>&1; then
        echo "[watchdog] $(date -u) Training not running — restarting..."
        start_training &
        sleep 30
    fi
done) &
echo "[boot] Watchdog running (auto-restarts training on crash)"

# Start training now
start_training &
TRAIN_PID=$!
echo "[boot] Training started PID=$TRAIN_PID"
