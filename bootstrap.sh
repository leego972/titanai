#!/bin/bash
# TitanAI — Vast.ai smart launcher
# Watchdog reads crash logs, fixes the root cause, THEN restarts.

LOG=/workspace/logs/titanai_full
REPO=/workspace/titanai
MASTER_LOG="$LOG/master.log"
WATCHDOG_LOG="$LOG/watchdog.log"

mkdir -p "$LOG"
exec >> "$LOG/bootstrap.log" 2>&1
echo "[boot] $(date -u) — TitanAI bootstrap started"

# ── 1. Free disk first ────────────────────────────────────────────────────
pip cache purge 2>/dev/null || true
find /workspace -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /root/.cache/huggingface 2>/dev/null || true
echo "[boot] Disk: $(df -h / | awk 'NR==2{print $3"/"$2" used ("$5")"}')"

# ── 2. Clone or update repo ───────────────────────────────────────────────
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull origin main 2>&1 || true
else
    git clone https://github.com/leego972/titanai.git "$REPO" 2>&1
fi
cd "$REPO"
echo "[boot] Repo: $(git rev-parse --short HEAD)"

# ── 3. Install ALL dependencies ───────────────────────────────────────────
pip install -q -r requirements.txt 2>&1 | tail -3
pip install flash-attn --no-build-isolation -q 2>&1 | tail -2 || echo "[WARN] FA2 unavailable — SDPA fallback"
pip cache purge 2>/dev/null || true

# ── 4. Prepare data if missing ────────────────────────────────────────────
if [ ! -d "$REPO/data/processed" ] || [ -z "$(ls -A $REPO/data/processed 2>/dev/null)" ]; then
    echo "[boot] Preparing training data..."
    python3 prepare_data.py 2>&1 | tee "$LOG/data_prep.log"
fi

# ── 5. Checkpoint rotation — keep 2 newest only ───────────────────────────
(while true; do
    sleep 300
    for CKDIR in checkpoints/titan_1b_pretrain checkpoints/titan_1b checkpoints/titan_1b_instruct checkpoints/titan_1b_dpo; do
        mapfile -t CKPTS < <(ls -t "$REPO/$CKDIR/"*.pt 2>/dev/null)
        if [ "${#CKPTS[@]}" -gt 2 ]; then
            for old in "${CKPTS[@]:2}"; do
                echo "[cleanup] $(date -u) Removed: $(basename $old)"
                rm -f "$old"
            done
        fi
    done
done) &

# ── 6. Smart crash handler ────────────────────────────────────────────────
diagnose_and_fix() {
    local last_lines
    last_lines=$(tail -60 "$MASTER_LOG" 2>/dev/null || true)
    echo "[watchdog] $(date -u) Diagnosing crash..." >> "$WATCHDOG_LOG"
    echo "$last_lines" | tail -10 >> "$WATCHDOG_LOG"

    # CUDA out-of-memory — reduce batch size in config
    if echo "$last_lines" | grep -qi "out of memory\|CUDA out of memory\|CUBLAS_STATUS"; then
        echo "[watchdog] OOM detected — reducing gradient_accumulation_steps by half" >> "$WATCHDOG_LOG"
        python3 - << 'PY'
import yaml, re
cfg = 'configs/titan_1b.yaml'
with open(cfg) as f: txt = f.read()
m = re.search(r'gradient_accumulation_steps:\s*(\d+)', txt)
if m:
    old = int(m.group(1))
    new = max(old // 2, 4)
    txt = txt.replace(f'gradient_accumulation_steps: {old}', f'gradient_accumulation_steps: {new}')
    with open(cfg,'w') as f: f.write(txt)
    print(f'[fix] grad_accum {old} -> {new}')
PY
        return 0

    # Disk full — run emergency cleanup
    if echo "$last_lines" | grep -qi "no space left\|disk full\|OSError.*28"; then
        echo "[watchdog] Disk full — running emergency cleanup" >> "$WATCHDOG_LOG"
        find "$REPO/checkpoints" -name "*.pt" | sort -t_ -k2 -n | head -n -2 | xargs rm -f 2>/dev/null || true
        pip cache purge 2>/dev/null || true
        find /workspace -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        df -h / >> "$WATCHDOG_LOG"
        return 0

    # Missing data — re-prepare
    if echo "$last_lines" | grep -qi "no such file.*processed\|data.*not found\|FileNotFoundError.*data"; then
        echo "[watchdog] Missing training data — re-running prepare_data.py" >> "$WATCHDOG_LOG"
        python3 "$REPO/prepare_data.py" >> "$LOG/data_prep.log" 2>&1
        return 0

    # Missing packages — reinstall
    if echo "$last_lines" | grep -qi "ModuleNotFoundError\|ImportError\|No module named"; then
        echo "[watchdog] Missing module — reinstalling requirements" >> "$WATCHDOG_LOG"
        pip install -q -r "$REPO/requirements.txt" 2>&1 | tail -3
        return 0

    # NaN loss / training diverged — revert to 2nd newest checkpoint
    if echo "$last_lines" | grep -qi "nan\|loss.*inf\|diverged"; then
        echo "[watchdog] NaN/diverged — rolling back to previous checkpoint" >> "$WATCHDOG_LOG"
        mapfile -t CKPTS < <(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null)
        [ "${#CKPTS[@]}" -ge 2 ] && rm -f "${CKPTS[0]}" && echo "[watchdog] Removed bad checkpoint, using ${CKPTS[1]}" >> "$WATCHDOG_LOG"
        return 0

    # Unknown error — log it and wait 2 min before retry
    else
        echo "[watchdog] Unknown crash — waiting 2 min before retry" >> "$WATCHDOG_LOG"
        sleep 120
        return 0
    fi
}

# ── 7. Training launcher with resume ──────────────────────────────────────
start_training() {
    local LATEST
    LATEST=$(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null | head -1)
    [ -n "$LATEST" ] && export TITAN_RESUME="$LATEST" && echo "[train] Resuming: $(basename $LATEST)"
    echo "[train] $(date -u) Launching train_all.sh"
    bash "$REPO/train_all.sh" >> "$MASTER_LOG" 2>&1
    echo "[train] $(date -u) Exited: $?"
}

# ── 8. Smart watchdog loop ────────────────────────────────────────────────
(while true; do
    sleep 90
    if ! pgrep -f "train\.py|trainer\.py|train_all" > /dev/null 2>&1; then
        echo "[watchdog] $(date -u) Training stopped" >> "$WATCHDOG_LOG"
        diagnose_and_fix
        echo "[watchdog] $(date -u) Restarting training after fix" >> "$WATCHDOG_LOG"
        start_training &
        sleep 60
    fi
done) &
echo "[boot] Smart watchdog running — diagnoses crash before restart"

# ── 9. Start training ─────────────────────────────────────────────────────
start_training &
echo "[boot] Training PID=$! — monitor: tail -f $MASTER_LOG"
