#!/bin/bash
# TitanAI — Vast.ai bootstrap (clean, no embedded credentials)
# Credentials must be set as env vars on the instance: TITAN_GITHUB_TOKEN, NOTIFY_TO

LOG=/workspace/logs/titanai_full
REPO=/workspace/titanai

mkdir -p "$LOG"
exec >> "$LOG/onstart.log" 2>&1
echo "[onstart] $(date -u) — starting"

# Free disk
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/huggingface 2>/dev/null || true
echo "[onstart] Disk: $(df -h / | awk 'NR==2{print $3"/"$2" ("$5" used)"}')"

# Clone if not present (train_all.sh will pull latest)
GH_URL="https://github.com/leego972/titanai.git"
[ -n "${TITAN_GITHUB_TOKEN:-}" ] && GH_URL="https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git"
if [ ! -d "$REPO/.git" ]; then
    git clone "$GH_URL" "$REPO" 2>&1 | tail -3
fi

# Checkpoint rotation — keep only 2 newest per phase (runs every 5 min)
(while true; do
    sleep 300
    for CKDIR in checkpoints/titan_1b_pretrain checkpoints/titan_1b checkpoints/titan_1b_instruct checkpoints/titan_1b_dpo; do
        mapfile -t CKPTS < <(ls -t "$REPO/$CKDIR/"*.pt 2>/dev/null)
        if [ "${#CKPTS[@]}" -gt 2 ]; then
            for old in "${CKPTS[@]:2}"; do
                echo "[cleanup] $(date -u) Removed $(basename "$old")"
                rm -f "$old"
            done
        fi
    done
done) &
echo "[onstart] Checkpoint rotation: PID=$!"

# GPU monitor + status push to GitHub (every 30 min)
(sleep 120 && while true; do
    cd "$REPO" 2>/dev/null || break
    git config user.email "titanai-bot@vast.ai" 2>/dev/null
    git config user.name  "TitanAI Status Bot" 2>/dev/null
    bash scripts/push_status_to_github.sh >> "$LOG/status_push.log" 2>&1 || true
    bash scripts/gpu_monitor.sh           >> "$LOG/gpu_monitor.log"  2>&1 || true
    sleep 1800
done) &
echo "[onstart] Status/GPU daemon: PID=$!"

# Smart watchdog — reads crash log, fixes root cause, then restarts
diagnose_and_fix() {
    local tail60; tail60=$(tail -60 "$LOG/master.log" 2>/dev/null || echo "")
    echo "[watchdog] $(date -u) Diagnosing crash:" >> "$LOG/watchdog.log"
    echo "$tail60" | tail -5 >> "$LOG/watchdog.log"
    if echo "$tail60" | grep -qi "out of memory\|CUDA out of memory\|CUBLAS"; then
        echo "[watchdog] OOM — halving gradient_accumulation_steps" >> "$LOG/watchdog.log"
        python3 -c "
import re; cfg='$REPO/configs/titan_1b.yaml'
txt=open(cfg).read(); m=re.search(r'gradient_accumulation_steps:\s*(\d+)',txt)
if m:
    old,new=int(m.group(1)),max(int(m.group(1))//2,4)
    open(cfg,'w').write(txt.replace(f'gradient_accumulation_steps: {old}',f'gradient_accumulation_steps: {new}'))
    print(f'[fix] grad_accum {old}->{new}')
" >> "$LOG/watchdog.log" 2>&1
    elif echo "$tail60" | grep -qi "no space left\|OSError.*28"; then
        echo "[watchdog] Disk full — emergency cleanup" >> "$LOG/watchdog.log"
        find "$REPO/checkpoints" -name "*.pt" | sort | head -n -2 | xargs rm -f 2>/dev/null || true
        pip cache purge 2>/dev/null || true
    elif echo "$tail60" | grep -qi "ModuleNotFoundError\|ImportError\|No module named"; then
        echo "[watchdog] Missing module — reinstalling" >> "$LOG/watchdog.log"
        pip install -q -r "$REPO/requirements.txt" 2>&1 | tail -3
    elif echo "$tail60" | grep -qi "nan\|loss.*inf\|diverged"; then
        echo "[watchdog] NaN — rolling back checkpoint" >> "$LOG/watchdog.log"
        mapfile -t CKPTS < <(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null)
        [ "${#CKPTS[@]}" -ge 2 ] && rm -f "${CKPTS[0]}"
    else
        echo "[watchdog] Unknown — waiting 3 min" >> "$LOG/watchdog.log"
        sleep 180
    fi
}

(while true; do
    sleep 90
    if ! pgrep -f "train_1b\.py\|pretrain_titan\|trainer\.py\|train_all" > /dev/null 2>&1; then
        if [ -f "$LOG/master.log" ]; then
            diagnose_and_fix
            echo "[watchdog] $(date -u) Restarting training" >> "$LOG/watchdog.log"
            LATEST=$(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null | head -1)
            [ -n "$LATEST" ] && export TITAN_RESUME="$LATEST"
            cd "$REPO" && 
# ─── Disk watchdog ────────────────────────────────────────────────────────────
# Runs in background; deletes old checkpoints when disk < 15GB free
disk_watchdog() {
    while true; do
        AVAIL=$(df /workspace --output=avail -BG 2>/dev/null | tail -1 | tr -d 'G ')
        AVAIL=${AVAIL:-99}
        if [ "$AVAIL" -lt 15 ] 2>/dev/null; then
            echo "[disk-watchdog] $(date -u) WARNING: ${AVAIL}GB free — purging old checkpoints..."
            for CKDIR in \
                "$REPO/checkpoints/titan_1b_pretrain" \
                "$REPO/checkpoints/titan_1b" \
                "$REPO/checkpoints/titan_1b_instruct" \
                "$REPO/checkpoints/titan_1b_dpo"; do
                [ -d "$CKDIR" ] || continue
                mapfile -t CKPTS < <(ls -t "$CKDIR"/*.pt 2>/dev/null)
                for old in "${CKPTS[@]:1}"; do
                    echo "[disk-watchdog] Removing $old"
                    rm -f "$old"
                done
            done
            pip cache purge 2>/dev/null || true
            rm -rf /root/.cache/huggingface 2>/dev/null || true
            AFTER=$(df /workspace --output=avail -BG 2>/dev/null | tail -1 | tr -d 'G ')
            echo "[disk-watchdog] After cleanup: ${AFTER}GB free"
        fi
        sleep 120
    done
}
disk_watchdog >> "$LOG_DIR/disk_watchdog.log" 2>&1 &
echo "[$(date -u +%T)] Disk watchdog started (PID $!)"

bash train_all.sh >> "$LOG/master.log" 2>&1 &
            sleep 60
        fi
    fi
done) &
echo "[onstart] Smart watchdog: PID=$!"

# Launch training
cd "$REPO"
LATEST=$(ls -t "$REPO/checkpoints/titan_1b_pretrain/"*.pt 2>/dev/null | head -1)
[ -n "$LATEST" ] && export TITAN_RESUME="$LATEST" && echo "[onstart] Resuming from: $(basename "$LATEST")"
echo "[onstart] Launching train_all.sh..."
nohup bash train_all.sh >> "$LOG/master.log" 2>&1 &
echo "[onstart] Training PID=$! — $(date -u)"
