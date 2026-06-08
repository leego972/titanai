#!/bin/bash
# gpu_monitor.sh — Background GPU watchdog for TitanAI training.
# Runs every 30 minutes. Logs stats, detects stalls, sends alerts.
# Launch as background process: nohup bash scripts/gpu_monitor.sh &

REPO="/workspace/titanai"
LOG_DIR="/workspace/logs/titanai_full"
GPU_LOG="${LOG_DIR}/gpu_stats.log"
CKPT_DIR="${REPO}/checkpoints"
CHECK_INTERVAL=1800  # 30 minutes

mkdir -p "${LOG_DIR}"

log_gpu() {
    local ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu  --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    local gpu_mem=$(nvidia-smi  --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    local gpu_name=$(nvidia-smi --query-gpu=name            --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")

    echo "[${ts}] GPU=${gpu_name} | Util=${gpu_util} | Mem=${gpu_mem} | Temp=${gpu_temp}°C" | tee -a "${GPU_LOG}"
}

detect_stall() {
    # Check if any checkpoint was updated in the last 3 hours
    local recent=$(find "${CKPT_DIR}" -name "*.pt" -newer "${LOG_DIR}/.last_check" 2>/dev/null | wc -l || echo 0)
    touch "${LOG_DIR}/.last_check"
    echo "${recent}"
}

notify() {
    local phase="$1"; local status="$2"; local detail="$3"
    cd "${REPO}" && python3 scripts/notify.py \
        --phase "${phase}" --status "${status}" --detail "${detail}" 2>/dev/null || true
}

echo "[gpu_monitor] Started. Checking every 30 min." | tee -a "${GPU_LOG}"
touch "${LOG_DIR}/.last_check"

STALL_COUNT=0

while true; do
    sleep ${CHECK_INTERVAL}
    log_gpu

    # Check for stalled training
    RECENT=$(detect_stall)
    if [ "${RECENT}" -eq 0 ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        echo "[gpu_monitor] WARNING: No new checkpoints in last 30 min (stall count: ${STALL_COUNT})" | tee -a "${GPU_LOG}"

        if [ "${STALL_COUNT}" -ge 2 ]; then
            # 2 consecutive stall checks (1 hour) = notify
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
            notify "Training may be stalled" "error" \
                "No checkpoint updates in ~1 hour. GPU util: ${GPU_UTIL}. Check master.log."
            STALL_COUNT=0
        fi
    else
        STALL_COUNT=0
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || echo "?")
        GPU_MEM=$(nvidia-smi  --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
        echo "[gpu_monitor] OK: ${RECENT} checkpoint(s) updated. GPU util: ${GPU_UTIL} | Mem: ${GPU_MEM}" | tee -a "${GPU_LOG}"
    fi
done
