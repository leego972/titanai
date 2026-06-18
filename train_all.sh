#!/bin/bash
# TitanAI — Complete Training Pipeline v2
# Phase 1: Pretraining → Phase 2: Upgrades → Phase 3: SFT → Phase 4: DPO
# RTX 4080S optimised | ~$0.25/hr
# Email notifications: set SMTP_USER + SMTP_PASS env vars (Gmail App Password)

set -e
# pipefail disabled — individual errors handled with || true to prevent silent crash

REPO="/workspace/titanai"
GH_TOKEN="${TITAN_GITHUB_TOKEN:-}"
LOG_DIR="/workspace/logs/titanai_full"
CKPT_PRETRAIN="${REPO}/checkpoints/titan_1b_pretrain"
CKPT_UPGRADES="${REPO}/checkpoints/titan_1b"
CKPT_SFT="${REPO}/checkpoints/titan_1b_instruct"
CKPT_DPO="${REPO}/checkpoints/titan_1b_dpo"

mkdir -p "${LOG_DIR}" "${CKPT_PRETRAIN}" "${CKPT_SFT}" "${CKPT_DPO}"
# ── DISK WATCHDOG ────────────────────────────────────────────────────────────
# Runs every 3 min in background. Keeps disk under 80% by pruning old ckpts.
disk_watchdog() {
    while true; do
        sleep 180
        USED=$(df /workspace --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
        [ -z "$USED" ] || [ "$USED" -lt 65 ] && continue

        log "[DISK] ${USED}% used — pruning old checkpoints..."

        # Delete pip cache first (safe, always reclaimable)
        pip cache purge 2>/dev/null || true
        rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null || true

        # Prune step_*.pt files — keep only the 2 most recent per dir
        for CKPT_DIR in "${CKPT_PRETRAIN}" "${CKPT_UPGRADES}" "${CKPT_SFT}" "${CKPT_DPO}"; do
            STEPS=$(ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null)
            COUNT=$(echo "$STEPS" | grep -c '.pt' 2>/dev/null || echo 0)
            if [ "$COUNT" -gt 2 ]; then
                echo "$STEPS" | tail -n +3 | xargs rm -f 2>/dev/null || true
                log "[DISK] Pruned $(( COUNT - 2 )) old step ckpts in ${CKPT_DIR}"
            fi
        done

        # Emergency: >88% — keep only 1 per dir
        USED2=$(df /workspace --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
        if [ "${USED2:-0}" -gt 88 ]; then
            log "[DISK][EMERGENCY] ${USED2}% — keeping only latest checkpoint per phase"
            for CKPT_DIR in "${CKPT_PRETRAIN}" "${CKPT_UPGRADES}" "${CKPT_SFT}" "${CKPT_DPO}"; do
                STEPS=$(ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null)
                echo "$STEPS" | tail -n +2 | xargs rm -f 2>/dev/null || true
            done
            # Truncate old log files > 100MB
            find "${LOG_DIR}" -name "*.log" -size +100M -exec truncate -s 50M {} \; 2>/dev/null || true
        fi

        USED3=$(df /workspace --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
        log "[DISK] After cleanup: ${USED3}% used"
    done
}
# ─────────────────────────────────────────────────────────────────────────────



log() { echo "[$(date -u '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/master.log"; }

notify() {
    local phase="$1"; local status="$2"; local detail="${3:-}"
    python3 "${REPO}/scripts/notify.py" \
        --phase "${phase}" --status "${status}" --detail "${detail}" 2>&1 | tee -a "${LOG_DIR}/notify.log" || true
}

# 1. Clone / update repo
REPO_URL="https://github.com/leego972/titanai.git"
[ -n "${GH_TOKEN}" ] && REPO_URL="https://${GH_TOKEN}@github.com/leego972/titanai.git"
if [ ! -d "${REPO}/.git" ]; then
    git clone "${REPO_URL}" "${REPO}" || { echo "[ERROR] git clone failed"; mkdir -p "${REPO}"; cd "${REPO}"; git init; } 
else
    cd "${REPO}" && git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
fi
cd "${REPO}" || { echo "[FATAL] Cannot cd to ${REPO}"; exit 1; }
log "Repo: $(git rev-parse --short HEAD)"

# ── MULTI-GPU DETECTION ──────────────────────────────────────────────────────
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
log "Detected ${NUM_GPUS} GPU(s)"
if [ "${NUM_GPUS}" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS}"
else
    TRAIN_CMD="python3"
fi
# ─────────────────────────────────────────────────────────────────────────────


# Set notification email target
export NOTIFY_TO="${NOTIFY_TO:-leego972@gmail.com}"

# 2. Dependencies
log "Installing Python packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1 || true
pip install -q -r requirements.txt 2>&1 | tail -1
pip install -q triton 2>&1 | tail -1 || true
pip cache purge 2>/dev/null || true
rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null || true
log "Disk after pip cleanup: $(df -h /workspace | tail -1)"
# FlashAttention-2 requires Ampere+ (sm_80+). Skip on Turing/Volta.
CC=$(python3 -c "import torch; cc=torch.cuda.get_device_capability(0); print(cc[0]*10+cc[1])" 2>/dev/null || echo 0)
if [ "${CC}" -ge 80 ]; then
    log "Building FlashAttention-2 (sm_${CC}, one-time ~10 min)..."
    pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || log "[WARN] FA2 failed — SDPA fallback active"
else
    log "[SKIP] FlashAttention-2 requires sm_80+, this GPU is sm_${CC} — using PyTorch SDPA instead"
fi

# 3. Speed env
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false

# 4. GPU info + time/cost estimate
GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown GPU")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
TRAIN_START=$(date +%s)

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║             TITANAI — FULL TRAINING STARTED                   ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
printf  "║  GPU      : %-49s║\n" "${GPU_NAME} (${GPU_VRAM})"
printf  "║  Started  : %-49s║\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
printf  "║  Notify   : %-49s║\n" "${NOTIFY_TO}"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  TIME & COST ESTIMATE  (RTX 4080S @ \$0.25/hr spot):          ║"
echo "║                                                               ║"
echo "║  Phase 1 — Pretraining  (305 000 steps, ~20B tokens)         ║"
echo "║            ≈ 40–48 hrs                        ≈ \$10–12       ║"
echo "║  Phase 2 — Upgrade pipeline (55 domains)                     ║"
echo "║            ≈ 28–35 hrs                        ≈  \$7–9        ║"
echo "║  Phase 3 — SFT instruction tuning                            ║"
echo "║            ≈  4–6 hrs                         ≈  \$1–2        ║"
echo "║  Phase 4 — DPO preference pass                               ║"
echo "║            ≈  4–5 hrs                         ≈  \$1          ║"
echo "║  ─────────────────────────────────────────────────────        ║"
echo "║  TOTAL     ≈ 76–94 hrs              TOTAL COST ≈ \$19–24      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Launch disk watchdog
disk_watchdog &
DISK_WATCHDOG_PID=$!
log "Disk watchdog started (PID ${DISK_WATCHDOG_PID})"

# 5. Launch GPU monitor in background (checks every 30 min)
nohup bash "${REPO}/scripts/gpu_monitor.sh" >> "${LOG_DIR}/gpu_monitor.log" 2>&1 &
GPU_MONITOR_PID=$!
log "GPU monitor started (PID ${GPU_MONITOR_PID})"

# Send start notification
notify "Training started" "info" \
    "GPU: ${GPU_NAME} (${GPU_VRAM}). Est. 76-94 hrs, ~\$19-24. Logs: /workspace/logs/titanai_full/master.log"

# 6. Inject cyber + film knowledge base
log "Injecting cybersecurity + film knowledge base..."
python3 "${REPO}/scripts/inject_knowledge_base.py" 2>&1 | tee "${LOG_DIR}/kb_inject.log"

# 7. PHASE 1 — Pretraining
log "=== PHASE 1: PRETRAINING ==="
INIT_PT="${CKPT_PRETRAIN}/init.pt"
if [ ! -f "${INIT_PT}" ]; then
    log "Creating init checkpoint..."
    python3 "${REPO}/upscale_to_1b.py" 2>&1 | tee "${LOG_DIR}/init.log" || \
    python3 - << 'PYINIT'
import torch, os, yaml
os.makedirs('/workspace/titanai/checkpoints/titan_1b_pretrain', exist_ok=True)
with open('/workspace/titanai/titan_1b.yaml') as f: cfg = yaml.safe_load(f)
torch.save({'model': {}, 'step': 0, 'cfg': cfg}, '/workspace/titanai/checkpoints/titan_1b_pretrain/init.pt')
print('[init] Random init saved.')
PYINIT
fi

RESUME_CKPT=$(ls -t "${CKPT_PRETRAIN}"/step_*.pt 2>/dev/null | head -1 || echo "")
START_CKPT="${RESUME_CKPT:-${INIT_PT}}"
RESUME_FLAG=""
[[ "${START_CKPT}" == *"step_"* ]] && RESUME_FLAG="--resume ${START_CKPT}"

${TRAIN_CMD} "${REPO}/scripts/pretrain_titan_v3.py" \
    --config "${REPO}/titan_1b.yaml" \
    --init-from "${START_CKPT}" \
    ${RESUME_FLAG} \
    --out-dir "${CKPT_PRETRAIN}" \
    --use-8bit-adam \
    2>&1 | tee "${LOG_DIR}/phase1_pretrain.log"

P1_CKPT_COUNT=$(ls "${CKPT_PRETRAIN}"/*.pt 2>/dev/null | wc -l || echo 0)
notify "Phase 1 COMPLETE — Pretraining" "ok" \
    "305k steps done. Checkpoints saved: ${P1_CKPT_COUNT}. Starting upgrade pipeline..."
log "=== PHASE 1 COMPLETE ==="

# 8. PHASE 2 — Upgrade pipeline
log "=== PHASE 2: UPGRADE PIPELINE (55 domains) ==="
BASE_CKPT=$(ls -t "${CKPT_PRETRAIN}/best.pt" "${CKPT_PRETRAIN}"/step_*.pt 2>/dev/null | head -1)
python3 "${REPO}/scripts/train_1b_pipeline.py" \
    --base-checkpoint "${BASE_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase2_upgrades.log"

notify "Phase 2 COMPLETE — Upgrade Pipeline" "ok" \
    "All 55 domain upgrade stages done. Starting SFT instruction tuning..."
log "=== PHASE 2 COMPLETE ==="

# 9. PHASE 3 — SFT
log "=== PHASE 3: SFT INSTRUCTION TUNING ==="
PHASE2_CKPT=$(ls -t \
    "${CKPT_UPGRADES}/upgrade_an/best.pt" \
    "${CKPT_UPGRADES}"/upgrade_*/best.pt \
    "${CKPT_PRETRAIN}"/step_*.pt \
    2>/dev/null | head -1)
python3 "${REPO}/scripts/run_sft_v2.py" \
    --config "${REPO}/titan_1b_instruct.yaml" \
    --checkpoint "${PHASE2_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase3_sft.log"

notify "Phase 3 COMPLETE — SFT Instruction Tuning" "ok" \
    "Cyber + film knowledge base + Alpaca + Dolly fine-tuning done. Starting DPO..."
log "=== PHASE 3 COMPLETE ==="

# 10. PHASE 4 — DPO
log "=== PHASE 4: DPO ==="
SFT_CKPT=$(ls -t "${CKPT_SFT}/best.pt" "${CKPT_SFT}"/step_*.pt 2>/dev/null | head -1)
python3 "${REPO}/scripts/run_dpo.py" \
    --config "${REPO}/titan_1b_dpo.yaml" \
    --checkpoint "${SFT_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase4_dpo.log"

log "=== PHASE 4 COMPLETE ==="

# 11. Final summary + notification
TRAIN_END=$(date +%s)
ELAPSED_S=$(( TRAIN_END - TRAIN_START ))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))
COST=$(awk "BEGIN {printf \"%.2f\", ${ELAPSED_H} * 0.25}")

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           TITANAI TRAINING — ALL PHASES COMPLETE             ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
printf  "║  Total time   : %-49s║\n" "${ELAPSED_H}h ${ELAPSED_M}m"
printf  "║  Actual cost  : ~\$%-48s║\n" "${COST}"
printf  "║  Final model  : %-49s║\n" "${CKPT_DPO}/"
printf  "║  Finished     : %-49s║\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
echo "╚═══════════════════════════════════════════════════════════════╝"

notify "ALL TRAINING COMPLETE" "ok" \
    "Total time: ${ELAPSED_H}h ${ELAPSED_M}m. Actual cost: ~\$${COST}. Final model: ${CKPT_DPO}/"

kill ${GPU_MONITOR_PID} 2>/dev/null || true
kill ${DISK_WATCHDOG_PID} 2>/dev/null || true
log "ALL DONE. Time: ${ELAPSED_H}h ${ELAPSED_M}m | Cost: ~\$${COST}"
