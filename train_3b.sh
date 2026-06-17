#!/bin/bash
# TitanAI — 3B Training Pipeline v1.0
# Upscales trained 1B model then continues training at 3B scale.
# Run AFTER train_all.sh (1B) has completed.

set -e

REPO="/workspace/titanai"
LOG_DIR="/workspace/logs/titanai_3b"
CKPT_1B_DPO="${REPO}/checkpoints/titan_1b_dpo"
CKPT_3B_PRETRAIN="${REPO}/checkpoints/titan_3b_pretrain"
CKPT_3B_UPGRADES="${REPO}/checkpoints/titan_3b"
CKPT_3B_SFT="${REPO}/checkpoints/titan_3b_instruct"
CKPT_3B_DPO="${REPO}/checkpoints/titan_3b_dpo"

mkdir -p "${LOG_DIR}" "${CKPT_3B_PRETRAIN}" "${CKPT_3B_SFT}" "${CKPT_3B_DPO}"

log() { echo "[$(date -u '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/master.log"; }

notify() {
    local phase="$1"; local status="$2"; local detail="${3:-}"
    python3 "${REPO}/scripts/notify.py" \
        --phase "${phase}" --status "${status}" --detail "${detail}" 2>&1 | tee -a "${LOG_DIR}/notify.log" || true
}

# Disk watchdog — same as 1B pipeline
disk_watchdog() {
    while true; do
        sleep 180
        USED=$(df /workspace --output=pcent 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
        [ -z "$USED" ] || [ "$USED" -lt 65 ] && continue
        log "[DISK] ${USED}% — pruning..."
        pip cache purge 2>/dev/null || true
        for D in "${CKPT_3B_PRETRAIN}" "${CKPT_3B_UPGRADES}" "${CKPT_3B_SFT}" "${CKPT_3B_DPO}"; do
            STEPS=$(ls -t "${D}"/step_*.pt 2>/dev/null)
            COUNT=$(echo "$STEPS" | grep -c '.pt' 2>/dev/null || echo 0)
            if [ "$COUNT" -gt 2 ]; then
                echo "$STEPS" | tail -n +3 | xargs rm -f 2>/dev/null || true
                log "[DISK] Pruned old ckpts in ${D}"
            fi
        done
    done
}

# ── Pull latest repo ──────────────────────────────────────────────────────────
cd "${REPO}" && git pull origin main 2>/dev/null || true
log "Repo: $(git rev-parse --short HEAD)"

# ── GPU detection ─────────────────────────────────────────────────────────────
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
log "Detected ${NUM_GPUS} GPU(s)"
if [ "${NUM_GPUS}" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=${NUM_GPUS}"
else
    TRAIN_CMD="python3"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name         --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "?")
TRAIN_START=$(date +%s)

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          TITANAI 3B — TRAINING STARTED (UPSCALED)            ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
printf  "║  GPU      : %-49s║\n" "${GPU_NAME} (${GPU_VRAM})"
printf  "║  Started  : %-49s║\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  3B = depth-scaled 1B (66 layers vs 24)                      ║"
echo "║  Phase 1  : 40B token continued pretraining  (~2x 1B steps)  ║"
echo "║  Phase 2  : 55-domain upgrade pipeline                        ║"
echo "║  Phase 3  : SFT instruction tuning                            ║"
echo "║  Phase 4  : DPO preference learning                           ║"
echo "║  TOTAL    : ~5.5x the 1B cost on same hardware               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TOKENIZERS_PARALLELISM=false

disk_watchdog &
DISK_WATCHDOG_PID=$!
log "Disk watchdog started (PID ${DISK_WATCHDOG_PID})"

notify "3B Training started" "info" \
    "GPU: ${GPU_NAME}. Upscaling 1B→3B then training 40B tokens. Est. 5.5x 1B cost."

# ── UPSCALE 1B → 3B ──────────────────────────────────────────────────────────
log "=== UPSCALING 1B → 3B ==="

# Find best 1B checkpoint: prefer DPO final, fall back through pipeline
SRC_CKPT=$(ls -t \
    "${CKPT_1B_DPO}/final.pt" \
    "${CKPT_1B_DPO}"/best.pt \
    "${CKPT_1B_DPO}"/step_*.pt \
    "${REPO}/checkpoints/titan_1b_instruct"/best.pt \
    "${REPO}/checkpoints/titan_1b_pretrain"/best.pt \
    "${REPO}/checkpoints/titan_1b_pretrain"/step_*.pt \
    2>/dev/null | head -1)

if [ -z "${SRC_CKPT}" ]; then
    log "[FATAL] No 1B checkpoint found. Run train_all.sh first."
    exit 1
fi

log "Using 1B source: ${SRC_CKPT}"

INIT_3B="${CKPT_3B_PRETRAIN}/init.pt"
if [ ! -f "${INIT_3B}" ]; then
    python3 "${REPO}/upscale_to_3b.py" \
        --src_checkpoint "${SRC_CKPT}" \
        --src_config     "${REPO}/titan_1b.yaml" \
        --dst_config     "${REPO}/titan_3b.yaml" \
        --dst_checkpoint "${INIT_3B}" \
        2>&1 | tee "${LOG_DIR}/upscale.log"
    log "Upscale complete: ${INIT_3B}"
else
    log "3B init checkpoint already exists, skipping upscale."
fi

# ── PHASE 1: Continued pretraining at 3B scale ───────────────────────────────
log "=== PHASE 1: 3B PRETRAINING (40B tokens) ==="

RESUME_CKPT=$(ls -t "${CKPT_3B_PRETRAIN}"/step_*.pt 2>/dev/null | head -1 || echo "")
START_CKPT="${RESUME_CKPT:-${INIT_3B}}"
RESUME_FLAG=""
[[ "${START_CKPT}" == *"step_"* ]] && RESUME_FLAG="--resume ${START_CKPT}"

${TRAIN_CMD} "${REPO}/scripts/pretrain_titan_v3.py" \
    --config   "${REPO}/titan_3b.yaml" \
    --init-from "${START_CKPT}" \
    ${RESUME_FLAG} \
    --out-dir  "${CKPT_3B_PRETRAIN}" \
    --use-8bit-adam \
    --compile \
    2>&1 | tee "${LOG_DIR}/phase1_pretrain.log"

notify "3B Phase 1 COMPLETE" "ok" "40B token pretraining done. Starting domain upgrades..."
log "=== PHASE 1 COMPLETE ==="

# ── PHASE 2: Domain upgrades ─────────────────────────────────────────────────
log "=== PHASE 2: 3B UPGRADE PIPELINE (55 domains) ==="
BASE_CKPT=$(ls -t "${CKPT_3B_PRETRAIN}/best.pt" "${CKPT_3B_PRETRAIN}"/step_*.pt 2>/dev/null | head -1)

python3 "${REPO}/scripts/train_1b_pipeline.py" \
    --base-checkpoint "${BASE_CKPT}" \
    --output-dir      "${CKPT_3B_UPGRADES}" \
    2>&1 | tee "${LOG_DIR}/phase2_upgrades.log"

notify "3B Phase 2 COMPLETE" "ok" "55-domain upgrades done. Starting SFT..."
log "=== PHASE 2 COMPLETE ==="

# ── PHASE 3: SFT ─────────────────────────────────────────────────────────────
log "=== PHASE 3: 3B SFT ==="
PHASE2_CKPT=$(ls -t \
    "${CKPT_3B_UPGRADES}"/upgrade_*/best.pt \
    "${CKPT_3B_PRETRAIN}"/step_*.pt \
    2>/dev/null | head -1)

python3 "${REPO}/scripts/run_sft_v2.py" \
    --config     "${REPO}/titan_1b_instruct.yaml" \
    --checkpoint "${PHASE2_CKPT}" \
    --out-dir    "${CKPT_3B_SFT}" \
    2>&1 | tee "${LOG_DIR}/phase3_sft.log"

notify "3B Phase 3 COMPLETE" "ok" "SFT done. Starting DPO..."
log "=== PHASE 3 COMPLETE ==="

# ── PHASE 4: DPO ─────────────────────────────────────────────────────────────
log "=== PHASE 4: 3B DPO ==="
SFT_CKPT=$(ls -t "${CKPT_3B_SFT}/best.pt" "${CKPT_3B_SFT}"/step_*.pt 2>/dev/null | head -1)

python3 "${REPO}/scripts/run_dpo.py" \
    --config     "${REPO}/titan_1b_dpo.yaml" \
    --checkpoint "${SFT_CKPT}" \
    --out-dir    "${CKPT_3B_DPO}" \
    2>&1 | tee "${LOG_DIR}/phase4_dpo.log"

log "=== PHASE 4 COMPLETE ==="

# ── Final summary ─────────────────────────────────────────────────────────────
TRAIN_END=$(date +%s)
ELAPSED_S=$(( TRAIN_END - TRAIN_START ))
ELAPSED_H=$(( ELAPSED_S / 3600 ))
ELAPSED_M=$(( (ELAPSED_S % 3600) / 60 ))
COST=$(awk "BEGIN {printf \"%.2f\", ${ELAPSED_H} * 4.481}")

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          TITANAI 3B — ALL PHASES COMPLETE                    ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
printf  "║  Total time  : %-49s║\n" "${ELAPSED_H}h ${ELAPSED_M}m"
printf  "║  Actual cost : ~\$%-48s║\n" "${COST}"
printf  "║  Final model : %-49s║\n" "${CKPT_3B_DPO}/"
printf  "║  Finished    : %-49s║\n" "$(date -u '+%Y-%m-%d %H:%M UTC')"
echo "╚═══════════════════════════════════════════════════════════════╝"

notify "3B ALL TRAINING COMPLETE" "ok" \
    "Time: ${ELAPSED_H}h ${ELAPSED_M}m. Cost: ~\$${COST}. Model: ${CKPT_3B_DPO}/"

kill ${DISK_WATCHDOG_PID} 2>/dev/null || true
log "ALL DONE. Time: ${ELAPSED_H}h ${ELAPSED_M}m | Cost: ~\$${COST}"
