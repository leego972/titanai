#!/bin/bash
# TitanAI — Complete Training Pipeline
# Phase 1: Pretraining → Phase 2: Upgrades → Phase 3: SFT → Phase 4: DPO
# RTX 4080S optimised | ~$0.25/hr

set -euo pipefail

REPO="/workspace/titanai"
GH_TOKEN="${TITAN_GITHUB_TOKEN:-}"
LOG_DIR="/workspace/logs/titanai_full"
CKPT_PRETRAIN="${REPO}/checkpoints/titan_1b_pretrain"
CKPT_UPGRADES="${REPO}/checkpoints/titan_1b"
CKPT_SFT="${REPO}/checkpoints/titan_1b_instruct"
CKPT_DPO="${REPO}/checkpoints/titan_1b_dpo"

mkdir -p "${LOG_DIR}" "${CKPT_PRETRAIN}" "${CKPT_SFT}" "${CKPT_DPO}"

log() { echo "[$(date -u '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/master.log"; }

# 1. Clone / update repo
REPO_URL="https://github.com/leego972/titanai.git"
[ -n "${GH_TOKEN}" ] && REPO_URL="https://${GH_TOKEN}@github.com/leego972/titanai.git"
if [ ! -d "${REPO}/.git" ]; then
    git clone "${REPO_URL}" "${REPO}"
else
    cd "${REPO}" && git fetch -q && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true
    git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || true
fi
cd "${REPO}"
log "Repo ready: $(git rev-parse --short HEAD)"

# 2. Dependencies
log "Installing Python packages..."
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1 || true
pip install -q -r requirements.txt 2>&1 | tail -1
pip install -q triton 2>&1 | tail -1 || true
log "Building FlashAttention-2 (one-time, ~10 min)..."
pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || log "[WARN] FA2 failed — SDPA fallback active"

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
log "Estimates shown. Training commencing..."

# 5. Inject cyber + film knowledge base into SFT data
log "Injecting cybersecurity + film knowledge base..."
python3 scripts/inject_knowledge_base.py 2>&1 | tee "${LOG_DIR}/kb_inject.log"

# 6. PHASE 1 — Pretraining
log "=== PHASE 1: PRETRAINING (305k steps / ~20B tokens) ==="
INIT_PT="${CKPT_PRETRAIN}/init.pt"
if [ ! -f "${INIT_PT}" ]; then
    log "Creating init checkpoint..."
    python3 upscale_to_1b.py 2>&1 | tee "${LOG_DIR}/init.log" || \
    python3 - << 'PYINIT'
import torch, os, yaml
os.makedirs('/workspace/titanai/checkpoints/titan_1b_pretrain', exist_ok=True)
with open('titan_1b.yaml') as f: cfg = yaml.safe_load(f)
print('[init] Saving random init placeholder...')
torch.save({'model': {}, 'step': 0, 'cfg': cfg}, '/workspace/titanai/checkpoints/titan_1b_pretrain/init.pt')
print('[init] Done.')
PYINIT
fi

RESUME_CKPT=$(ls -t "${CKPT_PRETRAIN}"/step_*.pt 2>/dev/null | head -1 || echo "")
START_CKPT="${RESUME_CKPT:-${INIT_PT}}"
RESUME_FLAG=""
[[ "${START_CKPT}" == *"step_"* ]] && RESUME_FLAG="--resume ${START_CKPT}"

python3 scripts/pretrain_titan_v3.py \
    --config "${REPO}/titan_1b.yaml" \
    --init-from "${START_CKPT}" \
    ${RESUME_FLAG} \
    --out-dir "${CKPT_PRETRAIN}" \
    --use-8bit-adam \
    --compile \
    2>&1 | tee "${LOG_DIR}/phase1_pretrain.log"
log "=== PHASE 1 COMPLETE ==="

# 7. PHASE 2 — Upgrade pipeline (all 55+ domain stages)
log "=== PHASE 2: UPGRADE PIPELINE (55 domains) ==="
BASE_CKPT=$(ls -t "${CKPT_PRETRAIN}/best.pt" "${CKPT_PRETRAIN}"/step_*.pt 2>/dev/null | head -1)
python3 scripts/train_1b_pipeline.py \
    --base-checkpoint "${BASE_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase2_upgrades.log"
log "=== PHASE 2 COMPLETE ==="

# 8. PHASE 3 — SFT (cyber + film knowledge + all upgrade data)
log "=== PHASE 3: SFT INSTRUCTION TUNING ==="
PHASE2_CKPT=$(ls -t \
    "${CKPT_UPGRADES}/upgrade_an/best.pt" \
    "${CKPT_UPGRADES}"/upgrade_*/best.pt \
    "${CKPT_PRETRAIN}"/step_*.pt \
    2>/dev/null | head -1)
python3 scripts/run_sft_v2.py \
    --config "${REPO}/titan_1b_instruct.yaml" \
    --checkpoint "${PHASE2_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase3_sft.log"
log "=== PHASE 3 COMPLETE ==="

# 9. PHASE 4 — DPO
log "=== PHASE 4: DPO ==="
SFT_CKPT=$(ls -t "${CKPT_SFT}/best.pt" "${CKPT_SFT}"/step_*.pt 2>/dev/null | head -1)
python3 scripts/run_dpo.py \
    --config "${REPO}/titan_1b_dpo.yaml" \
    --checkpoint "${SFT_CKPT}" \
    2>&1 | tee "${LOG_DIR}/phase4_dpo.log"
log "=== PHASE 4 COMPLETE ==="

# 10. Final summary
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
log "ALL DONE. Time: ${ELAPSED_H}h ${ELAPSED_M}m | Cost: ~\$${COST}"
