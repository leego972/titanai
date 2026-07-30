#!/bin/bash
# TitanAI — canonical 3B training pipeline
set -euo pipefail

REPO="${REPO:-/workspace/titanai}"
LOG_DIR="${LOG_DIR:-/workspace/logs/titanai_3b}"
CONFIG_1B="${REPO}/titan_1b.yaml"
CONFIG_3B="${REPO}/titan_3b.yaml"
CONFIG_3B_SFT="${REPO}/titan_3b_instruct.yaml"
CONFIG_3B_DPO="${REPO}/titan_3b_dpo.yaml"
CKPT_1B_DPO="${REPO}/checkpoints/titan_1b_dpo"
CKPT_3B_PRETRAIN="${REPO}/checkpoints/titan_3b_pretrain"
CKPT_3B_UPGRADES="${REPO}/checkpoints/titan_3b"
CKPT_3B_SFT="${REPO}/checkpoints/titan_3b_instruct"
CKPT_3B_DPO="${REPO}/checkpoints/titan_3b_dpo"

mkdir -p "${LOG_DIR}" "${CKPT_3B_PRETRAIN}" "${CKPT_3B_UPGRADES}" "${CKPT_3B_SFT}" "${CKPT_3B_DPO}"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/master.log"; }

cd "${REPO}"
python3 "${REPO}/scripts/validate_training_pipeline.py" --stage 3b

NUM_GPUS=$(python3 -c 'import torch; print(max(1, torch.cuda.device_count()))')
if [ "${NUM_GPUS}" -gt 1 ]; then
  TRAIN_CMD=(torchrun --nproc_per_node="${NUM_GPUS}")
else
  TRAIN_CMD=(python3)
fi
log "Detected ${NUM_GPUS} GPU(s)"

# Find the strongest available 1B checkpoint.
SRC_CKPT=$(find \
  "${CKPT_1B_DPO}" \
  "${REPO}/checkpoints/titan_1b_instruct" \
  "${REPO}/checkpoints/titan_1b_pretrain" \
  -maxdepth 2 -type f \( -name 'final.pt' -o -name 'best.pt' -o -name 'step_*.pt' \) \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
if [ -z "${SRC_CKPT}" ]; then
  log "FATAL: no 1B checkpoint found; complete the 1B pipeline first"
  exit 1
fi
log "Using 1B source checkpoint: ${SRC_CKPT}"

INIT_3B="${CKPT_3B_PRETRAIN}/init.pt"
if [ ! -f "${INIT_3B}" ]; then
  python3 "${REPO}/scripts/upscale_to_3b.py" \
    --src_checkpoint "${SRC_CKPT}" \
    --src_config "${CONFIG_1B}" \
    --dst_config "${CONFIG_3B}" \
    --dst_checkpoint "${INIT_3B}" \
    2>&1 | tee "${LOG_DIR}/upscale.log"
fi

# Continued pretraining.
RESUME_CKPT=$(find "${CKPT_3B_PRETRAIN}" -maxdepth 1 -type f -name 'step_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
START_CKPT="${RESUME_CKPT:-${INIT_3B}}"
PRETRAIN_ARGS=(
  "${REPO}/scripts/pretrain_titan_v3.py"
  --config "${CONFIG_3B}"
  --init-from "${START_CKPT}"
  --out-dir "${CKPT_3B_PRETRAIN}"
  --use-8bit-adam
  --compile
)
if [ -n "${RESUME_CKPT}" ]; then PRETRAIN_ARGS+=(--resume "${RESUME_CKPT}"); fi
"${TRAIN_CMD[@]}" "${PRETRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/phase1_pretrain.log"

BASE_CKPT=$(find "${CKPT_3B_PRETRAIN}" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'final.pt' -o -name 'step_*.pt' \) -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

# Sequential domain upgrades using the 3B architecture.
python3 "${REPO}/scripts/train_1b_pipeline.py" \
  --base-checkpoint "${BASE_CKPT}" \
  --output-dir "${CKPT_3B_UPGRADES}" \
  --model-config "${CONFIG_3B}" \
  2>&1 | tee "${LOG_DIR}/phase2_upgrades.log"

PHASE2_CKPT=$(python3 - <<'PY'
import json, os
p=os.environ.get('CKPT_3B_UPGRADES','/workspace/titanai/checkpoints/titan_3b')+'/status.json'
with open(p) as f: print(json.load(f)['final_checkpoint'])
PY
)

python3 "${REPO}/scripts/run_sft_v2.py" \
  --config "${CONFIG_3B_SFT}" \
  --checkpoint "${PHASE2_CKPT}" \
  --out-dir "${CKPT_3B_SFT}" \
  2>&1 | tee "${LOG_DIR}/phase3_sft.log"

SFT_CKPT=$(find "${CKPT_3B_SFT}" -maxdepth 1 -type f \( -name 'best.pt' -o -name 'final.pt' -o -name 'step_*.pt' \) -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
python3 "${REPO}/scripts/run_dpo.py" \
  --config "${CONFIG_3B_DPO}" \
  --checkpoint "${SFT_CKPT}" \
  --out-dir "${CKPT_3B_DPO}" \
  2>&1 | tee "${LOG_DIR}/phase4_dpo.log"

log "3B pipeline complete. Final checkpoints: ${CKPT_3B_DPO}"
