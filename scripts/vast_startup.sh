#!/bin/bash
# TitanAI Vast.ai Startup Script v0.5
# Dropbox is MANUAL via Vast.ai file manager — no rclone needed.
# Waits up to 2 hours for checkpoint transfer, then trains automatically.

set -euo pipefail
TRAINING_EXIT_CODE=0
REPO="/workspace/titanai"

shutdown_instance() {
    local ec=${1:-0}
    echo ""; echo "=============================="; echo "  TitanAI ended (exit: ${ec})"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"; echo "=============================="
    if [ -n "${TITAN_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "${TITAN_WEBHOOK_URL}" -H "Content-Type: application/json" \
            -d "{\"event\":\"training_ended\",\"exit_code\":${ec}}" --max-time 10 || true
    fi
    if [ "${TITAN_SKIP_SHUTDOWN:-false}" = "true" ]; then
        echo "[Shutdown] TITAN_SKIP_SHUTDOWN=true - staying alive"; exit ${ec}
    fi
    local ckpt_dir="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
    echo ""; echo "============================================================"
    echo "  TRAINING DONE - push checkpoints back to Dropbox NOW!"
    echo "  Source: ${REPO}/${ckpt_dir}/"
    echo "  Use Vast.ai file manager to copy to your Dropbox."
    echo "  Instance shuts down in 2 minutes."
    echo "============================================================"
    sleep 120
    shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
}
trap 'TRAINING_EXIT_CODE=$?; shutdown_instance $TRAINING_EXIT_CODE' EXIT

echo "=============================="; echo "  TitanAI Vast.ai v0.5"
echo "  Config: ${TITAN_CONFIG:-configs/titan_1b.yaml}"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"; echo "=============================="
mkdir -p "${REPO}/logs"

echo "[1/4] Auth check..."
[ "${TITAN_REQUIRE_AUTH:-false}" = "true" ] || { echo "[ERROR] Set TITAN_REQUIRE_AUTH=true"; exit 1; }
[ "${#TITAN_API_KEY}" -ge 32 ]              || { echo "[ERROR] TITAN_API_KEY too short";       exit 1; }
echo "  OK"

echo "[2/4] System packages..."
apt-get update -qq 2>/dev/null
apt-get install -y -qq wget curl python3-pip unzip 2>/dev/null || true

echo "[3/4] Python packages..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q 2>&1 | tail -1
pip install bitsandbytes tokenizers datasets numpy tqdm pyyaml requests huggingface_hub wandb -q 2>&1 | tail -1
if [ "${TITAN_SKIP_FLASH:-false}" != "true" ]; then
    echo "  FlashAttention-2: building (~10 min first time)..."
    pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || \
        echo "  [WARN] FA2 build failed - SDPA fallback active"
fi

echo "[4/4] Repository..."
REPO_URL="https://github.com/leego972/titanai.git"
[ -n "${TITAN_GITHUB_TOKEN:-}" ] && REPO_URL="https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git"
if [ ! -d "${REPO}/.git" ]; then
    git clone "${REPO_URL}" "${REPO}"
else
    cd "${REPO}"
    git remote set-url origin "${REPO_URL}" 2>/dev/null || true
    git fetch origin -q && git reset --hard origin/main 2>/dev/null || git reset --hard origin/master 2>/dev/null || true
fi
cd "${REPO}"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null)"

CONFIG_FILE="${TITAN_CONFIG:-configs/titan_1b.yaml}"
CKPT_DIR="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
mkdir -p "${REPO}/${CKPT_DIR}"

echo ""; echo "============================================================"
echo "  ACTION REQUIRED - Transfer checkpoint from Dropbox"
echo "  using Vast.ai file manager, to this exact path:"
echo ""; echo "    /workspace/titanai/${CKPT_DIR}/"
echo ""; echo "  Training starts automatically when file detected."
echo "  Waiting up to 2 hours..."; echo "============================================================"; echo ""

LATEST_CKPT=""
for i in $(seq 1 120); do
    LATEST_CKPT=$(ls -t "${REPO}/${CKPT_DIR}"/step_*.pt 2>/dev/null | head -1 || true)
    [ -z "${LATEST_CKPT}" ] && LATEST_CKPT=$(ls "${REPO}/${CKPT_DIR}/best.pt" 2>/dev/null || true)
    [ -z "${LATEST_CKPT}" ] && LATEST_CKPT=$(ls "${REPO}/${CKPT_DIR}/init.pt" 2>/dev/null || true)
    [ -z "${LATEST_CKPT}" ] && LATEST_CKPT="${TITAN_INIT_CHECKPOINT:-}"
    if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}" ]; then
        echo "  Checkpoint found: ${LATEST_CKPT}"; break
    fi
    echo "  [${i}/120] Waiting... $(( 120 - i )) min remaining"
    sleep 60
done

if [ -z "${LATEST_CKPT}" ] || [ ! -f "${LATEST_CKPT}" ]; then
    echo "[ERROR] No checkpoint after 2 hours. Shutting down."; exit 1
fi

RESUME_FLAG=""
echo "${LATEST_CKPT}" | grep -q "step_" && RESUME_FLAG="--resume ${LATEST_CKPT}"
LOG_DIR="${REPO}/logs/$(basename ${CONFIG_FILE} .yaml)"
mkdir -p "${LOG_DIR}"

echo ""; echo "=============================="; echo "  LAUNCHING TRAINING"
echo "  Checkpoint : ${LATEST_CKPT}"; echo "  Config     : ${CONFIG_FILE}"
echo "  Out dir    : ${CKPT_DIR}"; echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================="

python3 scripts/pretrain_titan_v3.py \
    --config "${REPO}/${CONFIG_FILE}" \
    --init-from "${LATEST_CKPT}" \
    ${RESUME_FLAG} \
    --out-dir "${REPO}/${CKPT_DIR}" \
    --use-8bit-adam \
    --compile \
    2>&1 | tee "${LOG_DIR}/training.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
[ $TRAINING_EXIT_CODE -eq 0 ] && echo "Training COMPLETE" || echo "Training FAILED (exit ${TRAINING_EXIT_CODE})"
exit $TRAINING_EXIT_CODE
