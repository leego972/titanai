#!/bin/bash
  # TitanAI — Vast.ai Startup Script v0.4 (FIXED)
  # ================================================
  # Fixes from v0.3:
  #   - Removed shard pipeline steps (rebalance_corpus + generate_shards)
  #     These were for the offline shard system; pretrain_titan_v3.py streams
  #     directly from HuggingFace and needs neither. They were wasting ~45 min
  #     of GPU billing time every run.
  #   - Fixed training invocation: now passes --config + --init-from correctly.
  #     v0.3 passed --config to a script that had no --config arg (instant crash).
  #   - Removed prerun_gates.py (requires interactive --budget-approved flag).
  #   - All paths are relative to /workspace/titanai (no hardcoding).
  #   - Dropbox wheel cache: FlashAttention-2 is cached after the first build
  #     so future runs skip the 10-minute compile.
  #
  # Required env vars (set in Vast.ai instance env before starting):
  #   TITAN_REQUIRE_AUTH=true
  #   TITAN_API_KEY=<any string >= 32 chars>
  #   DROPBOX_RCLONE_TOKEN=<base64-encoded rclone.conf>
  #
  # Optional env vars:
  #   TITAN_CONFIG=configs/titan_1b.yaml        (default shown)
  #   TITAN_CHECKPOINT_DIR=checkpoints/titan_1b_pretrain
  #   TITAN_INIT_CHECKPOINT=<path/to/init.pt>   (needed for first run)
  #   TITAN_SKIP_DROPBOX=true
  #   TITAN_SKIP_FLASH=true
  #   TITAN_SKIP_SHUTDOWN=true                  (debug: keep instance alive)
  #   TITAN_WEBHOOK_URL=<url>                   (POST JSON on completion)
  #   TITAN_GITHUB_TOKEN=<pat>
  # ================================================

  set -euo pipefail
  TRAINING_EXIT_CODE=0
  REPO="/workspace/titanai"

  # ── Shutdown hook ─────────────────────────────────────────────────────────────
  shutdown_instance() {
      local ec=${1:-0}
      echo ""
      echo "=============================="
      echo "  TitanAI ended (exit: ${ec})"
      echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
      echo "=============================="

      if [ "${TITAN_SKIP_DROPBOX:-false}" != "true" ] && command -v rclone &>/dev/null; then
          local ckpt_dir="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
          echo "[Dropbox] Pushing checkpoints..."
          rclone sync "${REPO}/${ckpt_dir}/" "dropbox:TitanAI/${ckpt_dir}/" \
              --transfers 4 --progress \
              --log-file "${REPO}/logs/rclone_push.log" 2>&1 || true
          echo "[Dropbox] Push complete."
      fi

      if [ -n "${TITAN_WEBHOOK_URL:-}" ]; then
          curl -s -X POST "${TITAN_WEBHOOK_URL}" -H "Content-Type: application/json" \
              -d "{\"event\":\"training_ended\",\"exit_code\":${ec}}" --max-time 10 || true
      fi

      if [ "${TITAN_SKIP_SHUTDOWN:-false}" = "true" ]; then
          echo "[Shutdown] TITAN_SKIP_SHUTDOWN=true — staying alive"
          exit ${ec}
      fi
      echo "[Shutdown] Powering off in 60 s..."
      sleep 60
      shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
  }
  trap 'TRAINING_EXIT_CODE=$?; shutdown_instance $TRAINING_EXIT_CODE' EXIT

  echo "=============================="
  echo "  TitanAI Vast.ai v0.4"
  echo "  Config: ${TITAN_CONFIG:-configs/titan_1b.yaml}"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "=============================="
  mkdir -p "${REPO}/logs"

  # ── 1. Auth ───────────────────────────────────────────────────────────────────
  echo "[1/6] Auth check..."
  [ "${TITAN_REQUIRE_AUTH:-false}" = "true" ] || { echo "[ERROR] Set TITAN_REQUIRE_AUTH=true"; exit 1; }
  [ "${#TITAN_API_KEY}" -ge 32 ]              || { echo "[ERROR] TITAN_API_KEY too short";       exit 1; }
  echo "  OK"

  # ── 2. System + rclone ───────────────────────────────────────────────────────
  echo "[2/6] System packages..."
  apt-get update -qq 2>/dev/null
  apt-get install -y -qq git wget curl python3-pip unzip 2>/dev/null || true
  command -v rclone &>/dev/null || curl -fsSL https://rclone.org/install.sh | bash 2>/dev/null || true

  # ── 3. Python packages + FlashAttention-2 ────────────────────────────────────
  echo "[3/6] Python packages..."
  pip install torch --index-url https://download.pytorch.org/whl/cu121 -q 2>&1 | tail -1
  pip install bitsandbytes tokenizers datasets numpy tqdm pyyaml requests \
              huggingface_hub wandb -q 2>&1 | tail -1

  if [ "${TITAN_SKIP_FLASH:-false}" != "true" ]; then
      # Try Dropbox wheel cache first (avoids 10-min build on every run)
      _FLASH_CACHED=false
      if rclone listremotes 2>/dev/null | grep -q "dropbox:"; then
          if rclone ls "dropbox:TitanAI/wheels/" 2>/dev/null | grep -q "flash_attn"; then
              echo "  FlashAttention-2: installing from Dropbox wheel cache..."
              rclone copy "dropbox:TitanAI/wheels/" /tmp/fa_wheels/ -q 2>/dev/null
              pip install /tmp/fa_wheels/flash_attn*.whl -q 2>&1 | tail -1 && _FLASH_CACHED=true || true
          fi
      fi
      if [ "${_FLASH_CACHED}" = "false" ]; then
          echo "  FlashAttention-2: building from source (~10 min first time)..."
          pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || \
              echo "  FlashAttention-2 build failed — SDPA fallback will be used"
          # Cache wheel to Dropbox so next run is instant
          if rclone listremotes 2>/dev/null | grep -q "dropbox:"; then
              FA_WHL=$(pip show -f flash_attn 2>/dev/null | grep ".whl" | head -1 || true)
              [ -n "${FA_WHL}" ] && rclone copy "${FA_WHL}" "dropbox:TitanAI/wheels/" -q 2>/dev/null || true
          fi
      fi
  fi

  # ── 4. Clone / update repo ────────────────────────────────────────────────────
  echo "[4/6] Repository..."
  REPO_URL="https://github.com/leego972/titanai.git"
  [ -n "${TITAN_GITHUB_TOKEN:-}" ] && REPO_URL="https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git"

  if [ ! -d "${REPO}/.git" ]; then
      git clone "${REPO_URL}" "${REPO}"
  else
      cd "${REPO}"
      git remote set-url origin "${REPO_URL}" 2>/dev/null || true
      git fetch origin -q && git reset --hard origin/main 2>/dev/null || \
                             git reset --hard origin/master 2>/dev/null || true
  fi
  cd "${REPO}"
  echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null)"

  # ── 5. Dropbox: pull checkpoint ───────────────────────────────────────────────
  echo "[5/6] Dropbox sync..."
  CONFIG_FILE="${TITAN_CONFIG:-configs/titan_1b.yaml}"
  CKPT_DIR="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"

  if [ "${TITAN_SKIP_DROPBOX:-false}" != "true" ]; then
      if [ -n "${DROPBOX_RCLONE_TOKEN:-}" ]; then
          mkdir -p ~/.config/rclone
          echo "${DROPBOX_RCLONE_TOKEN}" | base64 -d > ~/.config/rclone/rclone.conf
          echo "  rclone config loaded from env"
      fi
      if rclone listremotes 2>/dev/null | grep -q "dropbox:"; then
          echo "  Pulling checkpoint dir: ${CKPT_DIR}"
          mkdir -p "${REPO}/${CKPT_DIR}"
          rclone sync "dropbox:TitanAI/${CKPT_DIR}/" "${REPO}/${CKPT_DIR}/" \
              --transfers 4 --progress \
              --log-file "${REPO}/logs/rclone_pull.log" 2>&1 || true
          echo "  Pull done"
      else
          echo "  [WARN] No Dropbox remote. Set DROPBOX_RCLONE_TOKEN or TITAN_SKIP_DROPBOX=true"
      fi
  else
      echo "  Skipped (TITAN_SKIP_DROPBOX=true)"
  fi

  # ── 6. Resolve checkpoint + launch training ───────────────────────────────────
  echo "[6/6] Resolving checkpoint and launching..."
  cd "${REPO}"

  # Priority: step_N.pt (resume) > best.pt > init.pt > TITAN_INIT_CHECKPOINT env var
  LATEST_CKPT=$(ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null | head -1 || true)
  [ -z "${LATEST_CKPT}" ] && LATEST_CKPT=$(ls "${CKPT_DIR}/best.pt" 2>/dev/null || true)
  [ -z "${LATEST_CKPT}" ] && LATEST_CKPT=$(ls "${CKPT_DIR}/init.pt" 2>/dev/null || true)
  [ -z "${LATEST_CKPT}" ] && LATEST_CKPT="${TITAN_INIT_CHECKPOINT:-}"

  if [ -z "${LATEST_CKPT}" ] || [ ! -f "${LATEST_CKPT}" ]; then
      echo "[ERROR] No checkpoint found."
      echo "  Options:"
      echo "    1. Run upscale_to_1b.py and upload init.pt to Dropbox:TitanAI/${CKPT_DIR}/"
      echo "    2. Set TITAN_INIT_CHECKPOINT=/path/to/your/init.pt"
      exit 1
  fi

  echo "  Checkpoint: ${LATEST_CKPT}"

  # Build resume flag: use --resume if it's a step checkpoint (mid-run resume)
  RESUME_FLAG=""
  echo "${LATEST_CKPT}" | grep -q "step_" && RESUME_FLAG="--resume ${LATEST_CKPT}"

  LOG_DIR="${REPO}/logs/$(basename ${CONFIG_FILE} .yaml)"
  mkdir -p "${LOG_DIR}"

  echo "=============================="
  echo "  Config     : ${CONFIG_FILE}"
  echo "  Checkpoint : ${LATEST_CKPT}"
  echo "  Out dir    : ${CKPT_DIR}"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "=============================="

  # pretrain_titan_v3.py now accepts --config to read YAML hyperparameters,
  # --use-8bit-adam for memory-efficient optimizer, --compile for throughput.
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
  