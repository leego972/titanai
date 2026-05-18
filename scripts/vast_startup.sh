#!/bin/bash
  # TitanAI — Vast.ai Universal Startup Script v0.3
  # =================================================
  # Handles: dependency install, Dropbox sync, corpus rebalance,
  #          shard generation, pre-run gates, training launch,
  #          and checkpoint push back to Dropbox on shutdown.
  #
  # Required env vars (set in Vast.ai instance environment):
  #   TITAN_REQUIRE_AUTH=true
  #   TITAN_API_KEY=<min 32 chars>
  #   DROPBOX_RCLONE_TOKEN=<rclone OAuth token JSON, base64-encoded>
  #
  # Optional env vars:
  #   TITAN_CONFIG=configs/titan_1b.yaml     (default shown)
  #   TITAN_TRAIN_SCRIPT=scripts/pretrain_titan_v3.py
  #   TITAN_CHECKPOINT_DIR=checkpoints/titan_1b_pretrain
  #   TITAN_SKIP_SHUTDOWN=true               (debug: keep instance alive)
  #   TITAN_SKIP_DROPBOX=true                (skip Dropbox sync, use local data)
  #   TITAN_WEBHOOK_URL=<url>                (POST completion event)
  # =================================================

  set -euo pipefail

  TRAINING_EXIT_CODE=0

  # ── Auto-shutdown + Dropbox push on exit ─────────────────────────────────────
  shutdown_instance() {
      local exit_code=${1:-0}
      echo ""
      echo "======================================"
      echo "  TitanAI: Training ended (code: ${exit_code})"
      echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
      echo "======================================"

      # Push checkpoints back to Dropbox before shutdown
      if [ "${TITAN_SKIP_DROPBOX:-false}" != "true" ] && command -v rclone &>/dev/null; then
          echo "[Dropbox] Pushing checkpoints to Dropbox..."
          CKPT_DIR="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
          rclone sync "/workspace/titanai/${CKPT_DIR}/" \
              "dropbox:TitanAI/${CKPT_DIR}/" \
              --transfers 4 --checkers 8 --progress \
              --log-file /workspace/titanai/logs/rclone_push.log 2>&1 || true
          echo "[Dropbox] Checkpoint push complete."
      fi

      if [ -n "${TITAN_WEBHOOK_URL:-}" ]; then
          curl -s -X POST "${TITAN_WEBHOOK_URL}" \
              -H "Content-Type: application/json" \
              -d "{\"event\": \"training_ended\", \"exit_code\": ${exit_code}, \"timestamp\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"}" \
              --max-time 10 || true
      fi

      if [ "${TITAN_SKIP_SHUTDOWN:-false}" = "true" ]; then
          echo "[Shutdown] TITAN_SKIP_SHUTDOWN=true — instance kept alive (costs continue)"
          return
      fi

      echo "[Shutdown] Shutting down in 60 seconds..."
      sleep 60
      shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
  }

  trap 'TRAINING_EXIT_CODE=$?; shutdown_instance $TRAINING_EXIT_CODE' EXIT

  echo "======================================"
  echo "  TitanAI Vast.ai Startup v0.3"
  echo "  Config: ${TITAN_CONFIG:-configs/titan_1b.yaml}"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "======================================"

  # ── 1. Auth checks ───────────────────────────────────────────────────────────
  echo "[1/9] Checking required environment variables..."
  if [ "${TITAN_REQUIRE_AUTH:-false}" != "true" ]; then
      echo "[ERROR] TITAN_REQUIRE_AUTH must be 'true'. Set it in instance env."; exit 1
  fi
  if [ -z "${TITAN_API_KEY:-}" ] || [ "${#TITAN_API_KEY}" -lt 32 ]; then
      echo "[ERROR] TITAN_API_KEY is missing or too short (need >= 32 chars)."; exit 1
  fi
  echo "  Auth OK (key length=${#TITAN_API_KEY})"

  # ── 2. System dependencies ───────────────────────────────────────────────────
  echo "[2/9] Installing system dependencies..."
  apt-get update -qq && apt-get install -y -qq git wget curl python3-pip unzip 2>/dev/null || true

  # Install rclone for Dropbox sync
  if ! command -v rclone &>/dev/null; then
      echo "  Installing rclone..."
      curl -fsSL https://rclone.org/install.sh | bash 2>/dev/null || true
  fi
  echo "  rclone: $(rclone --version 2>/dev/null | head -1 || echo 'not available')"

  # ── 3. Python packages ───────────────────────────────────────────────────────
  echo "[3/9] Installing Python packages..."
  pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
  pip install bitsandbytes tokenizers datasets numpy tqdm pyyaml requests \
              huggingface_hub wandb fastapi uvicorn -q

  # Flash Attention 2 (build takes ~10 mins — skip with TITAN_SKIP_FLASH=true)
  if [ "${TITAN_SKIP_FLASH:-false}" != "true" ]; then
      echo "  Installing FlashAttention-2 (this takes ~10 min)..."
      pip install flash-attn --no-build-isolation -q 2>&1 | tail -3 || \
          echo "  FlashAttention-2 build failed — model will use PyTorch SDPA fallback"
  fi

  # ── 4. Clone / update repo ───────────────────────────────────────────────────
  echo "[4/9] Cloning/updating TitanAI repository..."
  REPO_URL="https://github.com/leego972/titanai.git"
  if [ -n "${TITAN_GITHUB_TOKEN:-}" ]; then
      REPO_URL="https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git"
  fi

  if [ ! -d "/workspace/titanai" ]; then
      git clone "${REPO_URL}" /workspace/titanai
  else
      cd /workspace/titanai
      git remote set-url origin "${REPO_URL}"
      git pull --ff-only origin main 2>/dev/null || git pull --ff-only origin master || true
  fi
  cd /workspace/titanai
  pip install -r requirements.txt -q 2>/dev/null || true

  # ── 5. Dropbox sync ──────────────────────────────────────────────────────────
  echo "[5/9] Dropbox sync..."

  if [ "${TITAN_SKIP_DROPBOX:-false}" = "true" ]; then
      echo "  TITAN_SKIP_DROPBOX=true — skipping Dropbox sync"
  else
      # Configure rclone with Dropbox token from env var
      if [ -n "${DROPBOX_RCLONE_TOKEN:-}" ]; then
          mkdir -p ~/.config/rclone
          # DROPBOX_RCLONE_TOKEN should be the full rclone.conf content, base64-encoded
          echo "${DROPBOX_RCLONE_TOKEN}" | base64 -d > ~/.config/rclone/rclone.conf
          echo "  rclone config loaded from DROPBOX_RCLONE_TOKEN env var"
      elif rclone listremotes 2>/dev/null | grep -q "dropbox:"; then
          echo "  rclone Dropbox remote already configured"
      else
          echo "  [WARN] No Dropbox credentials found."
          echo "  Set DROPBOX_RCLONE_TOKEN env var (base64-encoded rclone.conf) or"
          echo "  run 'rclone config' manually and set TITAN_SKIP_DROPBOX=true for now."
      fi

      # Pull corpus data
      if rclone listremotes 2>/dev/null | grep -q "dropbox:"; then
          echo "  Pulling corpus from Dropbox..."
          mkdir -p /workspace/titanai/data/raw
          rclone sync "dropbox:TitanAI/data/raw/" /workspace/titanai/data/raw/ \
              --transfers 8 --checkers 16 --progress \
              --log-file /workspace/titanai/logs/rclone_pull_data.log 2>&1 || true

          # Pull latest checkpoint (resume support)
          CKPT_DIR="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
          echo "  Pulling checkpoint from Dropbox: ${CKPT_DIR}"
          mkdir -p "/workspace/titanai/${CKPT_DIR}"
          rclone sync "dropbox:TitanAI/${CKPT_DIR}/" "/workspace/titanai/${CKPT_DIR}/" \
              --transfers 4 --progress \
              --log-file /workspace/titanai/logs/rclone_pull_ckpt.log 2>&1 || true

          echo "  Dropbox sync complete."
      fi
  fi

  # ── 6. Set resume_from in config if latest checkpoint exists ─────────────────
  echo "[6/9] Checking for latest checkpoint to resume..."
  CKPT_DIR="${TITAN_CHECKPOINT_DIR:-checkpoints/titan_1b_pretrain}"
  CONFIG_FILE="${TITAN_CONFIG:-configs/titan_1b.yaml}"

  LATEST_CKPT=$(ls -t "/workspace/titanai/${CKPT_DIR}"/step_*.pt 2>/dev/null | head -1 || true)
  if [ -n "${LATEST_CKPT}" ]; then
      echo "  Found checkpoint: ${LATEST_CKPT}"
      # Inject resume_from into config via python (safe YAML edit)
      python3 -c "
  import yaml, sys
  path = '/workspace/titanai/${CONFIG_FILE}'
  with open(path) as f: cfg = yaml.safe_load(f)
  cfg['training']['resume_from'] = '${LATEST_CKPT}'
  with open(path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
  print('  resume_from set to: ${LATEST_CKPT}')
  " || true
  else
      echo "  No checkpoint found — training from init checkpoint or scratch"
  fi

  # ── 7. Corpus rebalancing ────────────────────────────────────────────────────
  echo "[7/9] Running corpus rebalancer..."
  mkdir -p logs
  python3 scripts/rebalance_corpus.py 2>&1 | tee logs/rebalance.log
  REBALANCE_EXIT=${PIPESTATUS[0]}
  if [ $REBALANCE_EXIT -ne 0 ]; then
      echo "[ERROR] Corpus rebalancer failed. Check logs/rebalance.log"; exit 1
  fi

  # ── 8. Shard generation ──────────────────────────────────────────────────────
  echo "[8/9] Generating data shards..."
  python3 scripts/generate_shards.py --config "${CONFIG_FILE}" 2>&1 | tee logs/shard_gen.log

  # ── 9. Pre-run gates ─────────────────────────────────────────────────────────
  echo "[9/9] Pre-run gates..."
  python3 scripts/prerun_gates.py --config "${CONFIG_FILE}" 2>&1 | tee logs/prerun_gates.log
  GATES_EXIT=${PIPESTATUS[0]}
  if [ $GATES_EXIT -ne 0 ]; then
      echo "[ERROR] Pre-run gates failed. Check logs/prerun_gates.log"; exit 1
  fi

  # ── Training ─────────────────────────────────────────────────────────────────
  echo ""
  echo "======================================"
  echo "  Starting Training"
  echo "  Config  : ${CONFIG_FILE}"
  echo "  Script  : ${TITAN_TRAIN_SCRIPT:-scripts/pretrain_titan_v3.py}"
  echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "======================================"

  TRAIN_SCRIPT="${TITAN_TRAIN_SCRIPT:-scripts/pretrain_titan_v3.py}"
  LOG_DIR="logs/$(basename "${CONFIG_FILE}" .yaml)"
  mkdir -p "${LOG_DIR}"

  python3 "${TRAIN_SCRIPT}" --config "${CONFIG_FILE}" 2>&1 | tee "${LOG_DIR}/training.log"
  TRAINING_EXIT_CODE=${PIPESTATUS[0]}

  if [ $TRAINING_EXIT_CODE -eq 0 ]; then
      echo "======================================"
      echo "  Training COMPLETE [EXIT 0]"
      echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
      echo "======================================"
  else
      echo "======================================"
      echo "  Training FAILED [EXIT ${TRAINING_EXIT_CODE}]"
      echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
      echo "======================================"
  fi

  exit $TRAINING_EXIT_CODE
  