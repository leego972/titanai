#!/bin/bash
# TitanAI Probe Training Startup Script for Vast.AI
# ===================================================
# PRE-FLIGHT FIXES APPLIED:
#   1. ENV-VAR CHECKS: Required variables are validated before any work starts.
#   2. GIT AUTH: Supports TITAN_GITHUB_TOKEN for private repo access.
#   3. AUTO-SHUTDOWN: Instance shuts down automatically after training completes
#      or fails, preventing idle cost burn.
#   4. REBALANCE GATE: Runs corpus rebalancer before sharding to enforce ratios.
#   5. PRERUN GATES: Validates environment before starting the training loop.
#
# Required environment variables (set in Vast.AI instance env):
#   TITAN_REQUIRE_AUTH=true
#   TITAN_API_KEY=<strong-key-min-32-chars>
#
# Optional environment variables:
#   TITAN_GITHUB_TOKEN=<pat>    # Only needed if repo is private
#   TITAN_SKIP_SHUTDOWN=true    # Set to prevent auto-shutdown (for debugging)
#   TITAN_WEBHOOK_URL=<url>     # Optional: POST training result to this URL
# ===================================================

set -e

echo "======================================"
echo "  TitanAI Probe Training Setup"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================"

# ── STEP 0: Auto-shutdown trap (PRE-FLIGHT FIX: idle cost prevention) ────────
# This trap fires on EXIT (success, failure, or signal) and shuts down the
# Vast.AI instance unless TITAN_SKIP_SHUTDOWN=true is set.

TRAINING_EXIT_CODE=0

shutdown_instance() {
    local exit_code=$1
    echo ""
    echo "======================================"
    echo "  TitanAI: Training process ended"
    echo "  Exit code: $exit_code"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================"

    # Optional webhook notification
    if [ -n "${TITAN_WEBHOOK_URL:-}" ]; then
        echo "[Notify] Sending training completion webhook..."
        curl -s -X POST "${TITAN_WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"event\": \"training_ended\", \"exit_code\": ${exit_code}, \"timestamp\": \"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"}" \
            --max-time 10 || true
    fi

    if [ "${TITAN_SKIP_SHUTDOWN:-false}" = "true" ]; then
        echo "[Shutdown] TITAN_SKIP_SHUTDOWN=true -- skipping auto-shutdown"
        echo "[Shutdown] WARNING: Instance will continue to incur costs!"
        return
    fi

    echo "[Shutdown] Initiating instance shutdown in 60 seconds..."
    echo "[Shutdown] Set TITAN_SKIP_SHUTDOWN=true to prevent this."
    sleep 60
    echo "[Shutdown] Shutting down now."
    shutdown -h now 2>/dev/null || poweroff 2>/dev/null || true
}

trap 'TRAINING_EXIT_CODE=$?; shutdown_instance $TRAINING_EXIT_CODE' EXIT

# ── STEP 1: Security env-var checks ──────────────────────────────────────────
echo "[1/8] Checking required environment variables..."

if [ "${TITAN_REQUIRE_AUTH:-false}" != "true" ]; then
    echo "[ERROR] TITAN_REQUIRE_AUTH is not set to 'true'."
    echo "  Set TITAN_REQUIRE_AUTH=true in the Vast.AI instance environment."
    exit 1
fi

if [ -z "${TITAN_API_KEY:-}" ]; then
    echo "[ERROR] TITAN_API_KEY is not set."
    echo "  Generate one: openssl rand -hex 32"
    exit 1
fi

if [ "${#TITAN_API_KEY}" -lt 32 ]; then
    echo "[ERROR] TITAN_API_KEY is too short (${#TITAN_API_KEY} chars, minimum 32)."
    exit 1
fi

echo "  Auth: TITAN_REQUIRE_AUTH=true, key length=${#TITAN_API_KEY} chars [OK]"

# ── STEP 2: System dependencies ──────────────────────────────────────────────
echo "[2/8] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl python3-pip 2>/dev/null || true

# ── STEP 3: Python packages ───────────────────────────────────────────────────
echo "[3/8] Installing Python packages..."
pip install torch --index-url https://download.pytorch.org/whl/cu121 -q  # torchvision not needed (pure LM)
pip install transformers datasets tokenizers numpy tqdm pyyaml requests huggingface_hub -q

# ── STEP 4: Clone/update TitanAI repository (PRE-FLIGHT FIX: git auth) ───────
echo "[4/8] Cloning/updating TitanAI repository..."
REPO_URL="https://github.com/leego972/titanai.git"

if [ -n "${TITAN_GITHUB_TOKEN:-}" ]; then
    REPO_URL="https://${TITAN_GITHUB_TOKEN}@github.com/leego972/titanai.git"
    echo "  Using authenticated git clone (TITAN_GITHUB_TOKEN set)"
else
    echo "  Using public git clone (TITAN_GITHUB_TOKEN not set)"
fi

if [ ! -d "/workspace/titanai" ]; then
    git clone "${REPO_URL}" /workspace/titanai
else
    echo "  Repo already exists -- pulling latest changes..."
    cd /workspace/titanai
    git remote set-url origin "${REPO_URL}"
    git pull --ff-only origin main || git pull --ff-only origin master || true
fi

cd /workspace/titanai

# ── STEP 5: Install TitanAI requirements ─────────────────────────────────────
echo "[5/8] Installing TitanAI requirements..."
pip install -r requirements.txt -q 2>/dev/null || true

# ── STEP 6: Corpus rebalancing (PRE-FLIGHT FIX: enforce bucket ratios) ────────
echo "[6/8] Running corpus rebalancer..."
mkdir -p logs
python3 scripts/rebalance_corpus.py 2>&1 | tee logs/rebalance.log
REBALANCE_EXIT=${PIPESTATUS[0]}
if [ $REBALANCE_EXIT -ne 0 ]; then
    echo "[ERROR] Corpus rebalancer failed or ratio gate did not pass."
    echo "  Check logs/rebalance.log for details."
    exit 1
fi
echo "  Corpus rebalancing complete [OK]"

# ── STEP 7: Generate data shards ─────────────────────────────────────────────
echo "[7/8] Generating data shards..."
python3 scripts/generate_shards.py --config configs/titan_probe_v015.yaml 2>&1 | tee logs/shard_gen.log

# ── STEP 8: Pre-run gates ─────────────────────────────────────────────────────
echo "[8/8] Running pre-run environment gates..."
python3 scripts/prerun_gates.py --config configs/titan_probe_v015.yaml 2>&1 | tee logs/prerun_gates.log
GATES_EXIT=${PIPESTATUS[0]}
if [ $GATES_EXIT -ne 0 ]; then
    echo "[ERROR] Pre-run gates failed. Check logs/prerun_gates.log."
    exit 1
fi
echo "  Pre-run gates passed [OK]"

# ── START TRAINING ─────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  Starting Probe Training Run"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "======================================"

mkdir -p logs/probe_v015
python3 scripts/run_probe.py --config configs/titan_probe_v015.yaml 2>&1 | tee logs/probe_training.log
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "  Training complete! [EXIT 0]"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================"
else
    echo "======================================"
    echo "  Training FAILED [EXIT $TRAINING_EXIT_CODE]"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "======================================"
fi

# The EXIT trap handles shutdown automatically
exit $TRAINING_EXIT_CODE
