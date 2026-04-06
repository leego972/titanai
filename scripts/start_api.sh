#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TitanAI API Server — Launch Script
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/start_api.sh                     # defaults
#   TITAN_API_PORT=9000 ./scripts/start_api.sh # custom port
#   TITAN_REQUIRE_AUTH=true TITAN_API_KEY=mykey ./scripts/start_api.sh
#
# Environment variables:
#   TITAN_API_HOST          (default: 0.0.0.0)
#   TITAN_API_PORT          (default: 8000)
#   TITAN_CONFIG_PATH       (default: configs/titan_probe_v015.yaml)
#   TITAN_CHECKPOINT_PATH   (default: checkpoints/probe_v015/final.pt)
#   TITAN_DEVICE            (default: auto — uses CUDA if available)
#   TITAN_REQUIRE_AUTH      (default: false)
#   TITAN_API_KEY           (default: empty — no auth)
#   TITAN_API_LOG_LEVEL     (default: info)
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITANAI_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$TITANAI_ROOT"

echo "============================================================"
echo "  TitanAI API Server"
echo "  Root: $TITANAI_ROOT"
echo "  Port: ${TITAN_API_PORT:-8000}"
echo "  Device: ${TITAN_DEVICE:-auto}"
echo "  Checkpoint: ${TITAN_CHECKPOINT_PATH:-checkpoints/probe_v015/final.pt}"
echo "============================================================"

# Install API dependencies if not present
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "[Setup] Installing API dependencies..."
    pip install -r requirements-api.txt -q
fi

# Launch the API server
exec python3 -m api.main
