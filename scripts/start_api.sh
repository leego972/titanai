#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TitanAI API Server — Launch Script
# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT SECURITY FIX: Auth is now REQUIRED before the server starts.
# The API will refuse to launch without TITAN_REQUIRE_AUTH=true and a
# strong TITAN_API_KEY (minimum 32 characters).
#
# Usage:
#   export TITAN_REQUIRE_AUTH=true
#   export TITAN_API_KEY=$(openssl rand -hex 32)
#   ./scripts/start_api.sh
#
# Environment variables:
#   TITAN_API_HOST          (default: 0.0.0.0)
#   TITAN_API_PORT          (default: 8000)
#   TITAN_CONFIG_PATH       (default: configs/titan_probe_v015.yaml)
#   TITAN_CHECKPOINT_PATH   (default: checkpoints/probe_v015/final.pt)
#   TITAN_DEVICE            (default: auto — uses CUDA if available)
#   TITAN_REQUIRE_AUTH      (REQUIRED: must be "true")
#   TITAN_API_KEY           (REQUIRED: minimum 32 characters)
#   TITAN_API_LOG_LEVEL     (default: info)
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITANAI_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$TITANAI_ROOT"

# ── PRE-FLIGHT SECURITY CHECKS ───────────────────────────────────────────────
# The API must NOT start in unauthenticated mode on Vast.AI.
# These checks are mandatory and cannot be bypassed.

if [ "${TITAN_REQUIRE_AUTH:-false}" != "true" ]; then
    echo ""
    echo "[SECURITY ERROR] TITAN_REQUIRE_AUTH is not set to 'true'."
    echo "  The TitanAI API must not start in unauthenticated mode on Vast.AI."
    echo "  The API port (8000) is publicly exposed — open access is a security risk."
    echo ""
    echo "  Fix:"
    echo "    export TITAN_REQUIRE_AUTH=true"
    echo "    export TITAN_API_KEY=\$(openssl rand -hex 32)"
    echo "    ./scripts/start_api.sh"
    echo ""
    exit 1
fi

if [ -z "${TITAN_API_KEY:-}" ]; then
    echo ""
    echo "[SECURITY ERROR] TITAN_API_KEY is not set."
    echo "  A non-empty API key is required when TITAN_REQUIRE_AUTH=true."
    echo ""
    echo "  Fix:"
    echo "    export TITAN_API_KEY=\$(openssl rand -hex 32)"
    echo ""
    exit 1
fi

if [ "${#TITAN_API_KEY}" -lt 32 ]; then
    echo ""
    echo "[SECURITY ERROR] TITAN_API_KEY is too short (${#TITAN_API_KEY} chars, minimum 32)."
    echo "  Use a strong random key:"
    echo "    export TITAN_API_KEY=\$(openssl rand -hex 32)"
    echo ""
    exit 1
fi

echo "============================================================"
echo "  TitanAI API Server"
echo "  Root:       $TITANAI_ROOT"
echo "  Port:       ${TITAN_API_PORT:-8000}"
echo "  Device:     ${TITAN_DEVICE:-auto}"
echo "  Checkpoint: ${TITAN_CHECKPOINT_PATH:-checkpoints/probe_v015/final.pt}"
echo "  Auth:       REQUIRED (key length: ${#TITAN_API_KEY} chars)"
echo "============================================================"

# Install API dependencies if not present
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "[Setup] Installing API dependencies..."
    pip install -r requirements-api.txt -q
fi

# Launch the API server
exec python3 -m api.main
