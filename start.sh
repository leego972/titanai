#!/usr/bin/env bash
  # ============================================================
  # TitanAI Laptop Startup Script
  # Run from the titanai/ directory: bash start.sh
  # Requires: Python 3.10+, cloudflared installed
  #   Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
  # ============================================================
  set -e
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR"

  # ── Config ──────────────────────────────────────────────────
  export TITAN_CONFIG_PATH="${TITAN_CONFIG_PATH:-$SCRIPT_DIR/configs/titan_upgrade_an.yaml}"
  export TITAN_CHECKPOINT_PATH="${TITAN_CHECKPOINT_PATH:-$SCRIPT_DIR/checkpoints/upgrade_an/final.pt}"
  export TITAN_DEVICE="${TITAN_DEVICE:-cpu}"
  export TITAN_API_PORT="${TITAN_API_PORT:-8765}"
  export TITAN_REQUIRE_AUTH="false"
  export TITAN_API_WORKERS="1"

  echo ""
  echo "=========================================="
  echo "  TitanAI Laptop Server"
  echo "=========================================="
  echo "  Config:      $TITAN_CONFIG_PATH"
  echo "  Checkpoint:  $TITAN_CHECKPOINT_PATH"
  echo "  Device:      $TITAN_DEVICE"
  echo "  Port:        $TITAN_API_PORT"
  echo "=========================================="
  echo ""

  # ── Verify checkpoint exists ────────────────────────────────
  if [ ! -f "$TITAN_CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $TITAN_CHECKPOINT_PATH"
    echo "   Make sure Dropbox has finished syncing your checkpoints folder."
    exit 1
  fi
  echo "OK: Checkpoint found ($(du -sh "$TITAN_CHECKPOINT_PATH" | cut -f1))"

  # ── Verify config exists ────────────────────────────────────
  if [ ! -f "$TITAN_CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $TITAN_CONFIG_PATH"
    echo "   Try: TITAN_CONFIG_PATH=configs/titan_probe_v015.yaml bash start.sh"
    exit 1
  fi
  echo "OK: Config found"

  # ── Install dependencies if needed ─────────────────────────
  if ! python3 -c "import fastapi" 2>/dev/null; then
    echo ""
    echo "Installing API dependencies..."
    pip3 install -r requirements-api.txt -q
  fi

  # ── Start API server (background) ──────────────────────────
  echo ""
  echo "Starting TitanAI API on port $TITAN_API_PORT..."
  python3 -m uvicorn api.main:app --host 127.0.0.1 --port "$TITAN_API_PORT" --workers 1 &
  API_PID=$!
  echo "   API PID: $API_PID"

  # ── Wait for API to be ready ────────────────────────────────
  echo "   Waiting for model to load (may take 30-90 seconds on CPU)..."
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$TITAN_API_PORT/health" >/dev/null 2>&1; then
      echo "OK: API is ready!"
      break
    fi
    sleep 2
    if [ $i -eq 120 ]; then
      echo "ERROR: API failed to start within 4 minutes. Check logs above."
      kill $API_PID 2>/dev/null
      exit 1
    fi
  done

  # ── Start Cloudflare Tunnel ─────────────────────────────────
  echo ""
  echo "Starting Cloudflare Tunnel..."
  cloudflared tunnel --url "http://127.0.0.1:$TITAN_API_PORT" 2>&1 | tee /tmp/titan_tunnel.log &
  TUNNEL_PID=$!

  # Extract the public URL from cloudflared output
  echo "   Waiting for tunnel URL..."
  PUBLIC_URL=""
  for i in $(seq 1 30); do
    PUBLIC_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/titan_tunnel.log 2>/dev/null | head -1)
    if [ -n "$PUBLIC_URL" ]; then break; fi
    sleep 2
  done

  echo ""
  echo "=========================================="
  if [ -n "$PUBLIC_URL" ]; then
    echo "  TitanAI is LIVE!"
    echo ""
    echo "  Public URL:  $PUBLIC_URL"
    echo ""
    echo "  Set this in Virelle (Railway env vars):"
    echo "  TITAN_API_URL=$PUBLIC_URL"
    echo ""
    echo "  Test it:"
    echo "  curl $PUBLIC_URL/health"
  else
    echo "  WARNING: Tunnel URL not detected yet"
    echo "  Check: cat /tmp/titan_tunnel.log"
    echo "  Local URL: http://127.0.0.1:$TITAN_API_PORT"
  fi
  echo "=========================================="
  echo ""
  echo "  Press Ctrl+C to stop everything."
  echo ""

  # ── Keep running until Ctrl+C ───────────────────────────────
  cleanup() {
    echo ""
    echo "Shutting down TitanAI..."
    kill $API_PID $TUNNEL_PID 2>/dev/null
    exit 0
  }
  trap cleanup SIGINT SIGTERM
  wait
  