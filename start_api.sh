#!/bin/bash
cd /workspace/titanai
export TITAN_REQUIRE_AUTH=true
export TITAN_API_KEY=$(cat /root/.titan_api_key)
export TITAN_CHECKPOINT_PATH=/workspace/titanai/checkpoints/tool_v01/final.pt
export TITAN_CONFIG_PATH=/workspace/titanai/configs/titan_tool_v01.yaml
export TITAN_API_HOST=0.0.0.0
export PYTHONDONTWRITEBYTECODE=1
export TITAN_API_PORT=8000
exec python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
