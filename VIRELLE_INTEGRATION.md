# Virelle ↔ TitanAI Integration Guide

  ## Quick Start (Laptop)

  ```bash
  # From your titanai/ folder (Dropbox-synced from Vast.ai):
  bash start.sh
  ```

  The script will:
  1. Verify the checkpoint and config exist
  2. Load the model (30–60 seconds on CPU)
  3. Start the API on port 8765
  4. Open a Cloudflare Tunnel and print the public URL

  ## Connect to Virelle

  Once the tunnel is running, copy the public URL (looks like `https://xxxx.trycloudflare.com`)
  and set it as an environment variable in your Railway deployment:

  ```
  TITAN_API_URL=https://xxxx.trycloudflare.com
  ```

  Virelle will automatically detect this and route Director Chat messages to Titan.
  When your laptop is off, Virelle falls back to its existing LLM gracefully.

  ## Checkpoint Used

  | Checkpoint | Size | Notes |
  |-----------|------|-------|
  | `checkpoints/upgrade_an/final.pt` | ~1.4GB | Default — latest upgrade, fastest on CPU |
  | `checkpoints/titan_v3_phase3/best.pt` | ~3.7GB | Full 1B model — slower but most capable |

  To use the 1B model instead:
  ```bash
  TITAN_CHECKPOINT_PATH=checkpoints/titan_v3_phase3/best.pt \
  TITAN_CONFIG_PATH=configs/titan_crucible_v02.yaml \
  bash start.sh
  ```

  ## API Endpoints (OpenAI-Compatible)

  | Endpoint | Method | Use |
  |---------|--------|-----|
  | `/health` | GET | Check if model is loaded |
  | `/v1/chat/completions` | POST | Chat with Titan (used by Virelle) |
  | `/v1/models` | GET | List loaded checkpoints |
  | `/docs` | GET | Interactive API docs |

  ## System Prompt

  Virelle sends Titan this system prompt:
  > You are Titan, a film and cinema specialist AI built by Virelle Studios. 
  > You help directors, screenwriters, and filmmakers with creative decisions, 
  > storytelling, scene breakdowns, cinematography advice, and production guidance.
  > Be concise, creative, and speak like a seasoned industry professional.
  