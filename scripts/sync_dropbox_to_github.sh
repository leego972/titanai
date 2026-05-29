#!/bin/bash
# sync_dropbox_to_github.sh — Dropbox → GitHub sync
# Usage  : bash scripts/sync_dropbox_to_github.sh
# Cron   : */30 * * * * bash /workspace/titanai/scripts/sync_dropbox_to_github.sh >> /workspace/titanai/data/sync.log 2>&1

set -euo pipefail
REPO=/workspace/titanai
REMOTE="dropbox:"

cd "$REPO"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Dropbox → GitHub sync start"

# Sync metadata only — skip .pt / .bin / .npy (too large for GitHub)
rclone copy "${REMOTE}" "${REPO}/data/dropbox_mirror/" \
  --include "*.json" --include "*.yaml" --include "*.log" \
  --include "*.txt"  --include "*.md" \
  --exclude "*.pt"   --exclude "*.bin" --exclude "*.npy" \
  --max-size 50M --transfers 4 -v 2>&1 | tail -10

git config user.email "titanai-bot@vast.ai"  2>/dev/null || true
git config user.name  "TitanAI Sync Bot"     2>/dev/null || true
git add data/dropbox_mirror/ data/corpus_master.log data/corpus_injection_status.json 2>/dev/null || true

if ! git diff --cached --quiet; then
  git commit -m "sync: Dropbox → GitHub [$(date -u +%Y-%m-%dT%H:%M:%SZ)]"
  git push origin main
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pushed OK"
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Nothing new to sync"
fi
