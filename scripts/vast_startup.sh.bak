#!/bin/bash
# TitanAI Probe Training Startup Script for Vast.ai
# ===================================================
# This script sets up the environment and launches the Probe training run.
# Run this on the Vast.ai instance after connecting via SSH.

set -e
echo "======================================"
echo "  TitanAI Probe Training Setup"
echo "======================================"

# Install dependencies
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl python3-pip 2>/dev/null || true

echo "[2/6] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers datasets tokenizers numpy tqdm pyyaml requests huggingface_hub -q

echo "[3/6] Cloning TitanAI repository..."
if [ ! -d "/workspace/titanai" ]; then
    git clone https://github.com/leego972/titanai.git /workspace/titanai
fi
cd /workspace/titanai

echo "[4/6] Installing TitanAI requirements..."
pip install -r requirements.txt -q 2>/dev/null || true

echo "[5/6] Downloading corpus data..."
python3 scripts/load_corpus.py 2>&1 | tee logs/corpus_load.log

echo "[6/6] Generating data shards..."
python3 scripts/generate_shards.py --config configs/titan_probe_v015.yaml 2>&1 | tee logs/shard_gen.log

echo ""
echo "======================================"
echo "  Starting Probe Training Run"
echo "======================================"
python3 scripts/run_probe.py --config configs/titan_probe_v015.yaml 2>&1 | tee logs/probe_training.log

echo "Training complete!"
