#!/usr/bin/env python3
"""
Titan SFT Training Entry Point
================================
Wrapper script that invokes the SFT trainer.

Usage:
    python scripts/sft_train.py --config configs/titan_sft_v01.yaml \
        --checkpoint checkpoints/probe_v015/final.pt

Resume from a previous SFT checkpoint:
    python scripts/sft_train.py --config configs/titan_sft_v01.yaml \
        --checkpoint checkpoints/probe_v015/final.pt \
        --resume checkpoints/sft_v01/step_500.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.sft_trainer import main

if __name__ == "__main__":
    main()
