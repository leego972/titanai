#!/usr/bin/env python3
"""
Titan AI -- Tool Use Fine-Tuning Entry Point
Wraps the SFT trainer for the tool-use stage (titan_tool_v01.yaml).
Tool use fine-tuning is supervised fine-tuning on tool-formatted data.

Usage:
    python scripts/run_tool_tuning.py --config configs/titan_tool_v01.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.sft_trainer import main
if __name__ == "__main__":
    main()
