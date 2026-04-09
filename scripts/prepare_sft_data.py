#!/usr/bin/env python3
"""
Titan AI — SFT Dataset Preparation
Downloads and formats the instruction tuning datasets for Stage 1.
"""

import os
import json
import random
from pathlib import Path

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data" / "sft"

def download_alpaca():
    """Download the Alpaca 52k dataset (simulated for now)"""
    print("Downloading Alpaca-52k dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # In a real run, this would use datasets library:
    # from datasets import load_dataset
    # ds = load_dataset("tatsu-lab/alpaca")
    
    # For now, create a placeholder
    output_file = DATA_DIR / "alpaca_52k.json"
    with open(output_file, "w") as f:
        json.dump([
            {
                "instruction": "Explain what a buffer overflow attack is.",
                "input": "",
                "output": "A buffer overflow attack occurs when a program writes more data to a buffer than it can hold..."
            }
        ], f, indent=2)
    
    print(f"Saved to {output_file}")

def main():
    print("Preparing SFT datasets...")
    download_alpaca()
    print("SFT datasets ready.")

if __name__ == "__main__":
    main()
