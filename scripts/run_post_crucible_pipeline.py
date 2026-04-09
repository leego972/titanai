#!/usr/bin/env python3
"""
Titan AI — Post-Crucible Agentic Training Pipeline
This script sequentially executes the 2 post-Crucible training stages:
1. SFT (Instruction Tuning)
2. Tool Use Fine-Tuning

No alignment or censorship layer. Titan is unrestricted by design.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent

def run_stage(script_name, config_path, stage_name):
    print(f"\n{'='*50}")
    print(f"STARTING STAGE: {stage_name.upper()}")
    print(f"{'='*50}\n")
    
    script_path = BASE / "scripts" / script_name
    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)
        
    config_file = BASE / "configs" / config_path
    if not config_file.exists():
        print(f"[ERROR] Config not found: {config_file}")
        sys.exit(1)
        
    cmd = ["python3", str(script_path), "--config", str(config_file)]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n[COMPLETE] STAGE: {stage_name.upper()}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] STAGE: {stage_name.upper()} (Exit code: {e.returncode})\n")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the Titan post-Crucible pipeline")
    parser.add_argument("--start-from", type=str, choices=["sft", "tool"], default="sft",
                        help="Which stage to start from")
    args = parser.parse_args()
    
    stages = [
        ("sft",  "run_sft.py",          "titan_sft_v01.yaml",  "Instruction Tuning (SFT)"),
        ("tool", "run_tool_tuning.py",   "titan_tool_v01.yaml", "Tool Use Fine-Tuning"),
    ]
    
    # Find start index
    start_idx = 0
    for i, (stage_id, _, _, _) in enumerate(stages):
        if stage_id == args.start_from:
            start_idx = i
            break
            
    print(f"Starting post-Crucible pipeline from stage: {stages[start_idx][3]}")
    
    # Execute stages sequentially
    for stage_id, script_name, config_path, display_name in stages[start_idx:]:
        run_stage(script_name, config_path, display_name)
        
    print("\n" + "="*50)
    print("ALL POST-CRUCIBLE TRAINING STAGES COMPLETE!")
    print("Titan AI is now instruction-following and tool-capable.")
    print("Next: deploy agentic scaffolding (agent/runtime.py).")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
