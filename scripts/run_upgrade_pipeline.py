#!/usr/bin/env python3
"""
Titan AI — Complete Upgrade Pipeline
Runs all five expert domain depth-packs sequentially:

  Base:      SFT (Instruction Tuning)
  Base:      Tool Use Fine-Tuning
  Upgrade A: General Knowledge Depth Pack
  Upgrade B: Film & Cinema Production (Full Pipeline)
  Upgrade C: Cybersecurity Depth Pack
  Upgrade D: Creative Writing & Thinking
  Upgrade E: Proactive Problem Solving

Run after Crucible completes. Each stage builds on the previous checkpoint.
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent

STAGES = [
    ("sft",       "run_sft.py",          "titan_sft_v01.yaml",              "Instruction Tuning (SFT)"),
    ("tool",      "run_tool_tuning.py",   "titan_tool_v01.yaml",             "Tool Use Fine-Tuning"),
    ("general",   "run_domain_depth.py",  "titan_general_depth_v01.yaml",    "Upgrade A: General Knowledge"),
    ("cinema",    "run_domain_depth.py",  "titan_cinema_depth_v01.yaml",     "Upgrade B: Film & Cinema Production"),
    ("cyber",     "run_domain_depth.py",  "titan_cyber_depth_v01.yaml",      "Upgrade C: Cybersecurity"),
    ("creative",  "run_domain_depth.py",  "titan_creative_depth_v01.yaml",   "Upgrade D: Creative Writing & Thinking"),
    ("prosolve",  "run_domain_depth.py",  "titan_prosolve_depth_v01.yaml",   "Upgrade E: Proactive Problem Solving"),
]

def run_stage(script_name, config_path, stage_name):
    print(f"\n{'='*65}")
    print(f"  STARTING: {stage_name}")
    print(f"{'='*65}\n")

    script_path = BASE / "scripts" / script_name
    config_file = BASE / "configs" / config_path

    if not script_path.exists():
        print(f"[ERROR] Script not found: {script_path}")
        sys.exit(1)
    if not config_file.exists():
        print(f"[ERROR] Config not found: {config_file}")
        sys.exit(1)

    cmd = ["python3", str(script_path), "--config", str(config_file)]
    try:
        subprocess.run(cmd, check=True)
        print(f"\n  [COMPLETE] {stage_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n  [FAILED] {stage_name} (Exit code: {e.returncode})\n")
        sys.exit(e.returncode)

def main():
    valid_starts = [s[0] for s in STAGES]
    parser = argparse.ArgumentParser(description="Run the full Titan AI upgrade pipeline")
    parser.add_argument(
        "--start-from",
        type=str,
        choices=valid_starts,
        default="sft",
        help="Stage to start from (default: sft)"
    )
    args = parser.parse_args()

    start_idx = next(i for i, (sid, *_) in enumerate(STAGES) if sid == args.start_from)
    remaining = STAGES[start_idx:]

    print(f"\n{'='*65}")
    print(f"  TITAN AI — COMPLETE UPGRADE PIPELINE")
    print(f"  Starting from: {remaining[0][3]}")
    print(f"  Stages to run: {len(remaining)}")
    print(f"{'='*65}")

    for stage_id, script_name, config_path, display_name in remaining:
        run_stage(script_name, config_path, display_name)

    print(f"\n{'='*65}")
    print(f"  TITAN AI v1.5 — UPGRADE PIPELINE COMPLETE")
    print(f"")
    print(f"  Five pillars trained:")
    print(f"    General Knowledge")
    print(f"    Film & Cinema Production (full pipeline)")
    print(f"    Cybersecurity")
    print(f"    Creative Writing & Thinking")
    print(f"    Proactive Problem Solving")
    print(f"")
    print(f"  Next: deploy agentic scaffolding (agent/runtime.py)")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
