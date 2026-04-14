#!/usr/bin/env python3
"""
TitanAI — Full Post-Crucible Pipeline v2
==========================================
Sequentially executes all post-Crucible training stages with improved
instruction tuning (SFT v2) and alignment (DPO).

Pipeline stages:
  1. data_prep  — Download and format Alpaca Cleaned + Dolly 15K + DPO pairs
  2. sft_v2     — Instruction fine-tuning on 67K examples (5,000 steps)
  3. tool       — Tool-use fine-tuning
  4. dpo        — DPO alignment pass

Usage:
  # Full pipeline from crucible checkpoint:
  python scripts/run_full_pipeline_v2.py \\
      --checkpoint checkpoints/crucible_v02/final.pt

  # Resume from a specific stage:
  python scripts/run_full_pipeline_v2.py \\
      --checkpoint checkpoints/crucible_v02/final.pt \\
      --start-from sft_v2

  # Skip data prep (if already downloaded):
  python scripts/run_full_pipeline_v2.py \\
      --checkpoint checkpoints/crucible_v02/final.pt \\
      --skip-data-prep

Cost estimate (RTX 4090, ~$0.42/hr):
  data_prep : <5 min    | ~$0.03
  sft_v2    : ~2–3 hr   | ~$1–2
  tool      : ~15–20 hr | ~$6–9
  dpo       : ~4–6 hr   | ~$2–3
  Total                 : ~$10–15
"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent

STAGES = [
    {
        "id": "data_prep",
        "display": "Data Preparation (SFT v2 + DPO)",
        "script": None,  # handled inline
        "config": None,
    },
    {
        "id": "sft_v2",
        "display": "SFT v2 — Instruction Fine-Tuning (Alpaca + Dolly)",
        "script": "scripts/run_sft_v2.py",
        "config": "configs/titan_sft_v02.yaml",
        "checkpoint_in": None,     # filled from args
        "checkpoint_out": "checkpoints/sft_v02/final.pt",
    },
    {
        "id": "tool",
        "display": "Tool Use Fine-Tuning",
        "script": "scripts/run_tool_tuning.py",
        "config": "configs/titan_tool_v01.yaml",
        "checkpoint_in": "checkpoints/sft_v02/final.pt",
        "checkpoint_out": "checkpoints/tool_v01/final.pt",
    },
    {
        "id": "dpo",
        "display": "DPO — Preference Alignment",
        "script": "scripts/run_dpo.py",
        "config": "configs/titan_dpo_v01.yaml",
        "checkpoint_in": "checkpoints/sft_v02/final.pt",  # align from SFT, not tool
        "checkpoint_out": "checkpoints/dpo_v01/final.pt",
    },
]


def run_cmd(cmd, stage_name):
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"  CMD  : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] Stage '{stage_name}' exited with code {result.returncode}")
        print("Fix the error above and resume with --start-from {stage_id}")
        sys.exit(result.returncode)
    print(f"\n[COMPLETE] {stage_name}\n")


def validate_crucible_gates(checkpoint: str):
    """Check that the crucible checkpoint exists before starting."""
    path = BASE / checkpoint
    if not path.exists():
        print(f"\n[ERROR] Crucible checkpoint not found: {path}")
        print("Wait for crucible training to complete before running this pipeline.")
        sys.exit(1)
    print(f"[GATE] Crucible checkpoint found: {path}")


def main():
    parser = argparse.ArgumentParser(description="TitanAI Post-Crucible Pipeline v2")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/crucible_v02/final.pt",
        help="Crucible final checkpoint"
    )
    parser.add_argument(
        "--start-from",
        choices=["data_prep", "sft_v2", "tool", "dpo"],
        default="data_prep",
        help="Stage to start from"
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data download (if already done)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  TitanAI — Post-Crucible Pipeline v2")
    print("  Stages: data_prep → sft_v2 → tool → dpo")
    print("="*60)

    validate_crucible_gates(args.checkpoint)

    # Find start index
    stage_ids = [s["id"] for s in STAGES]
    start_idx = stage_ids.index(args.start_from)

    python = sys.executable

    for stage in STAGES[start_idx:]:
        sid = stage["id"]

        # ── Data prep ─────────────────────────────────────────────────────────
        if sid == "data_prep":
            if args.skip_data_prep:
                print("[SKIP] Data prep skipped (--skip-data-prep)")
                continue
            # Download SFT v2 data
            run_cmd([python, str(BASE / "scripts/prepare_sft_v2_data.py")], stage["display"] + " — SFT datasets")
            # Download DPO data
            run_cmd([python, str(BASE / "scripts/prepare_dpo_data.py")], stage["display"] + " — DPO pairs")
            continue

        # ── Training stages ───────────────────────────────────────────────────
        script = BASE / stage["script"]
        config = BASE / stage["config"]

        if not script.exists():
            print(f"[ERROR] Script not found: {script}")
            sys.exit(1)
        if not config.exists():
            print(f"[ERROR] Config not found: {config}")
            sys.exit(1)

        # Determine checkpoint input
        ckpt_in = stage.get("checkpoint_in") or args.checkpoint

        cmd = [python, str(script), "--config", str(config), "--checkpoint", str(BASE / ckpt_in)]
        run_cmd(cmd, stage["display"])

    print("\n" + "="*60)
    print("  ALL STAGES COMPLETE")
    print("  Titan is now:")
    print("    ✓ Pretrained      (checkpoints/crucible_v02/final.pt)")
    print("    ✓ Instruction-tuned (checkpoints/sft_v02/final.pt)")
    print("    ✓ Tool-capable    (checkpoints/tool_v01/final.pt)")
    print("    ✓ Aligned via DPO (checkpoints/dpo_v01/final.pt)")
    print()
    print("  Deploy the aligned model via the API server:")
    print("  TITAN_CHECKPOINT_PATH=checkpoints/dpo_v01/final.pt \\")
    print("  bash scripts/start_api.sh")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
