#!/usr/bin/env python3
"""
TitanAI — Resume Pipeline
Resumes from the first incomplete upgrade and runs all remaining upgrades.
upgrade_an runs LAST per project requirement.
Auto-cleans step_*.pt files after each upgrade to prevent disk fill.
"""
import subprocess
import sys
import glob
from pathlib import Path

BASE = Path("/workspace/titanai")
LOG = BASE / "checkpoints" / "pipeline_upgrades.log"

# Full training sequence — upgrade_an is intentionally LAST
SEQUENCE = [
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
    "aa","ab","ac","ad","ae","af","ag","ah","ai","aj","ak","al","am",
    "ao","ap","aq","ar","as","at","au","av","aw","ax","ay","az",
    "ba","bb","bc","bd","be","bf","bg","bh","bi","bj","bk","bl","bm","bn","bo","bp","bq",
    "an",
]

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def checkpoint(name):
    return BASE / "checkpoints" / f"upgrade_{name}" / "final.pt"

def cleanup_step_files(name):
    ckpt_dir = BASE / "checkpoints" / f"upgrade_{name}"
    removed = []
    for pat in ["step_*.pt", "step_*.pt.tmp", "*.pt.tmp"]:
        for f in ckpt_dir.glob(pat):
            f.unlink(missing_ok=True)
            removed.append(f.name)
    if removed:
        log(f"[CLEAN] upgrade_{name}: removed {len(removed)} step file(s) to free disk")

def run_upgrade(name, prev_name):
    config = BASE / "configs" / f"titan_upgrade_{name}.yaml"
    prev_ckpt = checkpoint(prev_name)
    out_log = Path(f"/tmp/train_{name}.log")

    if not config.exists():
        log(f"[SKIP] No config for upgrade_{name} — skipping")
        return True

    if not prev_ckpt.exists():
        log(f"[ERROR] Previous checkpoint missing: {prev_ckpt}")
        return False

    log(f"  UPGRADE {name.upper()} — loading from upgrade_{prev_name}/final.pt")

    cmd = [
        "python3", str(BASE / "scripts" / "run_upgrade.py"),
        "--config", str(config),
        "--checkpoint", str(prev_ckpt),
    ]

    with open(out_log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        log(f"[FAILED] upgrade_{name} (exit {result.returncode}) — check {out_log}")
        return False

    log(f"[titan-upgrade-{name}] COMPLETE -> checkpoints/upgrade_{name}/final.pt")
    cleanup_step_files(name)
    return True

def main():
    import os
    os.chdir(BASE)
    log("\n  TITAN UPGRADE PIPELINE — RESUME\n")

    for i, name in enumerate(SEQUENCE):
        ckpt = checkpoint(name)
        if ckpt.exists():
            log(f"[SKIP] upgrade_{name} already complete")
            cleanup_step_files(name)
            continue

        prev_name = SEQUENCE[i - 1] if i > 0 else None
        if prev_name is None:
            log(f"[ERROR] No previous checkpoint for first upgrade")
            sys.exit(1)

        ok = run_upgrade(name, prev_name)
        if not ok:
            sys.exit(1)

    log("\n  ALL 69 UPGRADES COMPLETE! (68 domains + upgrade_bq Hebrew/Kabbalah)")
    log("  upgrade_an ran LAST — owner loyalty passphrase locked in.")

if __name__ == "__main__":
    main()
