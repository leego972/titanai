#!/usr/bin/env python3
"""
TitanAI — Post-DPO Upgrade Runner
Waits for DPO to finish, then trains on ALL 168 upgrade files:
film, cinema, production, cyber (offensive + defensive), psychology,
mathematics, law, business, OSINT, CTF, exploit defense, and more.
"""
import sys, time, subprocess
from pathlib import Path

BASE       = Path(__file__).parent.parent
DPO_CKPT   = BASE / "checkpoints/dpo_v01/final.pt"
UPG_CKPT   = BASE / "checkpoints/upgrades_all/final.pt"
CONFIG     = BASE / "configs/titan_1b_upgrade_all.yaml"
LOG_DIR    = BASE / "logs/upgrades_all"
PYTHON     = sys.executable

LOG_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)

# ── Wait for DPO ────────────────────────────────────────────────────────────
log("Waiting for DPO checkpoint: checkpoints/dpo_v01/final.pt ...")
while not DPO_CKPT.exists():
    time.sleep(60)
    log("DPO still running — checking again in 60s ...")

log("DPO complete! Starting full upgrade training.")
log(f"Config  : {CONFIG}")
log(f"Base ckpt: {DPO_CKPT}")

# ── Skip if already done ─────────────────────────────────────────────────────
if UPG_CKPT.exists():
    log("Upgrades already complete — nothing to do.")
    sys.exit(0)

# ── Run upgrade training ─────────────────────────────────────────────────────
cmd = [
    PYTHON, str(BASE / "scripts/run_upgrade.py"),
    "--config",     str(CONFIG),
    "--checkpoint", str(DPO_CKPT),
]
log(f"Running: {' '.join(cmd)}")

result = subprocess.run(cmd, cwd=str(BASE))

if result.returncode == 0:
    log("=" * 60)
    log("ALL UPGRADE TRAINING COMPLETE")
    log(f"Final checkpoint: {UPG_CKPT}")
    log("Titan now has: film/cinema, cyber offense+defense, psychology,")
    log("  mathematics, law, business, OSINT, CTF, production, and more.")
    log("=" * 60)
else:
    log(f"[FAILED] Upgrade training exited with code {result.returncode}")
    sys.exit(result.returncode)
