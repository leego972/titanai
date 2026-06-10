#!/usr/bin/env python3
"""
TitanAI — Credit Watchdog
==========================
Monitors your Vast.ai account balance and automatically pauses the instance
when credits drop to or below the threshold.

What happens when threshold is hit:
  1. SIGTERM to all training processes (triggers graceful checkpoint save)
  2. Waits up to 5 minutes for clean exit (so the last checkpoint is valid)
  3. SIGKILL anything still running
  4. Calls Vast.ai API to stop (pause) the instance
  5. Instance is paused — you pay storage only, not compute

Resume when you have credits:
  ssh in, then:  bash /workspace/titanai/scripts/autorun.sh --start-from sft_v2

Launch:
  python3 scripts/credit_watchdog.py [--threshold 5.0] [--interval 120]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE        = Path(__file__).resolve().parent.parent
INSTANCE_ID = 40187240   # This Vast.ai instance


def load_api_key() -> str:
    key = os.environ.get("VASTAI_API_KEY", "").strip()
    if key:
        return key
    key_file = BASE / ".vast_api_key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    print("[watchdog] ERROR: Vast.ai API key not found.")
    print(f"  Set VASTAI_API_KEY env var, or write key to {key_file}")
    sys.exit(1)


def get_balance(api_key: str) -> float:
    url = f"https://console.vast.ai/api/v0/users/current/?api_key={api_key}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read())
    return float(data.get("credit", data.get("balance", 999)))


def stop_instance(api_key: str, instance_id: int) -> bool:
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/?api_key={api_key}"
    payload = json.dumps({"state": "stopped"}).encode()
    req = urllib.request.Request(url, data=payload, method="PUT",
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            return data.get("success", True)
    except Exception as e:
        print(f"[watchdog] API error: {e}")
        return False


def graceful_stop_training():
    procs = [
        "scripts/train.py",
        "scripts/run_sft_v2.py",
        "scripts/run_dpo.py",
        "scripts/run_tool_tuning.py",
        "scripts/autorun.py",
    ]
    killed = []
    for name in procs:
        r = subprocess.run(["pgrep", "-f", name], capture_output=True)
        if r.returncode == 0:
            for pid in r.stdout.decode().split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    killed.append(pid)
                    print(f"[watchdog] SIGTERM → PID {pid} ({name})")
                except ProcessLookupError:
                    pass

    if not killed:
        print("[watchdog] No training processes found to stop.")
        return

    print(f"[watchdog] Waiting up to 5 min for checkpoint save...")
    deadline = time.time() + 300
    while time.time() < deadline:
        alive = [p for p in killed if subprocess.run(
            ["kill", "-0", p], capture_output=True).returncode == 0]
        if not alive:
            print("[watchdog] All training processes exited cleanly.")
            return
        time.sleep(10)

    for pid in killed:
        try:
            os.kill(int(pid), signal.SIGKILL)
            print(f"[watchdog] SIGKILL → PID {pid} (timeout)")
        except ProcessLookupError:
            pass
    print("[watchdog] Force-killed remaining processes.")


def main():
    parser = argparse.ArgumentParser(description="TitanAI Credit Watchdog")
    parser.add_argument("--threshold", type=float, default=5.0,
        help="Pause instance when balance drops to this amount in USD (default: 5.0)")
    parser.add_argument("--interval", type=int, default=120,
        help="Seconds between balance checks (default: 120)")
    parser.add_argument("--instance-id", type=int, default=INSTANCE_ID,
        help=f"Vast.ai instance ID (default: {INSTANCE_ID})")
    args = parser.parse_args()

    api_key = load_api_key()
    log_file = BASE / "logs" / "credit_watchdog.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        ts   = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as lf:
            lf.write(line + "\n")

    log(f"Watchdog started | threshold=${args.threshold:.2f} | "
        f"interval={args.interval}s | instance={args.instance_id}")

    # Show current balance immediately on start
    try:
        balance = get_balance(api_key)
        log(f"Current balance: ${balance:.4f}")
    except Exception as e:
        log(f"WARNING: Could not fetch initial balance: {e}")

    while True:
        time.sleep(args.interval)
        try:
            balance = get_balance(api_key)
            log(f"Balance: ${balance:.4f}")

            if balance <= args.threshold:
                log(f"*** BALANCE ${balance:.4f} ≤ ${args.threshold:.2f} THRESHOLD ***")
                log("Stopping training processes to save checkpoints...")
                graceful_stop_training()

                log("Calling Vast.ai API to stop (pause) the instance...")
                ok = stop_instance(api_key, args.instance_id)

                if ok:
                    log("Instance paused successfully.")
                    log("Checkpoints are safe. When you have credits, restart the")
                    log("instance and run:  bash scripts/autorun.sh --start-from sft_v2")
                else:
                    log("WARNING: API call may have failed — check Vast.ai console.")
                log("Watchdog exiting.")
                sys.exit(0)

        except Exception as e:
            log(f"ERROR: {e} — will retry in {args.interval}s")


if __name__ == "__main__":
    main()
