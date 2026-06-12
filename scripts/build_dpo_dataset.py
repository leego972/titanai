"""
TitanAI — Build DPO Dataset from Feedback Logs
================================================
Reads data/dpo_feedback/feedback.jsonl → filters complete pairs
→ writes data/raw/custom_prefs.jsonl (ready for titan_1b_dpo.yaml).

Usage:
    python scripts/build_dpo_dataset.py
    python scripts/build_dpo_dataset.py --min-pairs 50
"""
import argparse
import json
from pathlib import Path

BASE         = Path(__file__).parent.parent
FEEDBACK_IN  = BASE / "data" / "dpo_feedback" / "feedback.jsonl"
DPO_OUT      = BASE / "data" / "raw" / "custom_prefs.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pairs", type=int, default=10,
                        help="Minimum complete pairs required before writing output")
    args = parser.parse_args()

    if not FEEDBACK_IN.exists():
        print(f"[DPO Builder] No feedback file at {FEEDBACK_IN}")
        return

    pairs = []
    skipped = 0

    with open(FEEDBACK_IN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            prompt   = r.get("prompt",   "").strip()
            chosen   = r.get("chosen",   "").strip() if r.get("chosen")   else ""
            rejected = r.get("rejected", "").strip() if r.get("rejected") else ""

            if prompt and chosen and rejected:
                pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            else:
                skipped += 1

    print(f"[DPO Builder] Found {len(pairs)} complete pairs ({skipped} skipped).")

    if len(pairs) < args.min_pairs:
        print(f"[DPO Builder] Below minimum of {args.min_pairs} pairs. Not writing output.")
        print(f"[DPO Builder] Collect more feedback via POST /v1/feedback")
        return

    DPO_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DPO_OUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[DPO Builder] Written {len(pairs)} pairs to {DPO_OUT}")
    print(f"[DPO Builder] Ready to run:")
    print(f"[DPO Builder]   pip install trl")
    print(f"[DPO Builder]   python train_dpo.py --config titan_1b_dpo.yaml")


if __name__ == "__main__":
    main()
