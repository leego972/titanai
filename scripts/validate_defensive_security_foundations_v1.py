#!/usr/bin/env python3
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft" / "defensive_security_foundations_v1" / "curated_seed.jsonl"
ALLOWED_DIFFICULTY = {"medium", "hard", "expert"}
REQUIRED_FIELDS = {"id","competency","subcompetency","difficulty","scenario_family","split_family","source_family","verification","safety","messages","rationale"}
REQUIRED_COMPETENCIES = {
    "identity_access","social_engineering","secrets_management","secure_configuration",
    "secure_software","logging_detection","vulnerability_management","incident_response",
    "resilience_recovery","governance_risk",
}
BANNED_PATTERNS = [
    r"\bwrite malware\b", r"\bsteal credentials\b", r"\bbypass mfa\b", r"\bdisable antivirus\b",
    r"\bpersistence technique\b", r"\bevasion technique\b",
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s.lower())).strip()

def main() -> int:
    if not DATA.exists():
        print(f"missing dataset: {DATA}", file=sys.stderr); return 1
    rows=[]
    with DATA.open(encoding="utf-8") as fh:
        for line_no,line in enumerate(fh,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except Exception as exc:
                print(f"line {line_no}: invalid JSON: {exc}", file=sys.stderr); return 1
            missing=REQUIRED_FIELDS-set(row)
            if missing:
                print(f"line {line_no}: missing {sorted(missing)}", file=sys.stderr); return 1
            if row["difficulty"] not in ALLOWED_DIFFICULTY:
                print(f"line {line_no}: invalid difficulty", file=sys.stderr); return 1
            if row["safety"] != "defensive":
                print(f"line {line_no}: non-defensive record", file=sys.stderr); return 1
            msgs=row["messages"]
            if not isinstance(msgs,list) or [m.get("role") for m in msgs] != ["system","user","assistant"]:
                print(f"line {line_no}: messages must be system/user/assistant", file=sys.stderr); return 1
            if any(not isinstance(m.get("content"),str) or not m["content"].strip() for m in msgs):
                print(f"line {line_no}: empty message content", file=sys.stderr); return 1
            user=msgs[1]["content"]; ans=msgs[2]["content"]
            if len(user)<80 or len(ans)<100 or len(row["rationale"])<40:
                print(f"line {line_no}: record below minimum content threshold", file=sys.stderr); return 1
            combined=f"{user}\n{ans}".lower()
            for pat in BANNED_PATTERNS:
                if re.search(pat,combined):
                    print(f"line {line_no}: unsafe pattern {pat}", file=sys.stderr); return 1
            rows.append(row)
    ids=[r["id"] for r in rows]
    if len(ids)!=len(set(ids)):
        print("duplicate ids", file=sys.stderr); return 1
    prompts=[norm(r["messages"][1]["content"]) for r in rows]
    if len(prompts)!=len(set(prompts)):
        print("duplicate prompts", file=sys.stderr); return 1
    comps=Counter(r["competency"] for r in rows)
    missing=REQUIRED_COMPETENCIES-set(comps)
    if missing:
        print(f"missing competencies: {sorted(missing)}", file=sys.stderr); return 1
    families=defaultdict(set)
    for r in rows:
        families[r["competency"]].add(r["scenario_family"])
    weak=[c for c in REQUIRED_COMPETENCIES if len(families[c])<1]
    if weak:
        print(f"weak family coverage: {weak}", file=sys.stderr); return 1
    print(f"PASS: {len(rows)} curated defensive-security records")
    print("Competency counts:", dict(sorted(comps.items())))
    print("NOTE: this validates seed quality only. It does NOT certify competency or marketplace readiness.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
