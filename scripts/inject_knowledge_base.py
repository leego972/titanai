#!/usr/bin/env python3
"""
inject_knowledge_base.py
Convert knowledge-base MD files into SFT instruction pairs (JSONL).
Output: data/sft/knowledge_base_cyber_film.jsonl
"""
import json, re, sys
from pathlib import Path

REPO = Path(__file__).parent.parent
KB   = REPO / "knowledge-base"
OUT  = REPO / "data" / "sft" / "knowledge_base_cyber_film.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

KB_FILES = [
    KB / "cybersecurity" / "cybersecurity.md",
    KB / "film-production" / "production.md",
    KB / "post-production" / "post-production.md",
]

def md_to_pairs(path: Path):
    if not path.exists():
        print(f"[WARN] Not found: {path}"); return []
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = re.split(r"(?m)^#{1,3} ", text)
    pairs = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 100: continue
        lines = sec.splitlines()
        title = lines[0].strip().rstrip("#").strip()
        body  = "\n".join(lines[1:]).strip()
        if len(body) < 80: continue
        # Pair 1: explain
        pairs.append({"prompt": f"Explain {title} in detail.", "response": body[:2000]})
        # Pair 2: how-to from code blocks
        code_blocks = re.findall(r"```[\w]*\n([\s\S]+?)```", body)
        for code in code_blocks:
            code = code.strip()
            if len(code) < 30: continue
            pairs.append({
                "prompt": f"Show me a practical example for {title}.",
                "response": f"Here is a practical example:\n\n```\n{code}\n```",
            })
        # Pair 3: bullet Q&A
        bullets = re.findall(r"^[-*] (.+)$", body, re.MULTILINE)
        if len(bullets) >= 3:
            q = f"What are the key points about {title}?"
            a = "\n".join(f"- {b}" for b in bullets[:10])
            pairs.append({"prompt": q, "response": a})
    return pairs

all_pairs = []
for kb_file in KB_FILES:
    pairs = md_to_pairs(kb_file)
    print(f"  {kb_file.name}: {len(pairs)} pairs")
    all_pairs.extend(pairs)

with open(OUT, "w", encoding="utf-8") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")

print(f"\n✓ Wrote {len(all_pairs)} instruction pairs → {OUT}")
