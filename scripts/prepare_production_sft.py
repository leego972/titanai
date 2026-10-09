#!/usr/bin/env python3
"""Prepare the production TitanAI 1B SFT mixture.

Downloads/streams established datasets and writes Titan-compatible chat JSONL.
The output is deterministic and train/validation membership is hash-based.

External sources:
- HuggingFaceH4/ultrachat_200k — general instruction/dialogue (MIT)
- open-r1/OpenR1-Math-220k — verified mathematical reasoning (Apache-2.0)
- nvidia/OpenCodeReasoning — coding reasoning (CC-BY-4.0 dataset release;
  per-row source license retained as metadata)

No behavioral-policy/refusal layer is added by this preparation script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "sft" / "production_external"


def bucket(key: str, mod: int = 1000) -> int:
    return int(hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:16], 16) % mod


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def normalize_messages(messages):
    out = []
    if not isinstance(messages, list):
        return out
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        if content is None:
            continue
        content = str(content).strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def valid_chat(messages) -> bool:
    roles = [m.get("role") for m in messages]
    return "user" in roles and "assistant" in roles


def prepare_ultrachat(train_cap: int, val_cap: int):
    print("[prepare] UltraChat 200k")
    train_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    try:
        val_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
    except Exception:
        val_ds = None

    def train_rows():
        n = 0
        for ex in train_ds:
            messages = normalize_messages(ex.get("messages"))
            if not valid_chat(messages):
                continue
            key = str(ex.get("prompt_id") or ex.get("id") or hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest())
            # Reserve bucket 0-19 as a fallback validation region; train excludes it.
            if bucket("ultrachat:" + key) < 20:
                continue
            yield {
                "id": "ultrachat:" + key,
                "source": "HuggingFaceH4/ultrachat_200k",
                "license": "MIT",
                "messages": messages,
            }
            n += 1
            if n >= train_cap:
                break

    def val_rows():
        n = 0
        ds = val_ds if val_ds is not None else load_dataset(
            "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
        )
        for ex in ds:
            messages = normalize_messages(ex.get("messages"))
            if not valid_chat(messages):
                continue
            key = str(ex.get("prompt_id") or ex.get("id") or hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest())
            if val_ds is None and bucket("ultrachat:" + key) >= 20:
                continue
            yield {
                "id": "ultrachat-val:" + key,
                "source": "HuggingFaceH4/ultrachat_200k",
                "license": "MIT",
                "messages": messages,
            }
            n += 1
            if n >= val_cap:
                break

    return train_rows(), val_rows()


def prepare_openr1(train_cap: int, val_cap: int):
    print("[prepare] OpenR1-Math-220k default")
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train", streaming=True)

    def rows(want_val: bool, cap: int):
        n = 0
        for ex in ds:
            messages = normalize_messages(ex.get("messages"))
            if not valid_chat(messages):
                problem = str(ex.get("problem") or "").strip()
                solution = str(ex.get("solution") or "").strip()
                if problem and solution:
                    messages = [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": solution},
                    ]
            if not valid_chat(messages):
                continue
            key = str(ex.get("uuid") or hashlib.sha256((str(ex.get("problem")) + str(ex.get("answer"))).encode()).hexdigest())
            is_val = bucket("openr1:" + key) < 20
            if is_val != want_val:
                continue
            yield {
                "id": "openr1:" + key,
                "source": "open-r1/OpenR1-Math-220k:default",
                "license": "Apache-2.0",
                "problem_type": ex.get("problem_type"),
                "messages": messages,
            }
            n += 1
            if n >= cap:
                break

    return rows(False, train_cap), rows(True, val_cap)


def prepare_code(train_cap: int, val_cap: int):
    print("[prepare] NVIDIA OpenCodeReasoning split_0")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)

    def rows(want_val: bool, cap: int):
        n = 0
        seen_questions = set()
        for ex in ds:
            prompt = str(ex.get("input") or "").strip()
            answer = str(ex.get("output") or "").strip()
            if not prompt or not answer or prompt == "-":
                continue
            qid = str(ex.get("id") or hashlib.sha256(prompt.encode()).hexdigest())
            # Keep only one reasoning trace per unique question ID to prevent the
            # same problem from dominating the SFT mixture.
            if qid in seen_questions:
                continue
            seen_questions.add(qid)
            is_val = bucket("opencode:" + qid) < 20
            if is_val != want_val:
                continue
            yield {
                "id": "opencode:" + qid,
                "source": "nvidia/OpenCodeReasoning:split_0",
                "source_platform": ex.get("source"),
                "source_dataset": ex.get("dataset"),
                "source_license": ex.get("license"),
                "difficulty": ex.get("difficulty"),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
            }
            n += 1
            if n >= cap:
                break

    return rows(False, train_cap), rows(True, val_cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultrachat-train", type=int, default=80000)
    ap.add_argument("--math-train", type=int, default=50000)
    ap.add_argument("--code-train", type=int, default=50000)
    ap.add_argument("--val-each", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}

    u_train, u_val = prepare_ultrachat(args.ultrachat_train, args.val_each)
    stats["ultrachat_train"] = write_jsonl(OUT / "ultrachat_train.jsonl", u_train)
    stats["ultrachat_val"] = write_jsonl(OUT / "ultrachat_val.jsonl", u_val)

    m_train, m_val = prepare_openr1(args.math_train, args.val_each)
    stats["openr1_math_train"] = write_jsonl(OUT / "openr1_math_train.jsonl", m_train)
    stats["openr1_math_val"] = write_jsonl(OUT / "openr1_math_val.jsonl", m_val)

    c_train, c_val = prepare_code(args.code_train, args.val_each)
    stats["opencode_train"] = write_jsonl(OUT / "opencode_train.jsonl", c_train)
    stats["opencode_val"] = write_jsonl(OUT / "opencode_val.jsonl", c_val)

    minimums = {
        "ultrachat_train": min(10000, args.ultrachat_train),
        "openr1_math_train": min(10000, args.math_train),
        "opencode_train": min(10000, args.code_train),
        "ultrachat_val": min(500, args.val_each),
        "openr1_math_val": min(500, args.val_each),
        "opencode_val": min(500, args.val_each),
    }
    failed = [k for k, minimum in minimums.items() if stats.get(k, 0) < minimum]

    manifest = {
        "version": "1.0.0",
        "split_rule": "SHA256 stable identity; validation bucket < 20/1000",
        "counts": stats,
        "sources": {
            "ultrachat": {"dataset": "HuggingFaceH4/ultrachat_200k", "license": "MIT"},
            "openr1_math": {"dataset": "open-r1/OpenR1-Math-220k", "config": "default", "license": "Apache-2.0"},
            "opencode": {"dataset": "nvidia/OpenCodeReasoning", "config": "split_0", "license": "CC-BY-4.0 release; per-row source license retained"},
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if failed:
        raise SystemExit(f"Production SFT preparation failed minimum-count checks: {failed}")
    print(f"[prepare] Production external SFT ready at {OUT}")


if __name__ == "__main__":
    main()
