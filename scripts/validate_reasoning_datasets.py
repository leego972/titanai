#!/usr/bin/env python3
"""Strict quality checks for TitanAI complex commonsense/reflection SFT datasets."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data/sft/titan_complex_commonsense_v1.jsonl",
    ROOT / "data/sft/titan_longitudinal_reflection_v1.jsonl",
]
REQUIRED_ROLES = ("system", "user", "assistant")
BANNED_MARKERS = ("placeholder", "todo", "lorem ipsum", "tbd")


def fail(message: str) -> None:
    raise AssertionError(message)


def validate(path: Path) -> dict[str, int]:
    if not path.exists():
        fail(f"missing dataset: {path}")

    ids: set[str] = set()
    prompts: set[str] = set()
    categories: Counter[str] = Counter()
    count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(f"{path.name}:{line_no}: invalid JSON: {exc}")

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                fail(f"{path.name}:{line_no}: expected exactly 3 messages")
            roles = tuple(m.get("role") for m in messages)
            if roles != REQUIRED_ROLES:
                fail(f"{path.name}:{line_no}: roles must be {REQUIRED_ROLES}, got {roles}")

            for msg in messages:
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    fail(f"{path.name}:{line_no}: empty message content")
                lowered = content.lower()
                if any(marker in lowered for marker in BANNED_MARKERS):
                    fail(f"{path.name}:{line_no}: placeholder marker found")

            user = messages[1]["content"].strip()
            answer = messages[2]["content"].strip()
            if len(user) < 20:
                fail(f"{path.name}:{line_no}: user prompt too short")
            if len(answer) < 25:
                fail(f"{path.name}:{line_no}: assistant answer too short")
            if user in prompts:
                fail(f"{path.name}:{line_no}: duplicate user prompt")
            prompts.add(user)

            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                fail(f"{path.name}:{line_no}: missing metadata")
            example_id = str(metadata.get("id", "")).strip()
            category = str(metadata.get("category", "")).strip()
            quality = str(metadata.get("quality", "")).strip()
            if not example_id or example_id in ids:
                fail(f"{path.name}:{line_no}: missing or duplicate id: {example_id!r}")
            if not category:
                fail(f"{path.name}:{line_no}: missing category")
            if quality != "curated":
                fail(f"{path.name}:{line_no}: quality must be curated")
            ids.add(example_id)
            categories[category] += 1
            count += 1

    if count < 20:
        fail(f"{path.name}: dataset too small ({count}); expected at least 20")
    if len(categories) < 12:
        fail(f"{path.name}: insufficient category diversity ({len(categories)})")

    return {"records": count, "categories": len(categories)}


def main() -> int:
    total = 0
    for path in FILES:
        stats = validate(path)
        total += stats["records"]
        print(f"OK {path.relative_to(ROOT)}: {stats['records']} records, {stats['categories']} categories")
    print(f"OK reasoning dataset suite: {total} records validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
