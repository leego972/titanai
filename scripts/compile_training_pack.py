#!/usr/bin/env python3
"""Compile uploaded TitanAI training packs into audited CPT/SFT/DPO/eval JSONL.

Standard-library only. Supports directories, JSON/JSONL/CSV/text/Markdown and
ZIP/TAR archives. It validates provenance, removes exact duplicates, detects
likely secrets, scores records, creates deterministic splits and writes a full
report. It does not add behavioural restrictions to the model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
]
SPACE_RE = re.compile(r"[ \t]+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]+")


@dataclass
class Audit:
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    files_seen: int = 0
    records_seen: int = 0
    reasons: Counter | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.splitlines()).strip()


def content_from_record(record: dict[str, Any]) -> str:
    for key in ("content", "text", "document", "body"):
        if record.get(key):
            return canonical_text(record[key])
    parts = []
    for key in ("instruction", "input", "response", "output", "chosen", "rejected", "code", "tests"):
        if record.get(key):
            parts.append(f"{key.upper()}:\n{canonical_text(record[key])}")
    return "\n\n".join(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def quality_score(text: str, record: dict[str, Any]) -> float:
    """Transparent heuristic quality score in [0, 1]."""
    chars = len(text)
    tokens = TOKEN_RE.findall(text)
    unique_ratio = len(set(t.lower() for t in tokens)) / max(1, len(tokens))
    lines = [line for line in text.splitlines() if line.strip()]
    score = 0.0
    score += min(0.24, math.log10(max(chars, 10)) / 4 * 0.24)
    score += min(0.16, unique_ratio * 0.28)
    score += 0.10 if len(lines) >= 4 else 0.03
    score += 0.08 if any(record.get(k) for k in ("source", "references")) else 0.0
    score += 0.08 if record.get("usage_basis") or record.get("license") else 0.0
    score += 0.08 if record.get("domain") else 0.0
    score += 0.08 if any(record.get(k) for k in ("code", "tests", "response", "chosen")) else 0.0
    score += 0.06 if any(mark in text for mark in ("because", "therefore", "trade-off", "root cause", "test", "verify")) else 0.0
    score += 0.06 if any(mark in text for mark in ("```", "## ", "1.", "- ")) else 0.0
    score += 0.06 if record.get("version") or record.get("retrieved_at") else 0.0
    if chars < 240:
        score -= 0.30
    if unique_ratio < 0.12 and len(tokens) > 80:
        score -= 0.20
    return round(max(0.0, min(1.0, score)), 4)


def infer_stage(record: dict[str, Any]) -> str:
    explicit = str(record.get("stage", "")).lower()
    if explicit in {"cpt", "sft", "dpo", "evaluation"}:
        return explicit
    if record.get("chosen") and record.get("rejected"):
        return "dpo"
    if record.get("instruction") and any(record.get(k) for k in ("response", "output")):
        return "sft"
    return "cpt"


def normalize(record: dict[str, Any], source_file: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    text = content_from_record(record)
    source = record.get("source") or defaults.get("source") or str(source_file)
    usage = record.get("usage_basis") or record.get("license") or defaults.get("usage_basis")
    domain = record.get("domain") or defaults.get("domain")
    normalized = dict(record)
    normalized.update({
        "content": text,
        "source": source,
        "usage_basis": usage,
        "domain": domain,
        "stage": infer_stage(record),
        "source_file": str(source_file),
        "content_sha256": sha256_text(text),
        "compiled_at": utc_now(),
    })
    normalized["quality_score"] = quality_score(text, normalized)
    return normalized


def iter_json(path: Path) -> Iterator[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(obj, list):
        for item in obj:
            yield item if isinstance(item, dict) else {"content": item}
    elif isinstance(obj, dict):
        rows = obj.get("records") or obj.get("data")
        if isinstance(rows, list):
            for item in rows:
                yield item if isinstance(item, dict) else {"content": item}
        else:
            yield obj


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                yield item if isinstance(item, dict) else {"content": item}
            except json.JSONDecodeError as exc:
                yield {"content": "", "_parse_error": f"line {line_no}: {exc}"}
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle)
    elif suffix in {".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".sql", ".sh", ".ps1"}:
        yield {"content": path.read_text(encoding="utf-8", errors="replace")}


def safe_extract(pack: Path, destination: Path) -> Path:
    if pack.is_dir():
        return pack
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(pack):
        with zipfile.ZipFile(pack) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise ValueError(f"unsafe archive member: {member.filename}")
            archive.extractall(destination)
        return destination
    if tarfile.is_tarfile(pack):
        with tarfile.open(pack) as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if destination.resolve() not in target.parents and target != destination.resolve():
                    raise ValueError(f"unsafe archive member: {member.name}")
            archive.extractall(destination)
        return destination
    return pack.parent


def deterministic_bucket(digest: str) -> int:
    return int(digest[:8], 16) % 10000


def compile_pack(pack: Path, manifest_path: Path, output_override: Path | None = None) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = manifest["quality"]
    output = output_override or Path(manifest["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    rejected_path = output / "rejected.jsonl"
    writers = {stage: (output / f"{stage}.jsonl").open("w", encoding="utf-8") for stage in ("cpt", "sft", "dpo", "evaluation")}
    rejected = rejected_path.open("w", encoding="utf-8")
    audit = Audit()
    seen: set[str] = set()
    defaults = manifest.get("defaults", {})

    with tempfile.TemporaryDirectory(prefix="titan-pack-") as temp:
        root = safe_extract(pack, Path(temp))
        files = [pack] if pack.is_file() and root == pack.parent and not (zipfile.is_zipfile(pack) or tarfile.is_tarfile(pack)) else [p for p in root.rglob("*") if p.is_file()]
        for source_file in sorted(files):
            if source_file.name.startswith("."):
                continue
            audit.files_seen += 1
            try:
                records = iter_records(source_file)
                for raw in records:
                    audit.records_seen += 1
                    rec = normalize(raw, source_file, defaults)
                    reason = None
                    text = rec["content"]
                    if raw.get("_parse_error"):
                        reason = "parse_error"
                    elif not text:
                        reason = "empty_content"
                    elif len(text) < int(quality["minimum_text_chars"]):
                        reason = "too_short"
                    elif len(text) > int(quality["maximum_text_chars"]):
                        reason = "too_long"
                    elif quality.get("require_provenance") and not rec.get("source"):
                        reason = "missing_provenance"
                    elif quality.get("require_usage_basis") and not rec.get("usage_basis"):
                        reason = "missing_usage_basis"
                    elif quality.get("reject_secrets") and contains_secret(text):
                        reason = "likely_secret"
                    elif rec["content_sha256"] in seen:
                        reason = "exact_duplicate"
                        audit.duplicates += 1
                    elif rec["quality_score"] < float(quality["minimum_quality_score"]):
                        reason = "below_quality_threshold"

                    if reason:
                        rec["rejection_reason"] = reason
                        rejected.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        audit.rejected += 1
                        audit.reasons[reason] += 1
                        continue

                    seen.add(rec["content_sha256"])
                    bucket = deterministic_bucket(rec["content_sha256"])
                    eval_cut = int(float(quality.get("holdout_fraction", 0.02)) * 10000)
                    val_cut = eval_cut + int(float(quality.get("validation_fraction", 0.01)) * 10000)
                    if bucket < eval_cut:
                        rec["split"] = "evaluation"
                        stage = "evaluation"
                    elif bucket < val_cut:
                        rec["split"] = "validation"
                        stage = rec["stage"]
                    else:
                        rec["split"] = "train"
                        stage = rec["stage"]
                    writers[stage].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    audit.accepted += 1
            except Exception as exc:
                audit.rejected += 1
                audit.reasons[f"file_error:{type(exc).__name__}"] += 1
                rejected.write(json.dumps({"source_file": str(source_file), "rejection_reason": str(exc)}, ensure_ascii=False) + "\n")

    for handle in writers.values():
        handle.close()
    rejected.close()
    report = {
        "pack_id": manifest["pack_id"],
        "target_model": manifest["target_model"],
        "compiled_at": utc_now(),
        "input": str(pack),
        "manifest": str(manifest_path),
        "output": str(output),
        **asdict(audit),
    }
    report["reasons"] = dict(audit.reasons or {})
    (output / "compile_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "READY").write_text("ready\n" if audit.accepted else "empty\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if audit.accepted else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, help="uploaded directory or archive")
    parser.add_argument("--manifest", type=Path, default=Path("configs/training_packs/batch_01_core_engineering_security.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.pack.exists():
        parser.error(f"pack does not exist: {args.pack}")
    if not args.manifest.exists():
        parser.error(f"manifest does not exist: {args.manifest}")
    return compile_pack(args.pack, args.manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
