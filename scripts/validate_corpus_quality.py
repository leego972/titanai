"""
TitanAI Corpus Quality Validator — Phase 2
============================================
Samples documents from every bucket, checks quality, confirms balance,
and produces a per-bucket quality review report.

Checks:
  1. Sample 20 random documents per bucket — inspect for garbage/spam
  2. Estimate token counts per bucket (word-based approximation)
  3. Confirm approximate target ratios (A:35 B:15 C:20 D:20 E:10)
  4. Check for minimum document length compliance
  5. Check for boilerplate/repetition signals
  6. Confirm Bucket D is defensive-oriented (no offensive keywords as primary content)
  7. Confirm Bucket E is real cinema craft (not fluff)
  8. Produce pass/fail per bucket
"""

import os
import sys
import json
import random
import re
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).parent.parent
RAW  = BASE / "data" / "raw"

random.seed(42)

BUCKETS = {
    "corpus_A_general":   {"target": 0.35, "name": "General Language"},
    "corpus_B_reasoning": {"target": 0.15, "name": "Reasoning / Planning"},
    "corpus_C_technical": {"target": 0.20, "name": "Technical / Systems"},
    "corpus_D_cyber":     {"target": 0.20, "name": "Cybersecurity"},
    "corpus_E_cinema":    {"target": 0.10, "name": "Film / Cinema"},
}

RATIO_TOLERANCE = 0.10  # ±10% for initial corpus (will tighten at shard stage)

# Offensive keywords that should NOT be the primary focus of Bucket D documents
OFFENSIVE_SIGNALS = [
    "how to hack", "step by step exploit", "bypass antivirus", "create malware",
    "phishing template", "credential harvester", "keylogger tutorial",
    "ransomware source", "botnet setup", "sql injection tutorial for beginners",
    "reverse shell one-liner", "privilege escalation cheat sheet"
]

# Quality signals that indicate low-quality / garbage documents
GARBAGE_SIGNALS = [
    "click here", "buy now", "limited offer", "subscribe to newsletter",
    "lorem ipsum", "placeholder text", "test test test",
    "aaaaaaa", "xxxxxxx", "null null null"
]

# Minimum expected content keywords per bucket
BUCKET_QUALITY_KEYWORDS = {
    "corpus_A_general":   ["the", "and", "of", "in", "is"],
    "corpus_B_reasoning": ["because", "therefore", "step", "solution", "answer",
                           "problem", "result", "prove", "calculate", "reasoning"],
    "corpus_C_technical": ["function", "class", "import", "return", "error",
                           "code", "system", "server", "api", "data"],
    "corpus_D_cyber":     ["security", "vulnerability", "attack", "detection",
                           "mitigation", "threat", "risk", "cve", "exploit",
                           "network", "authentication", "encryption"],
    "corpus_E_cinema":    ["film", "movie", "director", "scene", "character",
                           "story", "plot", "production", "cinema", "screen"],
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    words = len(text.split())
    return int(words * 0.75)


def check_document_quality(text: str, bucket_key: str) -> dict:
    text_lower = text.lower()
    issues = []

    # Length check
    if len(text.strip()) < 100:
        issues.append("too_short")

    # Garbage signals
    for signal in GARBAGE_SIGNALS:
        if signal in text_lower:
            issues.append(f"garbage_signal:{signal}")
            break

    # Offensive signals (only for Bucket D)
    if bucket_key == "corpus_D_cyber":
        for signal in OFFENSIVE_SIGNALS:
            if signal in text_lower:
                issues.append(f"offensive_signal:{signal}")
                break

    # Quality keyword presence — relax for short technical documents (CVE/CWE < 500 chars)
    quality_kws = BUCKET_QUALITY_KEYWORDS.get(bucket_key, [])
    kw_hits = sum(1 for kw in quality_kws if kw in text_lower)
    min_kw_hits = 1 if len(text) < 500 else 2
    if kw_hits < min_kw_hits:
        issues.append("low_quality_keyword_density")

    # Repetition check (simple)
    words = text.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            issues.append("high_repetition")

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "length": len(text),
        "est_tokens": estimate_tokens(text)
    }


def validate_bucket(bucket_key: str, bucket_info: dict) -> dict:
    bucket_dir = RAW / bucket_key
    files = sorted(bucket_dir.glob("*.txt"))
    n_files = len(files)

    if n_files == 0:
        return {
            "bucket": bucket_key,
            "name": bucket_info["name"],
            "status": "FAIL",
            "reason": "No documents found",
            "n_docs": 0,
            "est_tokens": 0,
            "size_mb": 0,
            "sample_pass_rate": 0,
            "issues": ["empty_bucket"]
        }

    # Sample up to 20 random files
    sample_size = min(20, n_files)
    sample_files = random.sample(files, sample_size)

    sample_results = []
    for f in sample_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            result = check_document_quality(text, bucket_key)
            result["file"] = f.name
            result["preview"] = text[:200].replace("\n", " ")
            sample_results.append(result)
        except Exception as e:
            sample_results.append({"pass": False, "issues": [f"read_error:{e}"],
                                    "file": f.name, "length": 0, "est_tokens": 0})

    pass_count = sum(1 for r in sample_results if r["pass"])
    pass_rate = pass_count / len(sample_results) if sample_results else 0

    # Estimate total tokens (sample-based)
    avg_tokens_per_doc = (
        sum(r.get("est_tokens", 0) for r in sample_results) / len(sample_results)
        if sample_results else 0
    )
    est_total_tokens = int(avg_tokens_per_doc * n_files)

    # Disk size
    total_bytes = sum(f.stat().st_size for f in files)
    size_mb = total_bytes / 1_048_576

    # Collect all issues
    all_issues = []
    for r in sample_results:
        all_issues.extend(r.get("issues", []))

    status = "PASS" if pass_rate >= 0.80 else "WARN" if pass_rate >= 0.60 else "FAIL"

    return {
        "bucket": bucket_key,
        "name": bucket_info["name"],
        "target_ratio": bucket_info["target"],
        "status": status,
        "n_docs": n_files,
        "est_tokens": est_total_tokens,
        "size_mb": round(size_mb, 2),
        "sample_size": sample_size,
        "sample_pass_count": pass_count,
        "sample_pass_rate": round(pass_rate, 3),
        "issues_found": list(set(all_issues)),
        "sample_previews": [
            {"file": r["file"], "pass": r["pass"],
             "preview": r.get("preview", "")[:150],
             "issues": r.get("issues", [])}
            for r in sample_results[:5]  # show first 5 samples
        ]
    }


def main():
    print("\n" + "="*70)
    print("  TitanAI Corpus Quality Validator — Phase 2")
    print("="*70)

    results = {}
    total_tokens = 0

    for bucket_key, bucket_info in BUCKETS.items():
        print(f"\n  Validating {bucket_info['name']} ({bucket_key})...")
        result = validate_bucket(bucket_key, bucket_info)
        results[bucket_key] = result
        total_tokens += result["est_tokens"]

        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(result["status"], "?")
        print(f"  [{icon}] {result['status']} — {result['n_docs']} docs, "
              f"~{result['est_tokens']:,} tokens, {result['size_mb']:.1f} MB, "
              f"sample pass rate: {result['sample_pass_rate']:.0%}")
        if result.get("issues_found"):
            print(f"      Issues: {result['issues_found'][:5]}")

    # Balance check
    print(f"\n  Total estimated tokens: {total_tokens:,}")
    print("\n  --- Ratio Check ---")
    balance_issues = []
    for bucket_key, result in results.items():
        if total_tokens > 0:
            actual_ratio = result["est_tokens"] / total_tokens
        else:
            actual_ratio = 0
        target = BUCKETS[bucket_key]["target"]
        diff = abs(actual_ratio - target)
        status = "OK" if diff <= RATIO_TOLERANCE else "OFF"
        result["actual_ratio"] = round(actual_ratio, 3)
        result["ratio_diff"] = round(diff, 3)
        print(f"  {bucket_key:30s}: actual={actual_ratio:.1%}, "
              f"target={target:.0%}, diff={diff:.1%} [{status}]")
        if status == "OFF":
            balance_issues.append(f"{bucket_key}: {actual_ratio:.1%} vs target {target:.0%}")

    # Overall verdict
    print("\n  --- Overall Verdict ---")
    all_pass = all(r["status"] in ("PASS", "WARN") for r in results.values())
    any_fail = any(r["status"] == "FAIL" for r in results.values())

    if any_fail:
        print("  [✗] CORPUS QUALITY: FAIL — one or more buckets failed quality check")
        overall = "FAIL"
    elif balance_issues:
        print(f"  [⚠] CORPUS QUALITY: WARN — balance issues: {balance_issues}")
        overall = "WARN"
    else:
        print("  [✓] CORPUS QUALITY: PASS — all buckets pass quality sampling")
        overall = "PASS"

    # Save report
    report = {
        "overall_status": overall,
        "total_estimated_tokens": total_tokens,
        "balance_issues": balance_issues,
        "buckets": results
    }
    report_path = BASE / "data" / "quality_review.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Quality review saved: {report_path}")

    # Also write the reviewed_buckets list for Gate 0-B
    report["reviewed_buckets"] = list(BUCKETS.keys())
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("="*70 + "\n")
    return overall


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result in ("PASS", "WARN") else 1)
