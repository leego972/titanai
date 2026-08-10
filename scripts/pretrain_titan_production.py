#!/usr/bin/env python3
"""TitanAI 1B production pretraining entry point.

This wrapper deliberately keeps the stable v3 optimization/training loop intact,
but fixes its local-corpus train/validation membership before execution.

The v3 LocalFileSource derives split membership from a counter whose value depends
on shuffled iteration order. Separate train/validation instances use different
shuffle seeds, so a document can land in both streams. For a production run that
invalidates validation loss.

This entry point rewrites only LocalFileSource into a temporary runtime copy. The
replacement hashes stable identities:
  * text file: source + repository-relative path
  * JSONL row: source + repository-relative path + line number
Thus train and validation membership are complementary and reproducible.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "scripts" / "pretrain_titan_v3.py"

REPLACEMENT = r'''class LocalFileSource:
    """Reads local .txt/.jsonl with deterministic, leakage-free split identity."""
    def __init__(self, name, directory, want_val, shard_seed):
        import glob as _glob, os as _os
        self.name = name
        self.directory = directory
        self.want_val = want_val
        self.shard_seed = shard_seed
        all_files = (sorted(_glob.glob(_os.path.join(directory, "**", "*.txt"), recursive=True)) +
                     sorted(_glob.glob(_os.path.join(directory, "**", "*.jsonl"), recursive=True)))
        self.files = all_files
        self.ok = len(self.files) > 0
        print(f"[pretrain] LocalFileSource '{name}': {len(self.files)} files @ {directory} split={'val' if want_val else 'train'}", flush=True)

    def _member(self, stable_id):
        return _is_val(stable_id) == self.want_val

    def __iter__(self):
        if not self.ok:
            return
        import random as _rnd, json as _json, os as _os
        rng = _rnd.Random(self.shard_seed)
        files = self.files[:]
        while True:
            rng.shuffle(files)
            yielded = 0
            for f in files:
                try:
                    rel = _os.path.relpath(f, self.directory).replace(_os.sep, "/")
                    with open(f, "r", errors="replace") as fp:
                        if f.endswith(".jsonl"):
                            for lineno, line in enumerate(fp, 1):
                                stable_id = f"local:{self.name}:{rel}:{lineno}"
                                if not self._member(stable_id):
                                    continue
                                try:
                                    obj = _json.loads(line)
                                    text = (obj.get("text") or obj.get("content") or
                                            " ".join(filter(None, [obj.get("instruction", ""), obj.get("output", "")])))
                                except Exception:
                                    text = line.strip()
                                if text:
                                    yielded += 1
                                    yield (self.name, text)
                        else:
                            stable_id = f"local:{self.name}:{rel}"
                            if not self._member(stable_id):
                                continue
                            text = fp.read().strip()
                            if text:
                                yielded += 1
                                yield (self.name, text)
                except Exception as exc:
                    print(f"[pretrain] local source read error {f}: {exc}", flush=True)
                    continue
            if yielded == 0:
                raise RuntimeError(
                    f"Local source {self.name} produced zero {'validation' if self.want_val else 'training'} items. "
                    "Increase corpus size or adjust --val-hash-mod."
                )


def build_packed_iter'''


def build_runtime() -> Path:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    src = SOURCE.read_text(encoding="utf-8")

    pattern = re.compile(
        r'class LocalFileSource:.*?\ndef build_packed_iter',
        flags=re.DOTALL,
    )
    patched, count = pattern.subn(REPLACEMENT, src, count=1)
    if count != 1:
        raise RuntimeError(
            "Could not locate exactly one LocalFileSource block in pretrain_titan_v3.py; "
            "refusing to run an unverified production patch."
        )

    # Make local corpus paths portable instead of hard-coding /workspace/titanai.
    patched = patched.replace(
        '"/workspace/titanai/data/raw/corpus_A_general"',
        'os.path.join(_REPO_ROOT, "data/raw/corpus_A_general")',
    ).replace(
        '"/workspace/titanai/data/raw/corpus_B_reasoning"',
        'os.path.join(_REPO_ROOT, "data/raw/corpus_B_reasoning")',
    ).replace(
        '"/workspace/titanai/data/raw/corpus_C_technical"',
        'os.path.join(_REPO_ROOT, "data/raw/corpus_C_technical")',
    ).replace(
        '"/workspace/titanai/data/raw/corpus_D_cyber"',
        'os.path.join(_REPO_ROOT, "data/raw/corpus_D_cyber")',
    ).replace(
        '"/workspace/titanai/data/raw/corpus_E_cinema"',
        'os.path.join(_REPO_ROOT, "data/raw/corpus_E_cinema")',
    )

    runtime = BASE / "scripts" / "_pretrain_titan_production_runtime.py"
    runtime.write_text(patched, encoding="utf-8")
    return runtime


def main() -> None:
    runtime = build_runtime()
    # v3 reads sys.argv itself. Preserve all caller arguments unchanged.
    sys.argv[0] = str(runtime)
    try:
        runpy.run_path(str(runtime), run_name="__main__")
    finally:
        try:
            runtime.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
