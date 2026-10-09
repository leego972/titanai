#!/usr/bin/env python3
"""TitanAI 1B production pretraining entry point.

Keeps the proven v3 optimization/checkpoint loop, while repairing data quality
and validation semantics for the production 1B run.

Production changes applied to the runtime copy:
1. Deterministic leakage-free train/validation membership for BOTH streamed and
   local sources. Membership hashes document content/identity rather than the
   shuffled iteration counter.
2. Adds large, verified pretraining sources so a 20B-token run does not recycle
   Titan's ~100M-token local corpus hundreds of times:
      - HuggingFaceFW/fineweb-edu, sample-10BT
      - open-web-math/open-web-math
3. Retains Titan's local specialist corpus for technical, cyber, cinema,
   reasoning and project-specific distribution shaping.
4. Makes local paths relative to the repository instead of /workspace.

The temporary runtime file is deleted when training exits.
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "scripts" / "pretrain_titan_v3.py"

# Total weights intentionally sum to 1.0.
# 75% high-diversity streamed corpus; 25% Titan specialist/local corpus.
STREAM_SOURCES = '''SOURCES_DEFAULT = [
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text", 0.55),
    ("openwebmath", "open-web-math/open-web-math", None, "text", 0.20),
]'''

LOCAL_SOURCES = '''LOCAL_CORPUS = [
    ("local_general",  os.path.join(_REPO_ROOT, "data/raw/corpus_A_general"),   0.03),
    ("local_reason",   os.path.join(_REPO_ROOT, "data/raw/corpus_B_reasoning"), 0.04),
    ("local_tech",     os.path.join(_REPO_ROOT, "data/raw/corpus_C_technical"), 0.08),
    ("local_cyber",    os.path.join(_REPO_ROOT, "data/raw/corpus_D_cyber"),     0.06),
    ("local_cinema",   os.path.join(_REPO_ROOT, "data/raw/corpus_E_cinema"),    0.04),
]'''

SOURCE_AND_LOCAL_CLASSES = r'''class SourceStream:
    """HF streaming source with deterministic content-hash train/val split."""
    def __init__(self, name, repo, cfg, text_field, want_val, shard_seed):
        self.name = name
        self.text_field = text_field
        self.want_val = want_val
        try:
            if cfg:
                self.ds = load_dataset(repo, cfg, split="train", streaming=True)
            else:
                self.ds = load_dataset(repo, split="train", streaming=True)
            # Shuffle affects presentation order only, never split membership.
            self.ds = self.ds.shuffle(seed=args.seed + shard_seed, buffer_size=10000)
            self.ok = True
        except Exception as exc:
            print(f"[pretrain] !!! source '{name}' failed to init: {exc}", flush=True)
            self.ok = False
            self.ds = None

    def __iter__(self):
        if not self.ok:
            return
        import time as _time
        err_count = 0
        ds_iter = iter(self.ds)
        while True:
            try:
                example = next(ds_iter)
                err_count = 0
            except StopIteration:
                ds_iter = iter(self.ds)
                continue
            except Exception as exc:
                err_count += 1
                wait = min(err_count * 3, 60)
                print(f"[pretrain] source '{self.name}' error #{err_count}: {exc} — retry in {wait}s", flush=True)
                if err_count > 30:
                    print(f"[pretrain] source '{self.name}' giving up after 30 errors", flush=True)
                    self.ok = False
                    return
                _time.sleep(wait)
                try:
                    ds_iter = iter(self.ds)
                except Exception:
                    pass
                continue

            text = example.get(self.text_field) or example.get("text") or example.get("content") or ""
            if not text:
                continue
            # The same content always belongs to exactly one split regardless of
            # shuffle order, restarts, worker count or source position.
            stable_id = "hf:" + self.name + ":" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            if _is_val(stable_id) != self.want_val:
                continue
            yield (self.name, text)


class LocalFileSource:
    """Reads local .txt/.jsonl with deterministic, leakage-free split identity."""
    def __init__(self, name, directory, want_val, shard_seed):
        import glob as _glob, os as _os
        self.name = name
        self.directory = directory
        self.want_val = want_val
        self.shard_seed = shard_seed
        self.files = (sorted(_glob.glob(_os.path.join(directory, "**", "*.txt"), recursive=True)) +
                      sorted(_glob.glob(_os.path.join(directory, "**", "*.jsonl"), recursive=True)))
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

    # Replace intentionally disabled HF source list.
    patched, source_count = re.subn(
        r'SOURCES_DEFAULT\s*=\s*\[\]\s*#.*',
        STREAM_SOURCES,
        src,
        count=1,
    )
    if source_count != 1:
        raise RuntimeError("Could not patch SOURCES_DEFAULT exactly once; refusing production run.")

    # Replace both stream implementations so split identity is stable.
    pattern = re.compile(
        r'class SourceStream:.*?\ndef build_packed_iter',
        flags=re.DOTALL,
    )
    patched, class_count = pattern.subn(SOURCE_AND_LOCAL_CLASSES, patched, count=1)
    if class_count != 1:
        raise RuntimeError("Could not patch source classes exactly once; refusing production run.")

    # Replace the local weighting/path block. Match through the closing list just
    # before local_ok, keeping the rest of the stable v3 loop unchanged.
    local_pattern = re.compile(
        r'LOCAL_CORPUS\s*=\s*\[.*?\]\nlocal_ok\s*=\s*\[\]',
        flags=re.DOTALL,
    )
    patched, local_count = local_pattern.subn(LOCAL_SOURCES + '\nlocal_ok = []', patched, count=1)
    if local_count != 1:
        raise RuntimeError("Could not patch LOCAL_CORPUS exactly once; refusing production run.")

    runtime = BASE / "scripts" / "_pretrain_titan_production_runtime.py"
    runtime.write_text(patched, encoding="utf-8")
    return runtime


def main() -> None:
    runtime = build_runtime()
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
