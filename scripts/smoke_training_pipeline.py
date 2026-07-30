#!/usr/bin/env python3
"""Run a CPU-sized end-to-end Titan training pipeline smoke test.

The test exercises the production code paths without production-scale weights:
1. Build a temporary tokenizer.
2. Create a tiny source checkpoint.
3. Expand source depth with scripts/upscale_to_3b.py.
4. Strict-load and forward-pass the expanded checkpoint.
5. Run one SFT optimizer step through scripts/run_sft_v2.py.
6. Run one DPO optimizer step through scripts/run_dpo.py.
7. Strict-load and forward-pass the final DPO checkpoint.

No repository datasets or checkpoints are modified. Outputs are written under
artifacts/smoke_training_pipeline and may be deleted after inspection.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "artifacts" / "smoke_training_pipeline"
TOKENIZER_PATH = OUT / "tokenizer.json"
SFT_DATA = OUT / "sft.jsonl"
DPO_DATA = OUT / "dpo.jsonl"
SRC_CONFIG = OUT / "tiny_source.yaml"
DST_CONFIG = OUT / "tiny_expanded.yaml"
SFT_CONFIG = OUT / "tiny_sft.yaml"
DPO_CONFIG = OUT / "tiny_dpo.yaml"
SRC_CKPT = OUT / "source.pt"
EXPANDED_CKPT = OUT / "expanded.pt"
SFT_OUT = OUT / "sft_checkpoints"
DPO_OUT = OUT / "dpo_checkpoints"

SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>",
    "<|system|>", "<|user|>", "<|assistant|>",
]


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("[smoke] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, check=True)


def build_tokenizer() -> int:
    corpus = [
        "system user assistant build inspect test repair validate security network code",
        "create a small service and verify the expected result",
        "chosen response is complete correct tested and concise",
        "rejected response is incomplete incorrect and fails validation",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=128, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    return tokenizer.get_vocab_size()


def architecture(vocab_size: int, layers: int) -> dict:
    return {
        "architecture": "decoder_transformer",
        "vocab_size": vocab_size,
        "d_model": 64,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_layers": layers,
        "d_ff": 192,
        "max_seq_len": 64,
        "rope_base": 10000.0,
        "dropout": 0.0,
        "tie_embeddings": True,
        "gradient_checkpointing": False,
    }


def common_config(model: dict) -> dict:
    return {
        "project": {"name": "titan-smoke", "version": "1.0"},
        "model": model,
        "tokenizer": {"path": str(TOKENIZER_PATH), "vocab_size": model["vocab_size"]},
        "data": {"tokenizer_path": str(TOKENIZER_PATH), "max_seq_len": 64, "num_workers": 0},
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "weight_decay": 0.0,
            "max_steps": 1,
            "warmup_steps": 0,
            "lr_scheduler": "cosine",
            "lr_min_ratio": 0.1,
            "clip_grad_norm": 1.0,
            "log_interval": 1,
            "eval_interval": 1,
            "save_interval": 1,
            "checkpoint_dir": str(OUT / "unused"),
        },
        "evaluation": {"val_batch_size": 2, "num_eval_batches": 1},
        "logging": {"log_dir": str(OUT / "logs"), "experiment_name": "titan-smoke"},
        "inference": {"default_max_new_tokens": 8, "default_temperature": 0.7},
    }


def build_data() -> None:
    with SFT_DATA.open("w", encoding="utf-8") as handle:
        for i in range(24):
            record = {
                "messages": [
                    {"role": "user", "content": f"Build and validate test component {i}."},
                    {"role": "assistant", "content": f"Component {i} is implemented, tested, and validated."},
                ]
            }
            handle.write(json.dumps(record) + "\n")

    with DPO_DATA.open("w", encoding="utf-8") as handle:
        for i in range(24):
            record = {
                "prompt": f"Review test component {i}.",
                "chosen": f"Component {i} was inspected, tested, and the result was verified.",
                "rejected": f"Component {i} is probably fine.",
            }
            handle.write(json.dumps(record) + "\n")


def strict_forward(checkpoint: Path, config_path: Path) -> None:
    from model.titan_model import TitanConfig, TitanLM

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = TitanLM(TitanConfig.from_dict(cfg))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, cfg["model"]["vocab_size"], (1, 12)))
    expected = (1, 12, cfg["model"]["vocab_size"])
    if tuple(logits.shape) != expected:
        raise AssertionError(f"unexpected logits shape {tuple(logits.shape)} != {expected}")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    vocab_size = build_tokenizer()
    build_data()

    source_model = architecture(vocab_size, layers=2)
    expanded_model = architecture(vocab_size, layers=4)

    source_cfg = common_config(source_model)
    expanded_cfg = common_config(expanded_model)
    write_yaml(SRC_CONFIG, source_cfg)
    write_yaml(DST_CONFIG, expanded_cfg)

    from model.titan_model import TitanConfig, TitanLM

    source = TitanLM(TitanConfig.from_dict(source_cfg))
    torch.save({"model_state_dict": source.state_dict(), "config": source_cfg, "step": 0}, SRC_CKPT)

    run(
        "scripts/upscale_to_3b.py",
        "--src_checkpoint", str(SRC_CKPT),
        "--src_config", str(SRC_CONFIG),
        "--dst_config", str(DST_CONFIG),
        "--dst_checkpoint", str(EXPANDED_CKPT),
    )
    strict_forward(EXPANDED_CKPT, DST_CONFIG)

    sft_cfg = common_config(expanded_model)
    sft_cfg["data"].update({"sft_files": [str(SFT_DATA)], "val_split": 0.2})
    sft_cfg["training"]["checkpoint_dir"] = str(SFT_OUT)
    sft_cfg["logging"] = {"log_dir": str(OUT / "logs_sft"), "experiment_name": "smoke-sft"}
    write_yaml(SFT_CONFIG, sft_cfg)

    run(
        "scripts/run_sft_v2.py",
        "--config", str(SFT_CONFIG),
        "--checkpoint", str(EXPANDED_CKPT),
        "--out-dir", str(SFT_OUT),
    )
    sft_final = SFT_OUT / "final.pt"
    strict_forward(sft_final, SFT_CONFIG)

    dpo_cfg = common_config(expanded_model)
    dpo_cfg["data"].update({"dpo_files": [str(DPO_DATA)], "val_split": 0.2})
    dpo_cfg["training"].update({"checkpoint_dir": str(DPO_OUT), "beta": 0.1})
    dpo_cfg["logging"] = {"log_dir": str(OUT / "logs_dpo"), "experiment_name": "smoke-dpo"}
    write_yaml(DPO_CONFIG, dpo_cfg)

    run(
        "scripts/run_dpo.py",
        "--config", str(DPO_CONFIG),
        "--checkpoint", str(sft_final),
        "--out-dir", str(DPO_OUT),
    )
    dpo_final = DPO_OUT / "final.pt"
    strict_forward(dpo_final, DPO_CONFIG)

    manifest = {
        "status": "passed",
        "vocab_size": vocab_size,
        "source_layers": 2,
        "expanded_layers": 4,
        "sft_checkpoint": str(sft_final.relative_to(BASE)),
        "dpo_checkpoint": str(dpo_final.relative_to(BASE)),
    }
    (OUT / "result.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n[smoke] PASSED —", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
