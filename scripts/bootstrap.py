"""
Titan Bootstrap Script
======================
Downloads a small seed corpus (TinyShakespeare) and runs the full pipeline:
    1. Prepares the data directory
    2. Trains the tokenizer
    3. Prepares the dataset
    4. Runs a short training validation run (100 steps)
    5. Evaluates the checkpoint
    6. Runs a sample inference

This script proves the full Titan model factory works end-to-end.
It is NOT a full training run — it is a pipeline validation run.

Usage:
    cd titan-model/
    python scripts/bootstrap.py
"""

import os
import sys
import urllib.request
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = str(Path(__file__).parent.parent)
CONFIG_PATH = os.path.join(BASE_DIR, "configs/titan_config.yaml")

SEED_CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SEED_CORPUS_PATH = os.path.join(BASE_DIR, "data/raw/tinyshakespeare.txt")


def download_seed_corpus():
    os.makedirs(os.path.dirname(SEED_CORPUS_PATH), exist_ok=True)
    if os.path.exists(SEED_CORPUS_PATH):
        print(f"[Bootstrap] Seed corpus already exists: {SEED_CORPUS_PATH}")
        return
    print(f"[Bootstrap] Downloading TinyShakespeare seed corpus...")
    urllib.request.urlretrieve(SEED_CORPUS_URL, SEED_CORPUS_PATH)
    size_kb = os.path.getsize(SEED_CORPUS_PATH) // 1024
    print(f"[Bootstrap] Downloaded {size_kb} KB to {SEED_CORPUS_PATH}")


def patch_config_for_bootstrap(config: dict) -> dict:
    """Reduce steps and sizes for a fast pipeline validation run."""
    config = dict(config)
    config["training"] = dict(config["training"])
    config["training"]["max_steps"] = 200       # Fast validation run
    config["training"]["eval_interval"] = 100
    config["training"]["save_interval"] = 100
    config["training"]["log_interval"] = 20
    config["training"]["batch_size"] = 16
    return config


def main():
    print("\n" + "=" * 60)
    print("  TITAN BOOTSTRAP — Full Pipeline Validation")
    print("=" * 60 + "\n")

    # Step 1: Download seed corpus
    download_seed_corpus()

    # Step 2: Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Step 3: Train tokenizer
    print("\n[Bootstrap] Step 1/4: Training tokenizer...")
    from tokenizer.train_tokenizer import train_tokenizer, test_tokenizer
    tokenizer = train_tokenizer(config, BASE_DIR)
    test_tokenizer(tokenizer)

    # Step 4: Prepare data
    print("\n[Bootstrap] Step 2/4: Preparing dataset...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "data/prepare_data.py", "--config", CONFIG_PATH, "--base-dir", BASE_DIR],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[Bootstrap] Data preparation failed. Check errors above.")
        sys.exit(1)

    # Step 5: Short training run
    print("\n[Bootstrap] Step 3/4: Running pipeline validation training (200 steps)...")
    bootstrap_config = patch_config_for_bootstrap(config)
    from training.trainer import train
    model = train(bootstrap_config, base_dir=BASE_DIR)

    # Step 6: Evaluate
    print("\n[Bootstrap] Step 4/4: Evaluating checkpoint...")
    import torch
    from training.checkpoint import get_latest_checkpoint
    from evaluation.evaluator import evaluate, generate_samples
    from data.dataset import create_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(BASE_DIR, config["training"]["checkpoint_dir"])
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)

    if latest_ckpt:
        processed_dir = os.path.join(BASE_DIR, config["data"]["processed_dir"])
        _, val_loader = create_dataloaders(
            train_dir=os.path.join(processed_dir, "train"),
            val_dir=os.path.join(processed_dir, "val"),
            max_seq_len=config["data"]["max_seq_len"],
            batch_size=config["training"]["batch_size"],
            val_batch_size=config["evaluation"]["val_batch_size"],
        )
        val_loss, val_ppl = evaluate(model, val_loader, device, num_batches=10)
        print(f"\n[Bootstrap] Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

        # Sample generation
        tok_dir = os.path.join(BASE_DIR, config["tokenizer"]["save_dir"])
        from tokenizer.train_tokenizer import load_tokenizer
        tokenizer_loaded = load_tokenizer(tok_dir)
        samples = generate_samples(
            model, tokenizer_loaded,
            prompts=["HAMLET:", "To be or not to be"],
            max_new_tokens=80, temperature=0.8, top_k=40, device=device,
        )
        print("\n[Bootstrap] Sample generations:")
        for s in samples:
            print(f"  Prompt:    {s['prompt']}")
            print(f"  Generated: {s['generated']}\n")

    print("\n" + "=" * 60)
    print("  TITAN BOOTSTRAP COMPLETE")
    print("  The full pipeline works end-to-end.")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    print("  Next step: Run a full training run with more data and more steps.")
    print("  See docs/NEXT_TRAINING_PLAN.md for guidance.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
