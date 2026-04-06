"""
Titan Inference Runtime
=======================
Load a trained Titan checkpoint and run interactive text generation.
Supports both CLI interactive mode and single-prompt mode.

Usage (interactive):
    python scripts/infer.py --config configs/titan_config.yaml --checkpoint checkpoints/final.pt

Usage (single prompt):
    python scripts/infer.py --config configs/titan_config.yaml --checkpoint checkpoints/final.pt \
        --prompt "The vulnerability was discovered" --max-tokens 150

Usage (as a module):
    from inference.infer import TitanInference
    titan = TitanInference(config, checkpoint_path)
    response = titan.generate("Hello, Titan.")
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.titan_model import TitanLM, build_model
from training.checkpoint import load_checkpoint
from tokenizer.train_tokenizer import load_tokenizer


class TitanInference:
    """
    Lightweight inference wrapper for TitanLM.
    Loads a checkpoint and provides a simple .generate() interface.
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        base_dir: str = ".",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[Titan] Loading model on {self.device}...")

        # Load tokenizer
        tok_dir = os.path.join(base_dir, config["tokenizer"]["save_dir"])
        self.tokenizer = load_tokenizer(tok_dir)
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")

        # Load model
        self.model = build_model(config).to(self.device)
        step = load_checkpoint(checkpoint_path, self.model, device=self.device)
        self.model.eval()
        print(f"[Titan] Model loaded from step {step}. Ready.")

        # Default generation settings from config
        inf_cfg = config.get("inference", {})
        self.default_max_new_tokens = inf_cfg.get("default_max_new_tokens", 200)
        self.default_temperature = inf_cfg.get("default_temperature", 0.8)
        self.default_top_k = inf_cfg.get("default_top_k", 50)
        self.default_top_p = inf_cfg.get("default_top_p", 0.95)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate text from a prompt string.
        Returns only the newly generated text (not the prompt).
        """
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        temperature = temperature if temperature is not None else self.default_temperature
        top_k = top_k if top_k is not None else self.default_top_k
        top_p = top_p if top_p is not None else self.default_top_p

        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(
            [[self.bos_id] + encoded.ids],
            dtype=torch.long,
            device=self.device,
        )

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=self.eos_id,
        )

        # Decode only new tokens
        new_ids = output_ids[0, input_ids.shape[1]:].tolist()
        if self.eos_id in new_ids:
            new_ids = new_ids[:new_ids.index(self.eos_id)]
        return self.tokenizer.decode(new_ids)


def interactive_mode(titan: TitanInference):
    """Run an interactive CLI session with Titan."""
    print("\n" + "=" * 60)
    print("  TITAN — Base Model Inference")
    print("  Type your prompt and press Enter.")
    print("  Commands: :quit, :temp <float>, :topk <int>, :tokens <int>")
    print("=" * 60 + "\n")

    temperature = titan.default_temperature
    top_k = titan.default_top_k
    max_new_tokens = titan.default_max_new_tokens

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Titan] Exiting.")
            break

        if not prompt:
            continue
        if prompt == ":quit":
            break
        if prompt.startswith(":temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"[Config] Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("[Config] Usage: :temp <float>")
            continue
        if prompt.startswith(":topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"[Config] Top-k set to {top_k}")
            except (IndexError, ValueError):
                print("[Config] Usage: :topk <int>")
            continue
        if prompt.startswith(":tokens "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"[Config] Max new tokens set to {max_new_tokens}")
            except (IndexError, ValueError):
                print("[Config] Usage: :tokens <int>")
            continue

        response = titan.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        print(f"Titan: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Run Titan inference")
    parser.add_argument("--config", default="configs/titan_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    titan = TitanInference(config, args.checkpoint, base_dir=args.base_dir)

    if args.prompt:
        # Single-shot mode
        response = titan.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"Prompt:    {args.prompt}")
        print(f"Generated: {response}")
    else:
        # Interactive mode
        interactive_mode(titan)


if __name__ == "__main__":
    main()
