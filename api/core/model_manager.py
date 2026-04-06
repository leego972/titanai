"""
TitanAI API — Model Manager
=============================
Singleton that loads and manages the TitanAI model checkpoint.
Provides thread-safe generation with a lock to prevent concurrent GPU access.
"""
import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, AsyncGenerator
import threading

import torch
import yaml

# Add titanai root to path
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

from inference.infer import TitanInference
from .config import config

log = logging.getLogger("titan.model_manager")

_STARTUP_TIME = time.time()


class ModelManager:
    """
    Singleton model manager. Loads TitanAI checkpoint once at startup
    and provides async-safe generate() and stream_generate() methods.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model: Optional[TitanInference] = None
        self._model_lock = asyncio.Lock()
        self._loaded_checkpoint: Optional[str] = None
        self._loaded_config_path: Optional[str] = None
        self._model_info: dict = {}
        self._generation_lock = threading.Lock()

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> bool:
        """
        Load the model. Returns True on success, False if checkpoint not found.
        Safe to call multiple times — reloads if paths differ.
        """
        cfg_path = config_path or config.CONFIG_PATH
        ckpt_path = checkpoint_path or config.CHECKPOINT_PATH
        dev = device or config.DEVICE

        if not Path(ckpt_path).exists():
            log.warning(f"[ModelManager] Checkpoint not found: {ckpt_path}")
            log.warning("[ModelManager] API will start without a loaded model.")
            log.warning("[ModelManager] POST /v1/models/load once training completes.")
            return False

        if not Path(cfg_path).exists():
            log.error(f"[ModelManager] Config not found: {cfg_path}")
            return False

        try:
            log.info(f"[ModelManager] Loading config: {cfg_path}")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            # Resolve device
            if dev == "auto":
                resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = dev

            log.info(f"[ModelManager] Loading checkpoint: {ckpt_path}")
            log.info(f"[ModelManager] Device: {resolved_device}")

            self._model = TitanInference(
                config=cfg,
                checkpoint_path=ckpt_path,
                base_dir=str(BASE),
                device=resolved_device,
            )

            self._loaded_checkpoint = ckpt_path
            self._loaded_config_path = cfg_path

            # Extract model info
            n_params = sum(p.numel() for p in self._model.model.parameters())
            self._model_info = {
                "parameters": n_params,
                "device": resolved_device,
                "checkpoint": ckpt_path,
                "config": cfg_path,
                "vocab_size": cfg["model"]["vocab_size"],
                "d_model": cfg["model"]["d_model"],
                "n_layers": cfg["model"]["n_layers"],
                "n_heads": cfg["model"]["n_heads"],
                "max_seq_len": cfg["model"]["max_seq_len"],
            }

            # Try to load training step from checkpoint
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                self._model_info["training_step"] = ckpt.get("step", None)
                self._model_info["train_loss"] = ckpt.get("loss", None)
            except Exception:
                pass

            log.info(f"[ModelManager] Model loaded. {n_params:,} parameters on {resolved_device}.")
            return True

        except Exception as e:
            log.error(f"[ModelManager] Failed to load model: {e}", exc_info=True)
            self._model = None
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        if self._model is None:
            return "none"
        return str(self._model.device)

    @property
    def checkpoint(self) -> Optional[str]:
        return self._loaded_checkpoint

    @property
    def model_info(self) -> dict:
        return self._model_info

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> tuple[str, int, int]:
        """
        Generate text. Returns (generated_text, prompt_tokens, generated_tokens).
        Thread-safe via generation lock.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. POST /v1/models/load first.")

        max_new_tokens = min(
            max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS,
            config.MAX_NEW_TOKENS_LIMIT,
        )
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        top_p = top_p if top_p is not None else config.DEFAULT_TOP_P

        with self._generation_lock:
            # Count prompt tokens
            encoded = self._model.tokenizer.encode(prompt)
            prompt_tokens = len(encoded.ids)

            generated = self._model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Count generated tokens
            gen_encoded = self._model.tokenizer.encode(generated)
            generated_tokens = len(gen_encoded.ids)

        return generated, prompt_tokens, generated_tokens

    async def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream generation token by token using asyncio.
        Runs the blocking generation in a thread pool.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        max_new_tokens = min(
            max_new_tokens or config.DEFAULT_MAX_NEW_TOKENS,
            config.MAX_NEW_TOKENS_LIMIT,
        )
        temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        top_k = top_k if top_k is not None else config.DEFAULT_TOP_K
        top_p = top_p if top_p is not None else config.DEFAULT_TOP_P

        # For true streaming we need token-by-token generation
        # The current TitanInference.generate() returns the full string
        # We simulate streaming by running full generation in a thread and yielding chunks
        loop = asyncio.get_event_loop()

        def _run():
            with self._generation_lock:
                return self._model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

        generated = await loop.run_in_executor(None, _run)

        # Yield word by word for streaming effect
        # TODO: implement true token-by-token streaming in TitanInference
        words = generated.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk
            await asyncio.sleep(0)  # yield control

    def get_uptime(self) -> float:
        return time.time() - _STARTUP_TIME

    def list_checkpoints(self) -> list[dict]:
        """Scan checkpoint directory and return available checkpoints."""
        ckpt_dir = Path(config.CHECKPOINT_DIR)
        checkpoints = []

        # Load run summary if available
        run_summary = {}
        summary_path = BASE / "logs" / "probe_v015" / "run_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    run_summary = json.load(f)
            except Exception:
                pass

        for pt_file in sorted(ckpt_dir.rglob("*.pt")):
            ckpt_info = {
                "id": pt_file.stem,
                "path": str(pt_file),
                "size_mb": round(pt_file.stat().st_size / 1_048_576, 1),
                "created": int(pt_file.stat().st_mtime),
                "is_loaded": str(pt_file) == self._loaded_checkpoint,
            }

            # Try to get step from filename (e.g. step_002000.pt)
            if pt_file.stem.startswith("step_"):
                try:
                    ckpt_info["training_step"] = int(pt_file.stem.split("_")[1])
                except (IndexError, ValueError):
                    pass

            # Attach final metrics if this is the final checkpoint
            if pt_file.stem == "final" and run_summary:
                ckpt_info["val_perplexity"] = run_summary.get("final_val_ppl")
                ckpt_info["val_loss"] = run_summary.get("final_val_loss")
                ckpt_info["training_step"] = run_summary.get("steps_completed")

            checkpoints.append(ckpt_info)

        return checkpoints


# Global singleton
manager = ModelManager.get()
