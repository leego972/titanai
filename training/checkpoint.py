"""
Titan Checkpoint System
=======================
Saves and loads full training state including:
    - Model weights
    - Optimizer state
    - LR scheduler state
    - Current training step
    - Full config snapshot

This ensures any interrupted training run can be resumed exactly
from where it stopped, with no progress loss.

Save:   save_checkpoint(path, model, optimizer, scheduler, step, config)
Load:   step = load_checkpoint(path, model, optimizer, scheduler, device)
"""

import os
import json
import torch
from typing import Optional


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: dict,
):
    """
    Save a full training checkpoint to disk.
    The checkpoint contains everything needed to resume training exactly.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    # Save to a temp file first, then rename — prevents corruption on interruption
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Load a checkpoint from disk and restore model/optimizer/scheduler state.
    Returns the step number stored in the checkpoint.

    If optimizer/scheduler are None, only model weights are restored
    (useful for inference-only loading).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    return step


def list_checkpoints(checkpoint_dir: str) -> list[str]:
    """List all checkpoint files in a directory, sorted by step number."""
    import glob
    ckpts = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    # Sort by step number
    def get_step(p):
        try:
            return int(os.path.basename(p).replace("step_", "").replace(".pt", ""))
        except ValueError:
            return 0
    ckpts.sort(key=get_step)
    return ckpts


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path to the most recent checkpoint, or None if none exist."""
    ckpts = list_checkpoints(checkpoint_dir)
    if not ckpts:
        # Check for final.pt
        final = os.path.join(checkpoint_dir, "final.pt")
        if os.path.exists(final):
            return final
        return None
    return ckpts[-1]
