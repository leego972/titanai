"""
Download TitanAI checkpoint from Hugging Face
===========================================

Default source:
  repo_id:  leego982/titanai
  filename: final.pt
  output:   checkpoints/final.pt

Usage:
  python scripts/download_checkpoint.py

  python scripts/download_checkpoint.py \
    --repo-id leego982/titanai \
    --filename final.pt \
    --output checkpoints/final.pt

Private Hugging Face repo:
  export HF_TOKEN=hf_xxx
  # or
  export TITAN_HF_TOKEN=hf_xxx
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = os.getenv("TITAN_HF_REPO_ID", "leego982/titanai")
DEFAULT_FILENAME = os.getenv("TITAN_HF_FILENAME", "final.pt")
DEFAULT_OUTPUT = os.getenv("TITAN_CHECKPOINT_PATH", "checkpoints/final.pt")


def download_checkpoint(repo_id: str, filename: str, output: str, revision: str | None = None) -> Path:
    token = os.getenv("TITAN_HF_TOKEN") or os.getenv("HF_TOKEN") or None
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TitanAI] Downloading checkpoint from Hugging Face: {repo_id}/{filename}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
        local_files_only=False,
    )

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != output_path.resolve():
        shutil.copyfile(downloaded_path, output_path)

    print(f"[TitanAI] Checkpoint ready: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TitanAI checkpoint from Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=os.getenv("TITAN_HF_REVISION") or None)
    args = parser.parse_args()
    download_checkpoint(args.repo_id, args.filename, args.output, args.revision)


if __name__ == "__main__":
    main()
