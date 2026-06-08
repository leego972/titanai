"""
Titan Dataset Loader
====================
PyTorch Dataset class that loads pre-processed shard files for training.
Shards are flat uint16 binary files (nanoGPT format) — each file is a
1-D array of token IDs written with numpy.ndarray.tofile().

Sequences of length (max_seq_len + 1) are sliced from the concatenated
token stream.  The +1 extra token lets __getitem__ return:
  input_ids = seq[:-1]   (max_seq_len tokens)
  labels    = seq[1:]    (max_seq_len tokens, shifted by 1)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TitanShardDataset(Dataset):
    """
    Loads tokenized shard files (.bin, flat uint16) from a directory.
    Each shard is a raw binary of uint16 token IDs written by
    numpy.ndarray.tofile() — the nanoGPT / Karpathy format.

    Sequences of (max_seq_len + 1) tokens are extracted; the dataset
    returns (input_ids, labels) pairs for causal language modelling.
    """

    def __init__(self, shard_dir: str, max_seq_len: int, pad_id: int = 0):
        self.shard_dir   = shard_dir
        self.max_seq_len = max_seq_len
        self.pad_id      = pad_id
        self.stride      = max_seq_len + 1   # tokens per sequence slot

        shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))
        if not shard_files:
            raise FileNotFoundError(
                f"No .bin shard files found in {shard_dir}\n"
                f"Run: python3 scripts/prep_local_corpus.py  (tokenises data/raw/)")

        all_seqs = []
        for sf in shard_files:
            # flat uint16 → int32 (token IDs can exceed int16 range at 32 k vocab)
            raw = np.fromfile(sf, dtype=np.uint16).astype(np.int32)
            n   = len(raw) // self.stride
            if n == 0:
                continue
            raw = raw[: n * self.stride].reshape(n, self.stride)
            all_seqs.append(raw)

        if not all_seqs:
            raise ValueError(f"All shards in {shard_dir} are too short (<{self.stride} tokens)")

        self.data = np.concatenate(all_seqs, axis=0)
        print(f"[Dataset] Loaded {len(self.data):,} sequences "
              f"from {len(shard_files)} shard(s) in {shard_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seq       = torch.tensor(self.data[idx], dtype=torch.long)
        input_ids = seq[:-1]          # first max_seq_len tokens
        labels    = seq[1:].clone()   # shifted by 1 (next-token prediction)
        labels[labels == self.pad_id] = -100   # mask padding from loss
        return input_ids, labels


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    max_seq_len: int,
    batch_size: int,
    val_batch_size: int,
    num_workers: int = 0,
    pad_id: int = 0,
):
    """Create train and validation DataLoaders."""
    train_dataset = TitanShardDataset(train_dir, max_seq_len, pad_id)
    val_dataset   = TitanShardDataset(val_dir,   max_seq_len, pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
