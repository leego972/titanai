"""
Titan Dataset Loader
====================
PyTorch Dataset class that loads pre-processed shard files for training.
Supports lazy loading across multiple shards for memory efficiency.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TitanShardDataset(Dataset):
    """
    Loads tokenized shard files (.npy) from a directory.
    Each shard is a numpy array of shape (N, seq_len) with int32 token IDs.
    Returns (input_ids, labels) pairs where labels = input_ids shifted by 1
    (standard causal language modeling objective).
    """

    def __init__(self, shard_dir: str, max_seq_len: int, pad_id: int = 0):
        self.shard_dir = shard_dir
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.npy")))
        if not shard_files:
            raise FileNotFoundError(f"No .npy shard files found in {shard_dir}")

        # Load all shards into memory (fine for small datasets)
        # For large datasets, replace with lazy per-shard loading
        all_data = []
        for sf in shard_files:
            data = np.load(sf)
            all_data.append(data)
        self.data = np.concatenate(all_data, axis=0)
        print(f"[Dataset] Loaded {len(self.data)} sequences from {len(shard_files)} shard(s) in {shard_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        # Input: all tokens except the last
        # Labels: all tokens except the first (shifted by 1)
        input_ids = seq[:-1]
        labels = seq[1:].clone()
        # Mask padding tokens in labels so they don't contribute to loss
        labels[labels == self.pad_id] = -100
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
    val_dataset = TitanShardDataset(val_dir, max_seq_len, pad_id)

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
