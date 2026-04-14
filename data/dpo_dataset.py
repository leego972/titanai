"""
TitanAI — DPO Dataset
======================
PyTorch Dataset for Direct Preference Optimization training.
Loads (prompt, chosen, rejected) triples and tokenizes them for
DPO loss computation.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

IGNORE_INDEX = -100


def _build_dpo_tokens(
    prompt: str,
    response: str,
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize a (prompt, response) pair.

    Returns:
        input_ids  : (max_seq_len,) — full sequence (prompt + response)
        labels     : (max_seq_len,) — IGNORE_INDEX for prompt tokens, token ids for response
    """
    bos_id = tokenizer.token_to_id("<bos>") or 1
    eos_id = tokenizer.token_to_id("<eos>") or 2

    prompt_ids = [bos_id] + tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids + [eos_id]

    # Truncate: keep as much response as possible
    max_prompt = max_seq_len - min(len(response_ids), max_seq_len // 2)
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    full_ids = prompt_ids + response_ids
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]

    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    labels = labels[:max_seq_len]

    # Pad to max_seq_len
    pad_len = max_seq_len - len(full_ids)
    input_ids = full_ids + [0] * pad_len
    labels = labels + [IGNORE_INDEX] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


class TitanDPODataset(Dataset):
    """
    Dataset of (prompt, chosen, rejected) preference triples for DPO.

    Each item returns:
        chosen_input_ids    : (max_seq_len,)
        chosen_labels       : (max_seq_len,)  — -100 on prompt tokens
        rejected_input_ids  : (max_seq_len,)
        rejected_labels     : (max_seq_len,)  — -100 on prompt tokens
    """

    def __init__(
        self,
        jsonl_paths: List[str],
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict] = []

        loaded = 0
        skipped = 0
        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"DPO data file not found: {p}\n"
                    f"Run: python scripts/prepare_dpo_data.py"
                )
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        assert "prompt" in item and item["prompt"]
                        assert "chosen" in item and item["chosen"]
                        assert "rejected" in item and item["rejected"]
                        self.examples.append(item)
                        loaded += 1
                    except Exception:
                        skipped += 1

        if verbose:
            print(f"[DPODataset] Loaded {loaded} preference pairs, {skipped} skipped")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        chosen_ids, chosen_labels = _build_dpo_tokens(
            prompt, chosen, self.tokenizer, self.max_seq_len
        )
        rejected_ids, rejected_labels = _build_dpo_tokens(
            prompt, rejected, self.tokenizer, self.max_seq_len
        )

        return chosen_ids, chosen_labels, rejected_ids, rejected_labels

    def get_stats(self) -> Dict:
        return {
            "total_pairs": len(self.examples),
            "sources": list({e.get("source", "unknown") for e in self.examples}),
        }
