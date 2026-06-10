"""
Titan SFT Dataset
=================
Loads instruction fine-tuning examples in chat JSONL format and formats them
for supervised fine-tuning with prompt masking.

Format expected in JSONL files:
  {"messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Prompt masking: loss is computed ONLY on assistant tokens.
System + user tokens are masked with IGNORE_INDEX (-100).
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional

# Sentinel value — PyTorch ignores positions set to this in cross-entropy loss
IGNORE_INDEX = -100

# Chat template tokens — must match exactly what the tokenizer was trained with.
# Titan uses simple delimiter tokens. Update if your tokenizer uses different ones.
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"


def format_chat_as_text(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of chat messages into a single string for tokenization.

    Format:
        <bos>System: {system}<sep>User: {user}<sep>Assistant: {assistant}<eos>

    The SEP token marks boundaries between turns. The EOS token marks the
    end of the full sequence. This format is simple and works with any BPE
    tokenizer that has these special tokens.
    """
    parts = [BOS]
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        parts.append(f"{role}: {content}")
        parts.append(SEP)
    # Replace trailing SEP with EOS to mark end of sequence
    if parts and parts[-1] == SEP:
        parts[-1] = EOS
    return "".join(f"{p}" if p in (BOS, EOS, SEP) else p for p in parts)


def build_labels_with_prompt_mask(
    input_ids: List[int],
    tokenizer,
    messages: List[Dict[str, str]],
) -> List[int]:
    """
    Build a labels tensor where all tokens EXCEPT the assistant's response
    are set to IGNORE_INDEX (-100). This ensures the loss is computed only
    on the tokens Titan should learn to predict.

    Strategy:
        1. Tokenize the full sequence to get input_ids.
        2. Tokenize the prompt (system + user) to find where it ends.
        3. Mask everything up to and including the prompt.
        4. The assistant response tokens remain unmasked (carry their real token IDs).
    """
    labels = list(input_ids)

    # Build the prompt-only string (everything before the assistant turn)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = format_chat_as_text(prompt_messages)
    # Remove trailing EOS that format_chat_as_text adds — the prompt isn't complete yet
    if prompt_text.endswith(EOS):
        prompt_text = prompt_text[: -len(EOS)] + SEP

    prompt_ids = tokenizer.encode(prompt_text).ids
    prompt_len = len(prompt_ids)

    # Mask prompt tokens
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX

    return labels


class TitanSFTDataset(Dataset):
    """
    Dataset for Titan instruction fine-tuning.

    Loads one or more JSONL files, each containing chat examples in the format:
        {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

    Each example is tokenized, truncated to max_seq_len, and returned with a
    labels tensor where prompt tokens are masked.
    """

    def __init__(
        self,
        jsonl_paths: List[str],
        tokenizer,
        max_seq_len: int = 2048,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict] = []

        total_loaded = 0
        total_skipped = 0

        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                if verbose:
                    print(f"[SFTDataset] WARNING: {path} not found, skipping.")
                continue

            with open(p, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        messages = example.get("messages", [])
                        if not messages:
                            total_skipped += 1
                            continue
                        # Verify required roles present
                        roles = {m["role"] for m in messages}
                        if "assistant" not in roles:
                            if verbose:
                                print(f"[SFTDataset] Skipping example at {path}:{line_num} — no assistant turn")
                            total_skipped += 1
                            continue
                        self.examples.append({"messages": messages, "source": str(p.name)})
                        total_loaded += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        if verbose:
                            print(f"[SFTDataset] Parse error at {path}:{line_num}: {e}")
                        total_skipped += 1

        if verbose:
            print(f"[SFTDataset] Loaded {total_loaded} examples ({total_skipped} skipped) from {len(jsonl_paths)} file(s)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        messages = example["messages"]

        # Format as text
        text = format_chat_as_text(messages)

        # Tokenize
        token_ids = self.tokenizer.encode(text).ids

        # Truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        # Build labels with prompt mask
        labels = build_labels_with_prompt_mask(token_ids, self.tokenizer, messages)

        # Truncate labels to same length
        labels = labels[: self.max_seq_len]

        # Pad if needed (rare — usually we just truncate)
        pad_len = self.max_seq_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def get_stats(self) -> Dict:
        """Return dataset composition statistics."""
        from collections import Counter
        sources = Counter(e["source"] for e in self.examples)
        return {
            "total": len(self.examples),
            "by_source": dict(sources),
        }
