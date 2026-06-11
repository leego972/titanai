"""
Titan SFT Dataset
=================
Loads instruction fine-tuning examples and formats them for supervised
fine-tuning with prompt masking.

Supported JSONL formats:
  1. Chat format (messages):
       {"messages": [{"role": "system|user|assistant", "content": "..."}]}

  2. Alpaca / Dolly format (instruction + response):
       {"instruction": "...", "input": "...", "response": "...", "source": "..."}
       {"instruction": "...", "output":  "...", "context": "..."}
       {"prompt": "...", "response": "..."}

All formats are normalised to the messages format before tokenisation.
Prompt masking: loss is computed ONLY on assistant tokens.
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, Dict, Optional

IGNORE_INDEX = -100

BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"

SYSTEM_PROMPT = "You are Titan, a helpful AI assistant."


def _normalise_to_messages(example: dict) -> Optional[List[Dict[str, str]]]:
    """
    Convert any supported record format to a messages list.
    Returns None if the record cannot be converted.
    """
    # ── Format 1: already in messages format ──────────────────────────────
    if "messages" in example:
        messages = example["messages"]
        if isinstance(messages, list) and messages:
            roles = {m.get("role") for m in messages}
            if "assistant" in roles:
                return messages
        return None

    # ── Format 2: alpaca / dolly ──────────────────────────────────────────
    # Field aliases
    instruction = (example.get("instruction") or example.get("prompt") or "").strip()
    response     = (example.get("response")    or example.get("output") or "").strip()
    context      = (example.get("input")       or example.get("context") or "").strip()

    if not instruction or not response:
        return None

    user_content = instruction
    if context:
        user_content = f"{instruction}\n\n{context}"

    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": response},
    ]


def format_chat_as_text(messages: List[Dict[str, str]]) -> str:
    parts = [BOS]
    for msg in messages:
        role    = msg["role"].capitalize()
        content = msg["content"].strip()
        parts.append(f"{role}: {content}")
        parts.append(SEP)
    if parts and parts[-1] == SEP:
        parts[-1] = EOS
    return "".join(f"{p}" if p in (BOS, EOS, SEP) else p for p in parts)


def build_labels_with_prompt_mask(
    input_ids: List[int],
    tokenizer,
    messages: List[Dict[str, str]],
) -> List[int]:
    labels = list(input_ids)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = format_chat_as_text(prompt_messages)
    if prompt_text.endswith(EOS):
        prompt_text = prompt_text[: -len(EOS)] + SEP
    prompt_ids  = tokenizer.encode(prompt_text).ids
    prompt_len  = len(prompt_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX
    return labels


class TitanSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: List[str],
        tokenizer,
        max_seq_len: int = 2048,
        verbose: bool = True,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: List[Dict] = []

        total_loaded  = 0
        total_skipped = 0

        for path in jsonl_paths:
            p = Path(path)
            if not p.exists():
                if verbose:
                    print(f"[SFTDataset] WARNING: {path} not found, skipping.")
                continue

            file_loaded  = 0
            file_skipped = 0
            with open(p, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example  = json.loads(line)
                        messages = _normalise_to_messages(example)
                        if messages is None:
                            file_skipped += 1
                            total_skipped += 1
                            continue
                        self.examples.append({
                            "messages": messages,
                            "source":   str(p.name),
                        })
                        file_loaded  += 1
                        total_loaded += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        file_skipped  += 1
                        total_skipped += 1

            if verbose:
                print(f"[SFTDataset]   {p.name}: {file_loaded} loaded, {file_skipped} skipped")

        if verbose:
            print(f"[SFTDataset] Loaded {total_loaded} examples ({total_skipped} skipped) from {len(jsonl_paths)} file(s)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example  = self.examples[idx]
        messages = example["messages"]
        text     = format_chat_as_text(messages)

        token_ids = self.tokenizer.encode(text).ids
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]

        labels = build_labels_with_prompt_mask(token_ids, self.tokenizer, messages)
        labels = labels[: self.max_seq_len]

        pad_len = self.max_seq_len - len(token_ids)
        if pad_len > 0:
            token_ids = token_ids + [0] * pad_len
            labels    = labels    + [IGNORE_INDEX] * pad_len

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }

    def get_stats(self) -> Dict:
        from collections import Counter
        sources = Counter(e["source"] for e in self.examples)
        return {"total": len(self.examples), "by_source": dict(sources)}
