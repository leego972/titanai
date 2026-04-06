"""
TitanAI Tokenizer Trainer — 32K Vocab
=======================================
Trains a BPE tokenizer on the combined corpus with 32,000 vocabulary size.
Saves artifacts to tokenizer/titan_32k/
"""

import os
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders, normalizers

BASE      = Path(__file__).parent.parent
CORPUS    = BASE / "data" / "raw" / "tokenizer_corpus.txt"
SAVE_DIR  = BASE / "tokenizer" / "titan_32k"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE    = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<mask>"]


def main():
    print(f"[Tokenizer] Training 32K BPE tokenizer on {CORPUS}")
    print(f"[Tokenizer] Corpus size: {CORPUS.stat().st_size / 1_048_576:.1f} MB")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: NFD unicode normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
    ])

    # Pre-tokenizer: ByteLevel
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Train
    tokenizer.train(files=[str(CORPUS)], trainer=trainer)

    # Post-processor
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    sep_id = tokenizer.token_to_id("<sep>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> <sep> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", bos_id),
            ("<eos>", eos_id),
            ("<sep>", sep_id),
        ],
    )

    # Save
    tok_path = SAVE_DIR / "tokenizer.json"
    tokenizer.save(str(tok_path))
    tokenizer.model.save(str(SAVE_DIR))

    special_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    with open(SAVE_DIR / "special_tokens.json", "w") as f:
        json.dump(special_ids, f, indent=2)

    print(f"\n[Tokenizer] Saved to {SAVE_DIR}")
    print(f"[Tokenizer] Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"[Tokenizer] Special token IDs: {special_ids}")

    # Sanity check
    tests = [
        "Hello, Titan. How are you?",
        "def train(model, optimizer, data):",
        "CVE-2024-1234: Buffer overflow vulnerability in OpenSSL.",
        "The cinematographer used a dolly shot to follow the protagonist.",
        "Prove that the sum of the first n natural numbers is n*(n+1)/2.",
    ]
    print("\n[Tokenizer] Sanity check:")
    for t in tests:
        enc = tokenizer.encode(t)
        dec = tokenizer.decode(enc.ids)
        print(f"  Input:   {t}")
        print(f"  Tokens:  {enc.tokens[:12]}{'...' if len(enc.tokens) > 12 else ''}")
        print(f"  Decoded: {dec}")
        print()

    # Write tokenizer info for Gate 0-F
    info = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_ids,
        "corpus_file": str(CORPUS),
        "save_dir": str(SAVE_DIR),
        "status": "trained"
    }
    with open(SAVE_DIR / "tokenizer_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[Tokenizer] Training complete. Tokenizer info saved.")


if __name__ == "__main__":
    main()
