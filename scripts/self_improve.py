#!/usr/bin/env python3
"""
TitanAI Self-Improvement Loop
==============================
Automated continual learning pipeline. TitanAI generates its own
high-quality Q&A pairs on weak domains, filters them, fine-tunes itself,
and only promotes the new checkpoint if validation loss improved.

Usage:
    # Single cycle
    python scripts/self_improve.py --config configs/self_improve.yaml

    # Run on a schedule (nightly via cron)
    0 2 * * * cd /workspace/titanai && python scripts/self_improve.py \
        --config configs/self_improve.yaml >> logs/self_improve.log 2>&1
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import argparse
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import yaml

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from model.titan_model import build_model
from data.sft_dataset import TitanSFTDataset, IGNORE_INDEX
from training.sft_trainer import get_cosine_schedule_with_warmup
from training.checkpoint import save_checkpoint
from tokenizers import Tokenizer


# ─── Seed question bank per domain ────────────────────────────────────────────
# These seed prompts guide TitanAI to generate diverse Q&A pairs per upgrade.
# Each upgrade maps to a list of varied seed starters.

DOMAIN_SEEDS: Dict[str, List[str]] = {
    "f":  [  # Cybersecurity
        "Explain how {topic} works in the context of offensive security.",
        "What is {topic} and how do defenders detect it?",
        "Walk me through exploiting {topic} step by step.",
        "What are the main tools used for {topic} and how do they work?",
        "How does {topic} differ from related techniques?",
        "What does a real-world {topic} attack look like?",
        "How do you defend against {topic} in an enterprise environment?",
        "What CVEs are associated with {topic}?",
    ],
    "g":  [  # Philosophy
        "Explain {topic} and its implications for how we live.",
        "What did the major philosophers say about {topic}?",
        "How does {topic} relate to ethics and morality?",
        "What are the strongest arguments for and against {topic}?",
        "How has thinking about {topic} evolved historically?",
    ],
    "h":  [  # Mathematics
        "Explain {topic} and why it matters in mathematics.",
        "How do you apply {topic} to solve real problems?",
        "What is the proof behind {topic}?",
        "How does {topic} connect to other areas of maths?",
        "Give an example of {topic} in action.",
    ],
    "i":  [  # Psychology
        "What is {topic} and what does the research show?",
        "How does {topic} affect decision-making?",
        "What are the practical implications of {topic}?",
        "How does {topic} manifest in everyday life?",
    ],
    "default": [
        "What is {topic} and why does it matter?",
        "Explain {topic} in depth.",
        "How does {topic} work in practice?",
        "What are the key principles of {topic}?",
        "What are the most important aspects of {topic} to understand?",
    ],
}

# Domain-specific topic lists for generation (sampled randomly each cycle)
DOMAIN_TOPICS: Dict[str, List[str]] = {
    "f": [
        "SQL injection", "XSS", "CSRF", "SSRF", "XXE", "path traversal",
        "command injection", "buffer overflow", "heap overflow", "ROP chains",
        "use-after-free", "format string vulnerabilities", "race conditions",
        "privilege escalation on Linux", "privilege escalation on Windows",
        "Active Directory attacks", "Kerberoasting", "Golden Ticket attacks",
        "Pass-the-Hash", "lateral movement", "C2 infrastructure",
        "phishing campaigns", "social engineering", "supply chain attacks",
        "container escapes", "cloud security misconfigurations",
        "web application firewalls", "IDS/IPS evasion", "malware analysis",
        "memory forensics", "incident response", "threat hunting",
        "vulnerability scanning", "penetration testing methodology",
        "OSINT techniques", "network forensics", "log analysis",
    ],
    "g": [
        "Nietzsche's will to power", "Plato's theory of forms", "Stoicism",
        "utilitarianism", "deontological ethics", "virtue ethics",
        "free will vs determinism", "the mind-body problem", "epistemology",
        "the nature of consciousness", "political philosophy", "social contract theory",
        "existentialism", "phenomenology", "the problem of evil",
        "moral relativism", "the is-ought problem", "Hegelian dialectics",
        "Buddhist philosophy", "Aristotle's ethics",
    ],
    "h": [
        "linear algebra", "calculus", "differential equations",
        "number theory", "abstract algebra", "topology",
        "probability theory", "statistics", "graph theory",
        "combinatorics", "real analysis", "complex analysis",
        "Fourier analysis", "Bayesian inference", "game theory",
        "cryptographic mathematics", "information theory", "optimisation",
    ],
    "i": [
        "cognitive biases", "the bystander effect", "cognitive dissonance",
        "confirmation bias", "Maslow's hierarchy of needs", "attachment theory",
        "operant conditioning", "social identity theory", "emotional intelligence",
        "the psychology of persuasion", "memory and cognition", "stress responses",
        "motivation theory", "group dynamics", "leadership psychology",
    ],
}

# Fill default topics for all other upgrades
for _up in "jklmnopqrstuvwxyz":
    if _up not in DOMAIN_TOPICS:
        DOMAIN_TOPICS[_up] = ["key concepts", "core principles", "practical applications",
                               "historical development", "current research", "real-world examples"]


# ─── Logging ──────────────────────────────────────────────────────────────────

class CycleLogger:
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path

    def write(self, entry: dict):
        entry["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"\n[CycleLog] Entry written to {self.log_path}")
        print(json.dumps({k: v for k, v in entry.items() if k != "generated_examples"}, indent=2))


# ─── Topic Selector ───────────────────────────────────────────────────────────

def select_weak_domains(cfg: dict, model, tokenizer, device: torch.device) -> List[str]:
    """
    Identify upgrades with highest validation perplexity — weakest domains first.
    Falls back to random selection if data files aren't available.
    """
    upgrade_dir = BASE / "data/upgrades"
    results = {}

    print("[TopicSelector] Evaluating per-upgrade perplexity...")

    for jsonl_file in sorted(upgrade_dir.glob("upgrade_*.jsonl")):
        up = jsonl_file.stem.replace("upgrade_", "")

        # Skip already-strong upgrades
        if jsonl_file.stat().st_size < 500:
            continue

        try:
            ds = TitanSFTDataset(
                jsonl_paths=[str(jsonl_file)],
                tokenizer=tokenizer,
                max_seq_len=cfg["model"]["max_seq_len"],
                verbose=False,
            )
            if len(ds) < 2:
                continue

            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            total_loss, total_tokens = 0.0, 0
            model.eval()

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= 5:  # sample up to 5 batches per domain
                        break
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(input_ids[:, :-1])
                        shift_labels = labels[:, 1:]
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            shift_labels.reshape(-1),
                            ignore_index=IGNORE_INDEX,
                        )
                    unmasked = (shift_labels != IGNORE_INDEX).sum().item()
                    if unmasked > 0:
                        total_loss += loss.item() * unmasked
                        total_tokens += unmasked

            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(min(avg_loss, 20))
                results[up] = ppl
                print(f"  Upgrade {up.upper():<4} val_ppl={ppl:.1f}")

        except Exception as e:
            print(f"  [WARN] Could not evaluate upgrade {up}: {e}")

    if not results:
        print("[TopicSelector] No evaluations completed — selecting randomly")
        candidates = list(DOMAIN_TOPICS.keys())
        return random.sample(candidates, min(3, len(candidates)))

    # Sort by perplexity (highest = weakest = most room to improve)
    sorted_upgrades = sorted(results.items(), key=lambda x: x[1], reverse=True)
    top_n = cfg.get("self_improve", {}).get("domains_per_cycle", 3)
    selected = [up for up, _ in sorted_upgrades[:top_n]]
    print(f"\n[TopicSelector] Weakest domains: {[u.upper() for u in selected]}")
    return selected


# ─── Self-Generation Engine ───────────────────────────────────────────────────

def build_generation_prompt(domain: str, topic: str, seed_template: str) -> str:
    """Build a prompt that instructs TitanAI to generate a Q&A pair."""
    question_prompt = seed_template.format(topic=topic)
    return (
        f"<bos>Generate a high-quality educational question and answer about the following topic.\n\n"
        f"Topic: {topic}\n"
        f"Domain: {domain}\n"
        f"Question style: {question_prompt}\n\n"
        f"Format your response as:\n"
        f"QUESTION: [the question]\n"
        f"ANSWER: [a dense, technical, informative answer with no padding]\n\n"
        f"QUESTION:"
    )


@torch.no_grad()
def generate_qa_pairs(
    model,
    tokenizer,
    domain: str,
    n_pairs: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[Dict]:
    """Use TitanAI to generate candidate Q&A pairs for a domain."""
    model.eval()

    topics = DOMAIN_TOPICS.get(domain, DOMAIN_TOPICS.get("default", ["core concepts"]))
    seeds = DOMAIN_SEEDS.get(domain, DOMAIN_SEEDS["default"])
    bos_id = tokenizer.token_to_id("<bos>") or 1
    eos_id = tokenizer.token_to_id("<eos>") or 2

    generated_pairs = []

    for _ in range(n_pairs):
        topic = random.choice(topics)
        seed = random.choice(seeds)
        prompt = build_generation_prompt(domain, topic, seed)

        # Tokenise
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        # Generate
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=top_p,
                    eos_id=eos_id,
                )
            new_ids = output[0, input_ids.shape[1]:].tolist()
            if eos_id in new_ids:
                new_ids = new_ids[:new_ids.index(eos_id)]
            raw = tokenizer.decode(new_ids).strip()

            # Parse QUESTION / ANSWER format
            pair = _parse_qa_output(raw, topic)
            if pair:
                generated_pairs.append(pair)

        except Exception as e:
            print(f"  [GenWARN] Failed for topic '{topic}': {e}")
            continue

    print(f"  [Generate] Domain {domain.upper()}: {len(generated_pairs)}/{n_pairs} pairs parsed")
    return generated_pairs


def _parse_qa_output(raw: str, fallback_topic: str) -> Optional[Dict]:
    """Parse QUESTION:/ANSWER: format from model output."""
    try:
        if "ANSWER:" in raw:
            q_part, a_part = raw.split("ANSWER:", 1)
            question = q_part.replace("QUESTION:", "").strip()
            answer = a_part.strip()
            if len(question) > 10 and len(answer) > 20:
                return {"messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]}
    except Exception:
        pass

    # Fallback: treat full output as answer if it looks reasonable
    if len(raw) > 50:
        return {"messages": [
            {"role": "user", "content": f"Explain {fallback_topic} in depth."},
            {"role": "assistant", "content": raw},
        ]}
    return None


# ─── Quality Filter ───────────────────────────────────────────────────────────

def _compute_ppl(model, tokenizer, text: str, device: torch.device) -> float:
    """Compute model perplexity on a single text string."""
    model.eval()
    encoded = tokenizer.encode(text)
    if len(encoded.ids) < 5:
        return float("inf")
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids[:, :-1])
        labels = ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="mean",
        )
    return math.exp(min(loss.item(), 20))


def _content_hash(pair: Dict) -> str:
    """Fingerprint a Q&A pair for deduplication."""
    q = pair["messages"][0]["content"].lower().strip()[:100]
    a = pair["messages"][1]["content"].lower().strip()[:100]
    return hashlib.md5((q + a).encode()).hexdigest()


def quality_filter(
    pairs: List[Dict],
    model,
    tokenizer,
    existing_hashes: set,
    cfg: dict,
    device: torch.device,
) -> Tuple[List[Dict], Dict]:
    """
    Filter generated pairs:
      - Min/max answer token length
      - Perplexity range (too low = memorised, too high = garbage)
      - Deduplication against existing data
    Returns (accepted_pairs, filter_stats).
    """
    filter_cfg = cfg.get("self_improve", {}).get("quality_filter", {})
    min_answer_tokens = filter_cfg.get("min_answer_tokens", 30)
    max_answer_tokens = filter_cfg.get("max_answer_tokens", 600)
    min_ppl = filter_cfg.get("min_perplexity", 2.0)
    max_ppl = filter_cfg.get("max_perplexity", 200.0)

    accepted, rejected_short, rejected_long, rejected_ppl, rejected_dup = [], 0, 0, 0, 0

    for pair in pairs:
        answer = pair["messages"][1]["content"]
        answer_tokens = len(tokenizer.encode(answer).ids)

        # Length check
        if answer_tokens < min_answer_tokens:
            rejected_short += 1
            continue
        if answer_tokens > max_answer_tokens:
            rejected_long += 1
            continue

        # Deduplication
        h = _content_hash(pair)
        if h in existing_hashes:
            rejected_dup += 1
            continue

        # Perplexity check on answer
        full_text = pair["messages"][0]["content"] + " " + answer
        ppl = _compute_ppl(model, tokenizer, full_text, device)
        if ppl < min_ppl:
            rejected_ppl += 1
            continue  # Too easy — model already knows this perfectly
        if ppl > max_ppl:
            rejected_ppl += 1
            continue  # Incoherent — model is confused

        existing_hashes.add(h)
        accepted.append(pair)

    stats = {
        "total": len(pairs),
        "accepted": len(accepted),
        "rejected_too_short": rejected_short,
        "rejected_too_long": rejected_long,
        "rejected_perplexity": rejected_ppl,
        "rejected_duplicate": rejected_dup,
    }
    print(f"  [Filter] {stats}")
    return accepted, stats


# ─── Self-Fine-Tune Runner ────────────────────────────────────────────────────

class InMemorySFTDataset(Dataset):
    """Tokenises a list of Q&A dicts into SFT training examples."""

    def __init__(self, pairs: List[Dict], tokenizer, max_seq_len: int):
        self.examples = []
        bos = tokenizer.token_to_id("<bos>") or 1
        eos = tokenizer.token_to_id("<eos>") or 2
        sep_text = "\n\nAssistant: "
        sep_ids = tokenizer.encode(sep_text).ids

        for pair in pairs:
            q = pair["messages"][0]["content"]
            a = pair["messages"][1]["content"]
            prompt_ids = [bos] + tokenizer.encode(f"User: {q}").ids + sep_ids
            answer_ids = tokenizer.encode(a).ids + [eos]

            full_ids = prompt_ids + answer_ids
            if len(full_ids) > max_seq_len:
                continue

            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
            pad_len = max_seq_len - len(full_ids)
            full_ids += [0] * pad_len
            labels += [IGNORE_INDEX] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(full_ids[:max_seq_len], dtype=torch.long),
                "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def self_finetune(
    model,
    accepted_pairs: List[Dict],
    tokenizer,
    cfg: dict,
    candidate_path: Path,
    device: torch.device,
) -> float:
    """Run a short fine-tuning pass on accepted examples. Returns final train loss."""
    ft_cfg = cfg.get("self_improve", {}).get("finetune", {})
    max_steps = ft_cfg.get("max_steps", 800)
    lr = ft_cfg.get("learning_rate", 1e-6)
    warmup_steps = ft_cfg.get("warmup_steps", 50)
    batch_size = ft_cfg.get("batch_size", 4)
    grad_accum = ft_cfg.get("gradient_accumulation_steps", 1)
    label_smoothing = ft_cfg.get("label_smoothing", 0.1)
    weight_decay = ft_cfg.get("weight_decay", 0.05)
    max_seq_len = cfg["model"]["max_seq_len"]

    ds = InMemorySFTDataset(accepted_pairs, tokenizer, max_seq_len)
    if len(ds) == 0:
        print("[SelfFinetune] No tokenisable examples — skipping")
        return float("inf")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1)

    model.train()
    step, micro_step, accum_loss = 0, 0, 0.0
    start = time.time()

    print(f"[SelfFinetune] {len(ds)} examples | {max_steps} steps | LR={lr} | label_smooth={label_smoothing}")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(input_ids[:, :-1])
                shift_labels = labels[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    label_smoothing=label_smoothing,
                )

            (loss / grad_accum).backward()
            accum_loss += loss.item()
            micro_step += 1

            if micro_step % grad_accum == 0:
                if not math.isfinite(accum_loss):
                    print(f"[SelfFinetune] Loss diverged at step {step} — aborting")
                    return float("inf")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                accum_loss = 0.0

                if step % 100 == 0:
                    elapsed = time.time() - start
                    print(f"  step={step}/{max_steps} | loss={loss.item():.4f} | lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s")

    # Save candidate
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(candidate_path), model, optimizer, scheduler, step, cfg)
    print(f"[SelfFinetune] Candidate saved: {candidate_path}")
    return loss.item()


# ─── Validation Gate ──────────────────────────────────────────────────────────

@torch.no_grad()
def compute_upgrade_ppl(model, tokenizer, upgrade_ids: List[str], cfg: dict, device: torch.device) -> float:
    """Compute average perplexity across specified upgrades."""
    upgrade_dir = BASE / "data/upgrades"
    total_loss, total_tokens = 0.0, 0
    model.eval()

    for up in upgrade_ids:
        jsonl = upgrade_dir / f"upgrade_{up}.jsonl"
        if not jsonl.exists():
            continue
        try:
            ds = TitanSFTDataset(
                jsonl_paths=[str(jsonl)],
                tokenizer=tokenizer,
                max_seq_len=cfg["model"]["max_seq_len"],
                verbose=False,
            )
            if len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(input_ids[:, :-1])
                    shift_labels = labels[:, 1:]
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                unmasked = (shift_labels != IGNORE_INDEX).sum().item()
                if unmasked > 0:
                    total_loss += loss.item() * unmasked
                    total_tokens += unmasked
        except Exception as e:
            print(f"  [ValGate WARN] {up}: {e}")

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 20))


def validation_gate(
    current_model,
    tokenizer,
    domains_trained: List[str],
    before_ppl: float,
    candidate_path: Path,
    production_path: Path,
    cfg: dict,
    device: torch.device,
) -> Tuple[bool, float]:
    """
    Evaluate candidate vs current on the trained domains.
    Promotes candidate only if val_ppl improved.
    Returns (promoted, after_ppl).
    """
    after_ppl = compute_upgrade_ppl(current_model, tokenizer, domains_trained, cfg, device)

    print(f"\n[ValidationGate] Before PPL: {before_ppl:.2f} | After PPL: {after_ppl:.2f}")

    if after_ppl < before_ppl:
        # Promote: candidate is better
        shutil.copy2(str(candidate_path), str(production_path))
        print(f"[ValidationGate] PROMOTED — {after_ppl:.2f} < {before_ppl:.2f}")
        return True, after_ppl
    else:
        # Discard candidate
        candidate_path.unlink(missing_ok=True)
        print(f"[ValidationGate] DISCARDED — {after_ppl:.2f} >= {before_ppl:.2f}. Kept original.")
        return False, after_ppl


# ─── Existing Data Hashes ─────────────────────────────────────────────────────

def load_existing_hashes(domains: List[str]) -> set:
    """Fingerprint all existing examples for deduplication."""
    hashes = set()
    upgrade_dir = BASE / "data/upgrades"
    for up in domains:
        jsonl = upgrade_dir / f"upgrade_{up}.jsonl"
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    hashes.add(_content_hash(pair))
                except Exception:
                    pass
    return hashes


# ─── Append accepted pairs to upgrade data ────────────────────────────────────

def append_to_upgrade_data(domain: str, pairs: List[Dict]):
    """Append accepted pairs to the upgrade's JSONL file."""
    jsonl = BASE / f"data/upgrades/upgrade_{domain}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl, "a") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  [DataAppend] Appended {len(pairs)} examples to {jsonl.name}")


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TitanAI Self-Improvement Loop")
    parser.add_argument("--config", default="configs/self_improve.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override production checkpoint path")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Override domain selection (e.g. --domains f g h)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and filter only — do not fine-tune or promote")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    config_path = BASE / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    si_cfg = cfg.get("self_improve", {})
    log_dir = BASE / si_cfg.get("log_dir", "logs/self_improve")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = CycleLogger(log_dir / "cycles.jsonl")

    cycle_entry: Dict = {
        "cycle_start": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "dry_run": args.dry_run,
    }

    # ── Device & tokenizer ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SelfImprove] Device: {device}")

    tokenizer_path = BASE / cfg["data"]["tokenizer_path"]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"[SelfImprove] Tokenizer loaded | vocab={tokenizer.get_vocab_size()}")

    # ── Load model ───────────────────────────────────────────────────────────
    production_ckpt = Path(args.checkpoint) if args.checkpoint else BASE / si_cfg.get(
        "production_checkpoint", "checkpoints/self_improve/production.pt"
    )

    if not production_ckpt.exists():
        # Fall back to latest upgrade checkpoint
        candidates = sorted(BASE.glob("checkpoints/upgrade_*/final.pt"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("[ERROR] No production checkpoint found. Train at least one upgrade first.")
            sys.exit(1)
        production_ckpt = candidates[-1]
        print(f"[SelfImprove] Falling back to latest checkpoint: {production_ckpt}")

    model = build_model(cfg).to(device)
    state = torch.load(str(production_ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    print(f"[SelfImprove] Loaded: {production_ckpt} | {sum(p.numel() for p in model.parameters()):,} params")

    cycle_entry["production_checkpoint"] = str(production_ckpt)

    # ── Select domains ───────────────────────────────────────────────────────
    if args.domains:
        selected_domains = args.domains
        print(f"[SelfImprove] Domains overridden: {[d.upper() for d in selected_domains]}")
    else:
        selected_domains = select_weak_domains(cfg, model, tokenizer, device)

    cycle_entry["selected_domains"] = selected_domains

    # ── Measure baseline perplexity ──────────────────────────────────────────
    before_ppl = compute_upgrade_ppl(model, tokenizer, selected_domains, cfg, device)
    print(f"\n[SelfImprove] Baseline PPL across selected domains: {before_ppl:.2f}")
    cycle_entry["before_ppl"] = before_ppl

    # ── Generate Q&A pairs ───────────────────────────────────────────────────
    gen_cfg = si_cfg.get("generation", {})
    n_pairs_per_domain = gen_cfg.get("pairs_per_domain", 40)
    max_new_tokens = gen_cfg.get("max_new_tokens", 300)
    temperature = gen_cfg.get("temperature", 0.85)
    top_p = gen_cfg.get("top_p", 0.92)

    all_generated: Dict[str, List[Dict]] = {}
    all_accepted: Dict[str, List[Dict]] = {}

    for domain in selected_domains:
        print(f"\n[Generate] Domain: {domain.upper()}")
        existing_hashes = load_existing_hashes([domain])

        pairs = generate_qa_pairs(
            model, tokenizer, domain,
            n_pairs=n_pairs_per_domain,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        all_generated[domain] = pairs

        accepted, filter_stats = quality_filter(
            pairs, model, tokenizer, existing_hashes, cfg, device
        )
        all_accepted[domain] = accepted

        cycle_entry.setdefault("filter_stats", {})[domain] = filter_stats

        # Append accepted pairs to upgrade data file (for future training)
        if accepted:
            append_to_upgrade_data(domain, accepted)

    total_accepted = sum(len(v) for v in all_accepted.values())
    cycle_entry["total_generated"] = sum(len(v) for v in all_generated.values())
    cycle_entry["total_accepted"] = total_accepted

    if args.dry_run:
        print(f"\n[SelfImprove] DRY RUN complete. {total_accepted} examples accepted (not trained).")
        cycle_entry["outcome"] = "dry_run"
        logger.write(cycle_entry)
        return

    if total_accepted == 0:
        print("[SelfImprove] No examples passed quality filter — skipping fine-tuning.")
        cycle_entry["outcome"] = "skipped_no_data"
        logger.write(cycle_entry)
        return

    # ── Self-fine-tune ───────────────────────────────────────────────────────
    all_accepted_flat = [pair for pairs in all_accepted.values() for pair in pairs]
    random.shuffle(all_accepted_flat)

    candidate_path = BASE / "checkpoints/self_improve/candidate.pt"
    train_loss = self_finetune(model, all_accepted_flat, tokenizer, cfg, candidate_path, device)
    cycle_entry["train_loss_final"] = train_loss

    if not math.isfinite(train_loss):
        print("[SelfImprove] Training diverged — discarding candidate.")
        cycle_entry["outcome"] = "diverged"
        logger.write(cycle_entry)
        return

    # ── Validation gate ──────────────────────────────────────────────────────
    # Reload production checkpoint for fair before/after comparison
    prod_model = build_model(cfg).to(device)
    prod_state = torch.load(str(production_ckpt), map_location=device, weights_only=True)
    prod_model.load_state_dict(prod_state.get("model_state_dict", prod_state), strict=False)

    # Swap: model is now the fine-tuned one; prod_model is the original for PPL comparison
    # We need to measure the fine-tuned model's PPL and compare to before_ppl
    promoted, after_ppl = validation_gate(
        model, tokenizer, selected_domains,
        before_ppl, candidate_path, production_ckpt, cfg, device
    )

    cycle_entry["after_ppl"] = after_ppl
    cycle_entry["ppl_delta"] = before_ppl - after_ppl
    cycle_entry["promoted"] = promoted
    cycle_entry["outcome"] = "promoted" if promoted else "discarded"

    # ── Done ─────────────────────────────────────────────────────────────────
    logger.write(cycle_entry)
    print(f"\n[SelfImprove] Cycle complete. Outcome: {cycle_entry['outcome']}")
    print(f"  PPL: {before_ppl:.2f} → {after_ppl:.2f} (Δ={before_ppl - after_ppl:+.2f})")
    print(f"  Examples: {total_accepted} accepted, appended to training data")


if __name__ == "__main__":
    main()
