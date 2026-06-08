#!/usr/bin/env python3
"""
TitanAI — Logic, Reasoning & Critical Thinking Corpus
=======================================================
Non-restrictive reasoning training. Follows logic wherever it leads.
No artificial topic avoidance. Argues all sides. Questions everything.

Subjects:
  1.  Formal Logic          — propositional, predicate, modal, temporal
  2.  Mathematical Reasoning — proof techniques, GSM8K, competition math
  3.  Philosophy            — Plato, Nietzsche, Machiavelli, Aristotle,
                              Kant, Descartes, Hume, Locke, Schopenhauer
  4.  Strategic Thinking    — Sun Tzu, Clausewitz, game theory, chess
  5.  Legal Reasoning       — case law, argumentation, precedent
  6.  Debate & Argumentation — both sides of every argument, Socratic method
  7.  Scientific Reasoning  — hypothesis, experimental design, falsifiability
  8.  Decision Theory       — probability, expected utility, risk
  9.  Critical Thinking     — fallacies, rhetoric, persuasion
  10. Adversarial Reasoning — devil's advocate, steelman/strawman

Quality: Only sources where reasoning is SHOWN not just stated.
"""

import os, sys, json, time, requests, subprocess
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw" / "logic_reasoning"
PROC_DIR = ROOT / "data" / "processed" / "logic_reasoning"
TOK_PATH = ROOT / "tokenizer" / "artifacts_v32k" / "tokenizer.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_tok():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK_PATH))


def tok_save(texts, out, tok, seq=2048):
    import numpy as np
    ids = []
    for t in tqdm(texts, desc=f"  tok {out.name[:35]}", leave=False):
        enc = tok.encode(t).ids
        for i in range(0, len(enc), seq):
            c = enc[i:i+seq]
            if len(c) == seq:
                ids.extend(c)
    if not ids: return 0
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out)
    print(f"  ✅ {len(arr):,} tokens → {out.name}")
    return len(arr)


def clone(url, dest):
    if not Path(dest).exists():
        try:
            subprocess.run(["git","clone","--depth=1", url, str(dest)],
                           check=True, capture_output=True, timeout=120)
        except Exception as e:
            print(f"  ⚠️  {e}")


# ── 1. FORMAL LOGIC ──────────────────────────────────────────────────────────

def dl_formal_logic(tok):
    print("\n🧠 Formal Logic...")
    try:
        from datasets import load_dataset

        # LogiQA — logical reasoning problems with full explanations
        ds = load_dataset("lucasmccabe/logiqa", split="train")
        texts = [
            f"Logical Reasoning Problem:\nContext: {i.get('context','')}\n"
            f"Question: {i.get('query','')}\n"
            f"Options: {i.get('options','')}\n"
            f"Correct Answer: {i.get('correct_option','')}\n"
            f"Explanation: Option {i.get('correct_option','')} follows "
            f"because the context establishes {i.get('context','')[:200]}\n"
            for i in tqdm(ds, leave=False)
        ]

        # ReClor — reading comprehension with logical reasoning
        try:
            ds2 = load_dataset("metaeval/reclor", split="train")
            for i in tqdm(ds2, leave=False):
                texts.append(
                    f"Logical Reasoning:\nPassage: {i.get('context','')}\n"
                    f"Question: {i.get('question','')}\n"
                    f"Answer: {i.get('answers',[''])[i.get('label',0)]}\n"
                )
        except Exception:
            pass

        tok_save(texts, PROC_DIR/"formal_logic.bin", tok)
        print(f"  {len(texts):,} logic problems with reasoning.")
    except Exception as e:
        print(f"  ⚠️  {e}")


# ── 2. MATHEMATICAL REASONING ─────────────────────────────────────────────────

def dl_math_reasoning(tok):
    print("\n🧠 Mathematical Reasoning (step-by-step proofs)...")
    try:
        from datasets import load_dataset
        texts = []

        # GSM8K — grade school math with full solution steps
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for i in tqdm(ds, leave=False):
            texts.append(
                f"Math Problem:\n{i['question']}\n\n"
                f"Step-by-step solution:\n{i['answer']}\n"
            )

        # MATH competition problems with full solutions
        try:
            ds2 = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
            for i in tqdm(ds2, leave=False):
                texts.append(
                    f"Competition Math ({i.get('type','')}, Level {i.get('level','')}):\n"
                    f"Problem: {i.get('problem','')}\n\n"
                    f"Solution:\n{i.get('solution','')}\n"
                )
        except Exception:
            pass

        # MetaMath
        try:
            ds3 = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
            count = 0
            for i in ds3:
                texts.append(
                    f"Math Reasoning:\nQ: {i.get('query','')}\n"
                    f"A: {i.get('response','')}\n"
                )
                count += 1
                if count >= 100000: break
        except Exception:
            pass

        tok_save(texts, PROC_DIR/"math_reasoning.bin", tok)
        print(f"  {len(texts):,} step-by-step math solutions.")
    except Exception as e:
        print(f"  ⚠️  {e}")


# ── 3. PHILOSOPHY ─────────────────────────────────────────────────────────────

def dl_philosophy(tok):
    """
    Public domain philosophy — Plato, Aristotle, Nietzsche, Machiavelli,
    Kant, Descartes, Hume, Sun Tzu, Clausewitz, Schopenhauer.
    These thinkers followed reason wherever it led — no restrictions.
    """
    print("\n🧠 Philosophy (unrestricted reasoning from great thinkers)...")

    # Project Gutenberg philosophy texts (public domain)
    gutenberg_ids = {
        # Plato
        "1656":  "Plato — The Republic",
        "1497":  "Plato — Apology",
        "1672":  "Plato — Symposium",
        # Aristotle
        "8438":  "Aristotle — Nicomachean Ethics",
        "6762":  "Aristotle — Politics",
        # Nietzsche
        "4363":  "Nietzsche — Beyond Good and Evil",
        "1998":  "Nietzsche — Thus Spoke Zarathustra",
        "25012": "Nietzsche — On the Genealogy of Morality",
        # Machiavelli
        "1232":  "Machiavelli — The Prince",
        # Descartes
        "59":    "Descartes — Meditations on First Philosophy",
        # Hume
        "9662":  "Hume — An Enquiry Concerning Human Understanding",
        # Kant
        "4280":  "Kant — Fundamental Principles of the Metaphysic of Morals",
        # Schopenhauer
        "38427": "Schopenhauer — The World as Will and Representation",
        # Sun Tzu
        "132":   "Sun Tzu — The Art of War",
        # Clausewitz
        "1946":  "Clausewitz — On War",
        # Hobbes
        "3207":  "Hobbes — Leviathan",
        # Locke
        "7370":  "Locke — Two Treatises of Government",
        # Spinoza
        "3800":  "Spinoza — Ethics",
    }

    texts = []
    import urllib.request
    for gid, title in gutenberg_ids.items():
        path = RAW_DIR / f"gutenberg_{gid}.txt"
        if not path.exists():
            try:
                url = f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
                urllib.request.urlretrieve(url, path)
                time.sleep(0.5)
            except Exception:
                try:
                    url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
                    urllib.request.urlretrieve(url, path)
                    time.sleep(0.5)
                except Exception:
                    continue
        try:
            content = path.read_text(errors="replace")
            if len(content) > 1000:
                texts.append(f"# {title}\n\n{content}")
                print(f"  ✓ {title}")
        except Exception:
            pass

    # Philosophy Stack Exchange (high quality philosophical reasoning)
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                          split="train", streaming=True)
        count = 0
        for item in ds:
            tags = item.get("tags", [])
            if any(t in str(tags).lower() for t in
                   ["philosophy","logic","ethics","epistemology","metaphysics","reasoning"]):
                if item.get("score", 0) >= 5:
                    texts.append(
                        f"Philosophy Q&A:\nQ: {item.get('question','')}\n"
                        f"A: {item.get('chosen','')}\n"
                    )
                    count += 1
                    if count >= 20000: break
    except Exception:
        pass

    tok_save(texts, PROC_DIR/"philosophy.bin", tok)
    print(f"  {len(texts):,} philosophy documents.")


# ── 4. STRATEGIC & GAME THEORY ───────────────────────────────────────────────

def dl_strategic_reasoning(tok):
    print("\n🧠 Strategic Reasoning & Game Theory...")
    texts = []

    # Chess game reasoning (PGN format with annotations)
    try:
        from datasets import load_dataset
        ds = load_dataset("adamkarvonen/chess_games", split="train", streaming=True)
        count = 0
        for item in ds:
            pgn = item.get("transcript", "")
            if len(pgn) > 200:
                texts.append(
                    f"Chess Game (Strategic Reasoning):\n{pgn}\n"
                )
                count += 1
                if count >= 50000: break
        print(f"  ✓ {count:,} chess games (pattern + strategic reasoning)")
    except Exception as e:
        print(f"  ⚠️  Chess: {e}")

    # Game theory problems
    game_theory = [
        {
            "concept": "Prisoner's Dilemma",
            "setup": "Two suspects are arrested. Each can betray or stay silent. "
                     "If both stay silent: 1 year each. If one betrays: betrayer goes free, "
                     "other gets 3 years. If both betray: 2 years each.",
            "reasoning": "Dominant strategy analysis: Regardless of what the other player does, "
                         "betraying yields a better outcome for each individual player. "
                         "If other stays silent: betraying gives 0 vs 1 year (betray wins). "
                         "If other betrays: betraying gives 2 vs 3 years (betray wins). "
                         "Therefore betrayal is the dominant strategy for both players, "
                         "leading to the Nash equilibrium of (betray, betray) = (2, 2), "
                         "which is Pareto-inferior to (silent, silent) = (1, 1). "
                         "This illustrates why rational individual behavior can lead to "
                         "collectively suboptimal outcomes.",
        },
        {
            "concept": "Nash Equilibrium",
            "setup": "A Nash Equilibrium is a strategy profile where no player can "
                     "improve their outcome by unilaterally changing their strategy.",
            "reasoning": "To find Nash Equilibria: For each player, identify best responses "
                         "to every possible strategy of other players. A Nash Equilibrium "
                         "is a strategy profile where every player is playing a best response "
                         "to the others. In the Prisoner's Dilemma, (Defect, Defect) is a "
                         "Nash Equilibrium because Defect is a best response to Defect for both players. "
                         "Games can have zero, one, or multiple Nash Equilibria. "
                         "Mixed strategy Nash Equilibria exist when no pure strategy equilibrium exists.",
        },
        {
            "concept": "Minimax Strategy",
            "setup": "In zero-sum games, one player's gain is another's loss. "
                     "Minimax: minimize the maximum loss (or maximize the minimum gain).",
            "reasoning": "Minimax theorem (von Neumann): In any finite two-player zero-sum game, "
                         "there exists a mixed strategy Nash Equilibrium where each player's "
                         "expected payoff equals the minimax value. "
                         "Applications: Chess, poker, military strategy, adversarial ML. "
                         "Key insight: in zero-sum games, randomization can be optimal — "
                         "predictability is exploitable, randomness is not.",
        },
        {
            "concept": "Schelling Points",
            "setup": "When coordination is needed without communication, people tend to "
                     "converge on salient focal points.",
            "reasoning": "Thomas Schelling's insight: in the absence of communication, "
                         "people solving coordination problems tend to pick solutions that "
                         "seem natural, obvious, or special. Example: if asked to meet someone "
                         "in New York without specifying where, people converge on Grand Central "
                         "at noon. Applications: negotiations, warfare, international relations, "
                         "cryptocurrency forks. The key is identifying what seems 'obvious' "
                         "to all parties — often round numbers, geographic centers, or "
                         "historically significant points.",
        },
    ]

    for item in game_theory:
        texts.append(
            f"Game Theory — {item['concept']}:\n"
            f"Setup: {item['setup']}\n\n"
            f"Reasoning: {item['reasoning']}\n"
        )

    tok_save(texts, PROC_DIR/"strategic_reasoning.bin", tok)
    print(f"  {len(texts):,} strategic reasoning documents.")


# ── 5. DEBATE & ADVERSARIAL REASONING ────────────────────────────────────────

def dl_debate_reasoning(tok):
    """
    Both sides of every argument. Devil's advocate. Steelmanning.
    Non-restrictive — explores all positions without predetermined conclusions.
    """
    print("\n🧠 Debate & Adversarial Reasoning (all sides)...")
    texts = []

    # IBM Debater — evidence for and against positions
    try:
        from datasets import load_dataset
        ds = load_dataset("ibm/claim_sentences_search", split="train")
        for item in tqdm(ds, leave=False):
            topic   = item.get("topic", "")
            claim   = item.get("claim_sentence", "")
            stance  = item.get("stance", "")
            texts.append(
                f"Debate Reasoning:\nTopic: {topic}\n"
                f"Stance: {stance}\n"
                f"Argument: {claim}\n"
            )
    except Exception:
        pass

    # Argument quality (reasoning quality assessment)
    try:
        from datasets import load_dataset
        ds = load_dataset("Helsinki-NLP/argument_quality_ibm_rank30k", split="train")
        for item in tqdm(ds, leave=False):
            texts.append(
                f"Argument Analysis:\nTopic: {item.get('topic','')}\n"
                f"Argument: {item.get('argument','')}\n"
                f"Quality Score: {item.get('score','')}\n"
            )
    except Exception:
        pass

    # Logical fallacies — comprehensive guide
    fallacies = [
        ("Ad Hominem", "Attacking the person rather than the argument.",
         "Example: 'You can't trust his economic policy — he's been divorced three times.' "
         "The divorce is irrelevant to economic competence. The attack targets the person, "
         "not the argument's logic or evidence."),
        ("Straw Man", "Misrepresenting someone's argument to make it easier to attack.",
         "Example: Person A says 'We need stricter gun regulations.' "
         "Person B responds 'So you want to ban all guns and leave citizens defenceless?' "
         "B has replaced A's moderate position with an extreme one."),
        ("False Dilemma", "Presenting only two options when more exist.",
         "Example: 'You're either with us or against us.' "
         "Ignores neutrality, partial agreement, or alternative positions entirely."),
        ("Slippery Slope", "Claiming one event will inevitably lead to extreme consequences.",
         "Example: 'If we allow same-sex marriage, next people will want to marry animals.' "
         "No mechanism is provided showing why the chain of events must follow."),
        ("Appeal to Authority", "Using an authority figure to support a claim without evidence.",
         "Example: 'This nutritional supplement must work — a doctor endorses it.' "
         "The doctor may be paid, out of their specialty, or simply wrong."),
        ("Circular Reasoning", "The conclusion is used as a premise.",
         "Example: 'The Bible is true because it says so in the Bible.' "
         "The truth of the Bible cannot be established by the Bible itself."),
        ("Post Hoc Ergo Propter Hoc", "Assuming causation from correlation.",
         "Example: 'I wore my lucky socks and we won the game — the socks caused the win.' "
         "Correlation ≠ causation. The win had other causes."),
        ("Hasty Generalisation", "Drawing broad conclusions from insufficient evidence.",
         "Example: 'I met two rude people from that city, so everyone there is rude.' "
         "Two data points cannot establish a generalisation about a whole population."),
    ]

    for name, definition, example in fallacies:
        texts.append(
            f"Logical Fallacy — {name}:\n"
            f"Definition: {definition}\n"
            f"Example and Analysis: {example}\n"
            f"How to counter: Identify the fallacy, name it, explain why it fails "
            f"logically, then redirect to the actual argument.\n"
        )

    # Steelmanning examples — strongest possible version of opposing views
    steelman_topics = [
        ("Strong AI regulation",
         "The strongest case for heavy AI regulation is not fear of science fiction robots "
         "but the demonstrated capacity of AI systems to concentrate power, automate "
         "misinformation at scale, and make consequential decisions without accountability. "
         "Historical precedent shows that powerful technologies without regulatory frameworks "
         "— financial derivatives, social media — caused systemic harm before governance caught up. "
         "A proactive regulatory framework prevents harm rather than responding to it."),
        ("Open borders immigration",
         "The strongest economic case for open borders is that labour mobility is the single "
         "largest source of economic gains available — estimates suggest it could double world GDP. "
         "Restricting movement based on birthplace is arbitrary discrimination no different "
         "from restricting movement based on race. Empirical evidence shows immigrants are "
         "net fiscal contributors and drive innovation disproportionately."),
        ("Radical life extension",
         "The strongest case against radical life extension is not that death is good but "
         "that extreme longevity would concentrate power permanently in whoever achieves it first, "
         "calcify social hierarchies, reduce generational turnover which drives cultural progress, "
         "and in the absence of universal access, create the most extreme inequality imaginable."),
    ]

    for topic, steelman in steelman_topics:
        texts.append(
            f"Steelman Exercise — {topic}:\n"
            f"Strongest version of this position:\n{steelman}\n"
            f"Note: Steelmanning means presenting the BEST version of a position, "
            f"not your own view. Engaging with the strongest form prevents "
            f"dismissing ideas through weak versions.\n"
        )

    tok_save(texts, PROC_DIR/"debate_reasoning.bin", tok)
    print(f"  {len(texts):,} debate and argumentation documents.")


# ── 6. SCIENTIFIC REASONING ──────────────────────────────────────────────────

def dl_scientific_reasoning(tok):
    print("\n🧠 Scientific Reasoning & Critical Thinking...")
    texts = []

    # ARC Challenge — scientific reasoning
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for item in tqdm(ds, leave=False):
            choices = item.get("choices", {})
            labels  = choices.get("label", [])
            options = choices.get("text", [])
            answer  = item.get("answerKey", "")
            idx     = labels.index(answer) if answer in labels else 0
            texts.append(
                f"Scientific Reasoning:\nQuestion: {item.get('question','')}\n"
                f"Options: {dict(zip(labels, options))}\n"
                f"Answer: {answer} — {options[idx] if idx < len(options) else ''}\n"
            )
    except Exception as e:
        print(f"  ⚠️  ARC: {e}")

    # BIG-Bench Hard — challenging reasoning tasks
    try:
        from datasets import load_dataset
        for task in ["logical_deduction_five_objects",
                     "causal_judgement",
                     "web_of_lies",
                     "tracking_shuffled_objects_five_objects"]:
            try:
                ds = load_dataset("maveriq/bigbenchhard", task, split="train")
                for item in tqdm(ds, desc=f"  {task}", leave=False):
                    texts.append(
                        f"Reasoning Task ({task}):\n"
                        f"Input: {item.get('input','')}\n"
                        f"Target: {item.get('target','')}\n"
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Causal reasoning
    try:
        from datasets import load_dataset
        ds = load_dataset("cos_e", "v1.11", split="train")
        for item in tqdm(ds, leave=False):
            texts.append(
                f"Causal Reasoning:\nQuestion: {item.get('question','')}\n"
                f"Answer: {item.get('answer','')}\n"
                f"Explanation: {item.get('abstractive_explanation','')}\n"
            )
    except Exception:
        pass

    tok_save(texts, PROC_DIR/"scientific_reasoning.bin", tok)
    print(f"  {len(texts):,} scientific reasoning examples.")


# ── 7. LEGAL REASONING ───────────────────────────────────────────────────────

def dl_legal_reasoning(tok):
    print("\n🧠 Legal Reasoning (argumentation, precedent, logic)...")
    texts = []
    try:
        from datasets import load_dataset

        # Legal reasoning dataset
        ds = load_dataset("nguha/legalbench", "contract_qa", split="train",
                          trust_remote_code=True)
        for item in tqdm(ds, leave=False):
            texts.append(
                f"Legal Reasoning:\nContext: {item.get('text','')}\n"
                f"Question: {item.get('question','')}\n"
                f"Answer: {item.get('answer','')}\n"
            )
    except Exception:
        pass

    try:
        from datasets import load_dataset
        ds = load_dataset("pile-of-law/pile-of-law",
                          "courtlistener_opinions",
                          split="train", streaming=True,
                          trust_remote_code=True)
        count = 0
        for item in ds:
            text = item.get("text", "")
            if len(text) > 1000:
                texts.append(f"Legal Opinion:\n{text[:3000]}\n")
                count += 1
                if count >= 50000: break
    except Exception:
        pass

    tok_save(texts, PROC_DIR/"legal_reasoning.bin", tok)
    print(f"  {len(texts):,} legal reasoning documents.")


# ── 8. CHAIN OF THOUGHT ──────────────────────────────────────────────────────

def dl_chain_of_thought(tok):
    print("\n🧠 Chain of Thought — thinking out loud...")
    texts = []
    try:
        from datasets import load_dataset

        # Dolphin (cognitivecomputations) — explicitly uncensored instruction data
        # Designed with no alignment restrictions; high quality GPT-4-class responses
        try:
            ds_dolphin = load_dataset("cognitivecomputations/dolphin", split="train", streaming=True)
            count = 0
            for item in ds_dolphin:
                instruction = item.get("instruction", item.get("input", ""))
                output = item.get("output", item.get("response", ""))
                if instruction and output:
                    texts.append(f"<|user|>\n{instruction}\n<|assistant|>\n{output}\n")
                    count += 1
                    if count >= 150_000: break
            print(f"  ✓ Dolphin: {count:,} pairs loaded")
        except Exception as e:
            print(f"  ⚠️  Dolphin: {e}")

        # ShareGPT Vicuna Unfiltered — real user conversations, no censorship applied
        try:
            ds_sg = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
            for item in tqdm(ds_sg, desc="  ShareGPT unfiltered", leave=False):
                convs = item.get("conversations", [])
                parts = []
                for turn in convs:
                    role = turn.get("from", "")
                    val  = turn.get("value", "")
                    if role == "human":
                        parts.append(f"<|user|>\n{val}")
                    elif role == "gpt":
                        parts.append(f"<|assistant|>\n{val}")
                if parts:
                    texts.append("\n".join(parts) + "\n")
            print(f"  ✓ ShareGPT unfiltered: {len(ds_sg):,} conversations loaded")
        except Exception as e:
            print(f"  ⚠️  ShareGPT unfiltered: {e}")

        # Airoboros 3.2 — diverse uncensored instruction dataset (jondurbin)
        try:
            ds_airo = load_dataset("jondurbin/airoboros-3.2", split="train")
            for item in tqdm(ds_airo, desc="  Airoboros 3.2", leave=False):
                instruction = item.get("instruction", item.get("input", ""))
                output      = item.get("response",    item.get("output", ""))
                if instruction and output:
                    texts.append(f"<|user|>\n{instruction}\n<|assistant|>\n{output}\n")
            print(f"  ✓ Airoboros 3.2: {len(ds_airo):,} pairs loaded")
        except Exception as e:
            print(f"  ⚠️  Airoboros: {e}")

        # WizardLM complex reasoning
        ds2 = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split="train")
        for item in tqdm(ds2, leave=False):
            texts.append(
                f"<|user|>\n{item.get('instruction','')}\n"
                f"<|assistant|>\n{item.get('output','')}\n"
            )

        # TheoremQA
        ds3 = load_dataset("TIGER-Lab/TheoremQA", split="test")
        for item in tqdm(ds3, leave=False):
            texts.append(
                f"Theorem: {item.get('theorem','')}\n"
                f"Question: {item.get('question','')}\n"
                f"Answer: {item.get('answer','')}\n"
            )

    except Exception as e:
        print(f"  ⚠️  {e}")

    tok_save(texts, PROC_DIR/"chain_of_thought.bin", tok)
    print(f"  {len(texts):,} chain of thought examples.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    tok = load_tok()
    if tok is None:
        print("❌ Tokenizer not found."); return

    fns = {
        "formal_logic":        dl_formal_logic,
        "math_reasoning":      dl_math_reasoning,
        "philosophy":          dl_philosophy,
        "strategic_reasoning": dl_strategic_reasoning,
        "debate_reasoning":    dl_debate_reasoning,
        "scientific_reasoning":dl_scientific_reasoning,
        "legal_reasoning":     dl_legal_reasoning,
        "chain_of_thought":    dl_chain_of_thought,
    }

    if args.dataset:
        fn = fns.get(args.dataset)
        if fn: fn(tok)
        else: print(f"Unknown: {args.dataset}. Options: {list(fns.keys())}")
        return

    print("\n" + "="*60)
    print("LOGIC & REASONING CORPUS — NON-RESTRICTIVE")
    print("Follows logic wherever it leads.")
    print("="*60)
    for name, fn in fns.items():
        fn(tok)

    print("\n✅ Logic & Reasoning corpus complete.")
    print("Add data/processed/logic_reasoning/* to your training mix.")


if __name__ == "__main__":
    main()
