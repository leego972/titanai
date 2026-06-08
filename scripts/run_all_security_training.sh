#!/bin/bash
# TitanAI — Complete Security Training Pipeline
# Runs ALL corpus loaders in correct order on Vast.ai
# Your existing scripts + new security additions

echo "=========================================="
echo " TITANAI COMPLETE SECURITY TRAINING"
echo "=========================================="
cd /workspace

echo ""
echo "PHASE 1 — Existing corpus loaders (your scripts)..."
echo "----------------------------------------------"
python scripts/load_corpus.py                          # Base: Wikipedia, Gutenberg, CVE defensive
python scripts/load_corpus_cyber_web_psych_motors.py   # Cyber systems, web, psychology, motors
python scripts/load_corpus_cs.py                       # CS + Cybersecurity & Cryptography
python scripts/load_corpus_supplement.py               # CVE supplement, CISA advisories
python scripts/load_corpus_engineering.py              # Electrical, mechanics, robotics
python scripts/load_corpus_applied_sciences.py         # Applied sciences
python scripts/load_corpus_business_rd.py              # Business, R&D
python scripts/load_corpus_premium_science.py          # Premium science
python scripts/load_corpus_world_politics.py           # World politics, conflict, cyber

echo ""
echo "PHASE 2 — New offensive security additions..."
echo "----------------------------------------------"
python scripts/prepare_data_final.py --phase 1         # All new: Exploit-DB, MITRE, HackTricks,
                                                       # CTF writeups, PayloadsAllTheThings,
                                                       # HackerOne, GTFOBins, LOLBAS, PEASS,
                                                       # Red team playbooks, OWASP, Phrack,
                                                       # Binary exploitation, Cloud attacks,
                                                       # Sigma rules, YARA, Threat intel,
                                                       # Stack Overflow, CommitPackFT,
                                                       # SWE-bench, Magicoder, CodeFeedback,
                                                       # OpenHermes, System Design, Cloud docs,
                                                       # GSM8K, MetaMath, Orca Math, SlimOrca,
                                                       # WizardLM, TheoremQA, RFCs, arXiv,
                                                       # The Stack v2, CodeSearchNet

echo ""
echo "PHASE 3 — Zero-day and zero-click knowledge..."
echo "----------------------------------------------"
python scripts/prepare_zeroday_knowledge.py            # Project Zero, famous zero-days,
                                                       # Zero-click cases (FORCEDENTRY etc),
                                                       # Vulnerability classes, disclosure ecosystem

echo ""
echo "PHASE 4 — Cybersecurity builder examples..."
echo "----------------------------------------------"
python scripts/gen_cyber_builder.py                    # Your existing pen test builder examples

echo ""
echo "=========================================="
echo " ALL DONE — Run training next:"
echo " python scripts/train_1b.py --config configs/titan_1b_v2.yaml"
echo "=========================================="

echo ""
echo "PHASE 5 — Logic and Reasoning (non-restrictive)..."
echo "----------------------------------------------"
python scripts/load_corpus_logic_reasoning.py          # Formal logic, math reasoning,
                                                       # Philosophy (Plato, Nietzsche,
                                                       # Machiavelli, Sun Tzu, Clausewitz),
                                                       # Game theory, debate both sides,
                                                       # Scientific reasoning, legal reasoning,
                                                       # Chain of thought

echo ""
echo "=========================================="
echo " Knowledge Base Folders Created:"
echo " knowledge-base/film-production/"
echo " knowledge-base/post-production/"
echo " knowledge-base/cybersecurity/"
echo "=========================================="
