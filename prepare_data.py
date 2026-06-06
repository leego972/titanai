#!/usr/bin/env python3
"""
TitanAI 1B — Data Preparation Pipeline
=======================================
Downloads, cleans, tokenizes, and saves all training datasets in the correct mix.

Dataset mix (Phase 1 - Continued Pretraining):
  35%  FineWeb          — High-quality web text (HuggingFace)
  25%  The Stack v2     — Python, TypeScript, Go, C, Bash code (BigCode)
  12%  NVD/CVE          — NIST vulnerability database (free JSON)
  10%  Exploit-DB       — Public exploit archive (GitHub)
   8%  MITRE ATT&CK     — Adversary tactics/