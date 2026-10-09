# Titan Dataset Competency Coverage Standard

Purpose: ensure every commercial Titan dataset contains enough high-quality examples to teach a capability, not merely demonstrate it.

## Core rule
A dataset is not complete because it reaches a total record count. Completion requires sufficient accepted examples for every material competency and sub-competency in its scope.

## Minimum accepted-example thresholds
- core competency: 500+ accepted examples
- important sub-competency: 200+ accepted examples
- narrow specialist skill: 100+ accepted examples when the scope is genuinely narrow
- high-risk or highly variable competency: increase quota until edge-case and transfer coverage is adequate

These are minimums, not targets. More examples are required where diversity, difficulty, regional variation, numerical variation, adversarial cases, ambiguity, or long-tail failure modes justify them.

## Coverage dimensions required per competency
Each competency must include multiple independent examples across:
- medium, hard and expert difficulty where applicable
- direct and indirect formulations
- clean and noisy evidence
- ordinary and edge-case scenarios
- single-step and multi-step variants where applicable
- multiple domains or contexts to test transfer
- positive, negative and counterexample cases
- ambiguity and insufficient-evidence cases where appropriate
- adversarial or misleading framing where appropriate

## Universal dataset minimum release sizes
These are planning floors only; quality gates may require larger corpora:
- Defensive Security Foundations: 5,000+ accepted records
- General Knowledge Core: 8,000+ accepted records
- World History & Historical Reasoning: 6,000+ accepted records
- Science & Technology Core: 6,000+ accepted records
- Data & Financial Literacy: 5,000+ accepted records
- Communication & Critical Thinking: 5,000+ accepted records

## Advanced reasoning datasets
Advanced reasoning datasets should generally contain at least 5,000 accepted records unless intentionally sold as a small specialist curated pack. A specialist pack must make its narrow scope explicit and cannot be represented as broad competency training.

## Rejection policy
Do not lower standards to satisfy quotas. Reject records that are:
- factually unsupported or unverifiable
- near-duplicates or paraphrase padding
- trivial relative to the labelled difficulty
- overly templated or distributionally repetitive
- dependent on hidden assumptions not stated in the prompt
- ambiguous without intentionally testing ambiguity
- contaminated across train/validation/test families
- unsafe for the stated product scope
- culturally or geographically narrow without a deliberate reason

If a competency cannot meet its quota at the required quality level, generation stops and the dataset remains unreleased.

## Release evidence
Every completed dataset must publish a machine-readable coverage report containing:
- total accepted/rejected record counts
- counts by competency and sub-competency
- counts by difficulty
- counts by scenario family
- split counts
- duplicate and near-duplicate results
- leakage test results
- factual/provenance verification summary
- safety review summary where relevant
- reviewer spot-check statistics
- SHA-256 integrity hashes

## Marketplace rule
No dataset can be listed as `verified` until its competency coverage report proves all required quotas and quality gates have passed.
