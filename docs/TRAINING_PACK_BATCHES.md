# TitanAI Professional Dataset Replacement Batches

TitanAI remains a single 3B target. This plan replaces weak legacy corpora in controlled batches so each stage can be audited, regenerated, resumed, and trained independently.

## Batch 1 — Core Engineering and Defensive Security

Priority: immediate.

Domains:
- Python, JavaScript/TypeScript, Rust, Go, Java, C/C++, SQL, Bash and PowerShell
- debugging, testing, code review, architecture, API design, databases, distributed systems and deployment
- secure coding, vulnerability analysis, hardening, detection engineering, incident response and threat modelling
- structured knowledge derived from public primary sources such as CWE, CAPEC, ATT&CK, NVD, CISA KEV, NIST and OWASP

Training forms:
- continued-pretraining documents
- professional instruction/response records
- repository and issue-to-patch trajectories
- preference pairs for review quality, correctness and maintainability
- evaluation-only records held out from training

Legacy replacement targets:
- repetitive `advanced_training_pack_*.txt` material
- shallow single-turn Q&A
- unverified synthetic explanations
- duplicate or template-expanded cyber records

## Batch 2 — Advanced Software Construction

Domains:
- repository-scale implementation
- refactoring and migration
- performance engineering
- concurrency and distributed systems
- CI/CD, containers, observability and cloud-neutral operations
- issue diagnosis, patch design, tests and rollback planning

## Batch 3 — Cybersecurity Depth

Domains:
- network, endpoint, identity, application, cloud and supply-chain security
- malware analysis concepts and defensive reverse engineering
- detection logic, telemetry, triage and incident reconstruction
- security architecture and red-team/blue-team reasoning
- exploit-root-cause understanding paired with remediation and regression tests

This batch does not introduce new behavioural restrictions. It improves technical depth, evidence quality and defensive competence.

## Batch 4 — Reasoning, Mathematics and Science

Domains:
- formal and quantitative reasoning
- mathematics, statistics and probability
- physics, computing fundamentals and systems reasoning
- evidence evaluation and uncertainty calibration

## Batch 5 — Professional Communication and Business Engineering

Domains:
- requirements analysis
- technical writing and documentation
- project planning and estimation
- product, operations, finance and legal-document comprehension
- stakeholder communication and decision records

## Batch 6 — Alignment to Titan Workflows

Domains:
- tool use
- multi-step planning
- repository navigation
- structured output
- self-checking and error correction
- DPO preference data based on correctness, completeness and engineering quality

## Mandatory pack standard

Every uploaded source pack must provide or receive generated metadata for:
- source and provenance
- licence or usage basis
- retrieval date/version
- content hash
- domain and training stage
- quality score and rejection reasons
- contamination status
- exact-duplicate and near-duplicate status

No pack becomes trainable until it passes schema, integrity, provenance, deduplication and quality checks.
