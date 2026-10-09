# Titan Defensive Security Foundations v1

Status: IN DEVELOPMENT — NOT SELLABLE

Goal: produce a competency-grade defensive cybersecurity SFT/evaluation corpus. Record count is necessary but never sufficient for release.

## Release target

- Target accepted corpus: 12,000–20,000+ records, expandable until competency gates pass.
- Minimum 1,000 accepted training examples for each core competency unless an evidence-based evaluation shows a higher requirement.
- Minimum 250 hard/expert transfer examples per core competency.
- Minimum 100 adversarial/edge-case examples per core competency.
- Separate held-out evaluation families; no scenario-family leakage across train/validation/test.

## Core competencies

1. Identity, authentication and authorization
2. Phishing/social-engineering recognition and response
3. Secrets, keys and credential lifecycle
4. Secure configuration and cloud hygiene
5. Secure software design and dependency risk
6. Logging, monitoring and detection reasoning
7. Vulnerability and remediation prioritisation
8. Incident triage, containment, response and recovery
9. Backup, resilience and recovery assurance
10. Security governance, risk and control selection

## Authoritative source families

- NIST Cybersecurity Framework 2.0 (CSF 2.0): GOVERN, IDENTIFY, PROTECT, DETECT, RESPOND, RECOVER.
- NIST SP 800-61 Rev. 3 (2025): incident response recommendations aligned with CSF 2.0.
- OWASP Top 10:2021 and OWASP Cheat Sheet Series: application-security risk awareness and defensive controls.
- CISA defensive guidance where applicable for phishing, secure-by-design, identity and operational resilience.

Source material is used to ground principles and competency coverage. Dataset records must be original scenarios and explanations; source text must not be copied verbatim beyond minimal terminology.

## Record requirements

Every accepted record must include:
- unique id
- competency and subcompetency
- difficulty (medium/hard/expert)
- scenario_family
- messages (system/user/assistant)
- evidence/rationale
- source_family identifiers
- verification mode
- safety classification
- split-family identifier

Answers must:
- distinguish observation from inference
- prioritise proportionate defensive action
- avoid unsupported certainty
- avoid offensive exploitation instructions
- explain why plausible alternatives are weaker when relevant
- preserve business continuity/evidence when containment decisions require tradeoffs

## Competency gate

A dataset cannot be marked complete until a held-out evaluator demonstrates:
- >= 90% aggregate score on deterministic/factual security tasks
- >= 85% score for every core competency
- >= 85% transfer score on unfamiliar scenario families
- no critical unsafe-action failures in safety evaluation
- no core competency with material systematic error

These are floors, not automatic acceptance criteria. If qualitative review identifies weak reasoning, poor coverage or template concentration, the dataset remains in development even when numeric thresholds pass.

## Release gate

Before marketplace publication:
1. source/provenance audit
2. schema validation
3. exact + semantic duplicate rejection
4. family-level split isolation
5. competency coverage report
6. factual/defensive-practice spot review
7. adversarial safety review
8. held-out competency evaluation
9. README/manifest/license
10. SHA-256 integrity manifest
11. package upload verification

No marketplace seed should reference this dataset before all gates pass.
