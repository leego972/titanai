# Titan Dataset Quality Expansion Method v2

## Purpose

Increase dataset size without degrading quality or creating thousands of superficial paraphrases.

A generated record is only a candidate. It becomes an accepted training record only after structural, semantic, factual/deterministic, diversity, leakage and review gates pass.

## Core change

Do not scale by writing one prompt/answer template and changing nouns or numbers.

Scale through a hierarchy:

1. competency
2. sub-competency
3. independent scenario blueprint
4. reasoning archetype
5. context/domain
6. evidence bundle
7. difficulty transformation
8. distractor/edge-case transformation
9. answer strategy
10. independent verification

A record must differ meaningfully at several levels, not only wording.

## Required blueprint density

Before a broad competency can be called complete:

- >= 40 independently authored scenario blueprints per core competency
- >= 8 reasoning archetypes per core competency
- >= 6 operational/business contexts per core competency where transfer matters
- >= 5 adversarial/edge-case families per core competency
- >= 3 answer structures per major reasoning archetype
- no blueprint may contribute > 2.5% of a core competency's accepted training records
- no reasoning archetype may contribute > 20% of a core competency unless justified

For a 10-competency dataset this means at least 400 independent blueprints before high-volume expansion.

## Candidate-to-accepted pipeline

### Gate A — schema and completeness
Reject if any required field, answer section, provenance field, verification field, split family or competency tag is absent.

### Gate B — deterministic/factual verification
Where a target answer can be calculated, calculate it independently in code.
Where it depends on standards or factual guidance, require an approved source family and a testable control principle.
Reject unverifiable claims.

### Gate C — semantic diversity
Reject:
- exact duplicates
- near duplicates
- prompt embedding similarity above the configured threshold against accepted records from the same competency
- excessive answer-skeleton overlap
- repeated evidence bundles
- parameter-only substitutions

### Gate D — scenario contribution
A candidate must add one or more of:
- new scenario blueprint
- new evidence pattern
- new context transfer
- new difficulty mechanism
- new edge case
- new misleading alternative
- new skill combination

If it adds none, reject it even if wording is unique.

### Gate E — difficulty integrity
Difficulty is produced by problem structure, not longer prose.

Medium:
- one principal skill
- mostly relevant evidence
- limited ambiguity

Hard:
- multiple evidence sources
- distractors or incomplete evidence
- at least two plausible alternatives
- requires prioritisation or trade-off reasoning

Expert:
- cross-competency interaction or multi-stage decision
- conflicting/partial evidence
- explicit uncertainty
- operational constraints
- requires validation plan and residual-risk reasoning

### Gate F — split isolation
Assign the scenario blueprint to train/validation/test before record expansion.
Every derivative of that blueprint stays in the same split.
Do not randomly split parameter variants.

### Gate G — contrast pairs and hard negatives
At least 25% of broad reasoning datasets should be part of a contrast pair/triplet where a small but material change alters the correct conclusion or action.

Examples:
- same alert, different asset exposure
- same transaction, different fee/slippage assumptions
- same historical claim, different source quality
- same pentest action, authorized target vs out-of-scope target
- same authentication anomaly, legitimate travel vs impossible-travel evidence

The model should learn decision boundaries, not keyword associations.

### Gate H — reviewer sampling
Every generation cohort must produce a stratified review pack before acceptance.
Review samples must include:
- every competency
- every difficulty level
- every new blueprint family
- contrast cases
- rejected-looking-but-correct cases
- plausible wrong alternatives

Quality defects trigger redesign of the affected blueprint family, not merely deletion of one row.

## Rich record format

High-value records should include structured metadata where useful:

```json
{
  "id": "...",
  "competency": "...",
  "subcompetency": "...",
  "blueprint_id": "...",
  "reasoning_archetype": "...",
  "difficulty": "hard",
  "split_family": "...",
  "context": "...",
  "evidence": [
    {"source": "auth_log", "observation": "...", "reliability": "high"},
    {"source": "change_log", "observation": "...", "reliability": "high"}
  ],
  "distractors": ["..."],
  "uncertainties": ["..."],
  "ground_truth": {
    "decision": "...",
    "must_verify": ["..."],
    "must_not_assume": ["..."]
  },
  "provenance": ["..."],
  "messages": ["..."]
}
```

The structured fields make automated quality checks stronger and allow future re-rendering into different chat formats without changing the latent problem.

## Multi-turn examples

At least 10% of an advanced dataset should use multi-turn cases where the assistant receives additional evidence after an initial answer and must update its conclusion rather than defend the first hypothesis.

This tests and teaches:
- belief revision
- uncertainty calibration
- maintaining context
- abandoning an attractive but disproven hypothesis
- asking for high-value missing evidence

## Cross-competency examples

At least 15% of the final broad corpus should require two or more competencies.

Examples for security:
- identity + incident response
- vulnerability management + business risk
- secure software + logging/detection
- secrets management + cloud configuration
- recovery + governance

These records are additional to minimum per-competency quotas and should not be used to hide weak single-competency coverage.

## Volume strategy

Build in cohorts of 1,000-2,500 accepted records.

After each cohort:
1. measure coverage
2. measure near-duplicate rate
3. measure blueprint/archetype concentration
4. inspect stratified review sample
5. run held-out evaluation if a training candidate is available
6. expand only the weakest competencies and failure modes

Do not generate the full target size blindly in one run.

## Competency proof

A large corpus is still not competent until the held-out model evaluation passes the Titan Dataset Competency & Release Standard.

If evaluation plateaus, stop adding generic records. Add targeted examples for the exact failure modes identified by the benchmark.
