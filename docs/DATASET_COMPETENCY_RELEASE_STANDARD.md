# Titan Dataset Competency & Release Standard

Purpose: prevent premature or low-value dataset releases. No dataset is considered complete because it reached a record-count target. Competence must be demonstrated on unseen evaluation data.

## Governing principle

Quality > coverage > volume > speed.

If a dataset fails any competency, factual, provenance, diversity, safety, leakage, or transfer requirement, it remains unreleased regardless of how many records have been generated.

## Minimum release criteria

A dataset may be marked `verified` only when all applicable gates pass.

### 1. Competency coverage
- Each core competency: >= 1,500 accepted training examples unless the skill is intrinsically narrow.
- Complex reasoning competencies: target 2,000-5,000+ accepted examples per core competency.
- Important sub-competency: >= 500 accepted examples.
- Narrow specialist sub-skill: >= 200 accepted examples where justified.
- At least 20% of accepted examples must be hard or expert for reasoning-oriented datasets.
- Each core skill must appear across multiple scenario families, domains, phrasings, and edge cases.
- No competency may be dominated by one template family.

These are floors, not completion guarantees. More data must be added when unseen evaluation performance is insufficient.

### 2. Unseen evaluation gate
A held-out evaluation set must be isolated before training or packaging.

Required:
- no prompt, answer, scenario-family or entity leakage from training into evaluation
- evaluation set size sufficient for statistical confidence; normally >= 500 items for a broad dataset and >= 100 per core competency
- >= 90% overall correctness for deterministic/factual tasks unless a stricter domain threshold is defined
- >= 85% per core competency
- no core competency below its floor even if the overall score passes
- materially lower thresholds are not permitted merely to ship a dataset

### 3. Transfer/generalisation gate
At least 20% of the evaluation set must use unfamiliar surface forms or combinations of known skills.

The target model must demonstrate that it can:
- solve equivalent problems with changed entities/numbers/contexts
- combine two or more trained sub-skills
- handle irrelevant information and distractors
- distinguish fact, inference, uncertainty and insufficient evidence
- refuse to infer beyond supplied evidence where appropriate

### 4. Adversarial and edge-case gate
Each core competency must include adversarial or edge cases covering applicable failure modes such as:
- misleading framing
- contradictory evidence
- missing data
- ambiguous terminology
- outliers
- boundary values
- tempting but invalid shortcuts
- confounders
- plausible-but-wrong alternatives

For security datasets, adversarial content remains defensive/diagnostic and must not become an offensive exploitation manual.

### 5. Diversity gate
Automated checks must measure and reject:
- exact duplicates
- near duplicates
- excessive lexical overlap
- repeated answer skeletons
- excessive template concentration
- scenario-family imbalance
- entity/name repetition that encourages memorisation

No single scenario family should normally exceed 10% of a broad dataset unless explicitly justified in the manifest.

### 6. Factual/provenance gate
For factual datasets:
- every factual record must have traceable provenance metadata
- volatile facts must include an as-of date or be excluded from weight-training products when RAG is more appropriate
- disputed historical/political interpretations must be labelled as interpretation, not settled fact
- source licensing/redistribution rights must be reviewed
- unverifiable generated claims are rejected

### 7. Human review gate
Before release:
- random stratified review across every competency and difficulty band
- minimum review sample: max(200, 2% of dataset), with at least 20 examples from every core competency
- zero tolerance for fabricated citations, broken answers, corrupt schema, or unsafe prohibited content in the reviewed sample
- material quality defects trigger remediation and a fresh review sample

### 8. Split integrity
Train/validation/test partitions must be isolated by semantic family where practical, not merely random row split.

Required checks:
- exact-hash leakage
- near-duplicate leakage
- scenario-family leakage
- answer-pattern leakage where measurable

### 9. Packaging/integrity
Before marketplace publication:
- README
- manifest
- schema description
- competency coverage report
- evaluation report
- provenance/licensing summary
- train/validation/test split description
- SHA-256 integrity hashes
- commercial licence
- reproducible package build

### 10. Marketplace release decision
`verified` means all gates above have passed. A dataset that only passes schema validation is `generated`, not `verified`.

Marketplace upload is the final step, never the quality-control mechanism.

## Target corpus sizes for current universal products

These are expected full-release ranges, not hard caps:
- Defensive Security Foundations: 12,000-20,000+
- General Knowledge Core: 30,000-60,000+
- World History & Historical Reasoning: 20,000-35,000+
- Science & Technology Core: 20,000-35,000+
- Data & Financial Literacy: 15,000-25,000+
- Communication & Critical Thinking: 15,000-25,000+
- Advanced Data Analysis: 15,000-25,000+
- Complex Reasoning: 20,000-40,000+

The corpus may exceed these ranges when evaluation results show that additional coverage is needed.

## Stop conditions
Generation must pause for redesign rather than continue if:
- duplicate/near-duplicate rate rises materially
- new records fail to add meaningful competency coverage
- factual verification failure rate increases
- eval gains plateau while volume continues to rise
- one template family dominates
- safety or provenance cannot be established

The correct response to a quality failure is to improve the generator, sources, or curriculum—not to lower the acceptance threshold.
