# Titan Dataset Competency Coverage Standard

This standard defines the minimum evidence required before a Titan dataset may be described as competency-building or listed for commercial sale.

## Governing rule

Total record count alone is not sufficient. Every major competency inside a dataset must have enough independent, high-quality examples to teach transfer across contexts rather than memorisation of one template family.

## Minimum competency quotas

- Core competency: minimum 500 accepted examples.
- Important sub-competency: minimum 200 accepted examples.
- Narrow specialist skill: minimum 100 accepted examples when it is genuinely bounded and deterministic.
- Universal foundational datasets should target at least 3,000-5,000 accepted records before a full v1 release.
- Advanced reasoning datasets should target at least 5,000 accepted records when the subject contains multiple reasoning modes.

These are minimums, not targets. A dataset should grow beyond them when coverage analysis identifies underrepresented contexts, edge cases or difficulty bands.

## Required variation within each competency

Each competency quota must include:
- multiple scenario families;
- multiple domains/contexts where transfer matters;
- normal cases and edge cases;
- misleading or irrelevant information where appropriate;
- counterexamples or negative cases;
- medium, hard and expert difficulty where the skill supports those levels;
- varied wording and problem structure;
- independent evidence or deterministic verification where feasible.

No single scenario/template family may contribute more than 20% of a competency's examples in a commercial release unless the skill is intrinsically narrow and this exception is documented.

## Split isolation

Train, validation and test sets must be separated by scenario family or latent problem structure, not random rows alone. Near-duplicates, paraphrase variants and parameter substitutions from the same family must remain within one split.

Target split:
- Train: 90%
- Validation: 5%
- Test: 5%

For competency evaluation, validation and test must each contain enough examples to measure performance meaningfully. Small competencies must use larger proportional holdouts if necessary.

## Competency readiness gate

A competency is not marked ready until:
1. minimum quota is reached;
2. duplicate and near-duplicate checks pass;
3. coverage matrix shows no material scenario gap;
4. factual/deterministic validation passes;
5. human spot-review passes on a stratified sample;
6. train/validation/test leakage checks pass;
7. answer quality and difficulty labels pass review.

## Marketplace rule

A dataset may be sold only when all advertised competencies pass this readiness gate. Seed milestones may be committed internally for review, but must not be marketed as complete competency datasets.

## Initial universal dataset targets

- Defensive Security Foundations v1: 5,000+ accepted examples across prevention, identity/access, phishing, malware recognition, incident response, vulnerability management, logging/monitoring, backup/recovery, cloud/application security and governance.
- General Knowledge Core v1: 8,000+ accepted examples across geography, civics, science, technology, economics, culture, institutions, environment and everyday quantitative/factual literacy.
- World History & Historical Reasoning v1: 6,000+ accepted examples across major periods/regions plus chronology, causality, continuity/change, source interpretation and competing explanations.
- Science & Technology Core v1: 6,000+ accepted examples.
- Data & Financial Literacy v1: 5,000+ accepted examples.
- Communication & Critical Thinking v1: 5,000+ accepted examples.

Quality gates override these numerical targets. If a field cannot sustain the target without repetition or weak records, generation stops and the gap is documented rather than padded.
