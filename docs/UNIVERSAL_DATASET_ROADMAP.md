# Titan Universal Dataset Roadmap

Purpose: build broadly useful, commercially sellable datasets that complement Titan's existing reasoning corpus. Quality takes priority over speed or raw record count.

## Priority 1 — Defensive Security Foundations
Target: 3,000–5,000 records in v1.

Coverage:
- identity and access control
- authentication/session security
- phishing and social-engineering recognition
- cloud configuration hygiene
- secrets management
- logging and incident triage
- vulnerability prioritisation
- secure software-development practices
- network/security monitoring
- backup/recovery and resilience
- defensive threat analysis

Restrictions:
- defensive, diagnostic and preventive content only
- no malware construction, credential theft, persistence, exploitation playbooks or evasion instructions
- every scenario should distinguish evidence from inference and recommend proportionate defensive action

## Priority 2 — General Knowledge Core
Target: 5,000–10,000 records in v1.

Coverage:
- geography
- civics and institutions
- science fundamentals
- technology fundamentals
- economics fundamentals
- arts and culture
- major world organisations
- everyday quantitative literacy
- environment and climate fundamentals
- media/information literacy

Quality rules:
- favour stable facts over trivia
- source-aware provenance metadata
- avoid culturally narrow question sets
- include explanation and context, not only fact recall

## Priority 3 — World History & Historical Reasoning
Target: 5,000–8,000 records in v1.

Coverage:
- ancient civilisations
- classical world
- medieval societies
- early modern period
- industrialisation
- imperialism and decolonisation
- world wars
- cold war
- political/economic transformations
- scientific and cultural history
- major regional histories across Africa, Asia, Europe, the Middle East, Oceania and the Americas

Quality rules:
- chronology plus causal reasoning
- distinguish primary fact from interpretation
- multiple regional perspectives where materially relevant
- avoid presenting disputed interpretations as settled fact

## Priority 4 — Science & Technology Core
Target: 5,000–8,000 records.

Coverage:
- physics
- chemistry
- biology
- earth science
- astronomy
- computing
- engineering fundamentals
- AI/data fundamentals
- scientific method and experimental reasoning

## Priority 5 — Data & Financial Literacy
Target: 4,000–6,000 records.

Coverage:
- percentages, ratios and growth
- charts/tables
- probability and uncertainty
- budgeting and cash flow
- unit economics
- basic accounting concepts
- inflation and interest
- misleading statistics
- business KPI interpretation

## Priority 6 — Communication & Critical Thinking
Target: 4,000–6,000 records.

Coverage:
- argument quality
- ambiguity detection
- logical fallacies
- evidence weighting
- concise professional communication
- negotiation/de-escalation
- reading instructions/contracts critically
- source reliability

## Release policy
No dataset is listed for sale until all of the following are true:
1. source/provenance checks pass
2. schema validation passes
3. duplicate/near-duplicate tests pass
4. train/validation/test leakage checks pass
5. factual spot checks pass
6. safety review passes where relevant
7. package manifest, README and license are complete
8. integrity hashes are generated
9. marketplace package upload succeeds

## Pricing
- hard floor: USD $5 for any paid dataset
- small specialist curated packs: $5–$9
- larger high-value datasets: generally $9–$19 at launch
- bundles may be priced above this while remaining volume-oriented
