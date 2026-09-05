# Biology Competency v1

## Purpose
Create a rigorous biology dataset for VCE tutoring, general science competence and foundation tertiary support. The corpus must teach biological mechanisms, experimental reasoning, data interpretation and evidence-based explanation rather than rote recall.

## Target scale
Full release target: 20,000-35,000+ accepted records.

## Core competencies
1. Cell structure and function
2. Biomolecules, enzymes and metabolism
3. Membranes, transport and homeostasis
4. Cellular respiration and photosynthesis
5. DNA, genes, chromosomes and gene expression
6. Cell division, inheritance and variation
7. Molecular genetics and biotechnology
8. Evolution and natural selection
9. Population genetics and speciation
10. Immunity, pathogens and disease response
11. Organism physiology and regulation
12. Ecology, populations and ecosystems
13. Experimental design, controls, uncertainty and data analysis
14. Scientific communication and evidence evaluation

## Example families
- mechanism explanation
- diagram-to-text reasoning
- sequence/process reconstruction
- genetics probability and pedigree reasoning
- experimental design
- controls and confounders
- graph/table interpretation
- compare biological models
- mutation-to-phenotype reasoning
- evolutionary scenario analysis
- immune response sequence reasoning
- ecology population scenario
- data-supported claim evaluation
- misconception diagnosis
- hard negative / plausible-but-wrong explanation
- mixed-topic transfer

## Quantitative verification
Genetics ratios, population calculations, rates, proportions and other deterministic tasks should be generated from structured variables and independently verified by code.

## VCE alignment
For VCE-facing subsets, map records to the current VCAA Biology study design and key science skills. VCAA resources define curriculum scope and assessment expectations only; training questions and solutions must be independently authored.

## Open-source strategy
Prioritise free/open authoritative sources after licence review, including NIH/NCBI resources where terms permit, CDC/WHO public educational data where licensing permits, NASA/NOAA/USGS environmental datasets, open biodiversity datasets such as GBIF where licence-compatible, OpenStax Biology, Wikidata CC0 and original synthetic experimental datasets.

## Release requirements
- >= 1,500 accepted examples per core competency unless a narrower cluster is justified
- broad clusters should generally exceed 2,000 examples
- >= 500 held-out evaluation items overall and >= 100 per core competency
- >= 90% overall deterministic correctness where applicable
- >= 85% per competency
- family-isolated train/validation/test split
- near-duplicate and answer-skeleton checks
- provenance/licensing audit
- stratified human review
- adversarial misconception and transfer tests

## Marketplace status
IN DEVELOPMENT. Not sellable until competency is proven.