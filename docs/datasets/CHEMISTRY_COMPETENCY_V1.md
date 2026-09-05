# Chemistry Competency v1

## Purpose
Create a rigorous chemistry training dataset suitable for secondary/VCE tutoring, general scientific reasoning and foundation-level tertiary support. The corpus must teach conceptual understanding, quantitative problem solving, experimental reasoning and evidence interpretation rather than rote recall.

## Target scale
Full release target: 20,000-35,000+ accepted records.

## Core competencies
1. Atomic structure, periodicity and bonding
2. Structure-property relationships
3. Stoichiometry and quantitative chemistry
4. Chemical reactions and equations
5. Thermochemistry and energy changes
6. Equilibrium and Le Chatelier reasoning
7. Acids, bases, buffers and pH
8. Redox and electrochemistry
9. Organic chemistry and reaction families
10. Analytical chemistry and spectroscopy/chromatography concepts
11. Fuels, materials and sustainability chemistry
12. Experimental design, uncertainty and data analysis
13. Scientific communication and evidence evaluation

## Example families
- concept explanation
- worked numerical problem
- unit/dimensional analysis
- error diagnosis from student working
- multiple representations: symbolic, particle-level, macroscopic
- graph/table interpretation
- reaction pathway reasoning
- equilibrium perturbation cases
- titration reasoning
- electrochemical cell interpretation
- chromatography/spectroscopy interpretation at appropriate level
- experimental design and controls
- uncertainty/significant-figure critique
- compare competing explanations
- real-world chemistry application
- hard negative / tempting misconception
- mixed-topic transfer

## Quantitative verification
All deterministic calculations should be generated from underlying structured variables and independently recomputed in code before acceptance. Reject records where the worked solution, units or significant figures are inconsistent.

## VCE alignment
For VCE-facing subsets, map records to the current VCAA Chemistry study design and key science skills. VCAA material is used as a curriculum blueprint; original questions and solutions must be authored independently.

## Open-source strategy
Prioritise free/open authoritative sources after licence review, including public scientific-agency data, PubChem/NIH material where terms permit, NIST chemistry reference data where permitted, IUPAC open nomenclature/reference resources, OpenStax chemistry where licence-compatible, and original synthetic numerical problems.

## Release requirements
- >= 1,500 accepted examples per core competency unless a narrower cluster is justified
- broad clusters should generally exceed 2,000 examples
- >= 500 held-out evaluation items overall and >= 100 per core competency
- >= 90% overall deterministic correctness
- >= 85% per competency
- family-isolated train/validation/test split
- near-duplicate and answer-skeleton checks
- provenance/licensing audit
- stratified human review
- adversarial misconception and transfer tests

## Marketplace status
IN DEVELOPMENT. Not sellable until competency is proven.