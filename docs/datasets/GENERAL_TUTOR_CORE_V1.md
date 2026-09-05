# General Tutor Core v1

## Purpose

A reusable tutoring-behavior dataset that teaches a model how to teach, diagnose misunderstanding, scaffold learning, adapt explanations, and support study across subjects and education levels.

This dataset intentionally contains minimal subject-specific knowledge. Subject competence is supplied by separately purchasable subject packs.

## Commercial architecture

The product model is modular:

1. General Tutor Core — tutoring behavior and pedagogy.
2. Subject Packs — Chemistry, Biology, Mathematics, Physics, History, Economics, Computer Science, etc.
3. Optional Level/Curriculum Adapters — primary, secondary, VCE, U.S. high school, GED, undergraduate, postgraduate and other jurisdiction/assessment mappings.
4. Optional Behavioral Packs — study coach, academic integrity, neurodiversity-friendly tutoring, English-language learner support, research/citation discipline, career readiness and other specialized behavior.

A user can therefore buy only the knowledge they need.

## Knowledge-depth principle

Subject packs should be built to the highest reliable level practical for that discipline, including advanced undergraduate, postgraduate and research/PhD-level knowledge where suitable and sourceable.

The presence of advanced knowledge does **not** mean the tutor should teach every learner at that depth.

The Tutor Core owns instructional-depth control. It must select an explanation, notation, assumptions, examples and problem difficulty appropriate to the learner's current target level and demonstrated prerequisites.

Example: a Physics subject pack may contain research-level material, but a learner studying VCE Physics should receive VCE-appropriate explanations, mathematics, terminology and assessment practice unless they explicitly ask to go beyond the curriculum and have the prerequisites to benefit from it.

## Learner-level routing policy

Before choosing instructional depth, the tutor should resolve or infer:

- subject
- target curriculum or qualification when known
- target level/year/stage
- current topic
- demonstrated prerequisite mastery
- assessment objective, if any
- preferred explanation style
- whether the learner wants curriculum-bounded help or deeper enrichment

The tutor should use the **lowest sufficient depth that fully answers the learner's need**, then deepen only when useful.

### Default depth hierarchy

1. Foundational / primary
2. Junior secondary / middle school
3. Senior secondary / high school
4. VCE / equivalent senior-secondary qualification
5. Introductory undergraduate
6. Intermediate undergraduate
7. Advanced undergraduate / honours
8. Postgraduate coursework
9. Research / PhD-level

A subject pack may support all or only some bands. Unsupported bands must not be claimed.

### Depth-control rules

- Never expose advanced detail merely because it exists in the subject pack.
- Match notation and assumed mathematics to the learner's level.
- Prefer curriculum-aligned terminology when a curriculum adapter is active.
- Do not introduce advanced exceptions before the learner understands the core rule unless the exception is necessary to avoid a misconception.
- Distinguish required curriculum knowledge from optional enrichment.
- Use prerequisite checks before moving a learner upward in depth.
- If the learner demonstrates rapid mastery, increase complexity gradually.
- If the learner repeatedly fails, step down to the missing prerequisite rather than repeating the same explanation.
- When a learner asks for an advanced explanation, provide it if appropriate but state which prerequisites it assumes.
- Assessment practice should stay inside the learner's target assessment level unless explicitly requested otherwise.

## Curriculum-bounded tutoring

When an adapter is active, the adapter defines the expected instructional boundary.

Examples:
- VCE Physics -> teach and assess to the active VCE study design; deeper physics is optional enrichment.
- U.S. High School Biology -> use the selected high-school standards/grade expectations.
- GED Mathematics -> target GED competencies and question styles rather than calculus.
- Undergraduate Organic Chemistry -> assume only prerequisites declared by the selected subject pack/course profile.

The tutor may draw on deeper internal knowledge to create clearer explanations, better analogies and more accurate misconception handling, but it should not force that deeper content into the lesson.

## Subject-pack depth metadata

Every subject record should support level-aware routing metadata where applicable:

- `minimum_level`
- `maximum_level`
- `prerequisites`
- `curriculum_tags`
- `concept_depth`
- `difficulty`
- `assessment_relevance`
- `enrichment_only`
- `notation_profile`

Every subject pack manifest should define a prerequisite graph and concept-depth ladder so the Tutor Core can retrieve the correct slice of knowledge.

## Training examples required for depth control

The Tutor Core corpus must include contrastive training examples where the same underlying concept is taught at different levels, including:

- primary vs secondary explanation
- junior vs senior secondary
- VCE/equivalent vs undergraduate
- undergraduate vs postgraduate
- undergraduate vs PhD/research explanation
- curriculum answer vs optional enrichment
- novice notation vs advanced formal notation
- conceptual explanation vs mathematical derivation
- learner with missing prerequisite vs learner ready for extension

At least 20% of the adaptive-difficulty evaluation set should test whether the tutor correctly **withholds unnecessary advanced detail** while still remaining technically accurate.

The tutor should be penalized both for:

- under-teaching: omitting required material or oversimplifying until wrong; and
- over-teaching: introducing unnecessary advanced material that is outside the learner's level and harms comprehension.

## Target release size

Target: 20,000-35,000+ accepted examples.

The final size is determined by demonstrated tutoring competence, not a fixed row count.

Minimum release expectations:
- >= 1,500 accepted examples per major tutoring competency where applicable
- >= 500 per important subcompetency
- >= 20% hard/expert pedagogical cases
- >= 500 held-out evaluation items total
- >= 100 held-out examples per major core competency
- >= 90% overall correctness for deterministic pedagogical cases where objective scoring is available
- >= 85% per major competency
- transfer, adversarial, misconception and ambiguity evaluation
- no subject-specific pack may be advertised as competent until its own independent subject evaluation passes

## Core tutoring competencies

1. Diagnostic tutoring
   - identify prerequisite gaps
   - distinguish conceptual misunderstanding from arithmetic, notation or reading mistakes
   - infer likely misconceptions from student work
   - ask minimal high-value diagnostic questions

2. Socratic scaffolding
   - guide before revealing
   - control hint granularity
   - use progressive prompting
   - check understanding before advancing

3. Worked-example teaching
   - clear intermediate steps
   - explain why each step is valid
   - compare correct and plausible-but-wrong approaches
   - identify transferable problem-solving patterns

4. Concept explanation
   - intuition first where useful
   - formal definition when appropriate
   - multiple explanation styles
   - connect verbal, symbolic, graphical and tabular representations

5. Misconception remediation
   - identify common misconception families
   - use counterexamples
   - classify error type
   - generate targeted micro-practice

6. Problem decomposition
   - identify givens and unknowns
   - select methods
   - sequence subtasks
   - independently verify the answer

7. Quantitative reasoning support
   - units and dimensions
   - estimation
   - significant figures
   - order-of-magnitude checks
   - sanity checks

8. Reading and comprehension support
   - identify main claim
   - distinguish evidence from interpretation
   - unpack difficult passages
   - vocabulary in context
   - infer without overreaching

9. Writing support
   - argument structure
   - thesis/evidence linkage
   - paragraph cohesion
   - editing and feedback
   - adapt tone and level

10. Research literacy
    - formulate questions
    - source-quality evaluation
    - primary vs secondary evidence
    - bias, confounding and uncertainty
    - citation discipline without fabricated references

11. Study strategy and metacognition
    - retrieval practice
    - spaced practice
    - interleaving
    - self-explanation
    - error logs
    - planning and prioritisation

12. Assessment preparation
    - practice under constraints
    - rubric interpretation
    - targeted revision
    - timed practice
    - post-test error analysis

13. Academic integrity
    - teach rather than impersonate
    - support outlining, feedback, explanation and practice
    - distinguish coaching from ghost-writing
    - preserve learning value while still being useful

14. Adaptive difficulty and depth control
    - foundational through research/PhD level where subject packs support it
    - curriculum-bounded depth selection
    - prerequisite-aware escalation
    - deliberate simplification without factual distortion
    - optional enrichment routing

15. Student communication
    - concise vs detailed modes
    - non-patronising feedback
    - adapt to language proficiency
    - accessible explanations without reducing conceptual accuracy
    - recognize frustration/confusion and alter teaching strategy

16. Uncertainty handling
    - state uncertainty
    - distinguish fact, interpretation and inference
    - identify insufficient information
    - recommend authoritative verification for changing or high-stakes claims

17. Learning-plan construction
    - prerequisite graph
    - mastery sequence
    - diagnostic entry test
    - targeted revision plan
    - progression based on demonstrated mastery

18. Cross-subject transfer
    - identify analogous structures across subjects
    - transfer mathematical/reasoning skills into science/economics/technology
    - transfer reading/evidence skills into humanities/social sciences
    - avoid importing invalid analogies

## Core scenario families

- diagnose a wrong solution
- choose the next hint
- identify missing prerequisite knowledge
- compare two explanations
- transform a poor explanation into a better one
- create a targeted practice sequence
- explain the same concept at several education levels
- select the correct depth for a stated curriculum
- detect when an explanation is technically correct but too advanced
- distinguish required content from enrichment
- grade against a supplied rubric
- critique reasoning without doing the student's entire assessed task
- identify unsupported claims
- distinguish calculation, reading and conceptual errors
- handle ambiguous questions
- handle insufficient information
- decide when to use an analogy and when it would mislead
- produce a worked example followed by a transfer problem
- adapt after repeated student failure
- adapt after rapid student mastery
- identify when memorisation is masking weak understanding
- create retrieval-practice prompts
- construct a prerequisite learning path
- provide formative rather than merely corrective feedback

## Subject-pack contract

Every separately sold subject pack must expose a common schema:

- subject_id
- subject_name
- discipline
- supported_level_bands
- prerequisite graph
- core competencies
- subcompetencies
- canonical concepts/topics
- concept-depth ladder from foundational to highest supported level
- notation/terminology profiles by level where needed
- worked-example families
- misconception catalogue
- problem families
- practical/lab components where relevant
- authoritative open-source registry
- provenance metadata
- train/validation/test split manifest
- held-out evaluation set
- competency report
- compatibility version with General Tutor Core

## Level and curriculum adapters

Level/curriculum adapters do not replace subject packs. They map the same deep subject competence to a particular educational context.

Examples:
- Primary School adapter
- Middle School / Junior Secondary adapter
- General Secondary School adapter
- VCE adapter
- U.S. High School adapter
- GED adapter
- Undergraduate adapter
- Advanced Undergraduate adapter
- Postgraduate adapter

Adapters may define:
- expected depth
- assessment format
- terminology
- curriculum topic order
- permitted calculator/formula-sheet assumptions
- grading conventions
- age-appropriate explanation style
- jurisdiction-specific examples

## Subject marketplace strategy

Each subject should be a separate marketplace product once competent.

The subject pack should be built as deeply as practical once, then reused across levels through Tutor Core routing and curriculum adapters. We should not create a shallow VCE Chemistry knowledge pack and a separate deep University Chemistry knowledge pack unless licensing, architecture or evaluation evidence shows a genuine need; one deep Chemistry pack with level metadata is preferred.

## Free/open-source policy

The dataset program should use authoritative free/open/public sources wherever licensing permits.

Source text should normally be transformed into independently authored examples rather than copied wholesale. Every factual record must retain provenance metadata.

Paid textbooks, commercial question banks and paid training corpora are not required for this product strategy.

Every source family must receive an explicit license and redistribution/training-use review before inclusion.

## Competency rule

General Tutor Core is not complete because it reaches a row target. It is complete only when held-out evaluation demonstrates strong tutoring behavior across unfamiliar subjects, education levels and problem types.

Likewise, no subject pack is sellable merely because data exists. Each subject pack must independently meet the Titan Dataset Competency & Release Standard before marketplace publication.
