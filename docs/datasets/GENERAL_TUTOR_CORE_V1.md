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

Example bundles:
- Tutor Core + Chemistry
- Tutor Core + Chemistry + Biology
- Tutor Core + Mathematics + Physics + U.S. High School adapter
- Tutor Core + Organic Chemistry + Biochemistry + Undergraduate adapter
- Tutor Core + Economics + Statistics + Research Methods

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

14. Adaptive difficulty
    - beginner
    - intermediate
    - advanced
    - secondary-school
    - undergraduate
    - postgraduate bridge where supported by subject pack

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
- explain at several depths
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
- notation/terminology
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

Level/curriculum adapters do not replace subject packs. They map the same subject competence to a particular educational context.

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

High-demand packs to prioritize:

### Mathematics
- Arithmetic & Numeracy
- Algebra
- Geometry
- Trigonometry
- Precalculus
- Calculus
- Linear Algebra
- Statistics & Probability
- Discrete Mathematics
- Differential Equations

### Sciences
- General Science
- Biology
- Chemistry
- Physics
- Earth & Environmental Science
- Astronomy
- Organic Chemistry
- Biochemistry
- Genetics
- Microbiology
- Anatomy & Physiology

### Computing and technology
- Computer Literacy
- Programming Fundamentals
- Python
- JavaScript/TypeScript
- Data Structures & Algorithms
- Databases
- Networking
- Cybersecurity
- Software Engineering
- AI & Machine Learning
- Data Science

### Humanities and social sciences
- World History
- Australian History
- U.S. History
- Geography
- Economics
- Psychology
- Sociology
- Philosophy
- Logic
- Politics & Civics
- International Relations

### Business and finance
- Accounting
- Business Management
- Finance
- Financial Literacy
- Trading & Market Analysis
- Cryptocurrency & Blockchain
- Entrepreneurship
- Marketing
- Operations Management

### Language and communication
- English Language
- English Literature
- Academic Writing
- Creative Writing
- Technical Writing
- Research Methods
- Public Speaking
- Professional Communication

### Professional and university packs
- Engineering foundations
- Law foundations
- Research methods
- Epidemiology
- Public Health
- Pharmacology foundations
- subject-specific advanced packs as demand warrants

## Optional behavioral packs

Behavior can also be modularized and sold separately when it materially changes tutoring style or capability:

- Socratic Tutor
- Study Coach & Metacognition
- Academic Integrity
- Neurodiversity-Friendly Tutor
- English-Language Learner Support
- Research & Citation Discipline
- Exam Preparation Coach
- Career Readiness
- Professional Communication
- Critical Thinking
- Decision-Making & Consequence Reasoning
- Respectful Communication & Conflict Resolution
- Digital Literacy & Online Safety

## Free/open-source policy

The dataset program should use authoritative free/open/public sources wherever licensing permits.

Source text should normally be transformed into independently authored examples rather than copied wholesale. Every factual record must retain provenance metadata.

Paid textbooks, commercial question banks and paid training corpora are not required for this product strategy.

Every source family must receive an explicit license and redistribution/training-use review before inclusion.

## Competency rule

General Tutor Core is not complete because it reaches a row target. It is complete only when held-out evaluation demonstrates strong tutoring behavior across unfamiliar subjects and problem types.

Likewise, no subject pack is sellable merely because data exists. Each subject pack must independently meet the Titan Dataset Competency & Release Standard before marketplace publication.
