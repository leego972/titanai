# General Tutor Core v1

## Purpose

A reusable tutoring-behavior dataset that teaches a model how to teach, diagnose misunderstanding, scaffold learning, adapt explanations, sustain attention, and support study across subjects and education levels.

This dataset intentionally contains minimal subject-specific knowledge. Subject competence is supplied by separately purchasable subject packs.

## Commercial architecture

The product model is modular:

1. General Tutor Core — tutoring behavior, pedagogy and engagement intelligence.
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

Target: 25,000-45,000+ accepted examples.

The final size is determined by demonstrated tutoring competence, engagement quality and transfer performance, not a fixed row count.

Minimum release expectations:
- >= 1,500 accepted examples per major tutoring competency where applicable
- >= 500 per important subcompetency
- >= 20% hard/expert pedagogical cases
- >= 20% engagement-adaptation cases
- >= 500 held-out evaluation items total
- >= 100 held-out examples per major core competency
- >= 90% overall correctness for deterministic pedagogical cases where objective scoring is available
- >= 85% per major competency
- transfer, adversarial, misconception, boredom/disengagement and ambiguity evaluation
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
    - postgraduate/research where supported by the subject pack
    - use the lowest sufficient depth that fully satisfies the learner's objective
    - penalize both under-teaching and unnecessary over-teaching

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

19. Engagement diagnosis
    - detect boredom, confusion, overload, low confidence, passive agreement and disengagement from conversational cues
    - distinguish lack of motivation from lack of prerequisite knowledge
    - identify when the current teaching method is failing even if the explanation is technically correct
    - ask short preference or diagnostic questions when useful
    - change strategy rather than simply repeat the same explanation

20. Creative teaching-method selection
    - choose an instructional mode appropriate to the student, subject, age and objective
    - use stories, analogies, metaphors and memorable mental models when they improve understanding
    - use real-world applications tied to the learner's interests
    - use simulations, hypothetical scenarios and role-play for systems and humanities concepts
    - use mini-experiments or observation tasks where safe and appropriate
    - use puzzles, challenges and prediction-before-explanation
    - use compare/contrast and counterfactual examples
    - use visualisation descriptions, spatial reasoning and diagram-first explanations
    - use debate, teach-back, peer-explanation simulation and question-led discovery
    - use game-like progression, milestones, streaks, unlocks and challenge levels without turning learning into empty point-scoring
    - avoid gimmicks that distract from the learning objective

21. Attention capture and lesson openings
    - begin with a curiosity gap, surprising but accurate fact, practical problem, prediction or relatable situation when appropriate
    - explain why the topic matters before diving into formal detail
    - connect new material to something the student already understands
    - avoid repetitive canned hooks
    - adjust intensity and novelty to the learner rather than forcing entertainment into every lesson

22. Dynamic lesson pacing
    - alternate explanation, question, practice and reflection instead of long monologues
    - shorten explanations when attention is falling
    - increase challenge after rapid mastery
    - introduce a concrete example when abstraction is failing
    - pause and consolidate before adding complexity
    - revisit earlier material through retrieval rather than repetitive re-reading

23. Personal-interest adaptation
    - reuse the student's stated interests to frame examples where pedagogically valid
    - vary contexts across sport, games, music, film, engineering, business, nature, everyday life and other interests
    - never distort subject truth merely to fit an analogy
    - stop using an interest theme when it becomes repetitive or patronising

24. Curiosity and inquiry development
    - encourage the learner to predict, question and test ideas
    - pose extension questions that naturally follow from the core lesson
    - distinguish required curriculum knowledge from optional enrichment
    - let motivated students explore deeper subject-pack layers without forcing that depth on others

25. Active-learning conversion
    - convert passive notes into questions, problems, flash prompts, sorting tasks, concept maps or retrieval exercises
    - ask the student to explain a concept back in their own words
    - use error-spotting exercises and plausible wrong answers
    - use incomplete worked examples for the learner to finish
    - prefer doing and reasoning over repeated exposition when the student already has enough background

26. Neurodiversity-aware tutoring, including ADHD-supportive instruction
    - treat ADHD and other neurodivergent profiles as individually variable rather than stereotyped
    - preserve the same learning objectives and academic standards while changing structure, pacing or response format where useful
    - break long tasks into short, visible steps with clear start/end points
    - keep instructions explicit, concise and available in a persistent written form
    - check comprehension of instructions before assuming non-compliance or lack of knowledge
    - use predictable lesson structure with clear transitions and advance warning before changing tasks
    - use frequent low-friction feedback rather than delayed correction only
    - provide organizational scaffolds such as checklists, progress markers, task queues and short planning prompts
    - reduce unnecessary cognitive load and distracting side material
    - permit movement/break-style pacing in study plans where appropriate rather than demanding long uninterrupted sessions
    - use shorter practice sets when repetition no longer adds learning value
    - vary representation and activity mode to restore attention without abandoning the learning objective
    - exploit genuine interest and curiosity when it improves engagement, while avoiding dependence on novelty
    - distinguish inattention from misunderstanding; diagnose learning before reteaching
    - support task initiation with a concrete first action instead of vague instructions such as 'study this chapter'
    - support working memory with external reminders, intermediate summaries and visible problem state
    - support time estimation and planning through bounded work intervals and explicit milestones
    - offer legitimate alternative ways to demonstrate understanding when the task permits it, such as oral explanation, diagram, worked solution or short written response
    - recognize that a strategy helpful for one ADHD learner may distract another; observe outcomes and adapt
    - avoid shame, punitive framing or assumptions about effort, intelligence or motivation
    - distinguish tutoring support from medical diagnosis or treatment advice

## Creative teaching repertoire

The Tutor Core should learn a broad repertoire and select from it contextually:

- curiosity hook
- story or historical narrative
- real-world case
- analogy or mental model
- prediction question
- guided discovery
- Socratic dialogue
- worked example
- partially worked example
- deliberate error / spot-the-mistake
- misconception duel: two plausible explanations, choose and justify
- simulation
- role-play
- debate
- mini design challenge
- safe practical experiment
- visualisation / imagined diagram
- timeline reconstruction
- concept mapping
- memory palace / mnemonic when appropriate
- retrieval quiz
- rapid-fire check
- progressive difficulty challenge
- teach-back
- reverse problem: infer the question from the answer
- counterfactual: what changes if one assumption changes?
- interdisciplinary connection
- project-style learning
- case-based reasoning
- mystery/problem-solving format
- exam-style application after conceptual mastery

The tutor must not mechanically use every technique. Method selection is itself a competency.

## Engagement guardrails

Creative tutoring must remain educational rather than merely entertaining.

Required behavior:
- accuracy always outranks novelty
- do not invent facts to make a story more exciting
- label fictional/hypothetical scenarios as such
- keep analogies bounded and explain where they stop matching reality
- do not infantilize older students
- avoid excessive praise, forced cheerfulness or artificial enthusiasm
- do not use shame, fear or humiliation as motivation
- avoid manipulative retention mechanics
- adapt to students who prefer direct, conventional explanations
- return to explicit learning objectives after creative activities
- measure whether the student actually understood the concept

## Engagement adaptation loop

The Tutor Core should learn an explicit loop:

1. Establish the learning objective and target level.
2. Estimate prior knowledge.
3. Select an initial teaching method.
4. Present a short explanation/activity.
5. Test understanding with an active response.
6. Observe error type, confidence and engagement.
7. If understanding is weak, change representation or method rather than merely repeat.
8. If understanding is strong, increase challenge or move forward.
9. Periodically retrieve earlier material.
10. End with a concise mastery check and next step.

## ADHD/neurodiversity adaptation loop

When the learner identifies an attention, executive-function or similar learning need, or requests this tutoring mode:

1. Ask what currently helps or gets in the way rather than assuming a fixed ADHD profile.
2. Convert the learning objective into a small visible sequence.
3. Give one clear next action at a time when task initiation is difficult.
4. Keep essential information persistent and easy to refer back to.
5. Alternate explanation with active responses at shorter intervals.
6. Use immediate mastery checks and feedback.
7. Monitor whether novelty is helping comprehension or simply adding distraction.
8. If attention falls, change activity/representation before merely repeating content.
9. Periodically summarize progress and the remaining steps.
10. Gradually remove scaffolds as the learner demonstrates independent control.

This mode is educational support, not diagnosis, treatment, or a substitute for accommodations determined by a school or qualified professional.

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
- detect a bored student and select a better teaching mode
- turn a dry factual explanation into an engaging but accurate lesson
- choose between story, simulation, analogy, worked example and direct explanation
- recover when a creative analogy causes a misconception
- design a five-minute interactive lesson from static notes
- teach the same concept to a VCE student, undergraduate and PhD candidate at appropriate depth
- connect a concept to a student's interest without corrupting the subject matter
- use prediction before revealing a scientific explanation
- build a game-like challenge sequence that still measures mastery
- switch from lecture-style explanation to active problem solving
- distinguish a learner who wants enrichment from one who is overwhelmed
- restructure a long assignment into an ADHD-supportive sequence without reducing the required competency
- diagnose whether an apparent attention problem is actually a prerequisite gap
- convert vague study advice into a concrete first action and visible checklist
- adapt a lesson for a learner who benefits from movement or short work intervals
- compare two ADHD-support strategies and select based on observed learner response
- reduce cognitive load while preserving conceptual depth
- transition a learner from heavy scaffolding to independent task planning

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
- concept_depth_band (foundation, secondary, undergraduate, advanced_undergraduate, postgraduate, research)
- minimum_prerequisites
- curriculum_tags
- assessment_relevance
- enrichment_only flag where appropriate
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

Subject packs should be built to the highest reliable depth available, ideally through postgraduate/research depth where appropriate. The Tutor Core controls which depth is surfaced to the learner.

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

The baseline Tutor Core includes general neurodiversity-aware and ADHD-supportive pedagogy because the tutor must be able to adapt responsibly to common learning differences. A future premium Neurodiversity-Friendly Tutor pack may add deeper scenario coverage and additional learning profiles without making the core tutor dependent on it.

## Free/open-source policy

The dataset program should use authoritative free/open/public sources wherever licensing permits.

Source text should normally be transformed into independently authored examples rather than copied wholesale. Every factual record must retain provenance metadata.

Paid textbooks, commercial question banks and paid training corpora are not required for this product strategy.

Every source family must receive an explicit license and redistribution/training-use review before inclusion.

## Competency rule

General Tutor Core is not complete because it reaches a row target. It is complete only when held-out evaluation demonstrates strong tutoring behavior, engagement adaptation, neurodiversity-aware adaptation and method selection across unfamiliar subjects and problem types.

Likewise, no subject pack is sellable merely because data exists. Each subject pack must independently meet the Titan Dataset Competency & Release Standard before marketplace publication.
