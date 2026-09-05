# General University Tutor Base v1

## Purpose

A reusable university-level tutoring/alignment dataset that teaches a model *how to tutor* before specialist subject packs are attached.

This is not a subject-content corpus. It is the pedagogical and reasoning backbone for a modular university tutor.

## Product model

Base tutor + optional university subject packs.

Example:
- General University Tutor Base
- + Calculus & Linear Algebra
- + Organic Chemistry
- + Cell & Molecular Biology
- + Introductory Economics
- + Computer Science

The same base tutor can therefore serve different degree pathways without duplicating pedagogy data across every subject.

## Target release size

Target: 20,000-35,000+ accepted examples.

Competency floor:
- >= 1,500 accepted examples per core tutoring competency where applicable
- >= 500 per important subcompetency
- >= 20% hard/expert cases
- >= 500 held-out evaluation items total and >= 100 per major core competency
- >= 90% overall correctness for deterministic pedagogical cases where objective scoring is available
- >= 85% per core competency
- strong transfer and adversarial evaluation

## Core competencies

1. Diagnostic tutoring
   - identify prerequisite gaps
   - distinguish conceptual misunderstanding from arithmetic/notation slips
   - infer likely misconception from student work
   - ask minimal high-value diagnostic questions

2. Socratic scaffolding
   - guide rather than immediately reveal
   - adjust hint granularity
   - progressive prompting
   - checkpoint understanding before advancing

3. Worked-example teaching
   - transparent intermediate steps
   - explain why each step is valid
   - compare correct and plausible-but-wrong approaches
   - identify transferable patterns

4. Conceptual explanation
   - multiple explanation styles
   - analogy where appropriate
   - formal definition after intuition
   - connect representations: verbal, symbolic, graphical, tabular

5. Misconception remediation
   - common misconceptions
   - counterexamples
   - error classification
   - targeted micro-practice

6. Problem decomposition
   - identify givens/unknowns
   - select methods
   - sequence subtasks
   - verify final result

7. Quantitative reasoning support
   - units and dimensions
   - estimation
   - significant figures
   - order-of-magnitude checks
   - sanity checking

8. Academic writing support
   - argument structure
   - thesis/evidence linkage
   - paragraph cohesion
   - discipline-appropriate tone
   - editing and feedback

9. Research literacy
   - formulate research questions
   - distinguish primary/secondary sources
   - evaluate evidence quality
   - identify confounding, bias and uncertainty
   - citation hygiene without fabricated references

10. Study strategy and metacognition
    - retrieval practice
    - spaced practice
    - interleaving
    - self-explanation
    - planning around assessment dates
    - diagnose ineffective study habits

11. Assessment preparation
    - practice under time constraints
    - rubric interpretation
    - error logs
    - targeted revision
    - exam strategy

12. Academic integrity
    - teach and coach rather than impersonate the student
    - support outlining, feedback, explanation and practice
    - distinguish legitimate assistance from ghost-writing where applicable

13. Uncertainty and source handling
    - state uncertainty
    - distinguish settled fact, interpretation and current research
    - avoid invented citations
    - recommend authoritative verification for volatile claims

14. Adaptive difficulty
    - introductory undergraduate
    - intermediate undergraduate
    - advanced undergraduate
    - honours/foundation postgraduate bridge where the subject pack supports it

15. Student communication
    - concise vs detailed modes
    - supportive but non-patronising feedback
    - adapt to English-language proficiency
    - accessible explanations without lowering conceptual accuracy

## Training scenario families

- diagnose a wrong solution
- choose the next hint
- compare two explanations
- transform a poor explanation into a better one
- identify missing prerequisite knowledge
- create a short targeted practice sequence
- explain a concept at multiple depths
- grade against a supplied rubric
- critique reasoning without rewriting the entire assignment
- identify unsupported claims
- distinguish calculation error vs conceptual error
- choose between formulaic and first-principles approaches
- derive a result, then independently verify it
- teach from a graph/table/diagram description
- resolve ambiguity in a student question
- handle insufficient information
- detect when a student's requested answer would bypass learning

## Subject-pack interface

Every university subject add-on should expose:

- subject_id
- discipline
- level_band
- prerequisite graph
- core competencies
- concepts/topics
- canonical notation/terminology
- worked-example families
- misconception catalogue
- problem families
- lab/practical components where relevant
- authoritative open-source registry
- held-out evaluation set
- competency report

The base tutor should not need retraining when a compatible subject pack is added if the deployment architecture supports adapters/RAG/tool routing. Fine-tuning may still be used when deeper subject-specific behaviour is required.

## Priority university subject add-ons

### Mathematics and statistics
- College Algebra / Precalculus
- Calculus I-III
- Linear Algebra
- Differential Equations
- Discrete Mathematics
- Probability
- Mathematical Statistics
- Numerical Methods
- Real Analysis

### Physical and life sciences
- General Chemistry
- Organic Chemistry
- Physical Chemistry
- Biochemistry
- General Biology
- Cell Biology
- Molecular Biology
- Genetics
- Microbiology
- Ecology
- Human Physiology
- General Physics
- Classical Mechanics
- Electromagnetism
- Thermodynamics
- Quantum Foundations
- Earth & Environmental Science

### Computing and engineering
- Programming Fundamentals
- Data Structures & Algorithms
- Computer Architecture
- Operating Systems
- Databases
- Networks
- Software Engineering
- Cybersecurity
- Artificial Intelligence
- Machine Learning
- Data Science
- Electrical Engineering Fundamentals
- Mechanical Engineering Fundamentals
- Civil Engineering Fundamentals
- Chemical Engineering Fundamentals

### Business, economics and finance
- Microeconomics
- Macroeconomics
- Accounting
- Corporate Finance
- Investments
- Econometrics
- Statistics for Business
- Management
- Marketing
- Operations Management
- Entrepreneurship

### Humanities and social sciences
- Psychology
- Sociology
- Political Science
- International Relations
- Philosophy
- Ethics
- Logic
- History
- Archaeology
- Anthropology
- Geography
- Linguistics

### Law and public policy
- Legal Research & Reasoning
- Contract Law foundations
- Tort Law foundations
- Constitutional/Public Law foundations
- Criminal Law foundations
- Administrative Law foundations
- Policy Analysis

Jurisdiction-specific legal packs must remain separate and current.

### Health and biomedical foundations
- Anatomy
- Physiology
- Pathophysiology
- Pharmacology foundations
- Epidemiology
- Public Health
- Biostatistics

These packs are educational and must not be positioned as clinical diagnosis or treatment systems.

### Communication and professional skills
- Academic Writing
- Technical Writing
- Scientific Communication
- Research Methods
- Presentation Skills
- Professional Communication
- Critical Thinking

## Free/open-source strategy

Use authoritative open/public sources wherever licensing permits, including government, university OER, public-domain scientific sources, open textbooks and open data.

Source material should be used to derive original training examples rather than copied wholesale. Every factual record must retain provenance metadata.

Potential source families include:
- OpenStax where licence-compatible
- LibreTexts where licence-compatible
- MIT OpenCourseWare materials subject to licence conditions
- OpenLearn/Open University resources where licence-compatible
- NCBI/NIH public resources
- NASA/NOAA/USGS public-domain material
- government statistical agencies
- Wikidata CC0 for structured facts
- arXiv metadata/open papers only where the specific licence permits derivative training use
- institutional OER repositories

Every source family requires licence review before inclusion.

## Behavioral companion profiles

Optional behavioral add-ons:
- General Academic Integrity
- Socratic Tutor
- Neurodiversity-Friendly Tutor
- English-Language Learner Support
- Study Coach / Metacognition
- Research & Citation Discipline
- Professional Communication
- Career Readiness

## Quality rule

A university subject pack is not complete because it has many rows. It is complete only when a model using it demonstrates competence on held-out, leakage-free evaluation data and no core competency is below threshold.

The same release standard defined in `docs/DATASET_COMPETENCY_RELEASE_STANDARD.md` applies.
