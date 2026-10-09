# VCE Tutor Core v1

## Purpose
Build a high-quality tutoring dataset for Victorian Certificate of Education students that can explain concepts, diagnose misconceptions, scaffold answers, generate original practice, and coach study/exam technique without reproducing copyrighted VCAA assessment material.

## Curriculum alignment
Use current VCAA study designs, assessment guidance, examination specifications, public curriculum descriptions and teacher-support pages as curriculum blueprints only. Generate original questions, worked examples and feedback. Do not copy substantial VCAA exam questions, copyrighted textbooks, commercial revision material or teacher resources into the training corpus.

2026 alignment must cover the currently accredited VCE study-design families, with subject/version metadata so the corpus can be refreshed when accreditation periods change.

## Target scale
Full release target: 45,000-80,000+ accepted records across the highest-demand VCE subjects, expandable as evaluation requires.

No subject is considered competent merely because its quota is filled. Each subject must pass its own held-out evaluation.

## Initial subject coverage
Priority subjects:
1. English / EAL
2. General Mathematics
3. Mathematical Methods
4. Specialist Mathematics
5. Chemistry
6. Biology
7. Physics
8. Psychology
9. Business Management
10. Legal Studies
11. Economics
12. Health and Human Development
13. History
14. Geography
15. Applied Computing

Later expansion can cover the remaining accredited studies.

## Tutoring competencies
Each supported subject must include examples that teach the model to:
- explain a concept at multiple difficulty levels
- diagnose a student's misconception from their attempted answer
- give a hint before revealing a solution when appropriate
- provide fully worked solutions for original problems
- show why distractors or incorrect approaches fail
- distinguish memorisation from application
- connect key knowledge to key skills
- interpret command verbs such as explain, analyse, evaluate, compare, justify and discuss in subject-appropriate ways
- provide concise exam-style feedback
- create original SAC-style and exam-style practice without copying protected assessment content
- adapt explanation depth to the student's demonstrated level
- handle uncertainty and say when supplied information is insufficient
- encourage academic integrity rather than completing live graded work deceptively

## Record families
Use many independent families, including:
- concept explanation
- misconception correction
- worked example
- Socratic hint sequence
- short-answer feedback
- extended-response feedback
- data/table/graph interpretation
- experimental design where relevant
- source interpretation where relevant
- calculation and dimensional reasoning
- compare/contrast
- evidence evaluation
- exam-time strategy
- spaced-recall prompt
- mixed-skill transfer problem
- deliberately tempting wrong answer
- student-generated draft critique

No single scenario family should normally exceed 8% of a subject corpus.

## Difficulty distribution
Target:
- 15% foundational remediation
- 35% standard VCE application
- 30% hard multi-step application
- 15% exam-pressure / transfer
- 5% expert extension

## Competency floor per supported subject
Before a subject is advertised as supported:
- >= 3,000 accepted training examples for a broad VCE subject
- >= 500 examples per major area of study or equivalent competency cluster
- >= 150 held-out evaluation items per major competency cluster
- >= 25% of evaluation uses unseen scenario families or cross-topic combinations
- >= 90% correctness on deterministic tasks
- >= 85% per major competency
- qualitative rubric pass for writing-heavy subjects
- no subject may hide a weak area behind a strong aggregate score

## Pedagogical behavior
The tutor should usually:
1. determine what the student is trying to learn
2. assess their current understanding from available evidence
3. explain the smallest missing concept
4. provide an example
5. ask or simulate a check-for-understanding question
6. increase difficulty only after demonstrated understanding

The dataset must include contrast examples showing poor tutoring behaviors to avoid: giving answers without explanation, excessive verbosity, unexplained jargon, falsely praising incorrect work, and replacing the student's reasoning rather than developing it.

## Academic integrity
Allowed:
- tutoring
- practice questions
- feedback on drafts
- study plans
- concept explanations
- worked examples created for learning

The dataset should discourage deceptive submission of model-generated work as a student's own where the task is clearly an active graded assessment. It should instead help the student understand, outline, revise or improve their own work.

## Sources
Use free/open sources wherever possible. Source families may include:
- VCAA curriculum/study-design pages for scope and outcome mapping
- Australian and international government education/science datasets
- OpenStax and other permissively licensed educational material after licence review
- Wikidata CC0 for structured facts
- government statistical/open-data portals
- NASA, NOAA, USGS and comparable public scientific resources
- public-domain literature for English examples
- original synthetic quantitative problems with code-verified answers

Every factual record must carry provenance metadata.

## Evaluation
A subject is released only after:
- schema validation
- exact and near-duplicate checks
- scenario-family leakage checks
- factual/provenance audit
- stratified human review
- unseen subject benchmark
- transfer benchmark
- adversarial misconception benchmark
- tutoring-quality rubric review

## Marketplace status
IN DEVELOPMENT. Do not list for sale until all advertised subject competencies are independently verified.