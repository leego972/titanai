# U.S. High School + GED Curriculum Adapter v1

## Purpose

Adapt the shared Titan/VIBA tutoring corpus for U.S. learners without duplicating the underlying high-quality subject datasets.

Important distinction:
- GED is a U.S. high-school equivalency credential/exam, not the ordinary curriculum used by American high schools.
- Therefore the product should expose two separate U.S. profiles:
  1. U.S. High School Core Tutor
  2. GED Preparation Tutor

## Architecture

Use a curriculum-adapter layer over the shared subject corpus.

Shared knowledge/skill corpora:
- English/language arts
- mathematics
- biology
- chemistry
- physics
- earth/space science
- history
- civics/government
- economics
- geography
- computer/digital literacy
- financial literacy
- study skills and metacognition
- behavioral/safety tutoring profiles

The adapter changes:
- curriculum tags
- grade/course mapping
- terminology/spelling where relevant
- expected depth
- assessment style
- rubric style
- worked-example conventions
- calculator/formula-sheet assumptions
- jurisdiction/context examples

It must NOT lower the competency or factual-quality standard of the base dataset.

## Profile A — U.S. High School Core

### English Language Arts
Align tutoring and evaluation to the publicly available Common Core ELA/Literacy framework where applicable, including:
- close reading
- evidence-based answers
- informational text
- literature analysis
- argumentative writing
- explanatory writing
- research and source evaluation
- speaking/listening concepts where text-only simulation is appropriate
- literacy in history/social studies, science and technical subjects

### Mathematics
Align to Common Core high-school mathematics concepts where applicable:
- algebra
- functions
- geometry
- statistics/probability
- number/quantity
- mathematical modeling
- mathematical practices such as reasoning, justification and problem solving

State-specific variants can later map the same competency graph to a state's standards when needed.

### Science
Align science tutoring to NGSS-style high-school performance expectations and three-dimensional reasoning where applicable:
- physical science / chemistry / physics
- life science / biology
- earth and space science
- engineering design
- science and engineering practices
- crosscutting concepts
- evidence/model/data-based reasoning

### Social Studies
Because U.S. social-studies requirements vary by state, maintain a national core covering:
- U.S. history
- world history
- civics/government
- economics
- geography
- primary-source reasoning
- evidence vs interpretation

Then add state adapters only when justified by demand.

### Additional common U.S. high-school subjects
- computer science
- environmental science
- psychology
- personal finance
- health
- career/technical literacy
- media and digital literacy

AP/IB and state end-of-course adapters may be added later as separate advanced profiles; they are not equivalent to the base high-school profile.

## Profile B — GED Preparation

GED currently assesses four separate subject areas:
1. Mathematical Reasoning
2. Reasoning Through Language Arts
3. Science
4. Social Studies

### GED Mathematical Reasoning
Cover:
- basic mathematics
- geometry
- algebra
- graphs and functions
- quantitative problem solving
- calculator/non-calculator reasoning
- interpretation of tables/charts
- real-world applications

### GED Reasoning Through Language Arts
Cover:
- reading comprehension
- evidence and inference
- argument analysis
- grammar/language conventions
- synthesis across passages
- extended-response planning and writing

### GED Science
Focus on scientific reasoning rather than memorization alone:
- life science
- physical science
- earth/space science
- experimental design
- variables and controls
- interpreting graphs/tables
- evidence-based conclusions
- scientific models

### GED Social Studies
Cover:
- civics/government
- U.S. history
- economics
- geography/world context
- source analysis
- charts, maps and political/economic data
- claims, evidence and inference

## Dataset strategy

Do not build four isolated GED corpora from scratch.

Instead:
1. train/maintain broad subject competence in the shared corpora;
2. tag examples with competency IDs;
3. build GED-specific assessment/task families over those skills;
4. build GED-held-out evaluation sets with unseen scenarios;
5. use GED profile metadata to select the appropriate depth and style.

This maximizes quality and reduces duplicate data.

## Quality requirements

Both U.S. profiles inherit Titan's Dataset Competency & Release Standard.

Additional U.S.-adapter gates:
- every mapped competency must trace to a current public curriculum/exam objective or documented subject rationale;
- no copied commercial test-prep questions;
- no copied proprietary textbook material;
- original questions/explanations generated from open/public-domain or license-compatible source material;
- state-specific legal/civics facts must carry jurisdiction and as-of metadata;
- terminology differences between Australian and U.S. education must be explicitly tagged;
- U.S. held-out test sets must be isolated from VCE held-out families so adaptation is measured, not memorized;
- regression tests must show that adding the U.S. adapter does not degrade VCE/Australian performance.

## Behavioral tutoring layer

Reuse the same behavioral tutor datasets, adjusted for U.S. school context:
- Socratic tutoring rather than answer dumping
- academic integrity
- misconception diagnosis
- study planning/metacognition
- age-appropriate communication
- respectful communication
- source/media literacy
- online safety
- career readiness
- financial literacy
- decision-making/consequence reasoning

## Product UX

Tutor profile selector can remain simple:
- Australia — VCE
- United States — High School
- United States — GED

The model should infer terminology and grading conventions from the selected profile.

## Open/free source policy

Use no paid curriculum or commercial test banks.
Prioritize current publicly accessible standards and government/nonprofit/open educational resources. Record source family, license/terms status, access date and allowed use in the provenance registry before material enters training.

## Release status

This document defines the adapter. It is not a competency certification. U.S. High School and GED profiles remain IN DEVELOPMENT until their mapped corpus coverage and held-out evaluation gates pass.
