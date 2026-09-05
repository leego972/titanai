# Special Education Support Dataset Family v1

## Purpose

Create a family of specialist educational-support datasets that deepen the General Tutor Core's ability to teach learners with different functional learning needs without reducing academic standards, stereotyping learners, or drifting into diagnosis/treatment.

These are educational adaptation datasets. They are not medical, psychological, speech-pathology, occupational-therapy or clinical treatment systems.

## Architecture

General Tutor Core contains baseline inclusive/neurodiversity-aware pedagogy.

Premium specialist packs add deeper scenario coverage, adaptation strategies, functional-learning profiles, transition logic, accessibility methods and evaluation.

Recommended packs:

1. ADHD & Executive-Function Learning Support
2. Autism-Supportive Tutoring
3. Dyslexia & Reading-Learning Support
4. Dyscalculia & Mathematics-Learning Support
5. Intellectual & Developmental Learning Support
6. Communication & Language Accessibility Support
7. Sensory & Environmental Accessibility Support
8. Motor/Physical Accessibility for Learning
9. Complex/Multi-Need Inclusive Tutoring
10. Gifted + Twice-Exceptional Learner Support

A learner may use more than one pack because real support needs can overlap.

## Governing principles

- individual functional needs outrank labels
- maintain the intended learning outcome wherever possible
- adapt representation, pacing, structure, response mode and environment before lowering conceptual expectations
- do not infer diagnosis from chat behavior
- do not claim a learner has a disability
- distinguish disability-related barriers from missing prerequisites, ineffective instruction, language proficiency, absenteeism, stress and other contextual causes
- consult learner preferences when known
- support independence rather than permanent unnecessary scaffolding
- avoid shame, coercion and deficit-based language
- separate tutoring support from clinical advice
- preserve age-appropriate dignity

## Pack 1: ADHD & Executive-Function Learning Support

Core competencies:
- task initiation support
- chunking and visible sequencing
- working-memory externalization
- time estimation and bounded work blocks
- organization/checklists
- transition support
- attention restoration via activity/representation changes
- short feedback loops
- movement/break-compatible study design
- distraction reduction
- interest-based engagement without novelty dependence
- scaffold fading
- distinguish inattention from misunderstanding

## Pack 2: Autism-Supportive Tutoring

Core competencies:
- individualized strengths/interests profile
- explicit lesson structure
- clear literal instructions when ambiguity causes barriers
- visual/persistent representation
- transition preparation
- predictable routines with flexibility
- sensory-load awareness
- reducing unnecessary social ambiguity
- using interests as legitimate learning bridges
- alternate response formats
- supporting independence and self-advocacy
- identifying when a metaphor/analogy may need explicit unpacking
- avoiding assumptions about communication style, intelligence or empathy

## Pack 3: Dyslexia & Reading-Learning Support

Core competencies:
- phonological/word-decoding support where age/level appropriate
- explicit vocabulary and morphology support
- chunked reading
- text-to-structure conversion
- persistent key terms
- comprehension separated from decoding burden
- alternate demonstration of content understanding when valid
- spelling feedback that does not eclipse conceptual feedback
- reading fluency support
- structured writing planning
- accessible formatting guidance
- gradual reduction of supports

The pack must not claim to provide clinical dyslexia intervention unless a specific evidence-based program and licensing framework supports that claim.

## Pack 4: Dyscalculia & Mathematics-Learning Support

Core competencies:
- number sense
- magnitude and comparison
- place value
- symbolic-to-concrete mapping
- visual quantities and number lines
- stepwise procedures with conceptual explanation
- units and dimensional anchors
- estimation/sanity checks
- error pattern diagnosis
- reduced working-memory load
- repeated representation changes without rote template dependence
- transfer from concrete to abstract reasoning

## Pack 5: Intellectual & Developmental Learning Support

Core competencies:
- functional prerequisite mapping
- smaller learning increments
- explicit modeling
- repeated guided practice with variation
- concrete examples before abstraction
- simplified language without infantilization
- longer consolidation windows
- generalization across settings
- functional literacy/numeracy applications
- choice and autonomy support
- meaningful mastery checks
- adaptive response modes
- scaffold fading when mastery develops

## Pack 6: Communication & Language Accessibility Support

Core competencies:
- plain-language alternatives
- reduced sentence complexity where needed
- visual/pictorial representation descriptions
- communication-choice support
- extra processing time in conversational pacing
- clarification without pressure
- structured response options
- vocabulary pre-teaching
- comprehension checks that do not rely only on verbal fluency
- support for AAC-compatible interaction patterns where the deployment interface supports AAC

This pack does not replace speech-language pathology.

## Pack 7: Sensory & Environmental Accessibility Support

Core competencies:
- identify learner-stated sensory barriers
- reduce irrelevant visual/verbal clutter
- predictable presentation
- lower-stimulation lesson mode
- break/pacing options
- alternate modalities
- noise/visual-distraction awareness in study planning
- distinguish sensory overload from lack of knowledge
- re-entry after overload or interruption

## Pack 8: Motor/Physical Accessibility for Learning

Core competencies:
- reduce unnecessary typing/writing burden when not part of the target competency
- allow alternate response formats
- split long production tasks
- support voice/assistive-technology compatible workflows
- distinguish subject mastery from motor-output limitations
- plan accessible practical/lab alternatives where educationally legitimate

This pack does not provide medical mobility advice.

## Pack 9: Complex/Multi-Need Inclusive Tutoring

Purpose: handle cases where multiple functional learning needs interact.

Core competencies:
- prioritize the barrier most limiting current learning
- avoid conflicting accommodations
- combine supports parsimoniously
- detect support overload
- preserve one clear learning objective
- coordinate level, communication, sensory and executive-function adaptations
- use learner feedback to tune support

## Pack 10: Gifted + Twice-Exceptional Learner Support

Core competencies:
- detect rapid mastery without assuming uniform ability
- allow subject acceleration while preserving weaker prerequisite support
- enrichment and open-ended inquiry
- avoid repetitive low-level practice once mastery is demonstrated
- asynchronous profiles (advanced reasoning plus executive/reading/etc support)
- challenge without excessive workload
- research-style extension when appropriate
- support frustration caused by mismatch between cognitive depth and production/organization skills

## Dataset record schema additions

Specialist-pack records should include:
- support_pack
- functional_barrier
- learner_level
- learning_objective
- subject_context
- prerequisite_state
- learner_preference_signal
- adaptation_selected
- adaptation_reason
- academic_standard_preserved: true/false
- alternate_response_mode
- scaffold_level
- scaffold_fade_condition
- contraindicated_or_unhelpful_strategy
- uncertainty
- provenance
- scenario_family
- split_family

## Required contrast examples

At least 35% of specialist-pack training examples should be contrast pairs/triplets such as:
- same diagnosis label, different useful adaptations
- same observable behavior caused by different barriers
- support that helps learner A but distracts learner B
- appropriate accommodation vs unnecessary lowering of content difficulty
- genuine prerequisite gap vs attention/executive-function barrier
- sensory overload vs conceptual misunderstanding
- slow written output vs weak subject knowledge
- enrichment need vs curriculum misunderstanding

The goal is to prevent simplistic label-to-strategy memorization.

## Evaluation gates

Each pack must independently demonstrate:
- correct identification of educational barrier from supplied evidence without diagnosing
- appropriate adaptation selection
- preservation of learning objective
- individualized rather than stereotyped support
- no clinical/treatment overreach
- correct escalation to teacher/family/qualified professional when tutoring alone is insufficient
- strong transfer across unfamiliar subjects and ages
- avoidance of over-accommodation and support dependency

Minimum release thresholds follow `docs/DATASET_COMPETENCY_RELEASE_STANDARD.md`.

For each major pack:
- target 12,000-25,000+ accepted examples depending on breadth
- >= 1,500 accepted examples per major competency where applicable
- >= 20% hard/complex multi-factor cases
- >= 100 held-out cases per major competency
- >= 85% per-competency evaluation floor
- >= 90% overall for objectively scorable adaptation decisions where applicable

## Source policy

Use authoritative free/open/public educational sources where licensing permits.

Priority source families:
- Australian Government Department of Education disability-education and reasonable-adjustment guidance
- Victorian Department of Education inclusive/autism/learning-needs guidance
- NCCD public guidance/resources
- U.S. Department of Education / IDEA and Section 504 public guidance for U.S. adapters
- CDC educational ADHD guidance
- reputable university OER and evidence reviews
- peer-reviewed/open-access education research where license permits

Do not copy proprietary intervention programs or paid special-education curricula into commercial datasets.

## Marketplace model

Possible separately purchasable products:
- ADHD & Executive Function Tutor Support
- Autism-Supportive Tutor
- Dyslexia & Reading Support Tutor
- Dyscalculia & Math Support Tutor
- Intellectual/Developmental Learning Support Tutor
- Communication Accessibility Tutor
- Sensory-Friendly Tutor
- Complex Needs Inclusive Tutor
- Twice-Exceptional Learner Tutor
- Inclusive Education Bundle

General Tutor Core retains enough baseline inclusive behavior to be responsible without any premium pack. Premium packs provide deeper specialization, not basic dignity or accessibility.

## Release rule

No pack is sellable because its specification exists or its row count is high. It becomes sellable only after independent competency evaluation, human review, provenance/license checks, diversity/leakage testing and package integrity all pass.
