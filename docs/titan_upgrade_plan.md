# Titan AI — Complete Upgrade Plan
## Five Pillars of Expert Intelligence

**Document Version:** 2.0  
**Prepared:** April 2026  
**Status:** Pre-prepared — executes after post-Crucible base pipeline  
**Crucible ETA:** ~April 15, 2026

---

## Titan's Identity

Titan AI is not a general-purpose chatbot. It is a **specialist intelligence** built around five pillars that define what it is and what it does better than anything else:

| Pillar | What Titan Excels At |
|--------|---------------------|
| **1. General Knowledge** | World history, geography, science, chemistry, mathematics — a vast, deep foundation |
| **2. Film & Cinema Production** | The complete filmmaking pipeline: development, pre-production, production, post-production, marketing, distribution |
| **3. Cybersecurity** | Offensive and defensive security, exploit development, penetration testing, vulnerability analysis, tool building |
| **4. Creative Writing & Thinking** | Original storytelling, screenwriting, narrative craft, lateral thinking, concept generation |
| **5. Proactive Problem Solving** | Anticipates problems before they occur, identifies root causes, proposes and executes solutions autonomously |

The goal is not breadth for its own sake. The goal is for Titan to be **genuinely useful** — to spot what is going wrong before you do, to know its domains deeply enough to give expert-level answers, and to act rather than just respond.

---

## Full Pipeline Overview

```
Crucible Base Model (~April 15)
        ↓
Stage 1: Instruction Tuning (SFT)
        ↓
Stage 2: Tool Use Fine-Tuning
        ↓
Upgrade A: General Knowledge Depth Pack
        ↓
Upgrade B: Film & Cinema Production (Full Pipeline)
        ↓
Upgrade C: Cybersecurity Depth Pack
        ↓
Upgrade D: Creative Writing & Thinking
        ↓
Upgrade E: Proactive Problem Solving
        ↓
Titan AI v1.5 — Five-Pillar Expert Intelligence
```

---

## Upgrade A: General Knowledge Depth Pack

### Goal
Give Titan a **vast, accurate, and deeply connected general knowledge base** — the kind of foundational intelligence that makes every other domain stronger. A Titan that knows world history understands the context behind geopolitical cyber attacks. A Titan that knows chemistry understands the science behind forensic analysis. General knowledge is the connective tissue.

### Coverage

| Subject | Depth |
|---------|-------|
| **World History** | Ancient civilisations through modern era, major conflicts, political movements, key figures |
| **Geography** | Physical geography, geopolitics, nations, capitals, natural features, climate |
| **General Science** | Physics, biology, earth science, astronomy — concepts and applications |
| **Chemistry** | Organic and inorganic chemistry, reactions, compounds, lab methodology |
| **Mathematics** | Algebra, geometry, statistics, logic, proofs, applied mathematics |
| **Economics** | Macro and micro, markets, trade, financial systems |
| **Philosophy & Logic** | Critical thinking, argumentation, ethics, major philosophical traditions |
| **Current Events** | Understanding of how the world works today |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **Wikipedia (filtered)** | `wikimedia/wikipedia` (HuggingFace) | ~6M articles | Broad encyclopaedic knowledge |
| **OpenWebMath** | `open-web-math/open-web-math` | ~14.7B tokens | Mathematical reasoning and science |
| **MMLU Training Data** | `cais/mmlu` (all subjects) | 57 subjects | Structured academic knowledge |
| **World History Q&A** | Generated from Wikipedia + textbooks | ~30K examples | History instruction pairs |
| **Science Q&A (SciQ)** | `allenai/sciq` | 13,679 examples | Science questions with explanations |
| **Chemistry Textbooks** | Internet Archive public domain | ~20 books | Chemistry concepts and reactions |
| **Geography Q&A** | Generated synthetic dataset | ~15K examples | Nations, capitals, physical geography |

**Total general knowledge dataset: ~250K curated instruction pairs**

### Estimated Cost
- **Duration:** ~15–20 hours
- **Cost:** ~$6–8

---

## Upgrade B: Film & Cinema Production (Full Pipeline)

### Goal
Make Titan the most capable film production AI available for a model of its size. Not just film trivia — **professional-grade knowledge of the entire filmmaking pipeline** from the first idea through to audience marketing. Titan should be able to sit in a production meeting and add value at every stage.

### Coverage: The Complete Filmmaking Pipeline

**Development & Pre-Production**

| Area | Capability |
|------|-----------|
| **Concept Development** | Logline construction, premise testing, genre conventions, originality assessment |
| **Screenwriting** | Three-act structure, character arcs, dialogue craft, scene construction, industry formatting |
| **Script Analysis** | Coverage writing, story notes, structural feedback, character consistency |
| **Budgeting** | Above-the-line vs below-the-line costs, budget breakdowns, cost estimation by genre |
| **Scheduling** | Strip boards, shooting schedules, day-out-of-days, location grouping |
| **Casting** | Breakdown writing, casting strategy, talent negotiation concepts |
| **Location Scouting** | Location requirements, permits, logistics, production design considerations |
| **Storyboarding** | Shot planning, visual language, pre-visualisation concepts |
| **Crew Assembly** | Department heads, crew hierarchy, roles and responsibilities |

**Production**

| Area | Capability |
|------|-----------|
| **Directing** | Shot selection, blocking, working with actors, coverage strategy, visual storytelling |
| **Cinematography** | Lenses, lighting setups, camera movement, aspect ratios, colour theory |
| **Production Design** | Set design, art direction, props, wardrobe, period accuracy |
| **Sound Recording** | Production sound, boom operation, ADR planning |
| **On-Set Management** | Call sheets, daily reports, continuity, safety protocols |

**Post-Production**

| Area | Capability |
|------|-----------|
| **Editing** | Continuity editing, montage theory, pacing, rhythm, assembly to fine cut |
| **Visual Effects** | VFX pipeline, compositing concepts, CGI vs practical |
| **Colour Grading** | Colour theory, LUTs, DI pipeline, delivery specifications |
| **Sound Design & Mixing** | Sound design, Foley, music scoring, mixing, delivery formats |
| **Music & Score** | Temp tracks, composer briefs, licensing, sync rights |

**Marketing, Distribution & Exhibition**

| Area | Capability |
|------|-----------|
| **Film Marketing** | Trailer construction, poster design principles, campaign strategy, social media |
| **Festival Strategy** | Major festivals (Cannes, Sundance, TIFF), submission strategy, premiere timing |
| **Distribution** | Theatrical vs streaming, sales agents, distribution deals, P&A |
| **Audience Research** | Test screenings, audience targeting, demographic analysis |
| **Pitching** | Pitch deck construction, logline delivery, investor presentations |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **Movie Scripts (IMSDB + Kaggle)** | Public domain + scraped | ~1,500 scripts | Screenplay format and storytelling |
| **OpenSubtitles** | `Helsinki-NLP/open_subtitles` | ~3M lines | Dialogue, character voice, tone |
| **Film Textbooks (processed)** | Internet Archive public domain | ~60 books | Directing, cinematography, editing, producing |
| **Film Reviews & Analysis** | RogerEbert.com, Criterion, Letterboxd | ~300K reviews | Critical analysis, film language |
| **Production Documents** | StudioBinder, Filmsourcing templates | ~8K documents | Call sheets, budgets, schedules, shot lists |
| **Film History & Theory** | Wikipedia film articles + essays | ~50K articles | Film movements, auteur theory, genre |
| **Marketing & Distribution** | Trade press (Variety, Deadline, THR) | ~30K articles | Industry business knowledge |
| **Cinema-Instruct (custom)** | Generated from all above sources | ~40K examples | Full-pipeline Q&A instruction pairs |
| **Pitching & Development** | Generated synthetic dataset | ~10K examples | Pitch decks, coverage, development notes |

**Total cinema dataset: ~300K high-quality examples**

### Estimated Cost
- **Duration:** ~20–25 hours
- **Cost:** ~$8–10

---

## Upgrade C: Cybersecurity Depth Pack

### Goal
Make Titan an **expert-level offensive and defensive security intelligence** — capable of deep vulnerability analysis, exploit reasoning, penetration testing methodology, malware analysis, and building security tools from scratch.

### Coverage

| Domain | Capability |
|--------|-----------|
| **Vulnerability Analysis** | CVE analysis, CVSS scoring, impact assessment, exploitation paths |
| **Penetration Testing** | Full methodology: recon, enumeration, exploitation, post-exploitation, reporting |
| **Exploit Development** | Buffer overflows, ROP chains, shellcode, format strings, heap exploitation |
| **Malware Analysis** | Static/dynamic analysis, reverse engineering, IOC identification, YARA rules |
| **Network Security** | Packet analysis, protocol exploitation, network forensics, traffic anomaly detection |
| **Web Application Security** | OWASP Top 10, SQL injection, XSS, SSRF, authentication bypass |
| **CTF Challenges** | Web, binary, crypto, forensics, reverse engineering — full solve methodology |
| **Tool Building** | Writing security tools in Python — scanners, exploits, automation scripts |
| **OSINT** | Target profiling, footprinting, open-source intelligence gathering |
| **MITRE ATT&CK** | Adversary tactics, techniques, and procedures across the full kill chain |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **CyberLLMInstruct** | `ElZemity/CyberLLMInstruct` | 54,928 examples | Malware, CVE, pentest Q&A |
| **Primus Cybersecurity Suite** | `CyberNative-AI/Primus` | ~100K examples | Broad security knowledge |
| **PenQA** | `PenQA/PenQA` | ~20K examples | Penetration testing Q&A |
| **NVD CVE Database** | nvd.nist.gov (public) | 250K+ CVEs | Vulnerability descriptions and analysis |
| **CTF Writeups** | CTFtime.org + GitHub | ~30K writeups | Real challenge solutions and reasoning |
| **Exploit-DB** | exploit-db.com (public) | ~50K exploits | Real exploit code and explanations |
| **MITRE ATT&CK** | attack.mitre.org (public) | ~700 techniques | Adversary tactics and techniques |
| **Security Tool Code** | GitHub public security repos | ~20K examples | Security tool implementation |
| **Cyber-Instruct (custom)** | Generated from all above | ~20K examples | Domain-specific instruction pairs |

**Total cybersecurity dataset: ~250K high-quality examples**

### Estimated Cost
- **Duration:** ~15–20 hours
- **Cost:** ~$6–8

---

## Upgrade D: Creative Writing & Thinking

### Goal
Give Titan **genuine creative intelligence** — the ability to generate original ideas, write with voice and style, construct compelling narratives, and think laterally across domains. This is what makes Titan interesting rather than just capable.

### Coverage

| Domain | Capability |
|--------|-----------|
| **Narrative Fiction** | Short stories, novel chapters, character development, plot architecture |
| **Screenwriting Voice** | Dialogue with subtext, scene tension, character distinctiveness, genre conventions |
| **Creative Ideation** | Brainstorming, lateral thinking, concept generation, "what if" reasoning |
| **Stylistic Range** | Literary, genre, experimental, commercial — adapts to any tone |
| **Poetry & Lyric Writing** | Verse forms, rhythm, imagery, metaphor, song lyrics |
| **World-Building** | Fictional universe construction, internal consistency, lore development |
| **Creative Non-Fiction** | Essays, personal narrative, long-form journalism style |
| **Conceptual Thinking** | Abstract reasoning, analogy, cross-domain idea synthesis |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **WritingPrompts** | `euclaise/writingprompts` | 300K+ stories | Creative fiction from prompts |
| **Project Gutenberg Fiction** | gutenberg.org (public domain) | ~60K books | Classic literary prose and style |
| **PoetryFoundation** | poetryfoundation.org | ~14K poems | Poetry, verse, lyric forms |
| **Creative-Instruct (custom)** | Generated via GPT-4.1-mini | ~25K examples | Creative writing instruction pairs |
| **Brainstorming & Ideation** | Generated synthetic dataset | ~15K examples | Lateral thinking, concept generation |
| **Screenplay Dialogue** | Extracted from movie scripts | ~50K exchanges | Authentic dialogue and subtext |

**Total creative writing dataset: ~200K high-quality examples**

### Estimated Cost
- **Duration:** ~15–18 hours
- **Cost:** ~$6–7

---

## Upgrade E: Proactive Problem Solving

### Goal
This is the most important upgrade. Titan should not wait to be asked — it should **anticipate problems, identify root causes, and propose solutions before the user even realises there is an issue**. This is trained through reasoning chains, diagnostic datasets, and multi-step problem decomposition examples.

### What This Means in Practice

- When analysing a film production schedule, Titan flags the location conflict on day 14 before you ask
- When reviewing a security architecture, Titan identifies the authentication gap before the audit
- When given a script, Titan notes the structural issue in act two before you finish reading
- When monitoring a system, Titan detects the anomaly pattern and explains what it predicts will happen next

### Coverage

| Domain | Capability |
|--------|-----------|
| **Diagnostic Reasoning** | Identify what is wrong from symptoms, trace to root cause |
| **Anticipatory Analysis** | Given a plan or system, identify what will fail and when |
| **Multi-Step Problem Decomposition** | Break complex problems into ordered, solvable steps |
| **Decision Trees** | Map out options, consequences, and optimal paths |
| **Risk Assessment** | Identify risks, rank by probability and impact, propose mitigations |
| **Iterative Refinement** | Propose a solution, test it mentally, refine it, repeat |
| **Cross-Domain Pattern Recognition** | Apply lessons from one domain to solve problems in another |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **Chain-of-Thought (CoT) Reasoning** | `reasoning-machines/gsm8k` + `cot_gsm8k` | ~100K examples | Step-by-step problem solving |
| **ARC Challenge** | `allenai/ai2_arc` | 7,787 examples | Hard reasoning and science problems |
| **StrategyQA** | `wics/strategy-qa` | 2,780 examples | Multi-step strategic reasoning |
| **ProofWriter** | `allenai/proofwriter` | ~130K examples | Logical deduction chains |
| **Diagnostic Reasoning (custom)** | Generated via GPT-4.1-mini | ~20K examples | Problem → diagnosis → solution chains |
| **Anticipatory Planning (custom)** | Generated synthetic dataset | ~15K examples | "What will go wrong and why" reasoning |
| **Cross-Domain Problem Solving** | Generated synthetic dataset | ~10K examples | Apply domain A knowledge to domain B problems |

**Total problem-solving dataset: ~150K high-quality examples**

### Estimated Cost
- **Duration:** ~12–15 hours
- **Cost:** ~$5–6

---

## Complete Upgrade Cost Summary

| Stage | Pillar | GPU Time | Cost |
|-------|--------|---------|------|
| Crucible (running) | Base language model | ~160 hrs | ~$64 |
| SFT | Instruction following | ~10 hrs | ~$5 |
| Tool Use | API and tool calling | ~5 hrs | ~$2 |
| **Upgrade A** | **General Knowledge** | ~18 hrs | ~$7 |
| **Upgrade B** | **Film & Cinema Production** | ~22 hrs | ~$9 |
| **Upgrade C** | **Cybersecurity** | ~18 hrs | ~$7 |
| **Upgrade D** | **Creative Writing** | ~16 hrs | ~$6 |
| **Upgrade E** | **Proactive Problem Solving** | ~14 hrs | ~$6 |
| Agentic Scaffolding | Runtime, memory, tools | Code only | ~$0 |
| **Grand Total** | **Titan AI v1.5** | **~263 hrs** | **~$106** |

**Starting balance:** $169.13  
**Estimated remaining after completion:** ~$63  
**No top-up required.**

---

## Execution Order

```bash
# Run the complete pipeline from start to finish:
python3 scripts/run_upgrade_pipeline.py --start-from sft

# Or resume from a specific stage if needed:
python3 scripts/run_upgrade_pipeline.py --start-from cyber
```

Stages run sequentially. Each builds on the checkpoint from the previous stage. Titan's knowledge compounds at every step — the problem-solving upgrade at the end benefits from everything Titan already knows about film, security, science, and creative thinking.

---

## What Titan Will Be at v1.5

A 109M parameter model that:

- Knows world history, geography, science, and chemistry at an educated-professional level
- Can manage a film production from the first logline through to the marketing campaign
- Can analyse vulnerabilities, write exploits, and build security tools
- Can write a screenplay, a short story, or a concept pitch with genuine creative voice
- **Proactively identifies problems and proposes solutions before being asked**

This is not a large model. But it will be a **sharp, focused, expert model** — trained deeply on exactly the domains that matter, with a reasoning capability that makes it genuinely useful rather than just impressive in demos.

---

*All configs and scripts for this upgrade pipeline are committed to `leego972/titanai` and ready to execute on Crucible completion.*
