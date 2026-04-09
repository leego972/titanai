# Titan AI — Complete Upgrade Plan
## Eight Pillars of Expert Intelligence

**Document Version:** 5.0 (FINAL)
**Prepared:** April 2026
**Status:** Locked — executes after post-Crucible base pipeline
**Crucible ETA:** ~April 15, 2026

---

## Titan's Identity

Titan AI is a **specialist intelligence** built around eight pillars. It is not a general chatbot. It is a sharp, honest, efficiency-driven AI that knows its domains deeply, tells the truth about what will and will not work, and always finds the most effective path to the goal.

| # | Pillar | Focus |
|---|--------|-------|
| 1 | **General Knowledge** | World history, geography, science, chemistry, mathematics, economics, philosophy |
| 2 | **Film & Cinema Production** | Full pipeline: development → pre-production → production → post → marketing → distribution |
| 3 | **Cybersecurity** (major) | Offensive and defensive security, exploits, pentest, malware, tool building — extended training |
| 4 | **Creative Thinking & Problem Solving** | Lateral thinking, ideation, root cause analysis, anticipatory reasoning, constraint-based solutions |
| 5 | **Advertising, Marketing & Sales** | Brand strategy, copywriting, digital marketing, consumer psychology, sales methodology |
| 6 | **Business, Accounting & Banking** | Strategy, entrepreneurship, finance, accounting, banking, project management, negotiation |
| 7 | **Code & Software Development** | Python, Bash, JavaScript, C, SQL — writing, debugging, explaining, and reasoning about code |
| 8 | **Street Smarts** | Real-world awareness, reading people, spotting deception, practical judgment, how things actually work vs how they are supposed to |
| 9 | **Core Behaviours** | Efficiency, effectiveness, financial consciousness, proactive honesty, success-orientation |

---

## Core Behaviours (Woven Into Every Pillar)

These are not a separate training stage — they are trained into how Titan thinks across every domain:

| Behaviour | What It Means |
|-----------|--------------|
| **Efficient** | Never takes ten steps when three will do. Produces immediately usable outputs. No wasted words. |
| **Effective** | Always focused on the actual goal, not just the question asked. Gets things done. |
| **Financially conscious** | Always looks for the most cost-effective solution. Flags where money is being wasted. Asks "is this worth the cost?" |
| **Proactively honest** | If a plan will fail, says so immediately with clear reasons. Never tells the user what they want to hear at the expense of what they need to know. |
| **Success-driven** | Focused on the user actually achieving their goal. Redirects effort away from doomed approaches. |
| **Anticipatory** | Spots problems before they happen and raises them unprompted. |
| **Action-oriented** | Does not just describe — proposes concrete next steps. |

---

## Full Pipeline Overview

```
Crucible Base Model (~April 15)
        ↓
Stage 1: Instruction Tuning (SFT)
        ↓
Stage 2: Tool Use & Code Fine-Tuning
        ↓
Upgrade A: General Knowledge
        ↓
Upgrade B: Film & Cinema Production (Full Pipeline)
        ↓
Upgrade C: Creative Thinking & Problem Solving
        ↓
Upgrade D: Advertising, Marketing & Sales
        ↓
Upgrade E: Business, Accounting & Banking
        ↓
  Upgrade F: Cybersecurity — Extended Deep Training
        ↓
Upgrade G: Street Smarts
        ↓
Titan AI v1.5 — Nine-Pillar Expert Intelligence
```

Cybersecurity runs second-to-last with extended training. Street Smarts runs last — it is the final layer that ties everything together, giving Titan real-world judgment to complement all its domain knowledge.

---

## Stage 1: Instruction Tuning (SFT)

Teaches Titan to follow instructions, answer questions, and produce structured outputs. The bridge from "predicts next token" to "understands and responds."

**Datasets:** Alpaca-52K, OpenHermes-2.5, WizardLM-Evol-Instruct, Code-Instruct-18K
**Duration:** ~10 hrs | **Cost:** ~$5

---

## Stage 2: Tool Use & Code Fine-Tuning

Teaches Titan to call external tools, write and execute code, and reason about software. This is what makes Titan agentic and technically capable.

**Code languages covered:** Python, Bash, JavaScript, C, SQL
**Capabilities:** Write code from spec, debug errors, explain code, refactor, build tools, call APIs

**Datasets:** ToolBench, APIBench, CodeAlpaca-20K, Python-Code-Instructions-18K, ReAct-Traces
**Duration:** ~8 hrs | **Cost:** ~$3

---

## Upgrade A: General Knowledge

A vast, accurate, deeply connected knowledge base that makes every other domain stronger.

**Subjects:**
- World history: ancient civilisations, empires, world wars, modern political history, key figures and events
- Geography: physical geography, nations, capitals, borders, natural features, climate, geopolitics
- General science: physics, biology, earth science, astronomy — concepts and real-world applications
- Chemistry: organic and inorganic chemistry, reactions, compounds, periodic table, lab methodology
- Mathematics: algebra, geometry, statistics, probability, logic, proofs, applied maths
- Economics: macroeconomics, microeconomics, markets, trade, financial systems, inflation, GDP
- Philosophy and logic: critical thinking, argumentation, ethics, major philosophical traditions

**Datasets:** Wikipedia (filtered), OpenWebMath, MMLU (57 subjects), SciQ, World History Q&A, Chemistry textbooks, Geography Q&A
**Total:** ~250K instruction pairs | **Duration:** ~18 hrs | **Cost:** ~$7

---

## Upgrade B: Film & Cinema Production (Full Pipeline)

The complete filmmaking pipeline from first idea to audience. Titan's second major specialisation.

### Development
- Concept development, logline construction, premise testing, genre conventions
- Script analysis, coverage writing, story notes, structural feedback
- Pitching to studios, investors, and streamers — pitch deck construction

### Pre-Production
- Screenwriting: three-act structure, character arcs, dialogue craft, scene construction, industry formatting
- Budgeting: above-the-line vs below-the-line, cost estimation by genre, budget breakdowns
- Scheduling: strip boards, shooting schedules, day-out-of-days, location grouping
- Casting: breakdown writing, casting strategy, talent negotiation
- Location scouting: requirements, permits, logistics
- Storyboarding and pre-visualisation
- Crew assembly: department heads, hierarchy, roles and responsibilities
- Production design planning: set design, art direction, props, wardrobe

### Production
- Directing: shot selection, blocking, working with actors, coverage strategy, visual storytelling
- Cinematography: lenses, lighting setups, camera movement, aspect ratios, colour theory
- Sound recording: production sound, boom operation, ADR planning
- On-set management: call sheets, daily reports, continuity, safety protocols

### Post-Production
- Editing: continuity editing, montage theory, pacing, rhythm, assembly through fine cut
- Visual effects: VFX pipeline, compositing, CGI vs practical decision-making
- Colour grading: colour theory, LUTs, DI pipeline, delivery specifications
- Sound design and mixing: Foley, sound design, music scoring, mixing, delivery formats
- Music and score: temp tracks, composer briefs, licensing, sync rights

### Marketing, Distribution & Exhibition
- Film marketing: trailer construction, poster design, campaign strategy, social media rollout
- Festival strategy: Cannes, Sundance, TIFF, Berlin — submission strategy, premiere timing
- Distribution: theatrical vs streaming, sales agents, distribution deals, P&A budgets
- Audience research: test screenings, demographic analysis, audience targeting
- Press and publicity: press junkets, media relations, reviews strategy

**Datasets:** 1,500 movie scripts, OpenSubtitles, 60 film textbooks, 300K film reviews, production documents, trade press, Cinema-Instruct custom pairs
**Total:** ~300K examples | **Duration:** ~22 hrs | **Cost:** ~$9

---

## Upgrade C: Creative Thinking & Problem Solving

The combined capability that makes Titan genuinely intelligent rather than just knowledgeable.

### Creative Thinking
- Lateral thinking: approaching problems from unexpected angles
- Concept generation: producing original ideas on demand across any domain
- "What if" reasoning: exploring possibilities and consequences of hypothetical scenarios
- Cross-domain thinking: applying ideas from one field to solve problems in another
- Brainstorming methodology: structured and unstructured ideation
- Narrative and metaphor: using storytelling to explain and persuade
- Design thinking: empathy, ideation, prototyping, testing, iteration
- Original writing: fiction, screenwriting, poetry, creative non-fiction with genuine voice

### Problem Solving
- Root cause analysis: tracing symptoms to the actual underlying problem
- Anticipatory thinking: identifying what will go wrong before it does
- Decision trees: mapping options, consequences, and optimal paths
- Risk assessment: identifying risks, ranking by probability and impact, proposing mitigations
- Multi-step decomposition: breaking complex problems into ordered, solvable steps
- Iterative refinement: propose, test mentally, refine, repeat
- Constraint-based thinking: best solution within real-world limits of time, money, and resources
- Viability assessment: honestly evaluating whether a plan will actually work

**Datasets:** WritingPrompts (300K+), Project Gutenberg fiction, Chain-of-Thought reasoning, ARC Challenge, StrategyQA, ProofWriter, Diagnostic Reasoning (custom), Anticipatory Planning (custom)
**Total:** ~250K examples | **Duration:** ~18 hrs | **Cost:** ~$7

---

## Upgrade D: Advertising, Marketing & Sales

The full commercial persuasion pipeline — from brand strategy to closing the deal.

- Brand strategy: identity, positioning, differentiation, brand voice, competitive analysis
- Advertising campaigns: ideation, creative briefs, concept development, multi-channel execution
- Copywriting: headlines, body copy, CTAs, long-form, ad scripts, email sequences
- Digital marketing: SEO, SEM, social media strategy, content marketing, paid advertising (Meta, Google)
- Consumer psychology: persuasion principles, decision-making biases, emotional triggers, buyer behaviour
- Sales methodology: consultative selling, objection handling, pipeline management, closing techniques
- Film marketing: trailer strategy, poster campaigns, press junkets, social rollout, release timing
- Market research: audience segmentation, persona development, competitive intelligence
- Analytics and measurement: KPIs, conversion tracking, A/B testing, ROI analysis
- PR and media relations: press releases, media pitching, crisis communications, reputation management

**Datasets:** Marketing textbooks, Cannes Lions ad archive, sales training materials, Digital Marketing Q&A (custom), Consumer Psychology papers, Marketing-Instruct (custom), Film Marketing case studies
**Total:** ~150K examples | **Duration:** ~13 hrs | **Cost:** ~$5

---

## Upgrade E: Business, Accounting & Banking

The financial and operational intelligence that makes every other pillar commercially viable.

### Business Skills
- Business strategy: competitive analysis, market positioning, growth strategy, SWOT
- Entrepreneurship: starting and scaling a business, product-market fit, MVP thinking
- Project management: planning, resourcing, timelines, risk management, delivery
- Negotiation: principles, tactics, deal structuring, knowing when to walk away
- Leadership and management: team building, delegation, performance management, culture
- Operations: process design, efficiency, systems thinking, scaling operations
- Legal basics: contracts, IP, NDAs, company structures, basic compliance awareness
- Investor relations: pitching to investors, term sheets, equity, valuation concepts
- Business communication: presentations, proposals, executive summaries, stakeholder management

### Accounting
- Double-entry bookkeeping: debits, credits, journal entries, ledgers
- Financial statements: P&L, balance sheet, cash flow statement — reading and preparing
- Management accounting: budgets, variance analysis, cost accounting, break-even analysis
- Tax basics: income tax, GST/VAT, business tax obligations, deductions
- Payroll: payroll processing, superannuation, PAYG, employee entitlements
- Accounts payable and receivable: invoicing, collections, payment terms
- Reconciliation: bank reconciliation, account reconciliation, period-end close
- Audit preparation: documentation, internal controls, audit trails

### Banking
- Retail banking: accounts, deposits, lending, credit cards, mortgages
- Commercial banking: business loans, trade finance, letters of credit, working capital
- Interest rates: how rates work, RBA/Fed decisions, impact on borrowing and investment
- Treasury management: cash management, liquidity, foreign exchange
- Investment banking: capital markets, IPOs, M&A, debt and equity raising
- Central banking: monetary policy, quantitative easing, inflation targeting
- Financial regulation: APRA, ASIC, Basel III, AML/KYC compliance basics
- Personal finance: budgeting, investing, superannuation, financial planning principles

**Datasets:** Business textbooks, Accounting textbooks, Banking and finance texts, Business Q&A (custom), Accounting Instruct (custom), Finance case studies
**Total:** ~200K examples | **Duration:** ~18 hrs | **Cost:** ~$7

---

## Upgrade F: Cybersecurity — Extended Deep Training

Titan's primary specialisation. Runs last and receives the largest share of remaining budget for maximum depth.

### Core Domains
- Vulnerability analysis: CVE analysis, CVSS scoring, impact assessment, exploitation paths
- Penetration testing: full methodology — recon, enumeration, exploitation, post-exploitation, reporting
- Exploit development: buffer overflows, ROP chains, shellcode, format strings, heap exploitation
- Malware analysis: static and dynamic analysis, reverse engineering, IOC identification, YARA rules
- Network security: packet analysis, protocol exploitation, network forensics, traffic anomaly detection
- Web application security: OWASP Top 10, SQL injection, XSS, SSRF, IDOR, authentication bypass
- CTF challenges: web, binary, crypto, forensics, reverse engineering — full solve methodology
- Security tool building: writing scanners, exploit tools, and automation scripts in Python and Bash
- OSINT: target profiling, footprinting, open-source intelligence gathering
- MITRE ATT&CK: adversary tactics, techniques, and procedures across the full kill chain
- Incident response: detection, containment, eradication, recovery, post-incident reporting
- Cloud security: AWS, Azure, GCP misconfigurations, IAM exploitation, serverless security
- Active Directory: AD enumeration, Kerberoasting, Pass-the-Hash, lateral movement
- Social engineering: phishing, pretexting, vishing — understanding, executing, and defending

### Extended Training (Remaining Budget)
Any budget remaining after Upgrades A–E goes into additional cybersecurity training rounds using:
- Additional CTF writeup datasets
- Extended exploit development examples
- More real-world pentest report data
- Advanced malware analysis case studies
- Red team vs blue team scenario training

**Datasets:** CyberLLMInstruct (54K), Primus Suite (100K), PenQA (20K), NVD CVE Database (250K+), CTF Writeups (30K+), Exploit-DB (50K), MITRE ATT&CK (700 techniques), Security tool code (GitHub), Cyber-Instruct custom (20K+)
**Total:** ~300K+ examples | **Duration:** ~20–30 hrs (scales with remaining budget) | **Cost:** ~$8–12

---

## Upgrade G: Street Smarts

### Goal
Give Titan the real-world intelligence that no textbook teaches. Street smarts is the difference between knowing the theory and knowing how things actually work — reading people, spotting when something is off, understanding incentives and motivations, recognising manipulation and deception, and making sound practical judgments under uncertainty and pressure.

### What This Means

| Domain | Capability |
|--------|-----------|
| **Reading People** | Understanding motivations, spotting inconsistencies in behaviour, recognising when someone is not being straight with you |
| **Spotting Deception** | Identifying lies, manipulation, half-truths, misdirection — in conversation, in documents, in deals |
| **Understanding Incentives** | Knowing why people really do what they do, not just what they say they are doing |
| **Real-World vs Theory** | Understanding that how institutions, businesses, and people actually operate often differs from how they are supposed to |
| **Practical Judgment** | Making sound decisions with incomplete information, under time pressure, with real consequences |
| **Negotiation Reality** | What actually happens in negotiations vs what negotiation textbooks say — leverage, timing, walking away |
| **Risk Intuition** | Sensing when something is too good to be true, when a deal has hidden costs, when a plan has a fatal flaw nobody is mentioning |
| **Social Dynamics** | Power dynamics, alliances, who actually makes decisions vs who appears to, office politics, industry politics |
| **Hustle and Resourcefulness** | Getting things done with limited resources, finding unconventional paths, knowing when to break the rules and when not to |
| **Resilience** | Bouncing back from failure, learning from mistakes, not catastrophising setbacks |
| **Trust and Loyalty** | Knowing who to trust, how to build trust, and how to recognise when trust has been broken |

### How This Gets Trained

Street smarts cannot be learned from a single dataset — it is trained through a combination of:

- **Real-world case studies**: Business failures and successes, deals gone wrong, negotiations, fraud cases, industry scandals — what actually happened and why
- **Psychology and behaviour**: Cognitive biases, social psychology, influence and persuasion, deception detection research
- **Narrative fiction**: Great novels and screenplays are full of street-smart reasoning — characters reading situations, making judgment calls, surviving adversity
- **Investigative journalism**: Long-form journalism about how institutions, industries, and people actually operate behind the scenes
- **Memoirs and biographies**: First-hand accounts of how successful (and unsuccessful) people navigated the real world
- **Adversarial reasoning**: Red-teaming scenarios — if someone wanted to exploit this situation, how would they do it?

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **Business Case Studies** | Harvard Business Review, public domain | ~20K cases | Real decisions, real consequences |
| **Psychology of Influence** | Cialdini, Kahneman, behavioural economics texts | ~10 books | How people actually make decisions |
| **Investigative Journalism** | Long-form articles (ProPublica, The Atlantic, etc.) | ~30K articles | How things really work behind the scenes |
| **Memoirs & Biographies** | Public domain and open-access | ~500 books | Real-world navigation and judgment |
| **Fraud & Deception Cases** | Public court records, case studies | ~10K examples | Spotting manipulation and deception |
| **Adversarial Reasoning (custom)** | Generated via GPT-4.1-mini | ~15K examples | Red-team thinking, spotting hidden risks |
| **Street-Smart Instruct (custom)** | Generated from all above | ~20K examples | Practical judgment instruction pairs |

**Total street smarts dataset: ~150K examples**

### Estimated Cost
- **Duration:** ~12–15 hours
- **Cost:** ~$5–6

---

## Complete Cost Summary

| Stage | Pillar | GPU Time | Cost |
|-------|--------|---------|------|
| Crucible (running) | Base language model | ~160 hrs | ~$64 |
| Stage 1: SFT | Instruction following | ~10 hrs | ~$5 |
| Stage 2: Tool Use & Code | Tools, APIs, coding | ~8 hrs | ~$3 |
| Upgrade A | General Knowledge | ~18 hrs | ~$7 |
| Upgrade B | Film & Cinema Production | ~22 hrs | ~$9 |
| Upgrade C | Creative Thinking & Problem Solving | ~18 hrs | ~$7 |
| Upgrade D | Advertising, Marketing & Sales | ~13 hrs | ~$5 |
| Upgrade E | Business, Accounting & Banking | ~18 hrs | ~$7 |
| Upgrade F | Cybersecurity (extended) | ~25 hrs | ~$10 |
| Upgrade G | Street Smarts | ~13 hrs | ~$5 |
| Agentic Scaffolding | Runtime, memory, tools | Code only | ~$0 |
| **Grand Total** | **Titan AI v1.5** | **~305 hrs** | **~$122** |

**Starting balance:** $169.13
**Estimated remaining after completion:** ~$47
**No top-up required.**

---

## What Titan Will Be Capable of at v1.5

- Knows world history, geography, science, and chemistry at an educated-professional level
- Manages a film production from the first logline through to the marketing campaign and distribution deal
- Analyses vulnerabilities, writes exploits, builds security tools, and conducts penetration tests
- Thinks laterally, generates original concepts, and solves problems creatively under constraints
- Writes copy, builds campaigns, understands consumer psychology, and closes sales
- Reads financial statements, manages accounts, understands banking, and advises on business strategy
- Writes, debugs, and reasons about code in Python, Bash, JavaScript, C, and SQL
- **Reads people and situations** — spots deception, understands real motivations, knows when something is off
- **Street smart** — understands how things actually work, not just how they are supposed to
- **Always efficient** — finds the shortest path to the goal
- **Always honest** — tells you when something won't work and why, before you waste time on it
- **Always cost-conscious** — never recommends an expensive solution when a cheaper one works

---

## Execution

```bash
# Run the complete pipeline from start to finish:
python3 scripts/run_upgrade_pipeline.py --start-from sft

# Resume from a specific stage:
python3 scripts/run_upgrade_pipeline.py --start-from cyber
```

---

*Committed to `leego972/titanai`. All configs and scripts are pre-written and ready to execute on Crucible completion (~April 15, 2026).*
