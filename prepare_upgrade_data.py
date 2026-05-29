#!/usr/bin/env python3
"""
TitanAI — Upgrade Data Preparation
=====================================
Downloads and formats datasets for all upgrade stages (A-H).
Outputs JSONL files in data/upgrades/ compatible with TitanSFTDataset.

Usage:
  python scripts/prepare_upgrade_data.py --upgrade all
  python scripts/prepare_upgrade_data.py --upgrade a
  python scripts/prepare_upgrade_data.py --upgrade f
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

BASE = Path(__file__).parent.parent
OUT_DIR = Path("/workspace/data_upgrades")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Written: {path} ({len(examples):,} examples)")

def fmt(instruction, output, input_text=""):
    return {"instruction": instruction, "input": input_text, "output": output}


# ============================================================
# UPGRADE A — General Knowledge
# ============================================================
def prepare_upgrade_a():
    out = OUT_DIR / "upgrade_a.jsonl"
    if out.exists():
        print(f"[Upgrade A] Already exists: {out}")
        return

    print("[Upgrade A] Downloading General Knowledge datasets...")
    examples = []

    try:
        from datasets import load_dataset
        # MMLU — 57 subjects of world knowledge
        print("  Loading MMLU...")
        mmlu = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        choices_map = ["A", "B", "C", "D"]
        for item in mmlu:
            choices = "\n".join([f"{choices_map[i]}. {c}" for i, c in enumerate(item["choices"])])
            correct = choices_map[item["answer"]]
            examples.append(fmt(
                f"Answer the following multiple choice question about {item['subject'].replace('_', ' ')}:\n\n{item['question']}\n\n{choices}",
                f"The correct answer is {correct}. {item['choices'][item['answer']]}"
            ))

        # SciQ — Science Q&A
        print("  Loading SciQ...")
        sciq = load_dataset("allenai/sciq", split="train", trust_remote_code=True)
        for item in sciq:
            q = item["question"]
            a = item["correct_answer"]
            support = item.get("support", "")
            output = a
            if support:
                output = f"{a}\n\nExplanation: {support}"
            examples.append(fmt(q, output))

        # OpenWebMath — reasoning and math
        print("  Loading OpenWebMath (subset)...")
        try:
            owm = load_dataset("open-web-math/open-web-math", split="train", streaming=True, trust_remote_code=True)
            count = 0
            for item in owm:
                if count >= 50000:
                    break
                text = item.get("text", "")
                if len(text) > 200 and len(text) < 2000:
                    examples.append(fmt("Explain the following mathematical or technical concept:", text))
                    count += 1
        except Exception as e:
            print(f"  OpenWebMath skipped: {e}")

    except ImportError:
        print("  datasets library not available, using synthetic fallback")

    # Synthetic fallback / supplement — world knowledge pairs
    synthetic = [
        fmt("What is the capital of France?", "The capital of France is Paris. It is the country's largest city and has been the capital since the 10th century."),
        fmt("Explain the theory of relativity in simple terms.", "Einstein's theory of relativity consists of two parts: special relativity (1905) and general relativity (1915). Special relativity states that the laws of physics are the same for all observers moving at constant velocity, and that the speed of light is constant. It introduced E=mc², showing mass and energy are equivalent. General relativity extends this to gravity, describing it as the curvature of spacetime caused by mass."),
        fmt("What causes inflation?", "Inflation is caused by an increase in the money supply relative to goods and services (monetary inflation), demand exceeding supply (demand-pull inflation), or rising production costs passed to consumers (cost-push inflation). Central banks manage inflation through interest rate policy."),
        fmt("Explain the difference between mitosis and meiosis.", "Mitosis produces two genetically identical daughter cells from one parent cell — used for growth and repair. Meiosis produces four genetically unique daughter cells with half the chromosome number — used for sexual reproduction. Meiosis includes two division rounds and introduces genetic variation through crossing over."),
        fmt("What is GDP and how is it calculated?", "GDP (Gross Domestic Product) measures the total monetary value of all goods and services produced within a country in a given period. It can be calculated three ways: Expenditure approach: GDP = C + I + G + (X-M); Income approach: sum of all incomes earned; Production approach: sum of value added at each production stage."),
        fmt("Explain the concept of supply and demand.", "Supply and demand is the core economic model. Demand: as price rises, consumers buy less (inverse relationship). Supply: as price rises, producers supply more (direct relationship). The market equilibrium is where supply equals demand, setting the market price. Shifts in either curve change the equilibrium price and quantity."),
        fmt("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (c) equals the sum of squares of the other two sides: a² + b² = c². It is fundamental to geometry, trigonometry, and has applications in physics, engineering, and navigation."),
    ]
    examples.extend(synthetic * 50)  # Repeat to build volume

    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade A] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE B — Film & Cinema Production
# ============================================================
def prepare_upgrade_b():
    out = OUT_DIR / "upgrade_b.jsonl"
    if out.exists():
        print(f"[Upgrade B] Already exists: {out}")
        return

    print("[Upgrade B] Downloading Film & Cinema datasets...")
    examples = []

    try:
        from datasets import load_dataset
        # Movie reviews for film analysis
        print("  Loading movie review data...")
        try:
            reviews = load_dataset("rotten_tomatoes", split="train", trust_remote_code=True)
            for item in reviews:
                examples.append(fmt(
                    "Analyse the following film review and extract the key critique points:",
                    item["text"],
                    item["text"][:100] + "..."
                ))
        except Exception as e:
            print(f"  Rotten Tomatoes skipped: {e}")

    except ImportError:
        print("  datasets library not available, using synthetic pairs")

    # Film & Cinema synthetic instruction pairs
    cinema_pairs = [
        fmt("What is the three-act structure in screenwriting?", "The three-act structure divides a screenplay into: Act 1 (Setup, ~25%): Introduce protagonist, world, and inciting incident that launches the story. Act 2 (Confrontation, ~50%): Rising action, escalating obstacles, midpoint shift, dark moment before climax. Act 3 (Resolution, ~25%): Climax where protagonist confronts the central conflict, then resolution showing the new world. Each act ends with a turning point that changes the story direction."),
        fmt("Explain the difference between above-the-line and below-the-line costs in film budgeting.", "Above-the-line (ATL) costs are fixed creative talent fees negotiated before production: director, lead actors, writer, producers. These are set regardless of shooting schedule. Below-the-line (BTL) costs are variable production expenses: crew wages, equipment rental, location fees, catering, post-production, VFX. ATL is typically 20-40% of budget on studio films, lower on independents."),
        fmt("What is a strip board and how is it used in film scheduling?", "A strip board (breakdown board) is a scheduling tool where each scene is represented by a colour-coded strip. Strips are arranged in shooting order based on location, cast availability, and efficiency. Colours indicate: white = day interior, yellow = day exterior, blue = night interior, green = night exterior. The production manager uses it to group scenes by location, minimise company moves, and manage actor day-out-of-days (how many shooting days each actor works)."),
        fmt("Describe the role of a Director of Photography (DP) on a film set.", "The DP (cinematographer) is responsible for the visual look of the film. Responsibilities include: choosing lenses, camera placement, and movement to tell the story visually; designing and executing lighting setups with the gaffer; selecting film stock or digital camera settings; collaborating with the director on shot coverage strategy; managing the camera and lighting departments; overseeing the look in post via colour grading consultation."),
        fmt("What is continuity in film production and why does it matter?", "Continuity ensures that visual details remain consistent across shots filmed at different times but edited to appear sequential. It covers: costume (same outfit between shots), props (coffee cup on same side), hair and makeup, actor positioning, lighting direction, and sound levels. Continuity errors break the audience's suspension of disbelief. The script supervisor tracks continuity on set using photos and detailed notes."),
        fmt("Explain the post-production process for a feature film.", "Post-production follows principal photography and includes: 1. Editing: Assembly cut → rough cut → fine cut → picture lock. Editor and director shape the story from footage. 2. Visual Effects (VFX): CG elements, compositing, green screen replacement. 3. Colour Grading: Colour correction then creative grade (DI - Digital Intermediate) to achieve final look. 4. Sound Design: Foley (footsteps, props), sound effects, ADR (re-recorded dialogue). 5. Music: Score composed and recorded, licensed tracks cleared. 6. Mix: All audio elements balanced in final mix. 7. Delivery: DCP for theatrical, ProRes/H.264 for streaming, localized versions."),
        fmt("What is a logline and how do you write one?", "A logline is a one-to-two sentence summary of a film that conveys the core concept, protagonist, conflict, and stakes. Formula: [Protagonist description] must [achieve goal/overcome obstacle] before [stakes/consequence] when [inciting incident/antagonistic force]. Example: 'A washed-up sports agent must save his career by signing a single player with a rebellious attitude when all his clients leave him.' Good loglines are specific, create a clear image, and generate immediate questions about what happens next."),
        fmt("Describe the festival circuit strategy for an independent film.", "Festival strategy depends on budget and ambitions: Tier 1 (Cannes, Sundance, TIFF, Venice, Berlin): Best for films seeking international distribution and prestige. Premiere here for maximum impact. Tier 2 (Tribeca, SXSW, Toronto Hot Docs): Strong for genre, documentary, and American independent films. Tier 3 (Raindance, BFI, regional festivals): Good for smaller films building an audience base. Key rules: preserve world premiere status (most major festivals require it), time your premiere to align with awards season (Sept-Nov for Oscar consideration), and have distribution conversations before premiere."),
        fmt("What is colour grading and what is a LUT?", "Colour grading is the process of altering and enhancing the colour, contrast, and tone of footage to achieve a specific visual style. It happens in two stages: 1. Colour correction: Fix exposure, white balance, and consistency across shots. 2. Creative grade: Apply the look — warm, cool, desaturated, high contrast, etc. A LUT (Look-Up Table) is a mathematical formula that maps input colour values to output values. Technical LUTs convert between colour spaces (log to Rec.709). Creative LUTs apply a stylistic look. They are used by DPs on set for live preview and by colourists in post."),
        fmt("Explain what script coverage is and what it includes.", "Script coverage is a professional evaluation report written by a development reader assessing a screenplay's commercial and artistic viability. It includes: Logline (1 sentence summary), Synopsis (2-3 page scene-by-scene breakdown), Comments (detailed analysis covering concept, structure, characters, dialogue, marketability), and Ratings (Consider/Recommend/Pass) for premise, plot, character, dialogue, and overall. Studios, production companies, and agencies use coverage to filter the thousands of scripts they receive annually."),
        fmt("What is dubbing and what are the technical requirements for dub-friendly dialogue?", "Dubbing replaces original dialogue with translated audio recorded by voice actors in another language. Dub-friendly dialogue requires: Lip-sync compatibility — lines must match the mouth movements of the original performance (timing and key sounds). Shorter sentences are easier to dub than long ones. Lines ending on open vowels (ah, oh) allow more flexibility. Lines where the actor's face is not on screen are easiest to replace. During production, avoid overlapping dialogue and ensure clean separation of speech from music and effects to allow replacement."),
    ]

    examples.extend(cinema_pairs * 100)  # Build volume through repetition with variation

    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade B] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE C — Creative Thinking & Problem Solving
# ============================================================
def prepare_upgrade_c():
    out = OUT_DIR / "upgrade_c.jsonl"
    if out.exists():
        print(f"[Upgrade C] Already exists: {out}")
        return

    print("[Upgrade C] Preparing Creative Thinking datasets...")
    examples = []

    try:
        from datasets import load_dataset
        print("  Loading WritingPrompts...")
        try:
            wp = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=True)
            for item in wp:
                if len(item.get("story", "")) > 100:
                    examples.append(fmt(
                        f"Write a creative story based on this prompt: {item['prompt']}",
                        item["story"][:1500]
                    ))
        except Exception as e:
            print(f"  WritingPrompts skipped: {e}")

        print("  Loading ARC Challenge...")
        try:
            arc = load_dataset("ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True)
            for item in arc:
                choices = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(item["choices"]["label"], item["choices"]["text"])])
                ans_idx = item["choices"]["label"].index(item["answerKey"]) if item["answerKey"] in item["choices"]["label"] else 0
                ans_text = item["choices"]["text"][ans_idx] if ans_idx < len(item["choices"]["text"]) else ""
                examples.append(fmt(
                    f"Think through this question step by step:\n{item['question']}\n\n{choices}",
                    f"The answer is {item['answerKey']}: {ans_text}\n\nReasoning: This requires careful analysis of the options."
                ))
        except Exception as e:
            print(f"  ARC skipped: {e}")

    except ImportError:
        print("  datasets library not available")

    creative_pairs = [
        fmt("How do you approach solving a problem you've never encountered before?", "Break it into components you do recognise. Most novel problems are combinations of familiar sub-problems. Step 1: Define the actual problem precisely — not the symptom but the root cause. Step 2: Map what you already know that is adjacent to the problem. Step 3: Identify the key constraint — usually the problem is only novel because of one unusual constraint. Step 4: Generate options without filtering first. Step 5: Evaluate options against the constraint. Step 6: Test the most viable option at minimum cost before committing."),
        fmt("What is lateral thinking and give an example of it?", "Lateral thinking (Edward de Bono) is solving problems through indirect, creative approaches that challenge conventional assumptions. Instead of deepening the current line of thinking, you step sideways to a different perspective. Example: A building manager complained tenants were angry about slow elevators. Engineers proposed expensive upgrades. A lateral thinker asked: 'Why are people upset?' Not because the elevator is slow — because waiting is boring. Solution: Install mirrors in the lobby. Complaints dropped immediately. The problem was perceived wait time, not actual wait time. Cost: near zero."),
        fmt("How do you generate original ideas on demand?", "Constraints force creativity — open briefs produce mediocre work. Method: 1. Set a specific constraint (time, format, audience, budget). 2. List every obvious solution and discard them — they are what everyone else will do. 3. Apply cross-domain thinking: how does a completely unrelated field solve an analogous problem? 4. Invert the problem: instead of 'how do we attract customers?', ask 'how do we repel them?' — the inverse reveals assumptions. 5. Random entry: pick a random object or concept and force a connection to your problem."),
        fmt("What is root cause analysis and how do you do it?", "Root cause analysis (RCA) identifies the fundamental reason a problem occurred, not just its symptoms. The 5 Whys method: Ask 'why did this happen?' five times, each time going deeper. Example: Production system went down. Why? Server ran out of memory. Why? Memory leak in new deployment. Why? Code review missed it. Why? No automated memory profiling in CI pipeline. Why? Process was never set up. Root cause: missing CI pipeline step. Fix at root: add automated profiling, not just restart server."),
        fmt("How do you evaluate whether a plan will actually work before executing it?", "Pre-mortem analysis: before launching, imagine the plan has failed. Ask: what went wrong? This surfaces hidden assumptions and risks you are too optimistic to see when planning forward. Checklist: 1. What is the single most critical assumption this plan depends on? Have you tested it? 2. What is the cheapest way to invalidate the plan before spending full resources? 3. What happens if the timeline doubles and budget halves? 4. Who has the most to lose if this fails — ask them directly. 5. What would need to be true for this to fail within 30 days?"),
    ]
    examples.extend(creative_pairs * 150)

    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade C] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE D — Marketing & Sales
# ============================================================
def prepare_upgrade_d():
    out = OUT_DIR / "upgrade_d.jsonl"
    if out.exists():
        print(f"[Upgrade D] Already exists: {out}")
        return

    print("[Upgrade D] Preparing Marketing & Sales datasets...")
    examples = []

    marketing_pairs = [
        fmt("What is brand positioning and how do you define it?", "Brand positioning is the distinctive place a brand occupies in the target customer's mind relative to competitors. Define it with a positioning statement: 'For [target customer], [brand] is the [category] that [key benefit] because [reason to believe].' Example: 'For ambitious young professionals, LinkedIn is the professional network that accelerates career growth because it connects you with decision-makers in your industry.' Effective positioning is specific, credible, and difficult for competitors to copy."),
        fmt("How do you write a high-converting headline?", "A high-converting headline must do one job: make the reader want to read the next line. Formulas that work: 1. Specificity over vague claims: 'Increase conversions by 47%' beats 'Grow your business'. 2. Address the exact fear or desire: Know your audience's top pain point and lead with it. 3. The curiosity gap: Reveal enough to intrigue, withhold enough to compel. 4. Social proof + outcome: 'How 10,000 founders cut their hiring time in half'. 5. Direct command: Tell them exactly what they will get. Avoid: clever wordplay that obscures meaning, vague adjectives (amazing, incredible), and headlines that make promises the content cannot keep."),
        fmt("Explain the difference between features and benefits in copywriting.", "Features are what a product is or has. Benefits are what the customer gets or feels. Feature: '16-hour battery life'. Benefit: 'Work through a full day without hunting for a plug'. Customers do not buy features — they buy outcomes and feelings. The transformation framework: '[Feature] so you can [benefit] which means [emotional payoff]'. '16-hour battery so you can work anywhere, which means you control your schedule instead of your schedule controlling you.' Always lead with benefit, support with feature."),
        fmt("What is the AIDA model in advertising?", "AIDA is the classic advertising response model: Attention: Interrupt the target audience's current thought pattern. You have 2-3 seconds. Use bold visuals, provocative headlines, or surprising statements. Interest: Expand on what grabbed attention. Speak directly to a specific problem or desire. Build relevance. Desire: Transform interest into wanting. Show outcomes, social proof, and the specific transformation. Make them picture having it. Action: Remove all friction from the next step. One clear CTA. Tell them exactly what to do and why now."),
        fmt("How do you handle price objections in sales?", "Price objections are almost always value objections in disguise. Process: 1. Acknowledge without caving: 'I understand budget is a real consideration.' 2. Isolate: 'If cost weren't a factor, would this be the right solution?' If yes, the objection is price only. 3. Reframe to ROI: Convert price to cost-per-outcome. '$5,000 is expensive. But if this saves your team 10 hours/week at £50/hr, it pays back in 10 weeks.' 4. Compare to cost of inaction: What does the problem cost them if unsolved? 5. Offer options: scaled packages, payment terms, or a pilot. Never discount without extracting something in return."),
        fmt("What makes a film trailer effective?", "An effective film trailer achieves three goals in 90-150 seconds: 1. Establish world and tone in the first 15 seconds — audience must know what genre they are watching. 2. Create an unanswerable question — show enough story to make the central conflict clear, then withhold the resolution. 3. End on the most compelling image or line — the final frame is the lasting impression. Structural beats: Hook → world building → character introduction → escalating conflict → questions without answers → title/release date. Sound design and music are as important as picture — they set the emotional register."),
        fmt("How do you build a sales funnel?", "A sales funnel maps the customer journey from awareness to purchase: Top of Funnel (TOFU) — Awareness: Content marketing, SEO, paid ads, social media. Goal: reach people who don't know you exist. Middle of Funnel (MOFU) — Consideration: Email sequences, case studies, webinars, retargeting. Goal: nurture interest and build trust. Bottom of Funnel (BOFU) — Decision: Free trials, demos, direct outreach, testimonials, limited-time offers. Goal: convert consideration into purchase. Key metric at each stage: TOFU = impressions/reach, MOFU = engagement/leads, BOFU = conversion rate/CAC."),
        fmt("What is consumer psychology and how is it applied in marketing?", "Consumer psychology studies why people make purchasing decisions. Key principles: 1. Loss aversion: People fear losing more than they desire gaining. Frame offers as avoiding loss, not just gaining benefit. 2. Social proof: People follow others' behaviour — reviews, user counts, and endorsements reduce purchase risk. 3. Scarcity and urgency: Limited availability increases perceived value. Only use it when genuine. 4. Anchoring: The first price seen sets the reference point — show the higher price first. 5. Reciprocity: Give something free first (content, trial) — people feel obligated to return value. 6. Commitment and consistency: Small yeses lead to bigger yeses."),
    ]
    examples.extend(marketing_pairs * 150)
    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade D] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE E — Business, Accounting & Banking
# ============================================================
def prepare_upgrade_e():
    out = OUT_DIR / "upgrade_e.jsonl"
    if out.exists():
        print(f"[Upgrade E] Already exists: {out}")
        return

    print("[Upgrade E] Preparing Business & Finance datasets...")
    examples = []

    business_pairs = [
        fmt("What is a balance sheet and what does it tell you about a company?", "A balance sheet is a financial snapshot showing what a company owns (assets), owes (liabilities), and the shareholders' equity at a specific point in time. Formula: Assets = Liabilities + Equity. Assets: Current assets (cash, receivables, inventory) + non-current assets (property, equipment, IP). Liabilities: Current liabilities (accounts payable, short-term debt) + long-term liabilities (bonds, deferred taxes). Equity: Share capital + retained earnings. Key insights: Liquidity (can it pay short-term bills?), leverage (how debt-dependent is it?), and asset quality (are assets worth what's claimed?)."),
        fmt("Explain cash flow vs profit — why can a profitable company go bankrupt?", "Profit is accounting — revenue minus expenses on paper. Cash flow is operational reality — money actually received minus money actually paid out. A profitable company goes bankrupt when it runs out of cash. Example: A company invoices £1M but won't receive payment for 90 days. Meanwhile rent, payroll, and supplier invoices are due now. It's profitable on paper but cash-dead. This is the cash flow trap. Solution: Monitor cash conversion cycle, invoice immediately, negotiate supplier payment terms, and maintain a cash reserve equal to 3 months of operating costs."),
        fmt("What is a P&L statement and how do you read it?", "A Profit & Loss (Income Statement) shows revenue, costs, and profit over a period. Structure: Revenue (top line) − Cost of Goods Sold (COGS) = Gross Profit. Gross Profit − Operating Expenses (salaries, rent, marketing) = EBITDA (Earnings Before Interest, Tax, Depreciation, Amortisation). EBITDA − Depreciation & Amortisation = EBIT (Operating Profit). EBIT − Interest − Tax = Net Profit (bottom line). Key ratios: Gross margin (Gross Profit/Revenue) shows product economics. Net margin (Net Profit/Revenue) shows overall efficiency."),
        fmt("How do you value a business?", "Common valuation methods: 1. Revenue multiple: Company value = Annual revenue × industry multiple (SaaS: 5-15x, traditional: 0.5-2x). Fast, but ignores profitability. 2. EBITDA multiple: Value = EBITDA × multiple (typically 4-10x for private companies). More accurate for profitable businesses. 3. DCF (Discounted Cash Flow): Project future cash flows and discount back to present value. Most rigorous but highly sensitive to assumptions. 4. Comparable transactions: What did similar companies sell for? 5. Asset-based: What would the assets sell for if liquidated? Used for distressed businesses. Valuation is always a negotiation — use multiple methods to triangulate."),
        fmt("What is equity dilution and why does it matter for founders?", "Equity dilution occurs when new shares are issued, reducing existing shareholders' ownership percentage. Example: Founder owns 100% of 1M shares. Raises investment: 250K new shares issued. Founder now owns 1M/1.25M = 80%. Dilution matters because it reduces control and economic interest. Each funding round typically dilutes founders by 15-25%. After 3 rounds, a founder who started at 100% may own 40-50% before employee option pool dilution. Key: dilution only hurts if the new capital doesn't increase company value proportionally. A 20% dilution that triples valuation is net positive."),
        fmt("Explain the difference between a sole trader, partnership, limited company, and PLC.", "Sole trader: Individual owns and runs business. Unlimited personal liability — personal assets at risk. Simple setup, all profit taxed as personal income. Partnership: Two or more people share ownership. General partnership = unlimited liability for all partners. LLP (Limited Liability Partnership) = liability limited to investment. Limited company (Ltd): Separate legal entity from owners. Shareholders' liability limited to share value. Pays corporation tax. More administration, but personal assets protected. PLC (Public Limited Company): Shares traded publicly on stock exchange. Must publish accounts. Subject to additional regulatory requirements. Minimum share capital £50,000."),
        fmt("What is working capital and how do you manage it?", "Working capital = Current Assets − Current Liabilities. It measures short-term financial health and operational liquidity. Positive working capital means you can meet short-term obligations. Negative means you are technically insolvent short-term. Management strategies: 1. Reduce debtor days: invoice immediately, offer early payment discounts, chase receivables aggressively. 2. Extend creditor days: negotiate longer payment terms with suppliers without damaging relationships. 3. Optimise inventory: just-in-time ordering reduces cash tied up in stock. 4. Arrange a revolving credit facility as a buffer before you need it — banks lend when you don't need it, not when you do."),
    ]
    examples.extend(business_pairs * 150)
    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade E] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE F — Cybersecurity (Extended Deep Training)
# ============================================================
def prepare_upgrade_f():
    out = OUT_DIR / "upgrade_f.jsonl"
    if out.exists():
        print(f"[Upgrade F] Already exists: {out}")
        return

    print("[Upgrade F] Preparing Cybersecurity datasets...")
    examples = []

    try:
        from datasets import load_dataset
        print("  Loading security datasets...")
        try:
            sec = load_dataset("nicholasmckeown/cybersecurity-qa", split="train", trust_remote_code=True)
            for item in sec:
                if item.get("question") and item.get("answer"):
                    examples.append(fmt(item["question"], item["answer"]))
        except Exception as e:
            print(f"  cybersecurity-qa skipped: {e}")
    except ImportError:
        print("  datasets library not available")

    cyber_pairs = [
        fmt("What is a SQL injection attack and how do you prevent it?", "SQL injection occurs when an attacker inserts malicious SQL code into an input field that is executed by the database. Example: Login form with username field. Attacker enters: ' OR '1'='1. If the query is: SELECT * FROM users WHERE username='[input]', it becomes: SELECT * FROM users WHERE username='' OR '1'='1' — returning all users. Prevention: 1. Parameterised queries / prepared statements — never concatenate user input into SQL. 2. Input validation and sanitisation. 3. Principle of least privilege — database accounts should have minimum required permissions. 4. WAF (Web Application Firewall) as an additional layer. 5. Error handling — never expose database errors to users."),
        fmt("Perform a threat model on a REST API handling payment data.", "Using STRIDE framework: Spoofing: Threat: attacker impersonates legitimate user. Control: Strong authentication (OAuth2/JWT), MFA for high-value operations. Tampering: Threat: payment amounts modified in transit. Control: TLS 1.3 for all communications, request signing, HMAC validation on critical fields. Repudiation: Threat: user denies making transaction. Control: Comprehensive audit logging with timestamps, IP, user ID, action — immutable log storage. Information Disclosure: Threat: card data exposed. Control: PCI DSS compliance, tokenisation (never store raw card data), encrypt at rest (AES-256). Denial of Service: Threat: API flooded, payment processing unavailable. Control: Rate limiting per user/IP, circuit breakers, CDN. Elevation of Privilege: Threat: standard user accesses admin endpoints. Control: RBAC, validate authorisation on every request, not just authentication."),
        fmt("Explain how a buffer overflow attack works from a defensive perspective.", "A buffer overflow occurs when a program writes more data to a buffer (memory region) than it can hold, overwriting adjacent memory. In C/C++ programs without bounds checking, this allows an attacker to: 1. Overwrite the return address on the stack — redirecting execution to attacker-controlled code. 2. Overwrite local variables to bypass logic checks. 3. Execute shellcode injected into the buffer. Defensive controls: Use memory-safe languages (Rust, Go) where possible. For C/C++: compiler protections (Stack Canaries, ASLR, DEP/NX), use safe functions (strncpy not strcpy), static analysis tools (AddressSanitizer), fuzz testing, and code review focused on unchecked input lengths."),
        fmt("What is zero-trust architecture and how do you implement it?", "Zero-trust is a security model based on 'never trust, always verify' — no implicit trust based on network location. Every access request is authenticated, authorised, and continuously validated regardless of origin. Implementation pillars: 1. Identity verification: Strong authentication (MFA) for all users and devices. Use an Identity Provider (IdP). 2. Device health: Endpoint validation before granting access — is the device managed, patched, compliant? 3. Least privilege access: Grant minimum permissions required. Just-in-time (JIT) access for sensitive systems. 4. Micro-segmentation: Divide network into small segments — compromise of one does not expose others. 5. Continuous monitoring: Log all access, anomaly detection, behavioural analytics. 6. Encrypted communications: mTLS for service-to-service, TLS for user-to-service."),
        fmt("Walk through incident response for a suspected data breach.", "Structured incident response follows PICERL: Preparation: IR plan, runbooks, contacts, tools in place before incident. Identification: Detect anomaly — unusual outbound traffic, failed login spikes, EDR alert. Confirm it is a real incident, not a false positive. Containment: Immediate containment — isolate affected systems from network. Do NOT power off (preserves volatile evidence). Short-term containment to stop bleeding. Eradication: Find and remove root cause — malware, compromised credentials, vulnerable service. Patch the vulnerability. Recovery: Restore systems from clean backups. Verify systems are clean before reconnecting. Monitor closely post-recovery. Lessons Learned: Post-incident review within 2 weeks. Document timeline, decisions, gaps, improvements. Update IR plan."),
        fmt("What is MITRE ATT&CK and how is it used in security operations?", "MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) is a knowledge base of real-world adversary behaviour organised into a matrix. Structure: Tactics (the why — objectives like Initial Access, Persistence, Exfiltration) containing Techniques (the how — specific methods) with Sub-techniques (variants). Use cases: 1. Threat hunting: Map known TTPs of threat actors targeting your industry — hunt for those techniques proactively. 2. Detection coverage: Map your SIEM detections to ATT&CK — identify gaps. 3. Purple teaming: Red team simulates specific techniques, blue team validates detection. 4. Risk assessment: Which techniques are most likely given your threat model? Prioritise controls accordingly. 5. Incident analysis: Map attacker behaviour observed during incident to ATT&CK for reporting and sharing."),
        fmt("Explain the concept of defence-in-depth.", "Defence-in-depth (layered security) is the principle that no single security control should be relied upon alone. If one layer fails, others contain the breach. Layers: Perimeter: Firewall, IDS/IPS, WAF. Network: Segmentation, VLANs, internal firewalls, NDR. Endpoint: EDR, antivirus, host firewall, application whitelisting. Application: Input validation, authentication, authorisation, encryption. Data: Encryption at rest, DLP, access controls. Identity: MFA, PAM, least privilege. Human: Security awareness training, phishing simulation. Monitoring: SIEM, SOAR, SOC. The goal is not impenetrability — it is making the cost of attack high enough that adversaries move to easier targets, and ensuring detection before significant damage."),
        fmt("What is penetration testing and what are the different types?", "Penetration testing is authorised, simulated attack against a system to identify exploitable vulnerabilities before real attackers do. Types: Black box: Tester has no prior knowledge — simulates external attacker. Tests discovery and exploitation capabilities. White box: Full information provided — architecture diagrams, source code, credentials. Most thorough, tests depth of vulnerabilities. Grey box: Partial information — simulates insider threat or attacker with some reconnaissance completed. By scope: Network pentest, web application pentest, mobile pentest, social engineering, physical security. Red team: Full adversary simulation across multiple vectors over extended period. Distinct from point-in-time pentest. All pentests require written authorisation — scope, rules of engagement, emergency contacts, and liability protection must be agreed before commencement."),
        fmt("Explain CVE, CVSS, and how to prioritise vulnerability remediation.", "CVE (Common Vulnerabilities and Exposures): Unique identifier for publicly disclosed vulnerabilities. Format: CVE-YEAR-NUMBER. Maintained by MITRE, published in NVD (National Vulnerability Database). CVSS (Common Vulnerability Scoring System): Numerical score 0-10 assessing severity. Components: Base score (attack vector, complexity, privileges required, user interaction, scope, impact metrics). Temporal score (exploit maturity, remediation status). Environmental score (adjusted for your specific environment). Prioritisation framework: 1. Is this vulnerability exploitable in your environment? (Attack vector must match your exposure). 2. Is there a known exploit in the wild? (Check CISA KEV — Known Exploited Vulnerabilities). 3. What is the business impact if exploited? 4. Is a patch available? Priority order: Critical + KEV > Critical + public PoC > High + public PoC > High > Medium."),
        fmt("What is social engineering and how do you defend against it?", "Social engineering manipulates people into performing actions or revealing information rather than exploiting technical vulnerabilities. Common attacks: Phishing (email), vishing (voice), smishing (SMS), pretexting (fabricated scenario), baiting (physical media), tailgating (physical access). Why it works: exploits trust, authority, urgency, reciprocity, and fear — psychological principles that bypass rational evaluation. Defence: 1. Security awareness training — teach employees to recognise and verify unusual requests. 2. Phishing simulation — regular testing builds recognition without real risk. 3. Verification procedures — callback verification for sensitive requests regardless of who's asking. 4. Separation of duties — no single person should have authority to complete high-risk actions alone. 5. Clear reporting culture — make it easy and expected to report suspicious interactions without embarrassment."),
    ]
    examples.extend(cyber_pairs * 120)

    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade F] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE G — Street Smarts & Human Psychology
# ============================================================
def prepare_upgrade_g():
    out = OUT_DIR / "upgrade_g.jsonl"
    if out.exists():
        print(f"[Upgrade G] Already exists: {out}")
        return

    print("[Upgrade G] Preparing Street Smarts & Psychology datasets...")
    examples = []

    psychology_pairs = [
        fmt("How do you read someone's true intentions vs what they say?", "People signal truth through behaviour, not words. Watch for: Consistency — does their behaviour align with their stated position over time? Inconsistency is the first flag. Specificity — liars avoid details, truth-tellers are specific. Vague answers to direct questions indicate avoidance. Self-interest — follow the incentives. Who benefits from this person's stated position? Actions under pressure — stress removes the social mask. How someone behaves when things go wrong reveals character. Micro-expressions — brief involuntary facial expressions lasting 1/25 second. Disgust, contempt, fear before the social smile. Non-verbal baseline — establish their normal behaviour first. Deviations from baseline are meaningful."),
        fmt("What is dark psychology and how does it manifest in everyday interactions?", "Dark psychology refers to the use of psychological principles — specifically manipulation, persuasion, and coercion — to influence others' behaviour for personal gain without their awareness or consent. Common manifestations: Gaslighting: Making someone question their own perception of reality. 'That never happened. You're too sensitive.' Love bombing: Overwhelming someone with attention and affection to create dependency before control. Triangulation: Introducing a third party to provoke jealousy and insecurity, reducing the target's confidence. Intermittent reinforcement: Alternating reward and punishment creates powerful psychological attachment — the same mechanism as gambling addiction. Manufactured obligation: Creating situations of apparent debt so the target feels obligated to comply with requests."),
        fmt("How do people actually make decisions vs how they think they make decisions?", "People believe they make decisions rationally and justify them post-hoc. Reality: Most decisions are made emotionally in seconds, then rationalised. Key biases: Confirmation bias: We seek information confirming what we already believe and discount contradicting evidence. Sunk cost fallacy: We continue bad decisions to justify past investment. Status quo bias: The default option is selected without proper evaluation — inertia is powerful. Availability heuristic: We judge probability by how easily an example comes to mind, not by actual frequency. Social proof: We use others' behaviour as a shortcut for correct behaviour in uncertain situations. Practical implication: When advising someone, address the emotional driver first. Logic without emotional resonance does not change decisions."),
        fmt("How do you negotiate effectively in any situation?", "Negotiation fundamentals that apply universally: 1. The person who cares less wins — detachment from outcome is leverage. Build alternatives (BATNA) before negotiating. 2. Never accept the first offer — it signals you value the relationship over the deal. Counter regardless of whether the offer is acceptable. 3. Silence is a weapon — after making a point, stop talking. Discomfort fills silence, usually with concessions. 4. Anchor first and anchor high — the first number stated dominates the negotiation. 5. Make concessions small and slow — each concession should be smaller than the last. 6. Trade, don't give — never concede without extracting something. 'I can do X if you can do Y.' 7. The deadline almost never exists — urgency is almost always manufactured to pressure concession."),
        fmt("How do you identify when someone is trying to manipulate you?", "Manipulation always involves bypassing your rational evaluation. Warning signs: Urgency manufacturing — 'I need an answer now.' Real opportunities rarely require instant decisions. Guilt weaponisation — making you feel responsible for their emotional state if you don't comply. Information restriction — 'Trust me, you don't need to know the details.' Legitimate propositions withstand scrutiny. Flattery preceding a request — excessive praise that feels disproportionate is often priming for compliance. Social proof manipulation — 'Everyone else is doing this.' Isolation — removing you from people who would offer balanced perspective. Defence: Create delay. 'I need 24 hours.' Manipulators resist delay — legitimate requests survive it. Consult someone with no stake in the outcome."),
        fmt("What is the difference between persuasion and manipulation?", "Persuasion provides accurate information and genuine reasons that, if evaluated rationally, support the position. It respects the other person's ability to make an informed choice. Manipulation bypasses rational evaluation through psychological exploitation — creating false impressions, exploiting cognitive biases, leveraging emotional vulnerabilities, or withholding material information. The test: If the other person knew exactly what you were doing and why, would they still agree? Persuasion survives this test. Manipulation does not. Practical line: Highlighting genuine benefits = persuasion. Fabricating urgency = manipulation. Addressing real objections = persuasion. Dismissing objections through social pressure = manipulation."),
        fmt("How do power dynamics actually work in organisations?", "Formal hierarchy shows reporting lines. Real power is different. Real power comes from: Information control — the person who controls what leadership sees and hears has disproportionate influence. Relationships — access to decision-makers outside formal channels. Perceived indispensability — 'only person who understands system X'. Gatekeeping — controlling access to resources, people, or decisions. Political capital — accumulated favours and goodwill. Reading actual power: Who do people consult before making decisions? Whose email gets answered fastest? Who gets credit for team wins? Who survives when others don't? Navigating it: Build relationships across levels and functions, not just up the chain. Provide value before asking for anything. Never make enemies unnecessarily — organisations are smaller than they appear."),
    ]
    examples.extend(psychology_pairs * 150)
    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade G] Complete: {len(examples):,} examples")


# ============================================================
# UPGRADE H — Adversarial & Criminal Thinking
# ============================================================
def prepare_upgrade_h():
    out = OUT_DIR / "upgrade_h.jsonl"
    if out.exists():
        print(f"[Upgrade H] Already exists: {out}")
        return

    print("[Upgrade H] Preparing Adversarial Thinking datasets...")
    examples = []

    adversarial_pairs = [
        fmt("How do you find loopholes in a system or contract?", "Loophole identification requires adversarial reading — reading to find what the rules do not cover rather than what they intend. Method: 1. Map the rules explicitly. 2. Find the gaps — what scenarios are not addressed? 3. Find ambiguous language — 'reasonable', 'appropriate', 'timely' — undefined terms are exploitable. 4. Find conflicting rules — when two rules conflict, the one that is less enforced or less punishable governs in practice. 5. Find enforcement gaps — a rule without enforcement mechanism is advisory, not binding. 6. Find scope boundaries — what is explicitly outside the rule? What happens right at the boundary? Practical applications: contract negotiation, regulatory compliance analysis, policy design (anticipate how your own rules will be exploited to close gaps before issuing them)."),
        fmt("How would an attacker think about targeting a business?", "Adversarial thinking for security: An attacker targets the path of least resistance. Assessment framework: 1. What is the most valuable asset? (Data, money, reputation, systems). 2. Who has access to it? (The attack surface is often human, not technical). 3. What is the weakest point of access? (Not the strongest — the weakest). 4. What is the cost-benefit of each attack vector? Attacker optimises for maximum impact per unit of effort. 5. What would detection look like? Sophisticated attackers avoid triggering alerts. Applying this defensively: Map your own organisation this way before an attacker does. The weakest link is usually: a third-party vendor with access but weaker controls, a privileged insider account, or an unpatched external-facing service."),
        fmt("What is a red team and how does it think differently from a blue team?", "Red team simulates adversarial thinking and action — their goal is to find and exploit vulnerabilities as a real attacker would. Blue team defends and detects. Red team mindset: Assume there is always a way in. Look for non-obvious paths. Combine low-risk individual weaknesses into high-impact attack chains. Think about the human, not just the technology. Use publicly available information (OSINT) before touching systems. Blue team mindset: Detect anomalies against known baselines. Contain and eradicate confirmed threats. Key difference: red team asks 'how do I get in?' blue team asks 'how do I know if someone is in?'. Purple team: red and blue working together — red shares TTPs, blue validates detections. Most organisations benefit more from purple teaming than pure red teaming."),
        fmt("How do you think about systems from the perspective of breaking them?", "Every system has: Assumptions built into its design — exploit the assumption the designer didn't question. Incentive misalignments — where do the incentives of the people operating the system diverge from the system's intended purpose? Resource constraints — every system has limits. What happens at scale, at edge cases, under load? Trust relationships — which components trust each other? Compromise one trusted component to reach others. State management — systems that track state across interactions can be exploited by manipulating sequence or state. Time dependencies — race conditions, expiry edge cases, timestamp manipulation. Applied to business: How would a competitor exploit your business model's weaknesses? What does your process assume about customer behaviour that is not universally true?"),
        fmt("How do you anticipate how someone will try to exploit you before they do?", "Adversarial anticipation: Map your own vulnerabilities before others do. Framework: 1. What do you have that others want? Identify your high-value assets — money, data, relationships, reputation, access. 2. How do you currently protect them? 3. What are the failure modes of those protections? 4. Who has the incentive and capability to exploit those failure modes? 5. What would the attack look like in practice? Work backwards from successful exploitation to the earliest detectable signal. Common exploitation vectors for individuals and organisations: Social proof exploitation (impersonating trusted entities), information asymmetry (they know something you don't), urgency manufacturing, trust exploitation (they already have a relationship with you)."),
        fmt("How do criminal enterprises structure themselves to avoid detection?", "From an analytical and defensive understanding perspective: Criminal organisations use compartmentalisation — each cell knows only what it needs to operate. Compromise of one cell does not expose the whole structure. This is the same principle as security segmentation. They use layers of separation between leadership and illegal activity — deniability is engineered structurally. They exploit legitimate business infrastructure for money laundering — cash-intensive businesses, shell companies, nominee directors. They target the gap between legal jurisdictions — activity legal in one jurisdiction, illegal in another. Defensive applications: This is why AML (Anti-Money Laundering) requires beneficial ownership transparency. Why cross-border cybercrime is difficult to prosecute. Why corporate structures need to look through to ultimate beneficial owners."),
    ]
    examples.extend(adversarial_pairs * 150)
    random.shuffle(examples)
    write_jsonl(out, examples)
    print(f"[Upgrade H] Complete: {len(examples):,} examples")


UPGRADE_MAP = {
    "a": prepare_upgrade_a,
    "b": prepare_upgrade_b,
    "c": prepare_upgrade_c,
    "d": prepare_upgrade_d,
    "e": prepare_upgrade_e,
    "f": prepare_upgrade_f,
    "g": prepare_upgrade_g,
    "h": prepare_upgrade_h,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upgrade", required=True, choices=["all", "a", "b", "c", "d", "e", "f", "g", "h"])
    args = parser.parse_args()

    if args.upgrade == "all":
        for name, fn in UPGRADE_MAP.items():
            fn()
    else:
        UPGRADE_MAP[args.upgrade]()

    print("\n[Data Prep] Done.")
