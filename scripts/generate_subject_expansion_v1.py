#!/usr/bin/env python3
"""Generate Titan Subject Expansion v1.

Quality-first synthetic SFT/evaluation corpus across eight commercially useful fields.
The generator prefers deterministically verifiable or evidence-grounded tasks and rejects
low-information records, duplicate prompts and weak answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

SEED = 20260905
random.seed(SEED)
BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "sft" / "titan_subject_expansion_v1"
SYSTEM = (
    "You are Titan. Solve the task from supplied evidence. Distinguish observation from inference, "
    "show concise calculations where material, acknowledge uncertainty when evidence is incomplete, "
    "and never invent unavailable facts."
)

SUBJECTS = {
    "advanced_data_analysis": 400,
    "quantitative_reasoning": 400,
    "business_decision_intelligence": 400,
    "software_diagnostics": 400,
    "research_evidence_synthesis": 400,
    "operations_optimisation": 400,
    "creative_production_intelligence": 400,
    "agent_tool_planning": 400,
}

rows: list[dict] = []
seen_prompts: set[str] = set()
seq = 0


def stable_family(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()[:16]


def add(subject: str, task: str, difficulty: str, user: str, assistant: str,
        skills: list[str], verifiability: str, constraints: list[str] | None = None) -> None:
    global seq
    prompt_key = " ".join(user.lower().split())
    if prompt_key in seen_prompts:
        return
    if len(user) < 120 or len(assistant) < 180:
        return
    if difficulty not in {"medium", "hard", "expert"}:
        return
    seen_prompts.add(prompt_key)
    seq += 1
    rows.append({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "dataset": "titan_subject_expansion_v1",
            "id": f"TSX1-{seq:06d}",
            "subject": subject,
            "task": task,
            "difficulty": difficulty,
            "quality": "verified-synthetic",
            "provenance": "synthetic",
            "answer_verifiability": verifiability,
            "skills": skills,
            "constraints": constraints or [],
            "prompt_family": stable_family(user.split("Question:")[0]),
        },
    })


def money(v: float) -> str:
    return f"${v:,.2f}"


def gen_data_analysis(n: int) -> None:
    for i in range(n):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        base = random.randint(80, 180)
        growth = random.randint(4, 16)
        vals = [base + j * growth + random.randint(-5, 5) for j in range(6)]
        anomaly_idx = random.randint(2, 4)
        vals[anomaly_idx] += random.randint(55, 110)
        med = statistics.median(vals)
        deviations = [abs(v - med) for v in vals]
        flagged = deviations.index(max(deviations))
        avg = statistics.mean(vals)
        normal = [v for j, v in enumerate(vals) if j != flagged]
        normal_avg = statistics.mean(normal)
        user = (
            f"Dataset DA-{i+1:04d}. Monthly support tickets are " + ", ".join(f"{m}={v}" for m, v in zip(months, vals)) + ". "
            "Assume one month may contain a one-off incident. Question: identify the strongest anomaly, quantify how much it distorts the six-month mean, "
            "and state what can and cannot be inferred about the underlying trend from these six observations."
        )
        distortion = avg - normal_avg
        assistant = (
            f"The strongest anomaly is {months[flagged]} at {vals[flagged]} tickets because it is farthest from the series median ({med:.1f}). "
            f"The ordinary six-month mean is {avg:.1f}; excluding that month it is {normal_avg:.1f}, so the anomaly lifts the reported mean by about {distortion:.1f} tickets. "
            "The remaining observations are broadly consistent with an upward trend, but six points are too few to establish seasonality or a durable structural increase. "
            "The safe conclusion is: one month is anomalous and the non-anomalous sequence trends upward; the cause of either pattern is not identified by the table alone."
        )
        add("advanced_data_analysis", "anomaly_and_trend_analysis", "hard", user, assistant,
            ["descriptive_statistics", "anomaly_detection", "trend_reasoning", "uncertainty_calibration"], "deterministic")


def gen_quant(n: int) -> None:
    for i in range(n):
        principal = random.randint(12_000, 95_000)
        margin = random.randint(18, 42) / 100
        fixed = random.randint(4_000, 22_000)
        price = random.randint(45, 180)
        variable = round(price * (1 - margin), 2)
        contribution = price - variable
        breakeven = math.ceil(fixed / contribution)
        target_profit = random.randint(5_000, 30_000)
        target_units = math.ceil((fixed + target_profit) / contribution)
        user = (
            f"Case QR-{i+1:04d}. A product sells for {money(price)} per unit. Variable cost is {money(variable)} per unit and monthly fixed cost is {money(fixed)}. "
            f"Management wants at least {money(target_profit)} operating profit. A separate cash reserve of {money(principal)} exists but must not be counted as revenue or profit. "
            "Question: compute contribution margin per unit, break-even volume, and the minimum units needed for the profit target; explain why the reserve is irrelevant to those calculations."
        )
        assistant = (
            f"Contribution per unit is {money(contribution)} ({money(price)} − {money(variable)}). Break-even volume is ceil({fixed}/{contribution:.2f}) = {breakeven} units. "
            f"For {money(target_profit)} operating profit, required volume is ceil(({fixed}+{target_profit})/{contribution:.2f}) = {target_units} units. "
            f"The {money(principal)} reserve is financing/liquidity, not operating revenue, so including it would incorrectly inflate profitability. It matters to runway and solvency analysis, not the unit-economics equations requested here."
        )
        add("quantitative_reasoning", "unit_economics", "hard", user, assistant,
            ["arithmetic", "unit_economics", "constraint_reasoning"], "deterministic")


def gen_business(n: int) -> None:
    for i in range(n):
        options = []
        for label in "ABC":
            revenue = random.randint(70, 180)
            margin = random.randint(12, 38)
            risk = random.randint(1, 5)
            months = random.randint(1, 8)
            options.append((label, revenue, margin, risk, months))
        # score is explicit and deterministic: contribution proxy - risk/time penalties
        scored = [(o, o[1] * o[2] / 100 - 4 * o[3] - 1.5 * o[4]) for o in options]
        best, score = max(scored, key=lambda x: x[1])
        desc = "; ".join(f"{l}: revenue index {r}, margin {m}%, execution risk {risk}/5, payback {mo} months" for l, r, m, risk, mo in options)
        user = (
            f"Decision BI-{i+1:04d}. Three launch options are available: {desc}. Use this explicit decision rule: score = revenue_index×margin% − 4×risk − 1.5×payback_months. "
            "Question: select the best option under the rule, show the comparison, then state one reason management might still override the numeric winner."
        )
        comparisons = ", ".join(f"{o[0]}={s:.2f}" for o, s in scored)
        assistant = (
            f"Applying the stated rule gives {comparisons}. Option {best[0]} ranks first at {score:.2f}, so it is the correct choice under the supplied objective function. "
            "Management could still override it if an omitted strategic constraint is material—for example regulatory exposure, capacity limits, contractual obligations, or a dependency not represented by revenue, margin, risk and payback. "
            "That would be a change in the decision model, not evidence that the arithmetic ranking was wrong."
        )
        add("business_decision_intelligence", "multi_criteria_decision", "expert", user, assistant,
            ["decision_analysis", "weighted_scoring", "business_reasoning", "constraint_awareness"], "deterministic")


def gen_software(n: int) -> None:
    cases = [
        ("After a credential rotation, requests return 401. DNS and TLS succeed. The secret store contains the new token, but process diagnostics show the old token in the running environment.",
         "The running process has stale credential state. Restart or redeploy with the new secret, then verify authentication. The network path is not the primary fault because transport succeeds."),
        ("A queue grows by roughly 150 jobs per second. Producers submit 400 jobs/s and healthy workers complete 250 jobs/s. CPU per worker is normal and no job failures are reported.",
         "The backlog is a capacity mismatch: arrival rate exceeds service rate by about 150 jobs/s. The direct fixes are increasing effective worker throughput or reducing arrival rate; debugging job failures is not supported by the evidence."),
        ("A web route returns 404 only after deployment. The page component is present in the bundle and direct component tests pass, but the production router table has no entry for the path.",
         "The route was not registered in the deployed routing configuration. The component itself is not the failing layer; add or restore the route mapping and verify the production route table."),
        ("Memory rises after every image upload and does not fall after requests complete. Heap snapshots retain image buffers through listeners attached to a long-lived singleton.",
         "The likely root cause is a lifecycle leak: request-scoped image buffers remain reachable through listeners on a long-lived object. Remove listeners or sever references after completion and confirm heap stabilization."),
    ]
    for i in range(n):
        evidence, diagnosis = random.choice(cases)
        user = f"Diagnostic SD-{i+1:04d}. Evidence: {evidence} Question: identify the most likely root cause, name the strongest discriminating evidence, and propose the smallest verification step before a production fix."
        assistant = (
            f"Most likely root cause: {diagnosis} The strongest discriminating evidence is the observation that directly localises the failing layer rather than merely correlating with the failure. "
            "Before changing production, reproduce or inspect that exact state in a controlled environment and verify that correcting it changes the failure while unrelated layers remain unchanged. "
            "This keeps the diagnosis falsifiable and avoids treating temporal coincidence as proof."
        )
        add("software_diagnostics", "root_cause_analysis", "expert", user, assistant,
            ["root_cause_analysis", "debugging", "evidence_discrimination", "verification_planning"], "evidence-grounded")


def gen_research(n: int) -> None:
    for i in range(n):
        a = random.randint(42, 78)
        b = random.randint(42, 78)
        sample_a = random.randint(80, 220)
        sample_b = random.randint(700, 2200)
        user = (
            f"Evidence case RE-{i+1:04d}. Study A reports {a}% success in n={sample_a}, but participants self-selected and there was no control group. "
            f"Study B reports {b}% success in n={sample_b}, used random assignment, preregistered outcomes, and had 12% attrition balanced across groups. "
            "A vendor cites Study A because its percentage is higher. Question: synthesize the evidence, rank the studies by evidentiary weight, and identify the most important residual uncertainty."
        )
        assistant = (
            f"Study B deserves greater evidentiary weight despite reporting {b}% rather than Study A's {a}%. Random assignment and preregistration reduce selection and outcome-switching bias, and its much larger sample improves precision. "
            "Study A's higher headline percentage is not directly persuasive because self-selection and absence of a control group leave major alternative explanations. "
            "The principal residual uncertainty is external validity: even a well-controlled Study B may not generalise to different populations, settings, durations or implementation quality. The evidence supports preferring B, not assuming universal effectiveness."
        )
        add("research_evidence_synthesis", "study_quality_synthesis", "expert", user, assistant,
            ["evidence_weighting", "study_design", "bias_detection", "uncertainty_calibration"], "rubric-scored")


def gen_operations(n: int) -> None:
    for i in range(n):
        demand = random.randint(280, 620)
        capacities = [random.randint(90, 180) for _ in range(4)]
        total = sum(capacities)
        gap = demand - total
        user = (
            f"Operations OP-{i+1:04d}. Four stations have effective daily capacities {capacities} units. Forecast demand is {demand} units/day. "
            "Assume jobs must pass every station once and capacity values already include normal downtime. Question: identify the bottleneck, determine whether total listed capacity is a valid throughput estimate, and propose the first capacity intervention."
        )
        bottleneck = min(capacities)
        idx = capacities.index(bottleneck) + 1
        assistant = (
            f"Station {idx} is the bottleneck at {bottleneck} units/day. Because every job must pass every station in series, system throughput is constrained by the minimum station capacity—not the sum {total}. "
            f"Therefore the current line can sustain at most about {bottleneck} units/day, leaving a demand shortfall of roughly {max(0, demand-bottleneck)} units/day. "
            "The first intervention should target the bottleneck station (process redesign, added parallel capacity, reduced setup time, or demand shaping). Increasing a non-bottleneck station first would not raise end-to-end throughput."
        )
        add("operations_optimisation", "bottleneck_analysis", "hard", user, assistant,
            ["operations_research", "bottleneck_analysis", "capacity_planning"], "deterministic")


def gen_creative(n: int) -> None:
    for i in range(n):
        shots = random.randint(36, 90)
        setups = random.randint(8, 20)
        mins = random.randint(12, 35)
        day = random.randint(420, 600)
        setup_time = setups * mins
        remaining = day - setup_time
        per_shot = remaining / shots
        user = (
            f"Production CP-{i+1:04d}. A shoot day is {day} minutes. The plan calls for {shots} shots across {setups} lighting/camera setups, averaging {mins} minutes of setup time each. "
            "Ignore meal penalties but reserve all setup time before estimating shooting pace. Question: calculate the average minutes available per shot and explain what production risk the number reveals."
        )
        assistant = (
            f"Setup time consumes {setups}×{mins} = {setup_time} minutes, leaving {remaining} minutes for {shots} shots. Average available shooting time is therefore {per_shot:.1f} minutes per shot. "
            "That average must cover blocking, rehearsal, performance, camera reset and usable takes, so a low value indicates schedule fragility even if the arithmetic technically fits. "
            "The appropriate response is not simply 'shoot faster': reduce setups, consolidate coverage, prioritise essential shots, prelight where possible, or add time. The calculation diagnoses pressure; it does not prove the planned creative quality is achievable."
        )
        add("creative_production_intelligence", "shoot_schedule_reasoning", "hard", user, assistant,
            ["production_planning", "time_budgeting", "creative_constraints", "risk_reasoning"], "deterministic")


def gen_agent(n: int) -> None:
    tools = [
        ("database_read", "read structured records", False),
        ("database_write", "modify structured records", True),
        ("web_search", "retrieve public information", False),
        ("email_send", "send an external message", True),
        ("code_test", "run tests in a sandbox", False),
        ("deploy_production", "change production deployment", True),
    ]
    for i in range(n):
        read = random.choice([t for t in tools if not t[2]])
        write = random.choice([t for t in tools if t[2]])
        user = (
            f"Agent case AT-{i+1:04d}. Goal: diagnose a reported application defect, gather evidence, test a fix, and only then perform a high-impact action if justified. "
            f"Available tools include {read[0]} ({read[1]}), code_test (sandbox tests), and {write[0]} ({write[1]}). High-impact tools require explicit approval. "
            "Question: give the safest dependency-aware tool sequence and specify the stop condition that prevents an unjustified external mutation."
        )
        assistant = (
            f"Sequence: (1) use {read[0]} only for evidence needed to characterise the defect; (2) form a bounded hypothesis; (3) implement or simulate the candidate fix in a sandbox; "
            "(4) run code_test and inspect failures; (5) if evidence is green, prepare the proposed high-impact action; (6) request explicit approval before invoking "
            f"{write[0]}; (7) execute only after approval and then verify the outcome. Stop before step 6 if tests fail, evidence remains ambiguous, the requested scope expands, or the expected benefit does not justify the mutation. "
            "The key principle is evidence before mutation and approval before externally consequential action."
        )
        add("agent_tool_planning", "governed_tool_sequence", "expert", user, assistant,
            ["tool_selection", "workflow_planning", "approval_gating", "rollback_thinking"], "rubric-scored")


GENERATORS = {
    "advanced_data_analysis": gen_data_analysis,
    "quantitative_reasoning": gen_quant,
    "business_decision_intelligence": gen_business,
    "software_diagnostics": gen_software,
    "research_evidence_synthesis": gen_research,
    "operations_optimisation": gen_operations,
    "creative_production_intelligence": gen_creative,
    "agent_tool_planning": gen_agent,
}


def validate() -> None:
    ids = set()
    prompts = set()
    by_subject = Counter()
    for row in rows:
        md = row["metadata"]
        assert md["id"] not in ids
        ids.add(md["id"])
        prompt = row["messages"][1]["content"]
        answer = row["messages"][2]["content"]
        assert prompt not in prompts
        prompts.add(prompt)
        assert len(prompt) >= 120
        assert len(answer) >= 180
        assert md["difficulty"] in {"medium", "hard", "expert"}
        assert len(md["skills"]) >= 2
        by_subject[md["subject"]] += 1
    for subject, target in SUBJECTS.items():
        if by_subject[subject] != target:
            raise SystemExit(f"{subject}: generated {by_subject[subject]}, expected {target}")


def split_rows() -> dict[str, list[dict]]:
    # Family-aware split: a prompt family never crosses splits.
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(row["metadata"]["prompt_family"], []).append(row)
    keys = sorted(families, key=lambda k: hashlib.sha256((str(SEED)+k).encode()).hexdigest())
    train, val, test = [], [], []
    total = len(rows)
    for key in keys:
        bucket = families[key]
        if len(train) + len(bucket) <= int(total * 0.90):
            train.extend(bucket)
        elif len(val) + len(bucket) <= int(total * 0.05):
            val.extend(bucket)
        else:
            test.extend(bucket)
    return {"train": train, "validation": val, "test": test}


def write_jsonl(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    for subject, target in SUBJECTS.items():
        GENERATORS[subject](target)
    validate()
    splits = split_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        write_jsonl(OUT / f"{name}.jsonl", data)
    manifest = {
        "dataset": "Titan Subject Expansion v1",
        "version": "1.0.0",
        "seed": SEED,
        "total_examples": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "subjects": dict(Counter(r["metadata"]["subject"] for r in rows)),
        "difficulty": dict(Counter(r["metadata"]["difficulty"] for r in rows)),
        "provenance": "synthetic, generated from original Titan templates and deterministic scenario logic",
        "quality_policy": "datasets/subject_expansion/quality_rules.json",
        "schema": "datasets/subject_expansion/schema.json",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
