# Titan Subject Expansion v1

High-quality commercial dataset programme for supervised fine-tuning and evaluation.

## Subject families

1. Advanced Data Analysis
2. Quantitative Reasoning
3. Business Decision Intelligence
4. Software Diagnostics & Root Cause
5. Research & Evidence Synthesis
6. Operations & Process Optimisation
7. Creative Production Intelligence
8. Agent Planning & Tool Selection

## Quality standard

Each accepted record must:
- require non-trivial inference rather than simple recall;
- contain enough evidence to support a defensible answer;
- separate observation, inference, uncertainty and recommendation where relevant;
- avoid unsupported factual claims;
- avoid duplicate or near-duplicate prompts;
- include subject, task type, difficulty and quality metadata;
- be safe for commercial redistribution and model training;
- pass schema, content-length, duplication and split-isolation validation.

## Dataset format

Chat-style JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"dataset":"...","id":"...","subject":"...","task":"...","difficulty":"hard","quality":"curated"}}
```

## Split policy

Target split: 90% train, 5% validation, 5% test. Near-duplicate prompt families must remain within one split to reduce leakage.

## Commercial catalogue target

Initial product targets are 10K-30K accepted examples per subject family. Quantity is subordinate to quality: generation should stop or reject examples when quality criteria are not met.
