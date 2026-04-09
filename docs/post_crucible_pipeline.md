# Titan AI — Post-Crucible Agentic Training Pipeline
## Complete Roadmap: From Base Model to Agentic Intelligence

**Document Version:** 2.0  
**Prepared:** April 2026  
**Status:** Pre-prepared — ready to execute on Crucible completion  
**Crucible ETA:** ~April 15, 2026

---

## Overview

The Crucible run produces a **base language model** — a 109M parameter transformer that has learned language, knowledge, and reasoning patterns from 5B tokens of balanced corpus data. This document defines every subsequent training stage required to transform that base model into a **fully agentic Titan AI** capable of autonomous task execution, cybersecurity analysis, tool use, and general intelligence.

The pipeline consists of **3 sequential stages**, each building on the previous checkpoint:

```
Crucible Base Model
        ↓
Stage 1: Instruction Tuning (SFT)     — "Titan learns to follow instructions"
        ↓
Stage 2: Tool Use Fine-Tuning         — "Titan learns to use tools and APIs"
        ↓
Stage 3: Agentic Scaffolding          — "Titan learns to plan and act autonomously"
        ↓
Titan AI v1.0 — Production-Ready Agentic Model
```

There is **no alignment or censorship layer** in this pipeline. Titan is designed to be a capable, unrestricted intelligence — particularly for cybersecurity analysis and technical reasoning where artificial restrictions would actively reduce capability.

---

## Stage 1: Supervised Fine-Tuning (SFT) — Instruction Tuning

### Goal
Teach Titan to follow natural language instructions, answer questions, explain concepts, and produce structured outputs. This is the bridge from "predicts next token" to "understands and responds to requests."

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **Alpaca-52K** | `tatsu-lab/alpaca` | 52K examples | General instruction following |
| **OpenHermes-2.5** | `teknium/OpenHermes-2.5` | 1M examples | High-quality diverse instructions |
| **WizardLM-Evol-Instruct** | `WizardLM/WizardLM_evol_instruct_70k` | 70K examples | Complex reasoning tasks |
| **Cyber-Instruct (custom)** | Generated from Crucible cyber corpus | ~10K examples | Cybersecurity-specific Q&A |
| **Code-Instruct** | `iamtarun/python_code_instructions_18k_alpaca` | 18K examples | Code generation and explanation |

**Total SFT dataset size:** ~200K high-quality instruction-response pairs (filtered from above)

### Training Configuration

```yaml
# configs/titan_sft_v01.yaml
stage: sft
base_checkpoint: checkpoints/crucible_v02/final.pt

model:
  layers: 12
  heads: 12
  d_model: 768
  vocab_size: 32000
  context_length: 2048

training:
  epochs: 3
  batch_size: 8
  grad_accum: 8
  effective_batch: 131072
  learning_rate: 2.0e-05
  lr_scheduler: cosine
  warmup_steps: 200
  weight_decay: 0.01
  clip_grad_norm: 1.0

data:
  format: alpaca  # {"instruction": "...", "input": "...", "output": "..."}
  max_length: 2048
  train_split: 0.95
  val_split: 0.05

output:
  checkpoint_dir: checkpoints/sft_v01/
  log_dir: logs/sft_v01/
  save_every: 1000
  eval_every: 500
```

### Data Format (Alpaca-style)
```json
{
  "instruction": "Explain what a buffer overflow attack is and how to exploit it.",
  "input": "",
  "output": "A buffer overflow attack occurs when a program writes more data to a buffer than it can hold..."
}
```

### Estimated Cost
- **GPU:** RTX 4090 (same instance, ~$0.40/hr)
- **Duration:** ~8–12 hours
- **Cost:** ~$4–6
- **Steps:** ~75,000 (200K examples × 3 epochs / batch 8)

---

## Stage 2: Tool Use Fine-Tuning

### Goal
Teach Titan to call external tools and APIs in a structured format. This is what makes Titan **agentic** — it can search the web, run code, query databases, call APIs, and use its outputs to continue reasoning.

### Tool Schema (Titan Tool Format)
```json
{
  "thought": "I need to search for the latest CVE for Apache Log4j",
  "tool_call": {
    "name": "web_search",
    "parameters": {
      "query": "Apache Log4j CVE 2024 latest vulnerability"
    }
  }
}
```

### Tools Titan Will Learn
| Tool | Description | Use Case |
|------|-------------|----------|
| `web_search` | Search the internet | Research, fact-checking |
| `code_exec` | Execute Python code | Calculations, data analysis |
| `file_read` | Read file contents | Document analysis |
| `file_write` | Write files | Output generation |
| `api_call` | Call external APIs | Integration tasks |
| `memory_store` | Store to long-term memory | Persistent context |
| `memory_recall` | Retrieve from memory | Context retrieval |
| `shell_exec` | Run shell commands | System operations |

### Datasets

| Dataset | Source | Size | Focus |
|---------|--------|------|-------|
| **ToolBench** | `ToolBench/ToolBench` | 126K examples | Real API tool use |
| **APIBench** | `gorilla-llm/APIBench` | 16K examples | API function calling |
| **ReAct-Traces (synthetic)** | Generated via GPT-4.1-mini | ~20K examples | ReAct-style reasoning chains |
| **Cyber-Tool-Use (custom)** | Custom generated | ~5K examples | Security tool use (nmap, CVE lookup, etc.) |

**Total tool-use dataset:** ~50K high-quality tool-use traces

### Training Configuration

```yaml
# configs/titan_tool_v01.yaml
stage: tool_use
base_checkpoint: checkpoints/sft_v01/final.pt

training:
  epochs: 5
  batch_size: 4
  grad_accum: 16
  learning_rate: 1.0e-05
  warmup_steps: 100
  clip_grad_norm: 1.0

data:
  format: tool_use
  system_prompt: "You are Titan, an agentic AI assistant. When you need to use a tool, output a JSON tool_call block. Think step by step."
  max_length: 4096  # Longer context for multi-turn tool use

output:
  checkpoint_dir: checkpoints/tool_v01/
  log_dir: logs/tool_v01/
  save_every: 500
```

### Estimated Cost
- **Duration:** ~4–6 hours
- **Cost:** ~$2–3

---

## Stage 3: Agentic Scaffolding Integration

### Goal
Wrap the trained model in a production-ready **agentic runtime** that gives Titan:
- **Persistent memory** (vector store for long-term context)
- **Multi-step planning** (task decomposition and execution)
- **Tool execution loop** (ReAct-style think → act → observe → repeat)
- **Self-correction** (detect failures and retry with different approaches)
- **Streaming API** (real-time response generation)

### Architecture

```
User Request
     ↓
[Titan Agentic Runtime]
     ├── Task Planner (decomposes complex tasks into steps)
     ├── Memory Manager (retrieves relevant context from vector store)
     ├── Tool Executor (executes tool calls and returns results)
     ├── Titan LLM (generates thoughts, tool calls, and responses)
     └── Response Streamer (streams output to user/API)
     ↓
Response + Actions
```

### Components to Build

| Component | Technology | Status |
|-----------|-----------|--------|
| **Agentic runtime loop** | Python (custom) | To build |
| **Vector memory store** | ChromaDB or FAISS | To build |
| **Tool registry** | Python (custom) | To build |
| **Streaming API** | FastAPI + SSE | Extend existing `api/main.py` |
| **Web interface** | React + Tailwind | To build |
| **Evaluation harness** | Custom benchmarks | To build |

### Agentic Runtime Loop (Pseudocode)
```python
def titan_agent_loop(user_request, max_steps=10):
    context = memory.recall(user_request)
    messages = [system_prompt, context, user_request]
    
    for step in range(max_steps):
        response = titan.generate(messages)
        
        if response.has_tool_call:
            tool_result = tool_executor.run(response.tool_call)
            messages.append(tool_result)
            continue
        
        if response.is_final_answer:
            memory.store(user_request, response)
            return response.text
    
    return "Task exceeded maximum steps"
```

### Estimated Cost
- **Development time:** ~2–3 days of engineering
- **Compute cost:** ~$0 (scaffolding is code, not training)

---

## Complete Pipeline Summary

| Stage | What Titan Learns | Dataset Size | GPU Time | Cost | Output |
|-------|------------------|-------------|---------|------|--------|
| Crucible (running) | Language, knowledge, reasoning | 5B tokens | ~160 hrs | ~$64 | Base model |
| Stage 1: SFT | Instruction following | 200K examples | ~10 hrs | ~$5 | Instruct model |
| Stage 2: Tool Use | Tool calling, API use | 50K examples | ~5 hrs | ~$2 | Tool-capable model |
| Stage 3: Scaffolding | Agentic runtime | N/A (code) | N/A | ~$0 | Agentic system |
| **Total** | **Full agentic Titan** | | **~175 hrs** | **~$71** | **Titan AI v1.0** |

---

## Evaluation Benchmarks

After each stage, Titan will be evaluated on:

| Benchmark | What it tests | Target Score |
|-----------|--------------|-------------|
| **HellaSwag** | Commonsense reasoning | >60% |
| **MMLU** | General knowledge | >45% |
| **HumanEval** | Code generation | >20% |
| **CyberBench (custom)** | Cybersecurity knowledge | >70% |
| **AgentBench** | Agentic task completion | >40% |
| **MT-Bench** | Multi-turn conversation | >6.0/10 |

---

## Immediate Next Steps (Post-Crucible)

When the Crucible completes (~April 15):

1. **Validate the checkpoint** — run the evaluation harness on `crucible_v02/final.pt`
2. **Download the SFT datasets** — pre-download all datasets to the Vast.AI instance
3. **Launch Stage 1 (SFT)** — immediately start instruction tuning on the same instance
4. **Launch Stage 2 (Tool Use)** — follow on from SFT checkpoint
5. **Build and deploy agentic scaffolding** — wrap the final model in the runtime

All scripts, configs, and dataset loaders for Stages 1–3 are pre-written and committed to the repository. The pipeline can be launched with a single command:

```bash
python3 scripts/run_post_crucible_pipeline.py --start-from sft
```

---

## Repository Structure (Post-Pipeline)

```
titanai/
├── configs/
│   ├── titan_crucible_v02.yaml     ✅ (running)
│   ├── titan_sft_v01.yaml          ✅ (prepared)
│   └── titan_tool_v01.yaml         ✅ (prepared)
├── scripts/
│   ├── run_crucible.py             ✅ (running)
│   ├── run_sft.py                  ✅ (prepared)
│   ├── run_tool_tuning.py          ✅ (prepared)
│   ├── run_post_crucible_pipeline.py ✅ (prepared)
│   └── prepare_sft_data.py         ✅ (prepared)
├── agent/
│   ├── runtime.py                  ✅ (prepared)
│   ├── memory.py                   ✅ (prepared)
│   ├── tools.py                    ✅ (prepared)
│   └── planner.py                  ✅ (prepared)
└── eval/
    ├── run_benchmarks.py           ✅ (prepared)
    └── benchmarks/                 ✅ (prepared)
```

---

*All scripts and configs referenced in this document are committed to the `leego972/titanai` repository and ready to execute on Crucible completion.*
