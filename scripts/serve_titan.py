"""
TitanAI Model Inference Server
================================
OpenAI-compatible REST API for the trained TitanAI model.
Set TITAN_MODEL_ENDPOINT=http://localhost:8000/v1 in the API server's .env
to route all traffic through your own model instead of Venice/OpenRouter/OpenAI.

Usage:
  python serve_titan.py --checkpoint /path/to/checkpoints/upgrade_an/final.pt \
                        --tokenizer  /path/to/tokenizer/artifacts_v32k/tokenizer.json \
                        --port       8000

Requirements:
  pip install fastapi uvicorn torch tokenizers
  (optionally: pip install flash-attn --no-build-isolation  # for GPU speedup)
"""

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer

# ── locate the titanai library ────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from model.titan_model import TitanConfig, build_model  # noqa: E402


# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="TitanAI OpenAI-compatible server")
parser.add_argument("--checkpoint", required=True, help="Path to final.pt")
parser.add_argument("--tokenizer",  required=True, help="Path to tokenizer.json")
parser.add_argument("--port",       type=int, default=8000)
parser.add_argument("--host",       default="0.0.0.0")
parser.add_argument("--device",     default="auto",
                    help="cuda | cpu | auto  (auto picks CUDA if available)")
args = parser.parse_args()

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
) if args.device == "auto" else args.device

print(f"[TitanAI] device={DEVICE}")


# ── load tokenizer ────────────────────────────────────────────────────────────
print(f"[TitanAI] loading tokenizer from {args.tokenizer}")
tokenizer: Tokenizer = Tokenizer.from_file(str(args.tokenizer))


# ── load model ────────────────────────────────────────────────────────────────
print(f"[TitanAI] loading checkpoint from {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

raw_cfg = ckpt.get("config", {})
cfg = TitanConfig(
    vocab_size=raw_cfg.get("vocab_size", 32000),
    d_model=raw_cfg.get("d_model", 768),
    n_heads=raw_cfg.get("n_heads", 12),
    n_layers=raw_cfg.get("n_layers", 12),
    d_ff=raw_cfg.get("d_ff", 3072),
    max_seq_len=raw_cfg.get("max_seq_len", 2048),
    dropout=0.0,
    tie_embeddings=raw_cfg.get("tie_embeddings", True),
)

model = build_model(cfg)
state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
model.load_state_dict(state, strict=False)
model.eval()
model.to(DEVICE)
print(f"[TitanAI] model ready — {sum(p.numel() for p in model.parameters()):,} params")


# ── chat prompt formatter ─────────────────────────────────────────────────────
SYSTEM_TAG  = "<|system|>"
USER_TAG    = "<|user|>"
ASST_TAG    = "<|assistant|>"
END_TAG     = "<|end|>"

def format_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"{SYSTEM_TAG}\n{content}{END_TAG}\n")
        elif role == "user":
            parts.append(f"{USER_TAG}\n{content}{END_TAG}\n")
        elif role == "assistant":
            parts.append(f"{ASST_TAG}\n{content}{END_TAG}\n")
    parts.append(f"{ASST_TAG}\n")   # model completes from here
    return "".join(parts)


# ── generation ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
) -> str:
    enc = tokenizer.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=DEVICE)

    eos_id: Optional[int] = tokenizer.token_to_id(END_TAG)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        ctx = ids[:, -cfg.max_seq_len:]
        logits = model(ctx)          # (1, seq, vocab)
        next_logits = logits[0, -1]  # (vocab,)

        if temperature > 0:
            next_logits = next_logits / temperature

        # top-k
        if top_k > 0:
            top_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < top_vals[-1]] = float("-inf")

        # top-p (nucleus)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumprob - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            next_logits = torch.zeros_like(next_logits).scatter_(0, sorted_idx, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()

        if eos_id is not None and next_id == eos_id:
            break

        generated.append(int(next_id))
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    return tokenizer.decode(generated)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="TitanAI Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "titan-1b-cyber"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    stream: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "titan-1b-cyber", "object": "model", "owned_by": "titanai"},
            {"id": "titan-1b-film",  "object": "model", "owned_by": "titanai"},
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="Streaming not yet supported. Set stream=false.")

    prompt = format_prompt([m.model_dump() for m in req.messages])
    t0 = time.time()
    text = generate(
        prompt,
        max_new_tokens=req.max_tokens or 512,
        temperature=req.temperature or 0.7,
        top_k=req.top_k or 50,
        top_p=req.top_p or 0.95,
    )
    elapsed = time.time() - t0

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens  = len(tokenizer.encode(prompt).ids)
    output_tokens  = len(tokenizer.encode(text).ids)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        },
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "elapsed_s": round(elapsed, 2),
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
