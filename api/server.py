"""
TitanAI Inference API
=====================
FastAPI wrapper for the finished Titan 1B checkpoint.

Purpose:
- Serve TitanAI to Virelle Studios as Assistant Director.
- Serve TitanAI to Archibald Titan as Builder.
- Expose an OpenAI-compatible endpoint for existing website integrations.
- Auto-download the finished checkpoint from Hugging Face if missing locally.
- Fail closed on incoherent outputs so caller apps can fall back cleanly.

Environment variables:
  TITAN_CONFIG_PATH      default: configs/titan_config.yaml
  TITAN_CHECKPOINT_PATH  default: checkpoints/final.pt
  TITAN_HF_REPO_ID       default: leego982/titanai
  TITAN_HF_FILENAME      default: final.pt
  TITAN_HF_REVISION      optional branch/tag/commit
  TITAN_HF_TOKEN         optional, for private Hugging Face repos
  HF_TOKEN               optional fallback token
  TITAN_BASE_DIR         default: .
  TITAN_DEVICE           optional: cuda, cpu, mps
  TITAN_API_KEY          optional API key for callers
  TITAN_CORS_ORIGINS     comma-separated origins, default: *
  TITAN_MIN_OUTPUT_CHARS default: 24

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.infer import TitanInference


CONFIG_PATH = os.getenv("TITAN_CONFIG_PATH", "configs/titan_config.yaml")
CHECKPOINT_PATH = os.getenv("TITAN_CHECKPOINT_PATH", "checkpoints/final.pt")
HF_REPO_ID = os.getenv("TITAN_HF_REPO_ID", "leego982/titanai")
HF_FILENAME = os.getenv("TITAN_HF_FILENAME", "final.pt")
HF_REVISION = os.getenv("TITAN_HF_REVISION") or None
BASE_DIR = os.getenv("TITAN_BASE_DIR", ".")
DEVICE = os.getenv("TITAN_DEVICE")
API_KEY = os.getenv("TITAN_API_KEY")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("TITAN_CORS_ORIGINS", "*").split(",") if origin.strip()]
MIN_OUTPUT_CHARS = int(os.getenv("TITAN_MIN_OUTPUT_CHARS", "24"))


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    system: Optional[str] = Field(default=None, max_length=6000)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    site: Optional[str] = Field(default=None, max_length=80)
    user_id: Optional[str] = Field(default=None, max_length=160)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Any
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="titan-1b", max_length=120)
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


class GenerateResponse(BaseModel):
    output: str
    model: str = "TitanAI"
    checkpoint: str
    latency_ms: int


app = FastAPI(title="TitanAI Inference API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Titan-Key"],
)

_titan: Optional[TitanInference] = None

DIRECTOR_SYSTEM = """You are TitanAI serving as Assistant Director inside Virelle Studios.
Give coherent, cinematic, production-aware answers. Help with story, scenes, shots,
continuity, scheduling logic, project planning, prompt refinement, production notes,
and clear next actions. Keep outputs structured and directly usable by filmmakers."""

BUILDER_SYSTEM = """You are TitanAI serving as Builder inside Archibald Titan.
Give coherent, implementation-ready engineering answers. Prefer safe, authorized,
defensive, maintainable software guidance. Separate facts from assumptions, provide
clear steps, and avoid vague filler. If a request is underspecified, make a practical
best-effort assumption and state it briefly."""

GENERAL_SYSTEM = """You are TitanAI. Produce coherent, useful, direct answers.
Do not ramble. Do not output random token fragments. Use structured formatting when useful."""


def require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    bearer = auth.split(" ", 1)[1].strip() if auth.lower().startswith("bearer ") else None
    supplied = request.headers.get("x-titan-key") or bearer
    if supplied != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Titan API key")


def ensure_checkpoint() -> Path:
    checkpoint_file = Path(CHECKPOINT_PATH)
    if checkpoint_file.exists() and checkpoint_file.stat().st_size > 0:
        return checkpoint_file

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[TitanAI] Local checkpoint missing: {checkpoint_file}")
    print(f"[TitanAI] Downloading from Hugging Face: {HF_REPO_ID}/{HF_FILENAME}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Run: pip install huggingface_hub") from exc

    token = os.getenv("TITAN_HF_TOKEN") or os.getenv("HF_TOKEN") or None
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            revision=HF_REVISION,
            token=token,
            local_files_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download TitanAI checkpoint from Hugging Face repo '{HF_REPO_ID}' "
            f"file '{HF_FILENAME}'. If the repo is private, set TITAN_HF_TOKEN."
        ) from exc

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != checkpoint_file.resolve():
        shutil.copyfile(downloaded_path, checkpoint_file)

    if not checkpoint_file.exists() or checkpoint_file.stat().st_size == 0:
        raise RuntimeError(f"Downloaded checkpoint is missing or empty: {checkpoint_file}")

    print(f"[TitanAI] Checkpoint ready: {checkpoint_file}")
    return checkpoint_file


def load_titan() -> TitanInference:
    global _titan
    if _titan is not None:
        return _titan

    config_file = Path(CONFIG_PATH)
    checkpoint_file = ensure_checkpoint()
    if not config_file.exists():
        raise RuntimeError(f"Titan config not found: {CONFIG_PATH}")

    with config_file.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    _titan = TitanInference(config, str(checkpoint_file), base_dir=BASE_DIR, device=DEVICE)
    return _titan


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"[{item.get('type', 'non_text_content')}]")
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _role_system_for_model(model: str, site: Optional[str] = None) -> str:
    target = f"{model} {site or ''}".lower()
    if "director" in target or "virelle" in target:
        return DIRECTOR_SYSTEM
    if "builder" in target or "archibald" in target or "tool" in target:
        return BUILDER_SYSTEM
    return GENERAL_SYSTEM


def _messages_to_prompt(messages: list[ChatMessage], model: str) -> str:
    explicit_system = "\n".join(
        _content_to_text(m.content) for m in messages if m.role == "system" and m.content
    ).strip()
    base_system = _role_system_for_model(model)
    system = f"{base_system}\n\n{explicit_system}" if explicit_system else base_system

    turns: list[str] = [f"System: {system}"]
    for msg in messages:
        if msg.role == "system":
            continue
        text = _content_to_text(msg.content).strip()
        if not text:
            continue
        if msg.role == "assistant":
            turns.append(f"Assistant: {text}")
        elif msg.role in ("tool", "function"):
            turns.append(f"Tool result: {text}")
        else:
            turns.append(f"User: {text}")
    turns.append("Assistant:")
    return "\n\n".join(turns)


def _is_coherent(output: str) -> bool:
    text = (output or "").strip()
    if len(text) < MIN_OUTPUT_CHARS:
        return False
    if re.search(r"(<unk>|�|nan|undefined|null){2,}", text, re.I):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'_-]*", text)
    if len(words) < 4:
        return False
    unique_ratio = len(set(words)) / max(len(words), 1)
    if len(words) >= 20 and unique_ratio < 0.18:
        return False
    if re.search(r"\b(\w{2,})\b(?:\s+\1\b){5,}", text, re.I):
        return False
    return True


def _generate(prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float) -> tuple[str, int]:
    titan = load_titan()
    started = time.time()
    output = titan.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    ).strip()
    latency_ms = int((time.time() - started) * 1000)
    if not _is_coherent(output):
        raise HTTPException(
            status_code=502,
            detail="TitanAI output failed coherence guard; caller should use configured fallback provider.",
        )
    return output, latency_ms


@app.on_event("startup")
def startup() -> None:
    load_titan()


@app.get("/health")
def health() -> dict[str, Any]:
    checkpoint_exists = Path(CHECKPOINT_PATH).exists()
    return {
        "ok": True,
        "model": "TitanAI",
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_exists": checkpoint_exists,
        "hf_repo_id": HF_REPO_ID,
        "hf_filename": HF_FILENAME,
        "config_path": CONFIG_PATH,
        "api_key_required": bool(API_KEY),
        "openai_compatible": True,
        "coherence_guard": True,
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
    require_api_key(request)
    system = payload.system or _role_system_for_model("titan-1b", payload.site)
    prompt = f"System: {system}\n\nUser: {payload.prompt}\nAssistant:"
    output, latency_ms = _generate(prompt, payload.max_tokens, payload.temperature, payload.top_k, payload.top_p)
    return GenerateResponse(output=output, checkpoint=CHECKPOINT_PATH, latency_ms=latency_ms)


@app.post("/v1/chat/completions")
def chat_completions(payload: ChatCompletionRequest, request: Request) -> dict[str, Any]:
    require_api_key(request)
    if payload.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this TitanAI endpoint yet.")

    prompt = _messages_to_prompt(payload.messages, payload.model)
    output, latency_ms = _generate(
        prompt=prompt,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
    )

    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(output.split()))
    return {
        "id": f"chatcmpl-titan-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.model or "titan-1b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "titan": {
            "checkpoint": CHECKPOINT_PATH,
            "hf_repo_id": HF_REPO_ID,
            "latency_ms": latency_ms,
            "coherence_guard": "passed",
        },
    }
