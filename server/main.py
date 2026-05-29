import os
import time
import uuid
import threading
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_URL: str = os.getenv(
    "MODEL_URL",
    "https://github.com/leego972/titanai/releases/download/titan-1b-v1/final_q8_0.gguf",
)
MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "/app/models/final_q8_0.gguf"))
TITAN_API_KEY: str = os.getenv("TITAN_API_KEY", "")
N_THREADS: int = int(os.getenv("N_THREADS", "4"))
N_CTX: int = int(os.getenv("N_CTX", "4096"))
MODEL_VERSION: str = "1.0.0"

# ── Global model state ────────────────────────────────────────────────────────

_llm = None
_model_ready: bool = False
_model_error: Optional[str] = None


def _download_model() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        logger.info("Model already on disk: %s", MODEL_PATH)
        return
    logger.info("Downloading model from %s", MODEL_URL)
    with httpx.Client(follow_redirects=True, timeout=None) as client:
        with client.stream("GET", MODEL_URL) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            last_pct = -1
            with open(MODEL_PATH, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        if pct // 10 != last_pct // 10:
                            last_pct = pct
                            logger.info("Download: %d%%", pct)
    logger.info("Download complete: %s", MODEL_PATH)


def _load_model() -> None:
    global _llm, _model_ready, _model_error
    try:
        _download_model()
        from llama_cpp import Llama  # noqa: PLC0415  (deferred import — llama_cpp is large)
        logger.info("Loading GGUF into memory …")
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False,
        )
        _model_ready = True
        logger.info("Model ready")
    except Exception as exc:
        _model_error = str(exc)
        logger.exception("Model load failed: %s", exc)


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: RUF029  (must be async per FastAPI)
    thread = threading.Thread(target=_load_model, daemon=True, name="model-loader")
    thread.start()
    yield


app = FastAPI(
    title="TitanAI Inference API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class TitanChatRequest(BaseModel):
    messages: List[ChatMessage]
    persona: str = "virelle"
    temperature: float = 0.8
    maxTokens: int = 2048


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "titan-1b"
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> None:
    if not TITAN_API_KEY:
        return
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {TITAN_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_model() -> None:
    if _model_error:
        raise HTTPException(
            status_code=503,
            detail=f"Model failed to load: {_model_error}",
        )
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading — please retry in a moment",
        )


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant" if msg.role == "assistant" else "System"
        lines.append(f"{role}: {msg.content}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "model_ready": _model_ready,
        "model_error": _model_error,
    }


@app.get("/titan/status")
def titan_status() -> dict:
    if _model_error:
        status = "error"
    elif _model_ready:
        status = "ready"
    else:
        status = "loading"
    return {
        "status": status,
        "training": False,
        "modelEndpoint": _model_ready,
        "personas": ["virelle", "archibald"],
        "version": MODEL_VERSION,
    }


_PERSONAS: dict = {
    "virelle": {
        "id": "virelle",
        "name": "Virelle",
        "domain": "virellestudios.com",
        "greeting": "I'm Virelle, your AI co-director. What story shall we tell?",
    },
    "archibald": {
        "id": "archibald",
        "name": "Archibald Titan",
        "domain": "archibaldtitan.com",
        "greeting": "Archibald Titan at your service. How can I assist?",
    },
}


@app.get("/titan/persona")
def titan_persona(id: str = "virelle") -> dict:
    persona = _PERSONAS.get(id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Unknown persona: {id!r}")
    return {**persona, "ready": _model_ready}


@app.post("/titan/chat")
def titan_chat(body: TitanChatRequest, request: Request) -> dict:
    _check_auth(request)
    _require_model()

    prompt = _messages_to_prompt(body.messages)
    result = _llm(
        prompt,
        max_tokens=body.maxTokens,
        temperature=body.temperature,
        stop=["User:", "\nUser:"],
        echo=False,
    )
    reply: str = result["choices"][0]["text"].strip()
    usage: dict = result.get("usage", {})

    return {
        "reply": reply,
        "persona": body.persona,
        "personaName": _PERSONAS.get(body.persona, {}).get("name", body.persona.capitalize()),
        "model": "titan-1b",
        "tokens": usage.get("total_tokens", 0),
    }


@app.post("/v1/chat/completions")
def openai_completions(body: OpenAIChatRequest, request: Request) -> dict:
    _check_auth(request)
    _require_model()

    # Convert OpenAI-style messages to ChatMessage for shared prompt builder
    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    prompt = _messages_to_prompt(chat_messages)

    result = _llm(
        prompt,
        max_tokens=body.max_tokens or 2048,
        temperature=body.temperature if body.temperature is not None else 0.8,
        stop=["User:", "\nUser:"],
        echo=False,
    )
    text: str = result["choices"][0]["text"].strip()
    usage: dict = result.get("usage", {})

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }
