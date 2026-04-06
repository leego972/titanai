"""
TitanAI API — Request & Response Schemas
==========================================
OpenAI-compatible schemas where possible, with TitanAI extensions.
"""
import time
import uuid
from typing import Optional, List, Union, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

def new_id(prefix: str = "titan") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ── /v1/completions ───────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "titan-probe-v0.1.5"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    n: int = Field(1, ge=1, le=4)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("prompt cannot be empty")
        return v


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Literal["stop", "length"] = "stop"
    logprobs: None = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "titan-probe-v0.1.5"
    choices: List[CompletionChoice]
    usage: UsageInfo


# ── /v1/chat/completions ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("message content cannot be empty")
        return v


class ChatCompletionRequest(BaseModel):
    model: str = "titan-probe-v0.1.5"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    n: int = Field(1, ge=1, le=4)

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "titan-probe-v0.1.5"
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# ── Streaming chunks (SSE) ────────────────────────────────────────────────────

class StreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int
    delta: StreamDelta
    finish_reason: Optional[Literal["stop", "length"]] = None


class StreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "titan-probe-v0.1.5"
    choices: List[StreamChoice]


# ── /v1/generate (native TitanAI) ────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stream: bool = False


class GenerateResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("gen"))
    prompt: str
    generated_text: str
    model: str = "titan-probe-v0.1.5"
    prompt_tokens: int
    generated_tokens: int
    created: int = Field(default_factory=lambda: int(time.time()))


# ── /v1/models ────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "titanai"
    checkpoint_path: str
    parameters: Optional[int] = None
    training_step: Optional[int] = None
    val_perplexity: Optional[float] = None
    is_loaded: bool = False


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    model_loaded: bool
    device: str
    checkpoint: Optional[str] = None
    uptime_seconds: float
    version: str = "0.1.5"


class GPUHealthResponse(BaseModel):
    status: Literal["ok", "no_gpu", "error"]
    device_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None
    cuda_version: Optional[str] = None


# ── Error responses ───────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    message: str
    type: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
