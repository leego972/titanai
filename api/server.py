"""
TitanAI Inference API
=====================
FastAPI wrapper for TitanInference so Virelle.life and ArchibaldTitan.com can call the
finished Titan 1B checkpoint over HTTP.

Environment variables:
  TITAN_CONFIG_PATH      default: configs/titan_config.yaml
  TITAN_CHECKPOINT_PATH  default: checkpoints/final.pt
  TITAN_BASE_DIR         default: .
  TITAN_DEVICE           optional: cuda, cpu, mps
  TITAN_API_KEY          required for production requests
  TITAN_CORS_ORIGINS     comma-separated origins, default: *

Run locally:
  uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.infer import TitanInference


CONFIG_PATH = os.getenv("TITAN_CONFIG_PATH", "configs/titan_config.yaml")
CHECKPOINT_PATH = os.getenv("TITAN_CHECKPOINT_PATH", "checkpoints/final.pt")
BASE_DIR = os.getenv("TITAN_BASE_DIR", ".")
DEVICE = os.getenv("TITAN_DEVICE")
API_KEY = os.getenv("TITAN_API_KEY")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("TITAN_CORS_ORIGINS", "*").split(",") if origin.strip()]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=12000)
    system: Optional[str] = Field(default=None, max_length=4000)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=500)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    site: Optional[str] = Field(default=None, max_length=80)
    user_id: Optional[str] = Field(default=None, max_length=160)


class GenerateResponse(BaseModel):
    output: str
    model: str = "TitanAI"
    checkpoint: str
    latency_ms: int


app = FastAPI(title="TitanAI Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Titan-Key"],
)

_titan: Optional[TitanInference] = None


def require_api_key(x_titan_key: Optional[str] = Header(default=None), authorization: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        # Development mode only. Set TITAN_API_KEY in production.
        return

    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization.split(" ", 1)[1].strip()

    supplied = x_titan_key or bearer
    if supplied != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Titan API key")


def load_titan() -> TitanInference:
    global _titan
    if _titan is not None:
        return _titan

    config_file = Path(CONFIG_PATH)
    checkpoint_file = Path(CHECKPOINT_PATH)

    if not config_file.exists():
        raise RuntimeError(f"Titan config not found: {CONFIG_PATH}")
    if not checkpoint_file.exists():
        raise RuntimeError(f"Titan checkpoint not found: {CHECKPOINT_PATH}")

    with config_file.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    _titan = TitanInference(config, str(checkpoint_file), base_dir=BASE_DIR, device=DEVICE)
    return _titan


@app.on_event("startup")
def startup() -> None:
    # Fail fast if checkpoint/config are missing.
    load_titan()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": "TitanAI",
        "checkpoint_path": CHECKPOINT_PATH,
        "config_path": CONFIG_PATH,
        "api_key_required": bool(API_KEY),
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest, request: Request, _: None = Header(default=None)) -> GenerateResponse:
    require_api_key(
        x_titan_key=request.headers.get("x-titan-key"),
        authorization=request.headers.get("authorization"),
    )

    titan = load_titan()
    prompt = payload.prompt
    if payload.system:
        prompt = f"System: {payload.system}\n\nUser: {payload.prompt}\nTitan:"

    started = time.time()
    output = titan.generate(
        prompt,
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
    )
    latency_ms = int((time.time() - started) * 1000)

    return GenerateResponse(
        output=output.strip(),
        checkpoint=CHECKPOINT_PATH,
        latency_ms=latency_ms,
    )
