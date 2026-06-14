"""
TitanAI API Server
==================
FastAPI application providing OpenAI-compatible inference endpoints
for the TitanAI base model.

Endpoints:
    POST /v1/chat/completions   — OpenAI-compatible chat (Archibald drop-in)
    POST /v1/completions        — OpenAI-compatible text completion
    POST /v1/generate           — Native TitanAI generation
    GET  /v1/models             — List available checkpoints
    GET  /v1/models/{id}        — Get model info
    POST /v1/models/load        — Hot-load a checkpoint
    GET  /health                — Liveness check
    GET  /health/gpu            — GPU stats
    GET  /health/model          — Model status
    GET  /metrics               — Training run metrics

Usage:
    python -m api.main
    # or
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add titanai root to path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from api.core.config import config
from api.core.model_manager import manager
from api.routes.chat import router as chat_router
from api.routes.completions import router as completions_router
from api.routes.generate import router as generate_router
from api.routes.models import router as models_router
from api.routes.health import router as health_router
from api.routes.audio import router as audio_router
from api.routes.feedback import router as feedback_router
from api.core.rag_manager import rag

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("titan.api")


def _background_startup():
    """Load model + RAG index in a background thread so healthcheck passes immediately."""
    # Model load
    try:
        loaded = manager.load()
        if loaded:
            log.info(f"[bg] Model loaded ({manager.model_info.get('parameters', 0):,} params)")
        else:
            log.warning("[bg] Model NOT loaded — checkpoint not found. API returns 503 on inference.")
            log.warning(f"[bg] Expected: {config.CHECKPOINT_PATH}")
    except Exception as e:
        log.error(f"[bg] Model load error: {e}")

    # RAG index
    try:
        n_docs = rag.build_index()
        log.info(f"[bg] RAG index built — {n_docs:,} documents")
    except Exception as e:
        log.error(f"[bg] RAG index error: {e}")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  TitanAI API Server — Starting Up")
    log.info("=" * 60)
    log.info(f"  Config:     {config.CONFIG_PATH}")
    log.info(f"  Checkpoint: {config.CHECKPOINT_PATH}")
    log.info(f"  Device:     {config.DEVICE}")
    log.info(f"  Auth:       {'enabled' if config.REQUIRE_AUTH else 'disabled (dev mode)'}")
    log.info(f"  Port:       {config.PORT}")
    log.info("  Model/RAG:  Loading in background — healthcheck will pass immediately")
    log.info("=" * 60)

    # Fire model + RAG loading in background so uvicorn is ready for healthcheck instantly
    t = threading.Thread(target=_background_startup, daemon=True)
    t.start()

    yield

    log.info("TitanAI API Server — Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TitanAI API",
    description=(
        "OpenAI-compatible REST API for the TitanAI base language model. "
        "Built from scratch — no third-party model wrappers."
    ),
    version="0.1.6",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Archibald and any local dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "server_error"}},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(completions_router)
app.include_router(generate_router)
app.include_router(models_router)
app.include_router(audio_router)
app.include_router(feedback_router)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "TitanAI API",
        "version": "0.1.6",
        "status": "ready" if manager.is_loaded else "model_not_loaded",
        "docs": "/docs",
        "health": "/health",
        "model_loaded": manager.is_loaded,
        "device": manager.device,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL,
        reload=False,
    )
