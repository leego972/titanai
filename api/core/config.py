"""
TitanAI API — Configuration
============================
All settings are read from environment variables with sensible defaults.

PRE-FLIGHT SECURITY FIX:
    TITAN_REQUIRE_AUTH now defaults to "true" (was "false").
    If REQUIRE_AUTH is true and API_KEY is empty, the server logs a loud
    warning at startup. start_api.sh enforces the key length check before
    the Python process even starts.

    To run in dev mode (local only, no auth):
        export TITAN_REQUIRE_AUTH=false
    Never set this on Vast.AI or any public-facing instance.
"""
import os
import logging
from pathlib import Path

log = logging.getLogger("titan.config")

BASE_DIR = Path(__file__).parent.parent.parent  # titanai/ root


class APIConfig:
    # Server
    HOST: str = os.getenv("TITAN_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("TITAN_API_PORT", "8000"))
    WORKERS: int = int(os.getenv("TITAN_API_WORKERS", "1"))  # 1 for GPU (shared memory)
    LOG_LEVEL: str = os.getenv("TITAN_API_LOG_LEVEL", "info")

    # Model
    CONFIG_PATH: str = os.getenv(
        "TITAN_CONFIG_PATH",
        str(BASE_DIR / "configs" / "titan_probe_v015.yaml")
    )
    CHECKPOINT_PATH: str = os.getenv(
        "TITAN_CHECKPOINT_PATH",
        str(BASE_DIR / "checkpoints" / "probe_v015" / "final.pt")
    )
    CHECKPOINT_DIR: str = os.getenv(
        "TITAN_CHECKPOINT_DIR",
        str(BASE_DIR / "checkpoints")
    )
    BASE_DIR: str = str(BASE_DIR)
    DEVICE: str = os.getenv("TITAN_DEVICE", "auto")  # auto | cuda | cpu

    # Auth (PRE-FLIGHT SECURITY FIX)
    # Default is now "true" — auth is on unless explicitly disabled for local dev.
    # start_api.sh enforces TITAN_REQUIRE_AUTH=true and key length >= 32 chars
    # before this process starts, so this default is a belt-and-suspenders guard.
    API_KEY: str = os.getenv("TITAN_API_KEY", "")
    REQUIRE_AUTH: bool = os.getenv("TITAN_REQUIRE_AUTH", "true").lower() == "true"

    # Generation defaults (can be overridden per request)
    DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("TITAN_DEFAULT_MAX_TOKENS", "512"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("TITAN_DEFAULT_TEMPERATURE", "0.8"))
    DEFAULT_TOP_K: int = int(os.getenv("TITAN_DEFAULT_TOP_K", "50"))
    DEFAULT_TOP_P: float = float(os.getenv("TITAN_DEFAULT_TOP_P", "0.95"))
    MAX_NEW_TOKENS_LIMIT: int = int(os.getenv("TITAN_MAX_TOKENS_LIMIT", "2048"))

    # Streaming
    STREAM_CHUNK_SIZE: int = int(os.getenv("TITAN_STREAM_CHUNK_SIZE", "1"))  # tokens per chunk

    # Metrics
    METRICS_LOG_PATH: str = os.getenv(
        "TITAN_METRICS_LOG",
        str(BASE_DIR / "logs" / "probe_v015" / "run_summary.json")
    )


config = APIConfig()

# ── Startup auth validation ───────────────────────────────────────────────────
# Warn loudly if auth is enabled but no key is set. This can happen if
# start_api.sh is bypassed and the server is started directly.
if config.REQUIRE_AUTH and not config.API_KEY:
    log.warning(
        "SECURITY WARNING: TITAN_REQUIRE_AUTH=true but TITAN_API_KEY is empty. "
        "All requests will be rejected with 401. "
        "Set TITAN_API_KEY to a strong random key (openssl rand -hex 32)."
    )
