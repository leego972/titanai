import os
import logging
from pathlib import Path

log = logging.getLogger("titan.config")

_BASE = Path(__file__).parent.parent.parent


class APIConfig:
    HOST: str = os.getenv("TITAN_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("TITAN_API_PORT", "8000"))
    WORKERS: int = int(os.getenv("TITAN_API_WORKERS", "1"))
    LOG_LEVEL: str = os.getenv("TITAN_API_LOG_LEVEL", "info")

    CONFIG_PATH: str = os.getenv("TITAN_CONFIG_PATH", str(_BASE / "configs" / "titan_1b.yaml"))
    CHECKPOINT_PATH: str = os.getenv("TITAN_CHECKPOINT_PATH", str(_BASE / "checkpoints" / "final.pt"))
    CHECKPOINT_DIR: str = os.getenv("TITAN_CHECKPOINT_DIR", str(_BASE / "checkpoints"))
    BASE_DIR: str = str(_BASE)
    DEVICE: str = os.getenv("TITAN_DEVICE", "auto")

    API_KEY: str = os.getenv("TITAN_API_KEY", "")
    REQUIRE_AUTH: bool = os.getenv("TITAN_REQUIRE_AUTH", "true").lower() == "true"

    DEFAULT_MAX_NEW_TOKENS: int = int(os.getenv("TITAN_DEFAULT_MAX_TOKENS", "512"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("TITAN_DEFAULT_TEMPERATURE", "0.8"))
    DEFAULT_TOP_K: int = int(os.getenv("TITAN_DEFAULT_TOP_K", "50"))
    DEFAULT_TOP_P: float = float(os.getenv("TITAN_DEFAULT_TOP_P", "0.95"))
    MAX_NEW_TOKENS_LIMIT: int = int(os.getenv("TITAN_MAX_TOKENS_LIMIT", "2048"))
    STREAM_CHUNK_SIZE: int = int(os.getenv("TITAN_STREAM_CHUNK_SIZE", "1"))
    METRICS_LOG_PATH: str = os.getenv("TITAN_METRICS_LOG", str(_BASE / "logs" / "api" / "run_summary.json"))


config = APIConfig()

if config.REQUIRE_AUTH and not config.API_KEY:
    log.warning("TITAN_REQUIRE_AUTH=true but TITAN_API_KEY is empty — all requests will return 401.")
