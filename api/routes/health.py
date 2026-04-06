"""
TitanAI API — Health & Metrics Routes
=======================================
GET /health          — Basic liveness check
GET /health/gpu      — GPU memory and utilization
GET /health/model    — Model loaded status and info
GET /metrics         — Training run metrics (from run_summary.json)
"""
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..core.model_manager import manager
from ..core.schemas import HealthResponse, GPUHealthResponse
from ..core.config import config
from ..middleware.auth import verify_api_key

log = logging.getLogger("titan.health")
router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Basic liveness check. Always returns 200 if the server is up."""
    return HealthResponse(
        status="ok" if manager.is_loaded else "degraded",
        model_loaded=manager.is_loaded,
        device=manager.device,
        checkpoint=manager.checkpoint,
        uptime_seconds=round(manager.get_uptime(), 1),
    )


@router.get("/health/gpu", response_model=GPUHealthResponse)
async def gpu_health():
    """GPU memory and utilization stats."""
    try:
        import torch
        if not torch.cuda.is_available():
            return GPUHealthResponse(status="no_gpu")

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        mem_total = props.total_memory / 1_073_741_824  # bytes → GB
        mem_reserved = torch.cuda.memory_reserved(device) / 1_073_741_824
        mem_allocated = torch.cuda.memory_allocated(device) / 1_073_741_824
        mem_free = mem_total - mem_reserved

        # Try to get utilization via pynvml
        gpu_util: Optional[float] = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = float(util.gpu)
        except Exception:
            pass

        return GPUHealthResponse(
            status="ok",
            device_name=props.name,
            vram_total_gb=round(mem_total, 2),
            vram_used_gb=round(mem_allocated, 2),
            vram_free_gb=round(mem_free, 2),
            gpu_utilization_pct=gpu_util,
            cuda_version=torch.version.cuda,
        )
    except Exception as e:
        log.error(f"GPU health check failed: {e}")
        return GPUHealthResponse(status="error")


@router.get("/health/model")
async def model_health():
    """Detailed model status including architecture and training info."""
    return {
        "loaded": manager.is_loaded,
        "checkpoint": manager.checkpoint,
        "device": manager.device,
        "info": manager.model_info,
        "uptime_seconds": round(manager.get_uptime(), 1),
    }


@router.get("/metrics")
async def metrics(_: Optional[str] = Depends(verify_api_key)):
    """
    Training run metrics from the last completed run.
    Returns the run_summary.json if available.
    """
    summary_path = Path(config.METRICS_LOG_PATH)
    if not summary_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "No training metrics available yet. Training may still be in progress."}
        )

    try:
        with open(summary_path) as f:
            summary = json.load(f)
        return summary
    except Exception as e:
        log.error(f"Failed to read metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to read metrics: {str(e)}"}
        )
