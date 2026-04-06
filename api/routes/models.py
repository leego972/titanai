"""
TitanAI API — Model Management Routes
=======================================
GET  /v1/models              — List available checkpoints
GET  /v1/models/{model_id}   — Get specific model info
POST /v1/models/load         — Load a checkpoint (hot-swap)
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ..core.model_manager import manager
from ..core.schemas import ModelInfo, ModelList
from ..core.config import config
from ..middleware.auth import verify_api_key

log = logging.getLogger("titan.models")
router = APIRouter(prefix="/v1/models", tags=["Models"])


@router.get("", response_model=ModelList)
async def list_models(_: Optional[str] = Depends(verify_api_key)):
    """List all available TitanAI checkpoints."""
    checkpoints = manager.list_checkpoints()

    models = []
    for ckpt in checkpoints:
        models.append(ModelInfo(
            id=f"titan-{ckpt['id']}",
            created=ckpt["created"],
            checkpoint_path=ckpt["path"],
            parameters=manager.model_info.get("parameters") if ckpt["is_loaded"] else None,
            training_step=ckpt.get("training_step"),
            val_perplexity=ckpt.get("val_perplexity"),
            is_loaded=ckpt["is_loaded"],
        ))

    # If no checkpoints found, return the configured default (even if not yet trained)
    if not models:
        models.append(ModelInfo(
            id="titan-probe-v0.1.5",
            created=0,
            checkpoint_path=config.CHECKPOINT_PATH,
            is_loaded=manager.is_loaded,
        ))

    return ModelList(data=models)


@router.get("/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str, _: Optional[str] = Depends(verify_api_key)):
    """Get info about a specific model checkpoint."""
    checkpoints = manager.list_checkpoints()

    for ckpt in checkpoints:
        if ckpt["id"] == model_id or f"titan-{ckpt['id']}" == model_id:
            return ModelInfo(
                id=f"titan-{ckpt['id']}",
                created=ckpt["created"],
                checkpoint_path=ckpt["path"],
                parameters=manager.model_info.get("parameters") if ckpt["is_loaded"] else None,
                training_step=ckpt.get("training_step"),
                val_perplexity=ckpt.get("val_perplexity"),
                is_loaded=ckpt["is_loaded"],
            )

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")


class LoadModelRequest(BaseModel):
    checkpoint_path: str
    config_path: Optional[str] = None
    device: Optional[str] = None


@router.post("/load")
async def load_model(
    request: LoadModelRequest,
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Hot-load a specific checkpoint. Useful after training completes.
    Requires auth if TITAN_REQUIRE_AUTH=true.
    """
    log.info(f"[Models] Loading checkpoint: {request.checkpoint_path}")
    success = manager.load(
        config_path=request.config_path,
        checkpoint_path=request.checkpoint_path,
        device=request.device,
    )

    if success:
        return {
            "success": True,
            "message": f"Model loaded from {request.checkpoint_path}",
            "info": manager.model_info,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load checkpoint: {request.checkpoint_path}. Check that the file exists."
        )
