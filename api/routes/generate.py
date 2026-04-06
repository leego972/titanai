"""
TitanAI API — /v1/generate (Native TitanAI endpoint)
======================================================
POST /v1/generate — Direct generation with full parameter control.
No chat template wrapping — raw prompt in, raw generated text out.
"""
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..core.model_manager import manager
from ..core.schemas import GenerateRequest, GenerateResponse
from ..middleware.auth import verify_api_key

log = logging.getLogger("titan.generate")
router = APIRouter(prefix="/v1/generate", tags=["Generate"])


@router.post("", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    _: Optional[str] = Depends(verify_api_key),
):
    """
    Native TitanAI generation endpoint.
    No chat template — raw prompt → raw generated continuation.
    Full control over all generation parameters.
    """
    if not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Training may still be in progress."
        )

    if request.stream:
        return _stream_generate(request)

    try:
        generated, prompt_tokens, gen_tokens = manager.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def _stream_generate(request: GenerateRequest) -> StreamingResponse:
    """Stream raw token generation."""
    request_id = f"gen-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_stream():
        try:
            async for token in manager.stream_generate(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                chunk = {
                    "id": request_id,
                    "created": created,
                    "token": token,
                    "object": "generate.chunk",
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            yield f"data: {json.dumps({'id': request_id, 'object': 'generate.done', 'finish_reason': 'stop'})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
