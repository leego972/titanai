"""
TitanAI API — /v1/completions (OpenAI-compatible)
==================================================
POST /v1/completions — Raw text completion
"""
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from ..core.model_manager import manager
from ..core.schemas import (
    CompletionRequest, CompletionResponse, CompletionChoice, UsageInfo,
    StreamChunk, StreamDelta, StreamChoice,
)
from ..middleware.auth import verify_api_key

log = logging.getLogger("titan.completions")
router = APIRouter(prefix="/v1/completions", tags=["Completions"])


@router.post("", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    _: Optional[str] = Depends(verify_api_key),
):
    """
    OpenAI-compatible text completion endpoint.
    Supports single prompt string or list of prompts (n=1 per prompt).
    """
    if not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Training may still be in progress. "
                   "POST /v1/models/load once the checkpoint is available."
        )

    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    if request.stream and len(prompts) == 1:
        return _stream_completion(request, prompts[0])

    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, prompt in enumerate(prompts):
        try:
            generated, prompt_tokens, gen_tokens = manager.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )

            # Apply stop sequences
            if request.stop:
                stops = [request.stop] if isinstance(request.stop, str) else request.stop
                for stop in stops:
                    if stop in generated:
                        generated = generated[:generated.index(stop)]

            choices.append(CompletionChoice(
                text=generated,
                index=i,
                finish_reason="stop",
            ))
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += gen_tokens

        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            log.error(f"Generation error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return CompletionResponse(
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


def _stream_completion(request: CompletionRequest, prompt: str) -> StreamingResponse:
    """Return a streaming SSE response for a single prompt."""
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_stream():
        try:
            async for token in manager.stream_generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [{"text": token, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final chunk with finish_reason
            final = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"Streaming error: {e}", exc_info=True)
            error = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
