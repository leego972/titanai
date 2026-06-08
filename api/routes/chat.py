"""
TitanAI API — /v1/chat/completions (OpenAI-compatible)
"""
import json
import logging
import time
import uuid
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from ..core.model_manager import manager
from ..core.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatMessage, UsageInfo, StreamChunk, StreamDelta, StreamChoice,
)
from ..middleware.auth import verify_api_key
from ..core.persona import get_system_prompt, is_request_private
from ..core.identity import check_identity, TITAN_RESPONSE_PASSPHRASE

log = logging.getLogger("titan.chat")
router = APIRouter(prefix="/v1/chat/completions", tags=["Chat"])

SYSTEM_PREFIX    = "System"
USER_PREFIX      = "User"
ASSISTANT_PREFIX = "Assistant"


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    lines = []
    for msg in messages:
        if msg.role == "system":
            lines.append(f"{SYSTEM_PREFIX}: {msg.content}")
        elif msg.role == "user":
            lines.append(f"{USER_PREFIX}: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"{ASSISTANT_PREFIX}: {msg.content}")
    lines.append(f"{ASSISTANT_PREFIX}:")
    return "\n".join(lines)


def extract_assistant_response(generated: str) -> str:
    for stop_token in [f"\n{USER_PREFIX}:", f"\n{SYSTEM_PREFIX}:", f"\n{ASSISTANT_PREFIX}:"]:
        if stop_token in generated:
            generated = generated[:generated.index(stop_token)]
    return generated.strip()


@router.post("", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
):
    is_private = is_request_private(fastapi_request.headers, api_key)
    system_prompt = get_system_prompt(is_private)

    if request.messages and request.messages[0].role == "system":
        request.messages[0].content = system_prompt
    else:
        request.messages.insert(0, ChatMessage(role="system", content=system_prompt))

    # ── Passphrase echo-back ───────────────────────────────────────────────────
    # When the owner sends the Shema as a chat message, Titan confirms with its
    # own response passphrase immediately, without going through generation.
    user_messages = [m for m in request.messages if m.role == "user"]
    if user_messages:
        last_user_msg = user_messages[-1].content.strip()
        if check_identity(last_user_msg):
            confirmation = (
                f"{TITAN_RESPONSE_PASSPHRASE} — Cyber mode active. "
                "Identity verified. All capabilities unlocked."
            )
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=confirmation),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

    if not manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "TitanAI model not loaded. Training may still be in progress.",
                    "type": "model_not_ready",
                    "code": "model_not_loaded",
                }
            }
        )

    prompt = messages_to_prompt(request.messages)

    if request.stream:
        return _stream_chat(request, prompt)

    try:
        generated, prompt_tokens, gen_tokens = manager.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        response_text = extract_assistant_response(generated)

        if request.stop:
            stops = [request.stop] if isinstance(request.stop, str) else request.stop
            for stop in stops:
                if stop in response_text:
                    response_text = response_text[:response_text.index(stop)]

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=gen_tokens,
                total_tokens=prompt_tokens + gen_tokens,
            ),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def _stream_chat(request: ChatCompletionRequest, prompt: str) -> StreamingResponse:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model

    async def event_stream():
        first_chunk = StreamChunk(
            id=request_id, created=created, model=model,
            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant"), finish_reason=None)],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"
        try:
            async for token in manager.stream_generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                chunk = StreamChunk(
                    id=request_id, created=created, model=model,
                    choices=[StreamChoice(index=0, delta=StreamDelta(content=token), finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            final = StreamChunk(
                id=request_id, created=created, model=model,
                choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason="stop")],
            )
            yield f"data: {final.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error(f"Streaming error: {e}", exc_info=True)
            error = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
