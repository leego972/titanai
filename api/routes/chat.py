"""
TitanAI API — /v1/chat/completions (OpenAI-compatible)
=======================================================
POST /v1/chat/completions — Chat completion with message history

Converts the OpenAI chat message format to a single prompt string
using a simple chat template, then runs TitanAI generation.

Chat template:
    System: <system_message>
    User: <user_message>
    Assistant: <assistant_message>
    User: <latest_user_message>
    Assistant:

This is a base model — it does not have RLHF/instruction tuning yet.
The chat format is a best-effort approximation for the Probe stage.
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

log = logging.getLogger("titan.chat")
router = APIRouter(prefix="/v1/chat/completions", tags=["Chat"])

# ── Chat template ─────────────────────────────────────────────────────────────

SYSTEM_PREFIX = "System"
USER_PREFIX = "User"
ASSISTANT_PREFIX = "Assistant"


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Convert OpenAI-style messages list to a single prompt string.
    Titan is a base model — this template primes it to continue as the assistant.
    """
    lines = []
    for msg in messages:
        if msg.role == "system":
            lines.append(f"{SYSTEM_PREFIX}: {msg.content}")
        elif msg.role == "user":
            lines.append(f"{USER_PREFIX}: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"{ASSISTANT_PREFIX}: {msg.content}")

    # Prime the model to respond as assistant
    lines.append(f"{ASSISTANT_PREFIX}:")
    return "\n".join(lines)


def extract_assistant_response(generated: str) -> str:
    """
    Extract only the assistant's response from the generated text.
    Stops at the next User: or System: turn.
    """
    # Stop at next turn boundary
    for stop_token in [f"\n{USER_PREFIX}:", f"\n{SYSTEM_PREFIX}:", f"\n{ASSISTANT_PREFIX}:"]:
        if stop_token in generated:
            generated = generated[:generated.index(stop_token)]
    return generated.strip()


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    fastapi_request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
):
    # Determine Persona based on Request Headers and Auth
    is_private = is_request_private(fastapi_request.headers, api_key)
    system_prompt = get_system_prompt(is_private)
    
    # Inject/Replace System Prompt
    if request.messages and request.messages[0].role == "system":
        request.messages[0].content = system_prompt
    else:
        request.messages.insert(0, ChatMessage(role="system", content=system_prompt))

    """
    OpenAI-compatible chat completion endpoint.
    Archibald can use this as a drop-in replacement for OpenAI API calls.
    """
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

    # Convert messages to prompt
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

        # Extract clean assistant response
        response_text = extract_assistant_response(generated)

        # Apply stop sequences
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
    """Return a streaming SSE response for chat."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = request.model

    async def event_stream():
        # First chunk: role announcement
        first_chunk = StreamChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[StreamChoice(
                index=0,
                delta=StreamDelta(role="assistant"),
                finish_reason=None,
            )],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        try:
            buffer = []
            async for token in manager.stream_generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            ):
                buffer.append(token)
                chunk = StreamChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[StreamChoice(
                        index=0,
                        delta=StreamDelta(content=token),
                        finish_reason=None,
                    )],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk
            final = StreamChunk(
                id=request_id,
                created=created,
                model=model,
                choices=[StreamChoice(
                    index=0,
                    delta=StreamDelta(),
                    finish_reason="stop",
                )],
            )
            yield f"data: {final.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            log.error(f"Streaming error: {e}", exc_info=True)
            error = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
