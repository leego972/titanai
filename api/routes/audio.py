import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from ..middleware.auth import verify_api_key
from ..core.persona import is_request_private
from ..core.identity import check_identity

# Whisper is imported lazily on first request so it does not crash the
# build or startup if openai-whisper is unavailable.
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model("base")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Whisper model unavailable: {exc}"
            )
    return _whisper_model


router = APIRouter(prefix="/v1/audio", tags=["Audio"])


@router.post("/transcriptions")
async def transcribe_audio(
    fastapi_request: Request,
    file: UploadFile = File(...),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """Transcribe audio file and perform identity verification."""
    model = _get_whisper()

    temp_dir  = "/tmp/titan_audio"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result   = model.transcribe(file_path)
        text     = result["text"].strip()
        is_owner = check_identity(text)
        is_private = is_request_private(fastapi_request.headers, api_key) or is_owner

        return {
            "text":         text,
            "is_verified":  is_owner,
            "access_level": "private" if is_private else "public",
        }
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
