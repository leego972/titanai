import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from ..middleware.auth import verify_api_key
from ..core.persona import is_request_private
from ..core.identity import check_identity

# Whisper model for transcription
import whisper
model = whisper.load_model("base")

router = APIRouter(prefix="/v1/audio", tags=["Audio"])

@router.post("/transcriptions")
async def transcribe_audio(
    fastapi_request: Request,
    file: UploadFile = File(...),
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Transcribe audio file and perform identity verification.
    """
    # 1. Save uploaded file temporarily
    temp_dir = "/tmp/titan_audio"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Transcribe
        result = model.transcribe(file_path)
        text = result["text"].strip()
        
        # 3. Check for Identity Passphrase in the transcription
        is_owner = check_identity(text)
        
        # 4. Determine if the request origin is private
        is_private = is_request_private(fastapi_request.headers, api_key) or is_owner
        
        return {
            "text": text,
            "is_verified": is_owner,
            "access_level": "private" if is_private else "public",
            "task": "transcription"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
