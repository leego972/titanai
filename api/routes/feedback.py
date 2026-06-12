"""
TitanAI — DPO Feedback Endpoint
=================================
Captures real usage feedback to build preference pairs for DPO training.

Two usage patterns:

1. EXPLICIT PAIR — you already have chosen/rejected:
   POST /v1/feedback
   {"prompt": "...", "chosen": "...", "rejected": "..."}

2. THUMBS — rate a live response up or down:
   POST /v1/feedback
   {"conversation_id": "...", "prompt": "...", "response": "...",
    "thumbs": "up",   # → response becomes "chosen", no rejected yet
    "correction": ""  # optional: your better answer (becomes "chosen",
                      #           original response becomes "rejected")
   }

All feedback is appended to data/dpo_feedback/feedback.jsonl.
Run scripts/build_dpo_dataset.py to compile into custom_prefs.jsonl for DPO.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

log = logging.getLogger("titan.feedback")
router = APIRouter(prefix="/v1/feedback", tags=["Feedback / DPO"])

FEEDBACK_FILE = Path(__file__).parent.parent.parent / "data" / "dpo_feedback" / "feedback.jsonl"
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ExplicitFeedback(BaseModel):
    """Direct preference pair — you supply both chosen and rejected."""
    prompt: str
    chosen: str
    rejected: str
    note: Optional[str] = None


class ThumbsFeedback(BaseModel):
    """Rate a live response. Correction becomes chosen; original becomes rejected."""
    conversation_id: Optional[str] = None
    prompt: str
    response: str
    thumbs: Literal["up", "down"]
    correction: Optional[str] = None   # required when thumbs=down for a full DPO pair
    note: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    status: str
    type: str
    message: str
    total_logged: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_feedback() -> int:
    try:
        return sum(1 for _ in open(FEEDBACK_FILE) if _.strip())
    except FileNotFoundError:
        return 0


def _append(record: dict):
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=FeedbackResponse)
async def submit_feedback(body: dict):
    """
    Accept either an ExplicitFeedback or ThumbsFeedback payload.
    Auto-detects by presence of 'chosen'/'rejected' vs 'thumbs'.
    """
    fid = f"fb-{uuid.uuid4().hex[:10]}"
    ts  = int(time.time())

    # ── Explicit pair ─────────────────────────────────────────────────────────
    if "chosen" in body and "rejected" in body:
        try:
            fb = ExplicitFeedback(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        record = {
            "id": fid, "ts": ts, "type": "explicit",
            "prompt": fb.prompt,
            "chosen": fb.chosen,
            "rejected": fb.rejected,
            "note": fb.note,
        }
        _append(record)
        log.info(f"[Feedback] Explicit pair logged ({fid})")
        return FeedbackResponse(
            id=fid, status="logged", type="explicit",
            message="Preference pair saved. Will be included in next DPO run.",
            total_logged=_count_feedback(),
        )

    # ── Thumbs ────────────────────────────────────────────────────────────────
    elif "thumbs" in body:
        try:
            fb = ThumbsFeedback(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        if fb.thumbs == "up":
            # Good response — log as positive signal. No rejected yet.
            record = {
                "id": fid, "ts": ts, "type": "thumbs_up",
                "conversation_id": fb.conversation_id,
                "prompt": fb.prompt,
                "chosen": fb.response,
                "rejected": None,   # incomplete pair — used for positive signal only
                "note": fb.note,
            }
            _append(record)
            log.info(f"[Feedback] Thumbs UP logged ({fid})")
            return FeedbackResponse(
                id=fid, status="logged", type="thumbs_up",
                message="Positive signal saved.",
                total_logged=_count_feedback(),
            )

        else:  # thumbs == "down"
            if not fb.correction:
                # Log the bad response for later pairing
                record = {
                    "id": fid, "ts": ts, "type": "thumbs_down_no_correction",
                    "conversation_id": fb.conversation_id,
                    "prompt": fb.prompt,
                    "rejected": fb.response,
                    "chosen": None,
                    "note": fb.note,
                }
                _append(record)
                log.info(f"[Feedback] Thumbs DOWN (no correction) logged ({fid})")
                return FeedbackResponse(
                    id=fid, status="logged", type="thumbs_down",
                    message="Bad response flagged. Add 'correction' field to create a full DPO pair.",
                    total_logged=_count_feedback(),
                )
            else:
                # Full DPO pair: correction=chosen, original=rejected
                record = {
                    "id": fid, "ts": ts, "type": "thumbs_down_corrected",
                    "conversation_id": fb.conversation_id,
                    "prompt": fb.prompt,
                    "chosen": fb.correction,      # your better answer
                    "rejected": fb.response,       # Titan's bad answer
                    "note": fb.note,
                }
                _append(record)
                log.info(f"[Feedback] Full DPO pair from correction logged ({fid})")
                return FeedbackResponse(
                    id=fid, status="logged", type="dpo_pair",
                    message="Full DPO pair saved. This will directly improve Titan's next training run.",
                    total_logged=_count_feedback(),
                )

    else:
        raise HTTPException(
            status_code=422,
            detail="Payload must contain either ('chosen'+'rejected') or 'thumbs' field."
        )


@router.get("/stats", tags=["Feedback / DPO"])
async def feedback_stats():
    """How much feedback has been collected."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "complete_pairs": 0, "thumbs_up": 0, "thumbs_down": 0, "file": str(FEEDBACK_FILE)}

    total = thumbs_up = thumbs_down = complete = 0
    try:
        with open(FEEDBACK_FILE) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    total += 1
                    t = r.get("type", "")
                    if "up" in t:      thumbs_up  += 1
                    if "down" in t:    thumbs_down += 1
                    if r.get("chosen") and r.get("rejected"):
                        complete += 1
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "total_entries": total,
        "complete_dpo_pairs": complete,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "feedback_file": str(FEEDBACK_FILE),
        "dpo_ready": complete >= 100,
        "note": "Run scripts/build_dpo_dataset.py when complete_dpo_pairs >= 100 to generate custom_prefs.jsonl"
    }
