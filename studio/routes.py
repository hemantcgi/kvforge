# studio/routes.py
"""FastAPI router for KVForge Studio — mounts at /studio in kvforge_portal.py."""

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from studio.migration import migrate_existing_use_cases
from studio.api import api_router
from studio.job_manager import get_manager
from studio.pipeline_runner import run_step_streaming

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates" / "studio"

router = APIRouter()
router.include_router(api_router)


# ── Auto-migrate on import ────────────────────────────────────────────────────
_migrated = False

def _ensure_migrated():
    global _migrated
    if not _migrated:
        migrate_existing_use_cases(root=ROOT)
        _migrated = True


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def studio_hub():
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()


@router.get("/uc/{uc_id}", response_class=HTMLResponse)
def uc_detail(uc_id: str):
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    jm = get_manager()
    job = jm.get(job_id)
    if not job:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type':'error','message':'job not found'})}\n\n"]),
            media_type="text/event-stream"
        )

    async def event_generator():
        async for chunk in run_step_streaming(
            job["uc_id"], job["step"], job_id, jm
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
