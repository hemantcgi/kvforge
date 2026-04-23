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
_UC_CONFIGS = ROOT / "uc_configs"

router = APIRouter()
router.include_router(api_router)

# Flywheel cross-UC analytics router
try:
    from studio.flywheel_routes import router as flywheel_router
    router.include_router(flywheel_router, prefix="/flywheel")
except ImportError:
    pass


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


# ── Dynamic PRS UC settings ───────────────────────────────────────────────────

_DYNAMIC_PRS_KEYS = {
    "deployment_mode", "prs_advancement_threshold", "prs_signal_weights",
    "prs_auto_weight", "brownfield_routing_threshold", "brownfield_confidence_floor",
    "brownfield_coverage_target", "difficulty_estimator",
}


def _uc_cfg_path(uc_name: str) -> Path:
    return _UC_CONFIGS / f"{uc_name}.json"


@router.get("/api/uc/{uc_name}/settings")
def get_uc_settings(uc_name: str):
    """Return Dynamic PRS settings for a use case."""
    p = _uc_cfg_path(uc_name)
    if not p.exists():
        return {"error": f"config not found for {uc_name}"}
    cfg = json.loads(p.read_text())
    return {k: cfg[k] for k in _DYNAMIC_PRS_KEYS if k in cfg}


@router.patch("/api/uc/{uc_name}/settings")
async def patch_uc_settings(uc_name: str, request: Request):
    """Update Dynamic PRS settings for a use case."""
    p = _uc_cfg_path(uc_name)
    if not p.exists():
        return {"error": f"config not found for {uc_name}"}
    updates = await request.json()
    cfg = json.loads(p.read_text())
    for k, v in updates.items():
        if k in _DYNAMIC_PRS_KEYS:
            cfg[k] = v
    p.write_text(json.dumps(cfg, indent=2))
    return {k: cfg[k] for k in _DYNAMIC_PRS_KEYS if k in cfg}


# ── ModelScout SSE ────────────────────────────────────────────────────────────

_scout_sessions: dict = {}


@router.post("/api/modelscout/{uc_name}/start")
async def start_modelscout(uc_name: str):
    """Launch a ModelScout session for a use case; returns a session_id."""
    import uuid
    session_id = str(uuid.uuid4())
    _scout_sessions[session_id] = {"uc_name": uc_name, "messages": [], "pending_ask": None}
    return {"session_id": session_id}


@router.get("/api/modelscout/{session_id}/stream")
async def modelscout_stream(session_id: str):
    """SSE stream of ModelScout messages for the given session."""
    import asyncio

    async def event_gen():
        session = _scout_sessions.get(session_id)
        if not session:
            yield f"data: {json.dumps({'type': 'error', 'message': 'session not found'})}\n\n"
            return
        seen = 0
        for _ in range(300):
            msgs = session["messages"]
            while seen < len(msgs):
                yield f"data: {json.dumps(msgs[seen])}\n\n"
                seen += 1
            if session.get("done"):
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache"})


@router.post("/api/modelscout/{session_id}/respond")
async def modelscout_respond(session_id: str, request: Request):
    """Inject a user response into a waiting ModelScout ask()."""
    body = await request.json()
    session = _scout_sessions.get(session_id)
    if not session:
        return {"error": "session not found"}
    session["pending_response"] = body.get("response", "")
    return {"ok": True}
