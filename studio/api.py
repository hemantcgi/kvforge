# studio/api.py
"""All /studio/api/* endpoint handlers — imported by routes.py."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from studio.migration import migrate_existing_use_cases, load_registry, add_to_registry
from studio.gpu_monitor import get_gpu_status, stop_vllm_process
from studio.job_manager import get_manager, DuplicateJobError

ROOT = Path(__file__).resolve().parent.parent

api_router = APIRouter(prefix="/api")


def _uc_path(uc_id: str) -> Path:
    """Return the UC directory path, raising 400 on path traversal attempts."""
    path = (ROOT / "examples" / uc_id).resolve()
    if not str(path).startswith(str((ROOT / "examples").resolve())):
        raise HTTPException(400, "Invalid use case ID")
    return path


# ── Registry ──────────────────────────────────────────────────────────────────

@api_router.get("/registry")
def get_registry():
    ucs = load_registry(root=ROOT)
    result = []
    for uc in ucs:
        uc_data = dict(uc)
        version_path = ROOT / "examples" / uc["id"] / "version.json"
        if version_path.exists():
            try:
                v = json.loads(version_path.read_text())
                uc_data["phase"] = v.get("phase", 1)
                history = v.get("prs_history", [])
                uc_data["prs"] = history[-1]["prs"] if history else None
            except Exception:
                uc_data["phase"] = 1
                uc_data["prs"] = None
        jm = get_manager()
        active = next((j for j in jm.list_active() if j["uc_id"] == uc["id"]), None)
        uc_data["active_job"] = active
        result.append(uc_data)
    return {"use_cases": result}


# ── UC Config ─────────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/config")
def get_uc_config(uc_id: str):
    path = _uc_path(uc_id) / "uc_config.json"
    if not path.exists():
        raise HTTPException(404, f"uc_config.json not found for {uc_id}")
    return json.loads(path.read_text())


@api_router.post("/uc/{uc_id}/config")
async def save_uc_config(uc_id: str, request: Request):
    path = _uc_path(uc_id) / "uc_config.json"
    if not path.exists():
        raise HTTPException(404, f"UC {uc_id} not found")
    existing = json.loads(path.read_text())
    updates = await request.json()
    for key, val in updates.items():
        if isinstance(val, dict) and key in existing:
            existing[key].update(val)
        else:
            existing[key] = val
    path.write_text(json.dumps(existing, indent=2))
    return {"status": "saved"}


class NewUCRequest(BaseModel):
    id: str
    display_name: str
    description: Optional[str] = ""


@api_router.post("/uc/new")
def create_new_uc(req: NewUCRequest):
    uc_dir = _uc_path(req.id)
    if (uc_dir / "uc_config.json").exists():
        raise HTTPException(409, f"Use case '{req.id}' already exists")
    uc_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone
    uc_config = {
        "id": req.id, "display_name": req.display_name,
        "type": "custom", "created_at": datetime.now(timezone.utc).isoformat(),
        "data":    {"source_type": "", "dataset_id": "", "split": "train",
                    "text_column": "text", "max_rows": 5000},
        "vectordb":{"store": "qdrant", "dimensions": 384, "chunk_size": 512,
                    "chunk_overlap": 64,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "index_type": "hnsw"},
        "llm":     {"local_model": "meta-llama/Llama-3.2-3B-Instruct",
                    "quantization": "4bit", "vllm_url": "",
                    "comparison_provider": "gemini", "comparison_model": "gemini-1.5-flash"},
    }
    (uc_dir / "uc_config.json").write_text(json.dumps(uc_config, indent=2))
    add_to_registry(req.id, req.display_name, root=ROOT)
    return {"status": "created", "id": req.id}


# ── GPU ───────────────────────────────────────────────────────────────────────

@api_router.post("/gpu-check")
def gpu_check():
    import studio.gpu_monitor as _gm
    return _gm.get_gpu_status()


class StopVllmRequest(BaseModel):
    port: int


@api_router.post("/gpu/stop-vllm")
def stop_vllm(req: StopVllmRequest):
    ok = stop_vllm_process(req.port)
    if not ok:
        raise HTTPException(500, f"Failed to stop vLLM on port {req.port}")
    return {"status": "stopped", "port": req.port}


# ── Pipeline Jobs ─────────────────────────────────────────────────────────────

class RunStepRequest(BaseModel):
    uc_id: str
    step: str  # "index" | "train" | "recompute" | "prs-eval" | "ab-eval"


@api_router.post("/run-step")
def run_step(req: RunStepRequest):
    from studio.pipeline_runner import STEP_MODULES
    if req.step not in STEP_MODULES:
        raise HTTPException(400, f"Unknown step: {req.step}. Valid: {list(STEP_MODULES)}")
    jm = get_manager()
    try:
        job_id = jm.create(req.uc_id, req.step)
    except DuplicateJobError as e:
        raise HTTPException(409, str(e))
    return {"job_id": job_id, "uc_id": req.uc_id, "step": req.step}


@api_router.delete("/job/{job_id}")
def stop_job(job_id: str):
    import signal, os
    jm = get_manager()
    job = jm.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    if job["status"] == "running" and job["pid"]:
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    jm.stop(job_id)
    return {"status": "stopped"}
