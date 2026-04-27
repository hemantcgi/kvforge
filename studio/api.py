# studio/api.py
"""All /studio/api/* endpoint handlers — imported by routes.py."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from studio.migration import migrate_existing_use_cases, load_registry, add_to_registry
from studio.gpu_monitor import get_gpu_status, stop_vllm_process, get_gpu_realtime
from studio.job_manager import get_manager, DuplicateJobError
from studio import settings_manager
from studio import curation_manager

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
        cfg_path = ROOT / "examples" / uc["id"] / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                if "dashboard_port" in cfg:
                    uc_data["dashboard_port"] = cfg["dashboard_port"]
            except Exception:
                pass
        faqs_path = ROOT / "examples" / uc["id"] / "faqs.json"
        uc_data["has_faqs"] = faqs_path.exists()
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
    data = json.loads(path.read_text())
    # Augment with fields from config.json (authoritative datasource config)
    cfg_path = _uc_path(uc_id) / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if "dashboard_port" in cfg:
            data["dashboard_port"] = cfg["dashboard_port"]
        # Expose authoritative vectordb fields for read-only display
        data["datasource_config"] = {k: cfg[k] for k in (
            "vector_store", "vector_dim", "chunk_size", "chunk_overlap",
            "embed_model", "embedder_backend", "collection", "loader",
        ) if k in cfg}
    return data


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
                    "comparison_provider": "gemini", "comparison_model": "gemini-2.5-flash",
                    "sleep_faq_provider": "gemini", "sleep_faq_model": "gemini-2.5-flash",
                    "sleep_faq_count": 50},
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


# ── Wizard Validate ───────────────────────────────────────────────────────────

VALID_STEPS = {"index", "train", "recompute", "prs-eval", "ab-eval", "sleep-faq"}

@api_router.post("/wizard-validate")
async def wizard_validate(request: Request):
    body = await request.json()
    errors = []
    step = body.get("step", "")
    if step not in VALID_STEPS:
        errors.append(f"Unknown step '{step}'. Valid steps: {sorted(VALID_STEPS)}")
    epochs = body.get("epochs")
    if epochs is not None and (not isinstance(epochs, int) or epochs < 1):
        errors.append("epochs must be an integer >= 1")
    top_k = body.get("top_k")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1 or top_k > 100):
        errors.append("top_k must be an integer between 1 and 100")
    faq_count = body.get("faq_count")
    if faq_count is not None and (not isinstance(faq_count, int) or faq_count < 5 or faq_count > 500):
        errors.append("faq_count must be an integer between 5 and 500")
    return {"ok": len(errors) == 0, "errors": errors}


# ── Pipeline Jobs ─────────────────────────────────────────────────────────────

class RunStepRequest(BaseModel):
    uc_id: str
    step: str  # "index" | "train" | "recompute" | "prs-eval" | "ab-eval" | "sleep-faq"


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


# ── Settings ──────────────────────────────────────────────────────────────────

@api_router.get("/settings")
def get_settings_endpoint():
    return JSONResponse(settings_manager.get_masked())


@api_router.post("/settings")
async def save_settings_endpoint(request: Request):
    body = await request.json()
    try:
        settings_manager.save(body)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return JSONResponse({"ok": True})


# ── GPU realtime ───────────────────────────────────────────────────────────────

@api_router.get("/gpu/realtime")
def gpu_realtime_endpoint():
    return JSONResponse(get_gpu_realtime())


# ── PRS history ────────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/prs-history")
def prs_history_endpoint(uc_id: str):
    version_path = _uc_path(uc_id) / "version.json"
    if not version_path.exists():
        return JSONResponse([])
    try:
        v = json.loads(version_path.read_text())
    except (json.JSONDecodeError, OSError):
        return JSONResponse([])
    raw = v.get("prs_history", [])
    result = []
    for entry in raw:
        if isinstance(entry, dict):
            result.append({
                "label": f"LoRA v{entry.get('round', '?')}",
                "round": entry.get("round"),
                "prs": entry.get("prs"),
            })
        elif isinstance(entry, (int, float)):
            result.append({"label": f"LoRA v{len(result)+1}", "round": len(result)+1, "prs": entry})
    return JSONResponse(result)


# ── Auto-curation ──────────────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/ab-curate")
async def ab_curate_endpoint(uc_id: str, request: Request):
    body = await request.json()
    question = body.get("question", "")
    answer = body.get("answer", "")
    source_model = body.get("source_model", "model_b")
    if not question or not answer:
        raise HTTPException(400, "question and answer are required")
    status = curation_manager.append(uc_id, question, answer, source_model)
    return JSONResponse(status)


@api_router.get("/uc/{uc_id}/curation-status")
def curation_status_endpoint(uc_id: str):
    return JSONResponse(curation_manager.get_status(uc_id))
