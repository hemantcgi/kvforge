# studio/api.py
"""All /studio/api/* endpoint handlers — imported by routes.py."""

import asyncio
import json
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import re as _re

from studio.migration import migrate_existing_use_cases, load_registry, add_to_registry
from studio.gpu_monitor import get_gpu_status, stop_vllm_process, get_gpu_realtime
from studio.job_manager import get_manager, DuplicateJobError
from studio import settings_manager
from studio import curation_manager
from studio import ab_runner
from studio import vdb_validator

ROOT = Path(__file__).resolve().parent.parent

api_router = APIRouter(prefix="/api")


def _uc_path(uc_id: str) -> Path:
    """Return the UC directory path, raising 400 on path traversal attempts."""
    path = (ROOT / "examples" / uc_id).resolve()
    if not path.is_relative_to((ROOT / "examples").resolve()):
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
                uc_data["lora_version"] = v.get("current_lora_version", 0)
                # Round number of the last PRS evaluation — used by UI to detect
                # whether training has happened since the last eval
                last_prs_entry = history[-1] if history else None
                uc_data["prs_round"] = (
                    last_prs_entry.get("round") if isinstance(last_prs_entry, dict) else len(history)
                ) if history else 0
            except Exception:
                uc_data["phase"] = 1
                uc_data["prs"] = None
                uc_data["prs_round"] = 0
        cfg_path = ROOT / "examples" / uc["id"] / "config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                if "dashboard_port" in cfg:
                    uc_data["dashboard_port"] = cfg["dashboard_port"]
            except Exception:
                pass
        uc_data["has_index"] = version_path.exists()
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
async def run_step(req: RunStepRequest, background_tasks: BackgroundTasks):
    from studio.pipeline_runner import STEP_MODULES, run_step_background
    if req.step not in STEP_MODULES:
        raise HTTPException(400, f"Unknown step: {req.step}. Valid: {list(STEP_MODULES)}")
    jm = get_manager()
    try:
        job_id = jm.create(req.uc_id, req.step)
    except DuplicateJobError as e:
        raise HTTPException(409, str(e))
    # Launch subprocess immediately as a background task, independent of any SSE connection
    background_tasks.add_task(run_step_background, req.uc_id, req.step, job_id, jm)
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


# ── Job logs ───────────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/logs")
def uc_logs_endpoint(uc_id: str):
    _uc_path(uc_id)  # path traversal guard
    jm = get_manager()
    job = jm.last_for_uc(uc_id)
    if job:
        return JSONResponse({
            "lines": job.get("last_lines", []),
            "status": job.get("status"),
            "step": job.get("step"),
            "job_id": job.get("job_id"),
        })
    # Fallback: read from disk log (survives studio restarts)
    from studio.pipeline_runner import _log_path
    log_file = _log_path(uc_id)
    if log_file.exists():
        try:
            lines = log_file.read_text().splitlines()
            # Parse step/status from header line written by run_step_background
            step = None
            status = "done"  # if file exists from a previous run, it completed
            if lines and lines[0].startswith("[studio] step="):
                parts = lines[0].split()
                for p in parts:
                    if p.startswith("step="):
                        step = p[5:]
            if lines and "[studio] failed" in lines[-1]:
                status = "failed"
            return JSONResponse({"lines": lines, "status": status, "step": step, "job_id": None})
        except OSError:
            pass
    return JSONResponse({"lines": [], "status": None, "step": None, "job_id": None})


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


@api_router.get("/uc/{uc_id}/sync-history")
def get_sync_history(uc_id: str):
    from core.sync_engine import SyncStateDB
    db_path = _uc_path(uc_id) / "sync.db"
    if not db_path.exists():
        return {"runs": []}
    try:
        runs = SyncStateDB(str(db_path)).get_sync_runs(uc_id)
    except Exception:
        runs = []
    return {"runs": runs}


@api_router.get("/uc/{uc_id}/eval-summary")
def eval_summary(uc_id: str):
    results_path = _uc_path(uc_id) / "ab_eval_results.json"
    if not results_path.exists():
        return JSONResponse({"has_results": False})
    try:
        results = json.loads(results_path.read_text())
    except Exception:
        return JSONResponse({"has_results": False})
    if not results:
        return JSONResponse({"has_results": False})

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    lat_a = [r["latency_a_ms"] for r in results if r.get("latency_a_ms", 0) > 0]
    lat_b = [r["latency_b_ms"] for r in results if r.get("latency_b_ms", 0) > 0]
    sem_a = [r["sem_sim_a"] for r in results if r.get("sem_sim_a") is not None]
    sem_b = [r["sem_sim_b"] for r in results if r.get("sem_sim_b") is not None]
    prs_scores = [r["prs_score"] for r in results if r.get("prs_score") is not None]
    wins = sum(1 for r in results if r.get("prs_score", 0) >= 0.75)

    avg_lat_a = avg(lat_a) or 0
    avg_lat_b = avg(lat_b) or 0
    avg_sem_a = avg(sem_a) or 0
    avg_sem_b = avg(sem_b) or 0

    # Speed gain: positive = KVForge faster, negative = KVForge slower
    speed_gain = round((avg_lat_b - avg_lat_a) / avg_lat_b * 100, 1) if avg_lat_b > 0 else None

    return JSONResponse({
        "has_results": True,
        "total": len(results),
        "wins": wins,
        "win_rate": round(wins / len(results) * 100, 1),
        "avg_prs": avg(prs_scores),
        "avg_sem_a": avg_sem_a,
        "avg_sem_b": avg_sem_b,
        "avg_lat_a_ms": round(avg_lat_a),
        "avg_lat_b_ms": round(avg_lat_b),
        "speed_gain_pct": speed_gain,  # negative means KVForge is slower
    })


# ── Auto-curation ──────────────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/ab-curate")
async def ab_curate_endpoint(uc_id: str, request: Request):
    _uc_path(uc_id)  # path traversal guard
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
    _uc_path(uc_id)  # path traversal guard
    return JSONResponse(curation_manager.get_status(uc_id))


# ── A/B query ──────────────────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/ab-query")
async def ab_query_endpoint(uc_id: str, request: Request):
    _uc_path(uc_id)  # path traversal guard
    body = await request.json()
    query = body.get("query", "")
    if not query:
        raise HTTPException(400, "query is required")
    result = await ab_runner.run_ab_query(
        uc_id=uc_id,
        query=query,
        model_a_settings=body.get("model_a_settings", {}),
        model_b_settings=body.get("model_b_settings", {}),
    )
    return JSONResponse(result)


# ── Wizard ─────────────────────────────────────────────────────────────────────

_KNOWN_PARAMS_B: dict[str, float] = {
    "meta-llama/Llama-3.2-3B": 3.2,
    "meta-llama/Llama-3.2-3B-Instruct": 3.2,
    "meta-llama/Llama-3.1-8B": 8.0,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "google/gemma-2-2b": 2.0,
    "google/gemma-2-2b-it": 2.0,
    "google/gemma-2-9b": 9.0,
    "Qwen/Qwen3-1.7B": 1.7,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen3-8B": 8.0,
}
_GPU_VRAM_GB = 22.0  # A10G default


@api_router.post("/wizard/validate-vdb")
async def wizard_validate_vdb(request: Request):
    body = await request.json()
    return JSONResponse(vdb_validator.validate(body))


_UC_ID_SAFE_RE = _re.compile(r'^[a-zA-Z0-9_-]{1,64}$')
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

@api_router.post("/wizard/upload-pdf")
async def wizard_upload_pdf(file: UploadFile, uc_id: str = Form("")):
    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
    safe_uc_id = uc_id if _UC_ID_SAFE_RE.match(uc_id) else "default"
    safe_name = Path(file.filename or "upload.pdf").name or "upload.pdf"
    size_mb = len(content) / (1024 * 1024)
    estimated_chunks = max(1, int(len(content) / 600))
    upload_dir = ROOT / "tmp" / "uploads" / safe_uc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name
    dest.write_bytes(content)
    return JSONResponse({
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "estimated_chunks": estimated_chunks,
        "path": str(dest),
    })


@api_router.post("/wizard/estimate-vram")
async def wizard_estimate_vram(request: Request):
    body = await request.json()
    model_id: str = body.get("model_id", "")
    lora_rank: int = int(body.get("lora_rank", 16))
    params_b = _KNOWN_PARAMS_B.get(model_id)
    if params_b is None:
        m = _re.search(r"(\d+(?:\.\d+)?)[Bb]", model_id.split("/")[-1])
        params_b = float(m.group(1)) if m else None
    if params_b is None:
        return JSONResponse({
            "vram_required_gb": None, "fits": False,
            "fits_with_reduced_batch": False,
            "error": "Unknown model — specify parameter count manually",
        })
    vram = round((params_b * 0.7) + 4.0, 1)
    vram_reduced = round((params_b * 0.7) + 2.5, 1)
    return JSONResponse({
        "vram_required_gb": vram,
        "fits": vram <= _GPU_VRAM_GB,
        "fits_with_reduced_batch": vram_reduced <= _GPU_VRAM_GB,
        "params_billions": params_b,
    })
