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

api_router = APIRouter()


def _uc_path(uc_id: str) -> Path:
    """Return the UC directory path, raising 400 on path traversal attempts."""
    if not uc_id or ".." in uc_id or "/" in uc_id or "\\" in uc_id:
        raise HTTPException(400, "Invalid use case ID")
    return ROOT / "examples" / uc_id


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
                uc_data["kv_lora_version"] = v.get("kv_computed_for_lora_version", 0)
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
                inf = cfg.get("addon_config", {}).get("inference", cfg)
                uc_data["vllm_url"]   = inf.get("vllm_url", "")
                uc_data["vllm_model"] = inf.get("vllm_model", "")
                uc_data["llm_model"]  = inf.get("llm_model", "")
            except Exception:
                pass
        uc_data["has_index"] = version_path.exists()
        faqs_path = ROOT / "examples" / uc["id"] / "faqs.json"
        try:
            import json as _json
            uc_data["has_faqs"] = faqs_path.exists() and bool(_json.loads(faqs_path.read_text()))
        except Exception:
            uc_data["has_faqs"] = False
        uc_cfg_path = ROOT / "examples" / uc["id"] / "uc_config.json"
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                uc_data["gpu_profile_id"] = uc_cfg.get("gpu_profile_id")
                # Surface llm section for A/B defaults
                llm = uc_cfg.get("llm", {})
                if llm.get("local_model"):
                    uc_data["llm_model"] = llm["local_model"]
                if llm.get("vllm_url"):
                    uc_data["vllm_url"] = llm["vllm_url"]
                # vllm_served_model is the --served-model-name used when launching vLLM
                # (separate from the HuggingFace model path in local_model)
                uc_data["vllm_served_model"] = llm.get("vllm_served_model", "kvforge-local")
                uc_data["comparison_provider"] = llm.get("comparison_provider", "anthropic")
                uc_data["comparison_model"]    = llm.get("comparison_model", "claude-haiku-4-5-20251001")
                uc_data["inference_system_prompt"] = llm.get("inference_system_prompt", "")
                # Resolve GPU profile host → derive vllm_url if not explicitly set
                profile_id = uc_cfg.get("gpu_profile_id")
                if profile_id and not uc_data.get("vllm_url"):
                    try:
                        from studio.remote_gpu import list_profiles
                        profile = next((p for p in list_profiles() if p["id"] == profile_id), None)
                        if profile and profile.get("host"):
                            vllm_port = profile.get("vllm_port", 8090)
                            uc_data["vllm_url"] = f"http://{profile['host']}:{vllm_port}/v1"
                            uc_data["gpu_profile_host"] = profile["host"]
                            uc_data["gpu_profile_name"] = profile.get("display_name", profile["host"])
                    except Exception:
                        pass
            except Exception:
                pass
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


# ── UC Archive / Delete ───────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/archive")
def archive_uc(uc_id: str):
    from studio.migration import set_archived_in_registry
    _uc_path(uc_id)  # path-traversal guard (raises 400 if invalid)
    set_archived_in_registry(uc_id, archived=True, root=ROOT)
    return {"status": "archived", "id": uc_id}


@api_router.post("/uc/{uc_id}/unarchive")
def unarchive_uc(uc_id: str):
    from studio.migration import set_archived_in_registry
    _uc_path(uc_id)
    set_archived_in_registry(uc_id, archived=False, root=ROOT)
    return {"status": "active", "id": uc_id}


@api_router.delete("/uc/{uc_id}")
def delete_uc(uc_id: str):
    import shutil
    from studio.migration import remove_from_registry
    uc_dir = _uc_path(uc_id)  # path-traversal guard; raises 400 if invalid
    remove_from_registry(uc_id, root=ROOT)
    if uc_dir.exists():
        shutil.rmtree(uc_dir)
    return {"status": "deleted", "id": uc_id}


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

VALID_STEPS = {"index", "train", "recompute", "prs-eval", "ab-eval", "sleep-faq", "setup"}

@api_router.get("/check-data/{uc_id}")
def check_data(uc_id: str):
    """Return whether the indexed data source exists for a use case."""
    import json as _json

    cfg_path = ROOT / "examples" / uc_id / "config.json"
    loader = "pdf"
    if cfg_path.exists():
        try:
            raw = _json.loads(cfg_path.read_text())
            loader = (
                raw.get("addon_config", {}).get("indexing", {}).get("loader", "pdf")
                or raw.get("loader", "pdf")
            )
        except Exception:
            pass

    uc_dir = ROOT / "examples" / uc_id
    corpus_path = uc_dir / "data" / "corpus.jsonl"
    faq_path = uc_dir / "faqs.json"

    if loader == "pdf":
        data_dir = uc_dir / "data"
        pdf_files = list(data_dir.glob("*.pdf")) if data_dir.exists() else []
        return {
            "loader": loader,
            "corpus_exists": bool(pdf_files),
            "corpus_path": str(pdf_files[0]) if pdf_files else None,
            "faq_exists": faq_path.exists(),
        }
    return {
        "loader": loader,
        "corpus_exists": corpus_path.exists(),
        "corpus_path": str(corpus_path) if corpus_path.exists() else None,
        "faq_exists": faq_path.exists(),
    }


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
    step: str  # "setup" | "index" | "train" | "recompute" | "prs-eval" | "ab-eval" | "sleep-faq"
    gpu_profile_id: str | None = None  # set when user chose a remote GPU profile


@api_router.post("/run-step")
async def run_step(req: RunStepRequest, background_tasks: BackgroundTasks):
    from studio.pipeline_runner import STEP_MODULES, run_step_background
    from studio.activity_log import log_event
    if req.step not in STEP_MODULES:
        raise HTTPException(400, f"Unknown step: {req.step}. Valid: {list(STEP_MODULES)}")
    jm = get_manager()
    try:
        job_id = jm.create(req.uc_id, req.step)
    except DuplicateJobError:
        existing_job_id = jm._uc_locks.get(req.uc_id)
        return JSONResponse(
            {"detail": "already_running", "job_id": existing_job_id},
            status_code=409,
        )
    log_event("pipeline", f"{req.step}.started",
              f"Pipeline step '{req.step}' started for use case '{req.uc_id}'",
              details={"step": req.step, "job_id": job_id, "gpu_profile_id": req.gpu_profile_id},
              uc_id=req.uc_id, severity="info")
    background_tasks.add_task(run_step_background, req.uc_id, req.step, job_id, jm, req.gpu_profile_id)
    return {"job_id": job_id, "uc_id": req.uc_id, "step": req.step}


@api_router.post("/uc/{uc_id}/reattach")
async def reattach_job(uc_id: str, request: Request, background_tasks: BackgroundTasks):
    """Reattach to an already-running nohup job on EC2 without relaunching it.

    Expects JSON body: {"job_id": "<hex>", "step": "<step>", "gpu_profile_id": "<id>"}
    The nohup process must still be running on EC2 (log + pid files present).
    Creates a fresh in-memory job that tails the existing log from the start.
    """
    from studio.pipeline_runner import run_step_background
    body = await request.json()
    job_id = body.get("job_id", "")
    step = body.get("step", "train")
    gpu_profile_id = body.get("gpu_profile_id")
    if not job_id:
        raise HTTPException(400, "job_id required")

    jm = get_manager()
    # Clear any stale lock so we can create a fresh job entry
    with jm._lock:
        jm._uc_locks.pop(uc_id, None)

    new_job_id = jm.create(uc_id, step)

    async def _reattach_tail():
        """Tail the existing remote log without re-launching the nohup process."""
        from studio.pipeline_runner import _log_path, _SSE_HEARTBEAT_INTERVAL
        from studio.remote_gpu import _load_profiles, _load_pkey, _decrypt
        import paramiko, time as _time

        profiles = _load_profiles()
        profile = next((p for p in profiles if p.get("id") == gpu_profile_id), None)
        if not profile:
            jm.fail(new_job_id, f"GPU profile '{gpu_profile_id}' not found")
            return

        pem_content = _decrypt(profile["pem_enc"])
        pkey = _load_pkey(pem_content)
        host, user, port = profile["host"], profile["user"], profile["port"]
        remote_log = f"/tmp/kvforge_{job_id}.log"
        remote_pid = f"/tmp/kvforge_{job_id}.pid"

        def _ssh_connect():
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(hostname=host, username=user, port=port, pkey=pkey, timeout=30)
            return c

        loop = asyncio.get_event_loop()

        def _append(line: str):
            jm.append_log(new_job_id, line)
            for _lp in (_log_path(uc_id), _log_path(uc_id, step)):
                try:
                    _lp.parent.mkdir(parents=True, exist_ok=True)
                    with open(_lp, "a") as _f:
                        _f.write(line + "\n")
                except Exception:
                    pass

        _append(f"[studio] reattached to job {job_id}; log → {remote_log}")

        poll_client = await loop.run_in_executor(None, _ssh_connect)

        def _open_tail():
            tc = _ssh_connect()
            # Tail from beginning so we replay history, then follow
            _, so, _ = tc.exec_command(f"tail -n +1 -F {remote_log} 2>/dev/null",
                                        timeout=None, get_pty=False)
            so.channel.setblocking(False)
            return tc, so

        tail_client, sout = await loop.run_in_executor(None, _open_tail)
        consecutive_poll_errors = 0

        while True:
            try:
                raw = await loop.run_in_executor(None, sout.channel.recv, 4096)
                if raw:
                    for ln in raw.decode("utf-8", errors="replace").splitlines():
                        _append(ln)
                elif sout.channel.closed:
                    raise EOFError("tail channel closed")
            except Exception:
                try:
                    sout.channel.close(); tail_client.close()
                except Exception:
                    pass
                try:
                    tail_client, sout = await loop.run_in_executor(None, _open_tail)
                except Exception:
                    pass

            try:
                _, chk, _ = poll_client.exec_command(
                    f"test -f {remote_pid}.exit && cat {remote_pid}.exit || echo running"
                )
                status_txt = (await loop.run_in_executor(None, chk.read)).decode().strip()
                consecutive_poll_errors = 0
            except Exception:
                consecutive_poll_errors += 1
                status_txt = "running"
                if consecutive_poll_errors >= 10:
                    try:
                        poll_client.close()
                    except Exception:
                        pass
                    try:
                        poll_client = await loop.run_in_executor(None, _ssh_connect)
                        consecutive_poll_errors = 0
                    except Exception:
                        pass

            if status_txt != "running":
                try:
                    exit_code = int(status_txt)
                except ValueError:
                    exit_code = 0
                await asyncio.sleep(1)
                try:
                    raw = await loop.run_in_executor(None, sout.channel.recv, 65536)
                    if raw:
                        for ln in raw.decode("utf-8", errors="replace").splitlines():
                            _append(ln)
                except Exception:
                    pass
                if exit_code == 0:
                    jm.complete(new_job_id, 0)
                    _append("[studio] done (exit 0)")
                else:
                    jm.fail(new_job_id, f"Remote process exited with code {exit_code}")
                    _append(f"[studio] failed (exit {exit_code})")
                break

            await asyncio.sleep(3)

        try:
            sout.channel.close(); tail_client.close()
        except Exception:
            pass
        try:
            poll_client.close()
        except Exception:
            pass

    background_tasks.add_task(_reattach_tail)
    return {"job_id": new_job_id, "reattached_from": job_id, "step": step}


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


@api_router.get("/gpu/remote-stats/{profile_id}")
async def remote_gpu_stats(profile_id: str):
    """SSH to a registered GPU profile and return live nvidia-smi stats."""
    import asyncio
    from studio.remote_gpu import _load_profiles, _decrypt, _make_client, _run_command

    profiles = _load_profiles()
    profile = next((p for p in profiles if p["id"] == profile_id), None)
    if not profile:
        return JSONResponse({"error": "Profile not found", "gpus": []}, status_code=404)

    def _fetch():
        pem = _decrypt(profile["pem_enc"])
        client = _make_client(profile["host"], profile["user"], profile["port"], pem)
        try:
            code, out, _ = _run_command(
                client,
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,power.draw "
                "--format=csv,noheader,nounits",
            )
            if code != 0:
                return {"error": "nvidia-smi not available on this host", "gpus": []}

            _, proc_out, _ = _run_command(
                client,
                "nvidia-smi --query-compute-apps=pid,used_memory,name "
                "--format=csv,noheader,nounits 2>/dev/null || true",
            )
        finally:
            client.close()

        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                gpus.append({
                    "id":       int(parts[0]),
                    "name":     parts[1],
                    "used_gb":  round(int(parts[2]) / 1024, 2),
                    "total_gb": round(int(parts[3]) / 1024, 2),
                    "util_pct": int(parts[4]),
                    "temp_c":   int(parts[5]),
                    "power_w":  round(float(parts[6]), 1),
                    "status":   "busy" if int(parts[4]) > 10 else "free",
                    "processes": [],
                })
            except (ValueError, IndexError):
                pass

        procs: list[dict] = []
        for line in proc_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    procs.append({"pid": int(parts[0]), "used_mem_mb": int(parts[1]), "name": parts[2]})
                except (ValueError, IndexError):
                    pass
        if gpus and procs:
            gpus[0]["processes"] = procs

        return {
            "gpus": gpus,
            "has_free_gpu": any(g["status"] == "free" for g in gpus),
            "host": profile["host"],
            "display_name": profile.get("display_name", profile["host"]),
        }

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_fetch), timeout=12)
        return JSONResponse(result)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "SSH timed out — check host reachability", "gpus": []})
    except Exception as e:
        return JSONResponse({"error": str(e)[:120], "gpus": []})


# ── Compute worker health ──────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/compute-worker/health")
async def compute_worker_health(uc_id: str):
    """Probe the remote compute worker configured for this UC."""
    cfg_path = _uc_path(uc_id) / "config.json"
    if not cfg_path.exists():
        return JSONResponse({"status": "no_config", "reachable": False})
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return JSONResponse({"status": "error", "reachable": False, "error": "invalid config"})

    compute_cfg = cfg.get("addon_config", {}).get("compute", {})
    if compute_cfg.get("backend", "local") != "remote":
        return JSONResponse({"status": "local", "reachable": None, "backend": "local"})

    worker_url = compute_cfg.get("worker_url", "").rstrip("/")
    if not worker_url:
        return JSONResponse({"status": "no_url", "reachable": False})

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{worker_url}/health")
        if resp.status_code == 200:
            return JSONResponse({"reachable": True, "worker_url": worker_url, **resp.json()})
        return JSONResponse({"reachable": False, "worker_url": worker_url, "http_status": resp.status_code})
    except Exception as e:
        return JSONResponse({"reachable": False, "worker_url": worker_url, "error": str(e)[:120]})


# ── Job logs ───────────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/logs")
def uc_logs_endpoint(uc_id: str, step: str | None = None):
    """Return logs for a UC run.

    If *step* is given (e.g. ``?step=train``), reads from the per-step
    persistent log (``logs/{step}.log``).  Without *step*, returns the most
    recent run log from the in-memory job manager or ``last_run.log``.
    """
    _uc_path(uc_id)  # path traversal guard
    from studio.pipeline_runner import _log_path

    # Per-step historical log requested
    if step:
        log_file = _log_path(uc_id, step)
        if log_file.exists():
            try:
                lines = log_file.read_text().splitlines()
                status = "failed" if lines and "[studio] failed" in lines[-1] else "done"
                return JSONResponse({"lines": lines, "status": status, "step": step, "job_id": None})
            except OSError:
                pass
        return JSONResponse({"lines": [], "status": None, "step": step, "job_id": None})

    # Most-recent run: check in-memory job manager first
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
    log_file = _log_path(uc_id)
    if log_file.exists():
        try:
            lines = log_file.read_text().splitlines()
            step_parsed = None
            status = "done"
            if lines and lines[0].startswith("[studio] step="):
                for p in lines[0].split():
                    if p.startswith("step="):
                        step_parsed = p[5:]
            if lines and "[studio] failed" in lines[-1]:
                status = "failed"
            return JSONResponse({"lines": lines, "status": status, "step": step_parsed, "job_id": None})
        except OSError:
            pass
    return JSONResponse({"lines": [], "status": None, "step": None, "job_id": None})


# ── Step summaries ────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/step-summaries")
def step_summaries_endpoint(uc_id: str):
    """Return a compact summary dict for each completed pipeline step."""
    import re as _re
    uc_dir = _uc_path(uc_id)

    cfg, ver, uc_cfg = {}, {}, {}
    for path, dest in [(uc_dir / "config.json", "cfg"),
                       (uc_dir / "version.json", "ver"),
                       (uc_dir / "uc_config.json", "uc_cfg")]:
        if path.exists():
            try:
                locals()[dest]  # noqa — reassign below
            except Exception:
                pass
    if (uc_dir / "config.json").exists():
        try: cfg = json.loads((uc_dir / "config.json").read_text())
        except Exception: pass
    if (uc_dir / "version.json").exists():
        try: ver = json.loads((uc_dir / "version.json").read_text())
        except Exception: pass
    if (uc_dir / "uc_config.json").exists():
        try: uc_cfg = json.loads((uc_dir / "uc_config.json").read_text())
        except Exception: pass

    indexing = cfg.get("addon_config", {}).get("indexing", cfg)
    inference = cfg.get("addon_config", {}).get("inference", cfg)
    training  = cfg.get("addon_config", {}).get("training", cfg)

    # ── Chunk count from VectorDB (fast native count) ─────────────────────────
    chunk_count = None
    collection = cfg.get("collection")
    if collection:
        try:
            from vectorstore.registry import get_store
            store = get_store(cfg)
            chunk_count = store.count(collection)
        except Exception:
            pass

    # ── Training loss from log ─────────────────────────────────────────────────
    training_loss = None
    train_log = uc_dir / "logs" / "train.log"
    if train_log.exists():
        try:
            matches = _re.findall(r"'train_loss':\s*'([0-9.]+)'", train_log.read_text())
            if matches:
                training_loss = float(matches[-1])
        except Exception:
            pass

    # ── FAQ count ─────────────────────────────────────────────────────────────
    faq_count = 0
    faqs_path = uc_dir / "faqs.json"
    if faqs_path.exists():
        try: faq_count = len(json.loads(faqs_path.read_text()))
        except Exception: pass

    llm = uc_cfg.get("llm", {})
    faq_provider = llm.get("sleep_faq_provider") or "claude"
    faq_model    = llm.get("sleep_faq_model") or "claude-haiku-4-5-20251001"
    # Shorten model name for display (strip org prefix)
    faq_model_short = faq_model.split("/")[-1] if "/" in faq_model else faq_model

    prs_history = ver.get("prs_history", [])
    last_prs = prs_history[-1] if prs_history else None

    # Shorten LLM model name for display
    llm_model = inference.get("llm_model", "")
    llm_model_short = llm_model.split("/")[-1] if "/" in llm_model else llm_model

    embed_model = indexing.get("embed_model", "")
    embed_model_short = embed_model.split("/")[-1] if "/" in embed_model else embed_model

    return JSONResponse({
        "index": {
            "chunk_count": chunk_count,
            "embed_model": embed_model_short,
            "vector_store": indexing.get("vector_store", "qdrant"),
            "chunk_size": indexing.get("chunk_size"),
        },
        "recompute": {
            "model": llm_model_short,
            "kv_version": ver.get("kv_computed_for_lora_version", 0),
            "chunk_count": chunk_count,
        },
        "faq-gen-cloud": {
            "count": faq_count,
            "provider": faq_provider,
            "model": faq_model_short,
            "per_chunk": round(faq_count / chunk_count, 2) if chunk_count else None,
        },
        "train": {
            "lora_version": ver.get("current_lora_version", 0),
            "lora_rank": training.get("lora_rank", 16),
            "lora_alpha": training.get("lora_alpha", 32),
            "lora_epochs": training.get("lora_epochs", 3),
            "training_loss": training_loss,
        },
        "prs-eval": {
            "prs": last_prs["prs"] if last_prs else None,
            "round": last_prs["round"] if last_prs else None,
            "samples": uc_cfg.get("prs_eval_sample", 20),
        },
    })


# ── LoRA training recommendations ─────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/training-recommendations")
def training_recommendations_endpoint(uc_id: str):
    """Return auto-derived LoRA hyperparameter recommendations with reasoning."""
    uc_dir = _uc_path(uc_id)
    cfg, ver = {}, {}
    if (uc_dir / "config.json").exists():
        try: cfg = json.loads((uc_dir / "config.json").read_text())
        except Exception: pass
    if (uc_dir / "version.json").exists():
        try: ver = json.loads((uc_dir / "version.json").read_text())
        except Exception: pass

    n_faqs = 0
    faqs_path = uc_dir / "faqs.json"
    if faqs_path.exists():
        try: n_faqs = len(json.loads(faqs_path.read_text()))
        except Exception: pass

    from core.auto_config import recommend
    recs = recommend(
        cfg=cfg,
        n_faqs=n_faqs,
        prs_history=ver.get("prs_history", []),
        lora_version=ver.get("current_lora_version", 0),
    )
    return JSONResponse(recs)


@api_router.post("/uc/{uc_id}/training-config")
async def save_training_config(uc_id: str, request: Request):
    """Persist LoRA hyperparameters to config.json addon_config.training."""
    cfg_path = _uc_path(uc_id) / "config.json"
    if not cfg_path.exists():
        raise HTTPException(404, f"config.json not found for {uc_id}")
    cfg = json.loads(cfg_path.read_text())
    updates = await request.json()
    allowed = {"lora_epochs", "lora_rank", "lora_alpha", "lora_lr",
               "lora_dropout", "lora_target_modules", "train_batch_size"}
    training = cfg.setdefault("addon_config", {}).setdefault("training", {})
    for k, v in updates.items():
        if k in allowed:
            training[k] = v
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return {"status": "saved"}


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


# ── Remote GPU ─────────────────────────────────────────────────────────────────

@api_router.post("/remote-gpu/session")
async def create_remote_gpu_session(request: Request):
    """Accept connection params + PEM content; return session_id for streaming."""
    from studio.remote_gpu import create_session
    body = await request.json()
    host = str(body.get("host", "")).strip()
    user = str(body.get("user", "ec2-user")).strip()
    port = int(body.get("port", 22))
    pem_key = str(body.get("pem_key", "")).strip()
    display_name = str(body.get("display_name", host)).strip()
    if not host or not pem_key:
        raise HTTPException(400, "host and pem_key are required")
    try:
        session_id = create_session(host, user, port, pem_key, display_name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"session_id": session_id})


@api_router.get("/remote-gpu/session/{session_id}/test-stream")
async def remote_gpu_test_stream(session_id: str):
    """SSE: SSH connect + nvidia-smi verification."""
    import asyncio
    from fastapi.responses import StreamingResponse
    from studio.remote_gpu import stream_test_connection

    async def gen():
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None, lambda: list(stream_test_connection(session_id))
        )
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api_router.get("/remote-gpu/session/{session_id}/setup-stream")
async def remote_gpu_setup_stream(session_id: str):
    """SSE: install KVForge dependencies on remote host."""
    import asyncio
    from fastapi.responses import StreamingResponse
    from studio.remote_gpu import stream_setup_gpu

    async def gen():
        loop = asyncio.get_event_loop()
        events = await loop.run_in_executor(
            None, lambda: list(stream_setup_gpu(session_id))
        )
        for ev in events:
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@api_router.post("/remote-gpu/session/{session_id}/save")
async def remote_gpu_save_profile(session_id: str, request: Request):
    """Persist the verified connection as an encrypted profile."""
    from studio.remote_gpu import get_session, save_profile
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    body = await request.json()
    display_name = str(body.get("display_name", sess["display_name"])).strip() or sess["host"]
    profile = save_profile(
        profile_id=session_id,
        host=sess["host"],
        user=sess["user"],
        port=sess["port"],
        display_name=display_name,
        pem_key=sess["pem_key"],
        fingerprint=sess.get("fingerprint") or "",
    )
    return JSONResponse(profile)


@api_router.get("/remote-gpu/profiles")
def list_remote_gpu_profiles():
    from studio.remote_gpu import list_profiles
    return JSONResponse(list_profiles())


@api_router.delete("/remote-gpu/profiles/{profile_id}")
def delete_remote_gpu_profile(profile_id: str):
    from studio.remote_gpu import delete_profile
    if not delete_profile(profile_id):
        raise HTTPException(404, "Profile not found")
    return JSONResponse({"ok": True})


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


# ── Resource Providers ────────────────────────────────────────────────────────

@api_router.get("/resources")
def list_resources(provider_type: str | None = None):
    from studio.resource_registry import list_providers
    return list_providers(provider_type)


@api_router.post("/resources")
async def create_resource(request: Request):
    from studio.resource_registry import create_provider, PROVIDER_TYPES, BACKENDS
    from studio.activity_log import log_event
    body = await request.json()
    missing = [f for f in ("type", "backend", "display_name") if not body.get(f)]
    if missing:
        return JSONResponse({"detail": f"missing fields: {missing}"}, status_code=400)
    if body["type"] not in PROVIDER_TYPES:
        return JSONResponse({"detail": f"type must be one of {PROVIDER_TYPES}"}, status_code=400)
    try:
        rec = create_provider(
            provider_type=body["type"],
            backend=body["backend"],
            display_name=body["display_name"],
            config=body.get("config", {}),
        )
        log_event("resource", "resource.created",
                  f"Resource provider '{rec['display_name']}' ({rec['backend']}) added",
                  details={"id": rec["id"], "type": rec["type"], "backend": rec["backend"]},
                  severity="success")
        return rec
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)


@api_router.put("/resources/{provider_id}")
async def update_resource(provider_id: str, request: Request):
    from studio.resource_registry import update_provider, get_provider
    from studio.activity_log import log_event
    body = await request.json()
    try:
        rec = update_provider(provider_id, **{k: body[k] for k in ("display_name", "config", "backend") if k in body})
        log_event("resource", "resource.updated",
                  f"Resource provider '{rec.get('display_name', provider_id)}' updated",
                  details={"id": provider_id, "fields": list(body.keys())},
                  severity="info")
        return rec
    except KeyError:
        return JSONResponse({"detail": "not found"}, status_code=404)


@api_router.delete("/resources/{provider_id}")
def delete_resource(provider_id: str):
    from studio.resource_registry import delete_provider, get_provider
    from studio.activity_log import log_event
    p = get_provider(provider_id)
    if p is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    log_event("resource", "resource.deleted",
              f"Resource provider '{p.get('display_name', provider_id)}' ({p.get('backend', '')}) removed",
              details={"id": provider_id, "type": p.get("type"), "backend": p.get("backend")},
              severity="warning")
    delete_provider(provider_id)
    return {"ok": True}


@api_router.post("/resources/{provider_id}/test")
async def test_resource(provider_id: str):
    from studio.resource_registry import test_provider, get_provider
    from studio.activity_log import log_event
    p = get_provider(provider_id)
    result = await __import__("asyncio").to_thread(test_provider, provider_id)
    if p:
        sev = "success" if result.get("ok") else "error"
        msg = f"Connectivity test for '{p.get('display_name', provider_id)}': " + \
              (result.get("detail", "OK") if result.get("ok") else result.get("error", "failed"))
        log_event("resource", "resource.tested", msg,
                  details={"id": provider_id, "backend": p.get("backend"), "result": result},
                  severity=sev)
    return result


# ── Activity Logs ──────────────────────────────────────────────────────────────

@api_router.get("/logs")
def get_logs(
    categories: str | None = None,
    severities: str | None = None,
    since: str | None = None,
    until: str | None = None,
    search: str | None = None,
    uc_id: str | None = None,
    limit: int = 500,
):
    from studio.activity_log import query_logs
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    sev_list = [s.strip() for s in severities.split(",")] if severities else None
    return query_logs(
        categories=cat_list,
        severities=sev_list,
        since=since,
        until=until,
        search=search or None,
        uc_id=uc_id or None,
        limit=min(limit, 2000),
    )


@api_router.get("/logs/stats")
def get_logs_stats():
    from studio.activity_log import get_stats
    return get_stats()


@api_router.get("/uc/{uc_id}/active-remote-job")
async def detect_active_remote_job(uc_id: str):
    """SSH into the remote GPU and find any live nohup kvforge jobs for this UC.

    Returns {"job_id": "<hex>", "step": "<step>", "gpu_profile_id": "<id>"}
    or {"job_id": null} if nothing is running.
    """
    from studio.remote_gpu import _load_profiles, _load_pkey, _decrypt
    import paramiko

    uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
    if not uc_cfg_path.exists():
        return JSONResponse({"job_id": None, "reason": "no uc_config"})

    uc_cfg = json.loads(uc_cfg_path.read_text())
    profile_id = uc_cfg.get("gpu_profile_id") or uc_cfg.get("remote_gpu_profile_id")
    if not profile_id:
        return JSONResponse({"job_id": None, "reason": "no gpu_profile configured"})

    profiles = _load_profiles()
    profile = next((p for p in profiles if p.get("id") == profile_id), None)
    if not profile:
        return JSONResponse({"job_id": None, "reason": "profile not found"})

    try:
        pem_content = _decrypt(profile["pem_enc"])
        pkey = _load_pkey(pem_content)
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(hostname=profile["host"], username=profile["user"],
                  port=profile["port"], pkey=pkey, timeout=15)

        # Find pid files for this uc_id whose process is still alive
        _, out, _ = c.exec_command(
            "for f in /tmp/kvforge_*.pid; do "
            "  [ -f \"$f\" ] || continue; "
            "  pid=$(cat \"$f\" 2>/dev/null); "
            "  kill -0 \"$pid\" 2>/dev/null && echo \"$f\"; "
            "done"
        )
        alive = out.read().decode().strip()
        c.close()

        if not alive:
            return JSONResponse({"job_id": None, "reason": "no running nohup jobs found"})

        # Pick the most recent pid file
        pid_file = alive.splitlines()[-1].strip()
        import re
        m = re.search(r"kvforge_([0-9a-f]+)\.pid", pid_file)
        if not m:
            return JSONResponse({"job_id": None, "reason": "could not parse job_id from pid file"})

        job_id = m.group(1)
        return JSONResponse({"job_id": job_id, "step": "train", "gpu_profile_id": profile_id})
    except Exception as e:
        return JSONResponse({"job_id": None, "reason": str(e)})
