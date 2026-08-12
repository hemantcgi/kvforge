# studio/pipeline_runner.py
"""Spawns pipeline subprocesses and streams their output as SSE events."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Maps step name → module path relative to project root
STEP_MODULES = {
    "index":     "pipeline.kv_indexer",
    "train":     "pipeline.lora_trainer",
    "recompute": "pipeline.kv_indexer",
    "prs-eval":  "pipeline.prs_evaluator",
    "ab-eval":   "pipeline.ab_evaluator",
    "sleep-faq": "pipeline.sleep_faq_generator",
    "faq-gen-cloud": "pipeline.sleep_faq_generator",
    "setup": "__setup__",  # handled specially in _build_cmd
}

# Steps that require a free GPU — sleep-faq calls a cloud REST API, no GPU needed
GPU_REQUIRED_STEPS = {"index", "train", "recompute", "prs-eval"}

# Static extra args per step (subcommands/flags appended after --config)
STEP_EXTRA_ARGS = {
    "index": ["index"],
    # recompute uses compute-kv; --stale-version is added dynamically in _build_cmd
    "recompute": ["compute-kv"],
}


def _ensure_config_json(uc_id: str) -> None:
    """Generate config.json from uc_config.json if it doesn't exist.

    Pipeline modules (kv_indexer, lora_trainer, etc.) all expect the
    addon_config-format config.json.  Wizard-created UCs only have
    uc_config.json, so we derive config.json automatically.
    """
    config_path = ROOT / "examples" / uc_id / "config.json"
    if config_path.exists():
        return  # already present (example UCs or previously generated)

    uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
    if not uc_cfg_path.exists():
        return  # nothing to derive from

    try:
        uc = json.loads(uc_cfg_path.read_text())
    except Exception:
        return

    data = uc.get("data", {})
    vdb = uc.get("vectordb", {})
    llm = uc.get("llm", {})

    source_type = data.get("source_type", "")
    loader_map = {
        "pdf": "pdf", "huggingface": "huggingface", "jsonl": "jsonl",
        "directory": "directory", "markdown": "markdown", "html": "html",
        "api": "jsonl",  # connector-synced data lands as jsonl
    }
    loader = loader_map.get(source_type, "jsonl")

    local_model = llm.get("local_model", "google/gemma-4-E2B-it")
    collection = uc_id.replace("_", "-")

    cfg = {
        "use_case_name": uc.get("display_name", uc_id),
        "collection": collection,
        "version_file": f"examples/{uc_id}/version.json",
        "addons": ["indexing", "inference", "training", "background", "sync", "monitoring"],
        "addon_config": {
            "indexing": {
                "loader": loader,
                "chunk_size": vdb.get("chunk_size", 512),
                "chunk_overlap": vdb.get("chunk_overlap", 64),
                "embed_batch": 64,
                "upsert_batch": 128,
                "embed_model": vdb.get("embedding_model", "BAAI/bge-small-en-v1.5"),
                "embedder_backend": "fastembed",
                "vector_dim": vdb.get("dimensions", 384),
                "vector_store": vdb.get("store", "qdrant"),
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
                "dataset_id": data.get("dataset_id", ""),
                "split": data.get("split", "train"),
                "jsonl_text_key": data.get("text_column", "text"),
                "max_rows": data.get("max_rows", 5000),
                "source_path": data.get("source_path", f"examples/{uc_id}/data/"),
                # HuggingFace-specific keys (used by ingestion/registry.py when loader=huggingface)
                **({"hf_config_name": data.get("hf_config_name"),
                    "hf_split": data.get("split", "train"),
                    "hf_text_column": data.get("text_column", "text"),
                    "hf_max_rows": data.get("max_rows", 5000)} if loader == "huggingface" else {}),
                "model_library": {
                    local_model: {"kv_num_layers": 28, "kv_num_heads": 8, "kv_head_dim": 128}
                },
            },
            "inference": {
                "top_k": 5,
                "llm_model": local_model,
                "quantization": llm.get("quantization", "4bit"),
                "vllm_url": llm.get("vllm_url", ""),
                "vllm_model": uc_id,
                "max_new_tokens": 256,
                "gate_threshold": 0.75,
            },
            "training": {
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
                "lora_dropout": 0.05,
                "lora_epochs": 3,
                "lora_lr": 0.0002,
                "checkpoint_dir": f"examples/{uc_id}/lora_checkpoints/",
                "replay_db": f"examples/{uc_id}/replay.db",
                "prs_threshold": 0.75,
                "prs_weights": {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2},
                "prs_advancement_threshold": 0.72,
                "prs_regression_threshold": 0.60,
                "faq_question_key": "question",
                "faq_answer_key": "answer",
            },
            "background": {"flush_seconds": 300, "flush_queries": 50},
            "sync": {
                "interval_minutes": 1440,
                "hitl_mode": "auto",
                "sync_regression_mode": "pct",
                "sync_regression_pct_threshold": 0.10,
                "sync_regression_tier_threshold": 0.15,
            },
            "monitoring": {"port": 8085},
        },
    }

    # Ensure data directory exists for connector-synced / local sources
    data_dir = ROOT / "examples" / uc_id / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    config_path.write_text(json.dumps(cfg, indent=2))


def _has_remote_compute_backend(uc_id: str) -> bool:
    """Return True if this UC's config.json uses the remote compute backend."""
    cfg_path = ROOT / "examples" / uc_id / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
        return cfg.get("addon_config", {}).get("compute", {}).get("backend") == "remote"
    except Exception:
        return False


def _build_cmd(uc_id: str, step: str) -> list[str]:
    if step == "setup":
        setup_script = str(ROOT / "examples" / uc_id / "setup.py")
        return [sys.executable, setup_script]
    module = STEP_MODULES[step]
    config = str(ROOT / "examples" / uc_id / "config.json")
    cmd = [sys.executable, "-m", module, "--config", config]
    cmd += STEP_EXTRA_ARGS.get(step, [])
    if step == "recompute":
        # Pass stale-version = current_lora_version + 1 so that chunks already
        # at kv_version == current_lora_version are also recomputed (handles the
        # case where a new checkpoint overwrites an existing version number).
        version_path = ROOT / "examples" / uc_id / "version.json"
        if version_path.exists():
            try:
                ver = json.loads(version_path.read_text())
                lora_ver = int(ver.get("current_lora_version", 1))
                cmd += ["--stale-version", str(lora_ver + 1)]
            except Exception:
                cmd += ["--stale-version", "2"]
        else:
            cmd += ["--stale-version", "2"]
    if step == "train":
        # Pass --faqs if faqs.json exists, otherwise fall back to --source-file
        faqs_path = ROOT / "examples" / uc_id / "faqs.json"
        if faqs_path.exists():
            cmd += ["--faqs", str(faqs_path)]
        else:
            source_path = ROOT / "examples" / uc_id / "data" / "train.jsonl"
            cmd += ["--source-file", str(source_path)]
    if step == "prs-eval":
        faqs_path = ROOT / "examples" / uc_id / "faqs.json"
        if faqs_path.exists():
            cmd += ["--faqs", str(faqs_path)]
        else:
            source_path = ROOT / "examples" / uc_id / "data" / "train.jsonl"
            cmd += ["--faqs", str(source_path)]
        # Pass explicit --sample only if prs_eval_sample is set in uc_config;
        # otherwise let prs_evaluator auto-compute max(10% of FAQs, 100)
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                raw = uc_cfg.get("prs_eval_sample")
                if raw is not None:
                    parsed = int(raw)
                    if parsed > 0:
                        cmd += ["--sample", str(parsed)]
            except Exception:
                pass
    if step in ("sleep-faq", "faq-gen-cloud"):
        output = str(ROOT / "examples" / uc_id / "faqs.json")
        cmd += ["--output", output]
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        faq_llm = {}
        if uc_cfg_path.exists():
            try:
                faq_llm = json.loads(uc_cfg_path.read_text()).get("llm", {})
            except Exception:
                pass
        # --count: only pass if > 0 (0 = no limit, cover all chunks)
        try:
            cap = int(faq_llm.get("sleep_faq_count", 0))
            if cap > 0:
                cmd += ["--count", str(cap)]
        except (TypeError, ValueError):
            pass
        # --n-per-chunk
        try:
            npc = int(faq_llm.get("sleep_faq_n_per_chunk", 3))
            cmd += ["--n-per-chunk", str(max(1, npc))]
        except (TypeError, ValueError):
            cmd += ["--n-per-chunk", "3"]
        # --delay
        try:
            delay = float(faq_llm.get("sleep_faq_delay", 2.0))
            cmd += ["--delay", str(max(0.0, delay))]
        except (TypeError, ValueError):
            cmd += ["--delay", "2.0"]
    if step == "ab-eval":
        # ab_evaluator requires --dashboard-url pointing to the per-UC monitoring dashboard
        cfg_path = ROOT / "examples" / uc_id / "config.json"
        port = 8081  # fallback
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                port = int(cfg.get("dashboard_port", 8081))
            except Exception:
                pass
        cmd += ["--dashboard-url", f"http://localhost:{port}"]
    return cmd


_SECRET_FLAGS = {"--api-key", "--token", "--password", "--secret"}


def _redact_cmd(cmd: list[str]) -> str:
    parts = []
    skip_next = False
    for part in cmd:
        if skip_next:
            parts.append("[REDACTED]")
            skip_next = False
        elif part in _SECRET_FLAGS:
            parts.append(part)
            skip_next = True
        else:
            parts.append(part)
    return " ".join(parts)


def _log_path(uc_id: str, step: str | None = None) -> Path:
    """Disk log path for a UC run.

    If *step* is given, returns the per-step log (``logs/{step}.log``) which
    is preserved across runs.  Without *step*, returns ``last_run.log`` which
    is always the most-recent run regardless of step.
    """
    if step:
        logs_dir = ROOT / "examples" / uc_id / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / f"{step}.log"
    return ROOT / "examples" / uc_id / "last_run.log"


# Files to sync TO EC2 per step (relative to UC dir)
_REMOTE_SYNC_TO: dict[str, list[str]] = {
    "train":    ["config.json", "faqs.json", "replay.db"],
    "prs-eval": ["config.json", "faqs.json"],
    "recompute":["config.json"],
    "index":    ["config.json"],
}
# Files/dirs to sync BACK from EC2 per step (relative to UC dir)
_REMOTE_SYNC_FROM: dict[str, list[str]] = {
    "train":    ["lora_checkpoints/", "version.json", "replay.db"],
    "prs-eval": ["version.json"],
}
_EC2_REPO   = "/home/ubuntu/kvforge"
_EC2_PYTHON = "/home/ubuntu/kvforge-env/bin/python"


async def _run_step_remote_ssh(
    uc_id: str, step: str, job_id: str, job_manager,
    profile_id: str, _append
) -> None:
    """Run a GPU pipeline step on a remote EC2 node via SSH."""
    import subprocess as _sp
    from studio.remote_gpu import _load_profiles, _load_pkey, _get_pem, _decrypt

    # Load profile (use _load_profiles to get pem_enc, then decrypt)
    profiles = _load_profiles()
    profile = next((p for p in profiles if p["id"] == profile_id), None)
    if not profile:
        raise RuntimeError(f"Remote GPU profile '{profile_id}' not found")

    pem_content = _decrypt(profile["pem_enc"])

    host = profile["host"]
    user = profile["user"]
    port = profile["port"]
    uc_local  = ROOT / "examples" / uc_id
    uc_remote = f"{_EC2_REPO}/examples/{uc_id}"

    # Write PEM to a temp file for rsync/ssh
    import tempfile, stat
    pem_file = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    pem_file.write(pem_content.encode())
    pem_file.close()
    os.chmod(pem_file.name, stat.S_IRUSR | stat.S_IWUSR)

    ssh_opts = ["-i", pem_file.name, "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-p", str(port)]

    try:
        # ── 1. Ensure remote UC dir exists ────────────────────────────────────
        _append(f"[remote] syncing files to {host}:{uc_remote}")
        _sp.run(["ssh"] + ssh_opts + [f"{user}@{host}", f"mkdir -p {uc_remote}"],
                capture_output=True)

        # ── 2. Rsync input files to EC2 ───────────────────────────────────────
        for rel in _REMOTE_SYNC_TO.get(step, ["config.json"]):
            local_path = uc_local / rel
            if local_path.exists():
                _sp.run(["rsync", "-az", "-e", "ssh " + " ".join(ssh_opts),
                         str(local_path), f"{user}@{host}:{uc_remote}/{rel}"],
                        capture_output=True)

        # ── 3. Build remote command ────────────────────────────────────────────
        remote_config = f"{uc_remote}/config.json"
        remote_cmd = f"{_EC2_PYTHON} -m {STEP_MODULES[step]} --config {remote_config}"
        for arg in STEP_EXTRA_ARGS.get(step, []):
            remote_cmd += f" {arg}"
        if step == "recompute":
            version_path = uc_local / "version.json"
            lora_ver = 1
            if version_path.exists():
                try:
                    lora_ver = int(json.loads(version_path.read_text()).get("current_lora_version", 1))
                except Exception:
                    pass
            remote_cmd += f" --stale-version {lora_ver + 1}"
        if step in ("train", "prs-eval"):
            faqs_remote = f"{uc_remote}/faqs.json"
            remote_cmd += f" --faqs {faqs_remote}"
        if step == "prs-eval":
            # Let prs_evaluator auto-compute max(10% of FAQs, 100) unless overridden
            try:
                uc_cfg = json.loads((ROOT / "examples" / uc_id / "uc_config.json").read_text())
                override = uc_cfg.get("prs_eval_sample")
                if override is not None:
                    remote_cmd += f" --sample {int(override)}"
            except Exception:
                pass

        _append(f"[remote] running on {host}: {remote_cmd}")

        # ── 4. Launch via nohup so the job survives SSH disconnection ─────────
        #
        # Strategy: one nohup launch that writes exit code to a sentinel file.
        # A separate persistent SSH connection tails the log; a second persistent
        # connection polls the sentinel every 3 s.  If the tail channel dies we
        # reconnect it rather than failing the job — the nohup process on EC2
        # keeps running regardless of Studio/SSH state.
        import paramiko
        pkey = _load_pkey(pem_content)

        def _ssh_connect():
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(hostname=host, username=user, port=port, pkey=pkey, timeout=30)
            return c

        loop = asyncio.get_event_loop()
        remote_log = f"/tmp/kvforge_{job_id}.log"
        remote_pid = f"/tmp/kvforge_{job_id}.pid"

        # Single launch with exit-code sentinel tracking.
        wrap_cmd = (
            f"cd {_EC2_REPO} && "
            f"nohup bash -c '{remote_cmd} > {remote_log} 2>&1; echo $? > {remote_pid}.exit' & "
            f"echo $! > {remote_pid} && echo started"
        )
        launch_client = await loop.run_in_executor(None, _ssh_connect)
        # Remove stale pid/log from any previous attempt for this job_id
        launch_client.exec_command(f"rm -f {remote_log} {remote_pid} {remote_pid}.exit")
        await asyncio.sleep(0.3)
        _, lout, _ = launch_client.exec_command(wrap_cmd)
        launch_out = await loop.run_in_executor(None, lout.read)
        launch_client.close()
        if b"started" not in launch_out:
            raise RuntimeError(f"Failed to launch remote job: {launch_out!r}")
        _append(f"[remote] job running (nohup); log → {remote_log}")

        # Persistent connection for exit-status polling (reused every 3 s).
        poll_client = await loop.run_in_executor(None, _ssh_connect)

        def _open_tail():
            """Open a fresh tail -F channel; return (client, channel_stdout)."""
            tc = _ssh_connect()
            _, so, _ = tc.exec_command(f"tail -F {remote_log} 2>/dev/null",
                                        timeout=None, get_pty=False)
            so.channel.setblocking(False)
            return tc, so

        tail_client, sout = await loop.run_in_executor(None, _open_tail)

        exit_code = None
        consecutive_poll_errors = 0
        while True:
            # Read available log bytes without blocking; reconnect tail on EOF/error.
            try:
                raw = await loop.run_in_executor(None, sout.channel.recv, 4096)
                if raw:
                    for ln in raw.decode("utf-8", errors="replace").splitlines():
                        _append(ln)
                elif sout.channel.closed:
                    raise EOFError("tail channel closed")
            except Exception:
                # Tail channel died — reconnect silently; job continues on EC2.
                try:
                    sout.channel.close()
                    tail_client.close()
                except Exception:
                    pass
                try:
                    tail_client, sout = await loop.run_in_executor(None, _open_tail)
                except Exception:
                    pass

            # Poll exit sentinel using the persistent poll_client.
            try:
                _, chk, _ = poll_client.exec_command(
                    f"test -f {remote_pid}.exit && cat {remote_pid}.exit || echo running"
                )
                status_raw = await loop.run_in_executor(None, chk.read)
                status_txt = status_raw.decode().strip()
                consecutive_poll_errors = 0
            except Exception:
                consecutive_poll_errors += 1
                status_txt = "running"
                if consecutive_poll_errors >= 10:
                    # Re-establish poll connection after repeated failures.
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
                # Drain remaining log output
                await asyncio.sleep(1)
                try:
                    raw = await loop.run_in_executor(None, sout.channel.recv, 65536)
                    if raw:
                        for ln in raw.decode("utf-8", errors="replace").splitlines():
                            _append(ln)
                except Exception:
                    pass
                break

            await asyncio.sleep(3)

        sout.channel.close()
        tail_client.close()
        try:
            poll_client.close()
        except Exception:
            pass

        # Cleanup remote temp files
        try:
            cl = await loop.run_in_executor(None, _ssh_connect)
            cl.exec_command(f"rm -f {remote_log} {remote_pid} {remote_pid}.exit")
            cl.close()
        except Exception:
            pass

        # ── 5. Rsync results back ─────────────────────────────────────────────
        sync_back = _REMOTE_SYNC_FROM.get(step, [])
        if sync_back:
            _append(f"[remote] syncing results back from {host}")

        # Capture local version.json BEFORE sync so we can preserve phase
        ver_path = uc_local / "version.json"
        local_phase_before = 1
        if "version.json" in sync_back and ver_path.exists():
            try:
                local_phase_before = json.loads(ver_path.read_text()).get("phase", 1)
            except Exception:
                pass

        for rel in sync_back:
            remote_path = f"{user}@{host}:{uc_remote}/{rel}"
            local_dest  = str(uc_local / rel)
            _sp.run(["rsync", "-az", "-e", "ssh " + " ".join(ssh_opts),
                     remote_path, local_dest],
                    capture_output=True)

        # After sync, restore local phase — EC2 starts fresh and doesn't track
        # KV indexing phase advances that happened locally.
        if "version.json" in sync_back and ver_path.exists():
            try:
                synced = json.loads(ver_path.read_text())
                if synced.get("phase", 1) < local_phase_before:
                    synced["phase"] = local_phase_before
                    ver_path.write_text(json.dumps(synced, indent=2))
            except Exception:
                pass

        if exit_code == 0:
            job_manager.complete(job_id, 0)
            _append("[studio] done (exit 0)")
        else:
            job_manager.fail(job_id, f"Remote process exited with code {exit_code}")
            _append(f"[studio] failed (exit {exit_code})")

    finally:
        os.unlink(pem_file.name)


async def run_step_background(
    uc_id: str, step: str, job_id: str, job_manager, gpu_profile_id: str | None = None
) -> None:
    """
    Run pipeline step subprocess as a background task, independent of any SSE
    connection. Buffers all output in job_manager AND writes to disk so logs
    survive studio restarts.
    """
    # Ensure config.json exists (wizard UCs only have uc_config.json)
    _ensure_config_json(uc_id)

    _step_log = _log_path(uc_id, step)  # per-step persistent log

    def _append(line: str):
        job_manager.append_log(job_id, line)
        for _lp in (_log_path(uc_id), _step_log):
            try:
                with open(_lp, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    # Truncate both logs at the start of every run
    _header = f"[studio] step={step} job={job_id}\n"
    for _lp in (_log_path(uc_id), _step_log):
        try:
            _lp.write_text(_header)
        except OSError:
            pass

    from studio.gpu_monitor import get_gpu_status

    if step in GPU_REQUIRED_STEPS and gpu_profile_id is None:
        # Recompute can bypass the local GPU check when config uses a remote compute worker
        if step == "recompute" and _has_remote_compute_backend(uc_id):
            pass  # remote worker handles GPU — no local GPU needed
        else:
            gpu_status = get_gpu_status()
            if not gpu_status.get("has_free_gpu"):
                msg = "No free GPU available — select a Remote GPU profile before running GPU steps"
                job_manager.fail(job_id, msg)
                _append(f"[studio] error: {msg}")
                return

    # Route GPU steps to remote EC2 when a profile is selected.
    # Recompute and prs-eval are excluded: they need local Qdrant access and
    # call the remote GPU via HTTP (compute worker / vLLM), not SSH.
    _LOCAL_STEPS = {"recompute", "prs-eval"}
    if gpu_profile_id and step in GPU_REQUIRED_STEPS and step not in _LOCAL_STEPS:
        try:
            await _run_step_remote_ssh(uc_id, step, job_id, job_manager,
                                       gpu_profile_id, _append)
        except Exception as e:
            job_manager.fail(job_id, str(e))
            _append(f"[studio] error: {e}")
        return

    cmd = _build_cmd(uc_id, step)
    env = _build_env(uc_id, step)

    _append(f"[studio] starting: {_redact_cmd(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
            limit=1024 * 1024,  # 1 MB per line — handles long model-loader progress bars
        )
        job_manager.set_pid(job_id, proc.pid)

        while True:
            try:
                raw_line = await proc.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                _append(line)
            except asyncio.LimitOverrunError:
                # Line exceeded buffer; drain and log a truncated placeholder
                await proc.stdout.read(1024 * 1024)
                _append("[studio] (line too long — truncated)")

        exit_code = await proc.wait()
        if exit_code == 0:
            job_manager.complete(job_id, exit_code)
            _append("[studio] done (exit 0)")
        else:
            job_manager.fail(job_id, f"Process exited with code {exit_code}")
            _append(f"[studio] failed (exit {exit_code})")

    except Exception as e:
        job_manager.fail(job_id, str(e))
        _append(f"[studio] error: {e}")


_SSE_HEARTBEAT_INTERVAL = 15  # seconds between SSE keepalive comments

async def run_step_streaming(uc_id: str, step: str, job_id: str, job_manager):
    """
    SSE relay: replays buffered log lines then tails new ones until the job
    finishes. Does NOT spawn a subprocess — call run_step_background for that.
    Sends SSE comment heartbeats every 15s so browsers don't time out during
    silent inference phases.
    """
    offset = 0
    last_heartbeat = time.monotonic()
    while True:
        job = job_manager.get(job_id)
        if not job:
            yield _sse({"type": "error", "message": "job not found"})
            return

        lines = job.get("last_lines", [])
        while offset < len(lines):
            yield _sse({"type": "log", "line": lines[offset]})
            offset += 1
            last_heartbeat = time.monotonic()  # data counts as activity

        status = job.get("status")
        if status != "running":
            if status == "done":
                yield _sse({"type": "done", "exit_code": 0})
            else:
                yield _sse({"type": "error", "exit_code": -1,
                             "message": job.get("error", "failed")})
            return

        now = time.monotonic()
        if now - last_heartbeat >= _SSE_HEARTBEAT_INTERVAL:
            yield ": heartbeat\n\n"
            last_heartbeat = now

        await asyncio.sleep(0.3)


def _build_env(uc_id: str, step: str) -> dict:
    """Return subprocess env with CUDA_VISIBLE_DEVICES set for GPU steps."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # ensure print() output streams immediately through the pipe
    if step in GPU_REQUIRED_STEPS:
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        gpu_id = 0  # default
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                gpu_id = int(uc_cfg.get("gpu_id", 0))
            except Exception:
                pass
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if step in {"index", "train", "recompute"}:
        from studio.settings_manager import get_setting
        hf_token = get_setting("huggingface_token") or ""
        if hf_token:
            env["HF_TOKEN"] = hf_token
    if step == "faq-gen-cloud":
        from studio.settings_manager import get_setting
        import json as _json
        _PROVIDER_KEY_MAP = {
            "claude": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
        }
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        provider = "claude"
        if uc_cfg_path.exists():
            try:
                uc_cfg = _json.loads(uc_cfg_path.read_text())
                provider = uc_cfg.get("llm", {}).get("cloud_provider", "claude")
            except Exception:
                pass
        settings_key, env_var = _PROVIDER_KEY_MAP.get(provider, ("anthropic_api_key", "ANTHROPIC_API_KEY"))
        api_key = get_setting(settings_key) or ""
        if api_key:
            env[env_var] = api_key
    return env


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
