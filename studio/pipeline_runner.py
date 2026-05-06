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
}

# Steps that require a free GPU — sleep-faq calls a cloud REST API, no GPU needed
GPU_REQUIRED_STEPS = {"index", "train", "recompute", "prs-eval"}

# Static extra args per step (subcommands/flags appended after --config)
STEP_EXTRA_ARGS = {
    "index": ["index"],
    # recompute uses compute-kv; --stale-version is added dynamically in _build_cmd
    "recompute": ["compute-kv"],
}


def _build_cmd(uc_id: str, step: str) -> list[str]:
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
        # Read prs_eval_sample from uc_config; default 20 (5 inference calls × 20 FAQs ≈ 30 min)
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        sample = 20
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                raw = uc_cfg.get("prs_eval_sample", 20)
                parsed = int(raw)
                if parsed > 0:
                    sample = parsed
            except Exception:
                pass
        cmd += ["--sample", str(sample)]
    if step == "sleep-faq":
        # --output is dynamic (depends on uc_id)
        output = str(ROOT / "examples" / uc_id / "faqs.json")
        cmd += ["--output", output]
        # --count is read from uc_config.json if present
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                try:
                    count = int(uc_cfg.get("llm", {}).get("sleep_faq_count", 50))
                    if count <= 0:
                        count = 50
                except (TypeError, ValueError):
                    count = 50
                cmd += ["--count", str(count)]
            except Exception:
                cmd += ["--count", "50"]
        else:
            cmd += ["--count", "50"]
    if step == "faq-gen-cloud":
        import json as _json
        output = str(ROOT / "examples" / uc_id / "faqs.json")
        cmd += ["--output", output]
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        count = 50
        if uc_cfg_path.exists():
            try:
                uc_cfg = _json.loads(uc_cfg_path.read_text())
                raw_count = uc_cfg.get("llm", {}).get("sleep_faq_count", 50)
                parsed = int(raw_count)
                if parsed > 0:
                    count = parsed
            except (TypeError, ValueError, KeyError):
                pass
            except Exception:
                pass
        cmd += ["--count", str(count)]
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


def _log_path(uc_id: str) -> Path:
    """Disk log file for the most recent run of a UC — survives studio restarts."""
    return ROOT / "examples" / uc_id / "last_run.log"


async def run_step_background(uc_id: str, step: str, job_id: str, job_manager) -> None:
    """
    Run pipeline step subprocess as a background task, independent of any SSE
    connection. Buffers all output in job_manager AND writes to disk so logs
    survive studio restarts.
    """
    from studio.gpu_monitor import get_gpu_status

    if step in GPU_REQUIRED_STEPS:
        gpu_status = get_gpu_status()
        if not gpu_status.get("has_free_gpu"):
            job_manager.fail(job_id, "No free GPU available")
            return

    cmd = _build_cmd(uc_id, step)
    env = _build_env(uc_id, step)

    def _append(line: str):
        job_manager.append_log(job_id, line)
        try:
            with open(_log_path(uc_id), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # Truncate log file for new run
    try:
        _log_path(uc_id).write_text(f"[studio] step={step} job={job_id}\n")
    except OSError:
        pass

    _append(f"[studio] starting: {_redact_cmd(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
        )
        job_manager.set_pid(job_id, proc.pid)

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            _append(line)

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
