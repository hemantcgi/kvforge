# studio/pipeline_runner.py
"""Spawns pipeline subprocesses and streams their output as SSE events."""

import asyncio
import json
import os
import sys
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
            # Fall back to train data if no faqs generated yet
            source_path = ROOT / "examples" / uc_id / "data" / "train.jsonl"
            cmd += ["--faqs", str(source_path)]
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
    return cmd


async def run_step_streaming(uc_id: str, step: str, job_id: str, job_manager):
    """
    Spawn pipeline step subprocess and yield SSE-formatted lines.
    Also appends log lines to job_manager.
    """
    from studio.gpu_monitor import get_gpu_status

    # GPU pre-check — skipped for steps that don't need a GPU
    if step in GPU_REQUIRED_STEPS:
        gpu_status = get_gpu_status()
        if not gpu_status.get("has_free_gpu"):
            job_manager.fail(job_id, "No free GPU available")
            yield _sse({"type": "error", "message": "No free GPU available"})
            return

    cmd = _build_cmd(uc_id, step)
    env = _build_env(uc_id, step)
    yield _sse({"type": "start", "cmd": " ".join(cmd)})

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
            job_manager.append_log(job_id, line)
            yield _sse({"type": "log", "line": line})

        exit_code = await proc.wait()

        if exit_code == 0:
            job_manager.complete(job_id, exit_code)
            yield _sse({"type": "done", "exit_code": 0})
        else:
            job_manager.fail(job_id, f"Process exited with code {exit_code}")
            yield _sse({"type": "error", "exit_code": exit_code,
                         "message": f"Process exited with code {exit_code}"})

    except Exception as e:
        job_manager.fail(job_id, str(e))
        yield _sse({"type": "error", "message": str(e)})


def _build_env(uc_id: str, step: str) -> dict:
    """Return subprocess env with CUDA_VISIBLE_DEVICES set for GPU steps."""
    env = os.environ.copy()
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
    return env


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
