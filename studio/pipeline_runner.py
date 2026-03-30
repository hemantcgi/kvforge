# studio/pipeline_runner.py
"""Spawns pipeline subprocesses and streams their output as SSE events."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Maps step name → module path relative to project root
STEP_MODULES = {
    "index":    "pipeline.kv_indexer",
    "train":    "pipeline.lora_trainer",
    "recompute":"pipeline.kv_indexer",
    "prs-eval": "pipeline.prs_evaluator",
    "ab-eval":  "pipeline.ab_evaluator",
}

# Extra args per step
STEP_EXTRA_ARGS = {
    "recompute": ["--recompute"],
}


def _build_cmd(uc_id: str, step: str) -> list[str]:
    module = STEP_MODULES[step]
    config = str(ROOT / "examples" / uc_id / "config.json")
    cmd = [sys.executable, "-m", module, "--config", config]
    cmd += STEP_EXTRA_ARGS.get(step, [])
    return cmd


async def run_step_streaming(uc_id: str, step: str, job_id: str, job_manager):
    """
    Spawn pipeline step subprocess and yield SSE-formatted lines.
    Also appends log lines to job_manager.
    """
    from studio.gpu_monitor import get_gpu_status

    # GPU pre-check
    gpu_status = get_gpu_status()
    if not gpu_status.get("has_free_gpu") and not gpu_status.get("error"):
        job_manager.fail(job_id, "No free GPU available")
        yield _sse({"type": "error", "message": "No free GPU available"})
        return

    cmd = _build_cmd(uc_id, step)
    yield _sse({"type": "start", "cmd": " ".join(cmd)})

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        job_manager.set_pid(job_id, proc.pid)

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            job_manager.append_log(job_id, line)
            yield _sse({"type": "log", "line": line})

        exit_code = await proc.wait()
        job_manager.complete(job_id, exit_code)

        if exit_code == 0:
            yield _sse({"type": "done", "exit_code": 0})
        else:
            yield _sse({"type": "error", "exit_code": exit_code,
                         "message": f"Process exited with code {exit_code}"})

    except Exception as e:
        job_manager.fail(job_id, str(e))
        yield _sse({"type": "error", "message": str(e)})


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
