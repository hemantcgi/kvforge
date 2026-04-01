# studio/pipeline_runner.py
"""Spawns pipeline subprocesses and streams their output as SSE events."""

import asyncio
import json
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

# Static extra args per step
STEP_EXTRA_ARGS = {
    "recompute": ["--recompute"],
}


def _build_cmd(uc_id: str, step: str) -> list[str]:
    module = STEP_MODULES[step]
    config = str(ROOT / "examples" / uc_id / "config.json")
    cmd = [sys.executable, "-m", module, "--config", config]
    cmd += STEP_EXTRA_ARGS.get(step, [])
    if step == "sleep-faq":
        # --output is dynamic (depends on uc_id)
        output = str(ROOT / "examples" / uc_id / "faqs.json")
        cmd += ["--output", output]
        # --count is read from uc_config.json if present
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                count = uc_cfg.get("llm", {}).get("sleep_faq_count", 50)
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


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
