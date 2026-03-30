# studio/gpu_monitor.py
"""GPU availability monitor: parses nvidia-smi and detects vLLM processes."""

import os
import re
import signal
import subprocess
import time
from typing import Optional


def _run_nvidia_smi() -> str:
    """Run nvidia-smi and return CSV output. Raises FileNotFoundError if unavailable."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _run_ps() -> str:
    """Return ps output for vLLM processes."""
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True, text=True, timeout=5
    )
    lines = [l for l in result.stdout.splitlines() if "vllm.entrypoints" in l]
    return "\n".join(lines)


def parse_nvidia_smi(output: str) -> list[dict]:
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx        = int(parts[0])
        name       = parts[1]
        total_mib  = int(parts[2])
        used_mib   = int(parts[3])
        free_mib   = total_mib - used_mib
        # Consider "free" if less than 4 GB used (headroom for system)
        status = "free" if used_mib < 4096 else "busy"
        gpus.append({
            "id":       idx,
            "name":     name,
            "total_gb": round(total_mib / 1024, 1),
            "used_gb":  round(used_mib  / 1024, 1),
            "free_gb":  round(free_mib  / 1024, 1),
            "status":   status,
            "process":  None,  # filled in by get_gpu_status
        })
    return gpus


def find_vllm_processes(ps_output: str) -> list[dict]:
    """Parse `ps aux` output lines (already filtered to vLLM lines by _run_ps)."""
    procs = []
    for line in ps_output.strip().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        # ps aux format: USER PID %CPU %MEM ... CMD
        # Need at least USER + PID + some command tokens
        if len(tokens) < 3:
            continue
        try:
            pid = int(tokens[1])  # tokens[0]=USER, tokens[1]=PID
        except (ValueError, IndexError):
            continue
        # Use full line for regex matching to handle both condensed test data
        # and real ps aux output (which has 11 fixed columns before CMD)
        port_m = re.search(r"--port\s+(\d+)", line)
        lora_m = re.search(r"--lora-modules\s+(\S+)", line)
        cmd = " ".join(tokens[10:]) if len(tokens) > 10 else " ".join(tokens[2:])

        port  = int(port_m.group(1)) if port_m else None
        label = lora_m.group(1).split("=")[0] if lora_m else "vllm"

        procs.append({"pid": pid, "port": port, "label": label, "cmd": cmd})
    return procs


def get_gpu_status() -> dict:
    try:
        smi_out = _run_nvidia_smi()
    except FileNotFoundError:
        return {"error": "nvidia-smi not found", "gpus": [], "has_free_gpu": False}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out", "gpus": [], "has_free_gpu": False}
    except RuntimeError as e:
        return {"error": str(e), "gpus": [], "has_free_gpu": False}

    gpus   = parse_nvidia_smi(smi_out)
    ps_out = _run_ps()
    procs  = find_vllm_processes(ps_out)

    # Map vLLM processes to GPU cards by port range heuristic
    # Ports 8090-8093 → UC4/UC1/UC2/UC3; annotate busy GPUs
    port_to_label = {p["port"]: p["label"] for p in procs}
    # nvidia-smi reports GPUs in physical order; vLLM CUDA_VISIBLE_DEVICES
    # assignment is best-effort here — mark all "busy" GPUs with vLLM info
    vllm_labels = list(port_to_label.values())
    vllm_idx = 0
    for gpu in gpus:
        if gpu["status"] == "busy" and vllm_idx < len(vllm_labels):
            gpu["process"] = f"vLLM {vllm_labels[vllm_idx]}"
            vllm_idx += 1

    has_free = any(g["status"] == "free" for g in gpus)
    return {"gpus": gpus, "has_free_gpu": has_free, "vllm_processes": procs}


def stop_vllm_process(port: int, timeout: int = 10) -> bool:
    """Send SIGTERM to the vLLM process on the given port. Returns True on success."""
    ps_out = _run_ps()
    procs = find_vllm_processes(ps_out)
    target = next((p for p in procs if p["port"] == port), None)
    if not target:
        return False
    try:
        os.kill(target["pid"], signal.SIGTERM)
        for _ in range(timeout):
            time.sleep(1)
            ps_out2 = _run_ps()
            if not any(p["pid"] == target["pid"] for p in find_vllm_processes(ps_out2)):
                return True
        return False
    except ProcessLookupError:
        return True  # Already gone
    except PermissionError:
        return False  # Cannot kill process owned by another user
