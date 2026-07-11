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


def _run_smi_realtime() -> tuple[str, str, str]:
    """Run three nvidia-smi queries: extended GPU stats, UUID map, compute processes."""
    def _smi(*args):
        r = subprocess.run(["nvidia-smi"] + list(args), capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {r.stderr.strip()}")
        return r.stdout.strip()

    stats = _smi(
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    uuids = _smi("--query-gpu=index,uuid", "--format=csv,noheader")
    procs = _smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    return stats, uuids, procs


def parse_gpu_realtime(stats_csv: str, uuid_csv: str, procs_csv: str) -> dict:
    gpus: list[dict] = []
    for line in stats_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            used_mib = int(parts[2])
            total_mib = int(parts[3])
            power_str = parts[6].replace("[N/A]", "0").replace("N/A", "0")
            gpus.append({
                "id": int(parts[0]),
                "name": parts[1],
                "used_gb": round(used_mib / 1024, 1),
                "total_gb": round(total_mib / 1024, 1),
                "util_pct": int(parts[4]),
                "temp_c": int(parts[5]),
                "power_w": int(float(power_str)) if power_str else None,
                "status": "free" if used_mib < 4096 else "busy",
                "processes": [],
            })
        except (ValueError, IndexError):
            continue

    uuid_to_idx: dict[str, int] = {}
    for line in uuid_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_to_idx[parts[1]] = int(parts[0])

    idx_map = {g["id"]: g for g in gpus}
    for line in procs_csv.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_idx = uuid_to_idx.get(parts[0])
        if gpu_idx is None or gpu_idx not in idx_map:
            continue
        try:
            idx_map[gpu_idx]["processes"].append({
                "pid": int(parts[1]),
                "type": "C",
                "name": parts[2],
                "mem_mib": int(parts[3]),
            })
        except (ValueError, IndexError):
            continue

    return {"gpus": gpus, "has_free_gpu": any(g["status"] == "free" for g in gpus)}


def get_gpu_realtime() -> dict:
    """Extended GPU status: util%, temp, power, and per-GPU process list."""
    try:
        stats, uuids, procs = _run_smi_realtime()
    except FileNotFoundError:
        return {"error": "nvidia-smi not found", "gpus": [], "has_free_gpu": False}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timed out", "gpus": [], "has_free_gpu": False}
    except RuntimeError as e:
        return {"error": str(e), "gpus": [], "has_free_gpu": False}
    return parse_gpu_realtime(stats, uuids, procs)
