# KVForge Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build KVForge Studio — a generalized web UI at `/studio` that lets users configure and run the full KVForge pipeline on any dataset, with live GPU monitoring, pipeline progress streaming, and A/B evaluation comparison.

**Architecture:** A `studio/` Python package (FastAPI router + API + pipeline runner + job manager + GPU monitor) is mounted into the existing `kvforge_portal.py` at port 8080. HTML pages are served as static files from `templates/studio/`. Use cases are tracked in `kvforge_registry.json` with per-UC config in `examples/<uc_id>/uc_config.json`.

**Tech Stack:** FastAPI, Python 3.10+, SSE (via `StreamingResponse` + manual event formatting), `subprocess`, `nvidia-smi`, vanilla JS (no framework), existing pipeline scripts (`kv_indexer`, `lora_trainer`, `prs_evaluator`, `ab_evaluator`).

**Spec:** `docs/superpowers/specs/2026-03-29-kvforge-studio-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `studio/__init__.py` | Create | Package marker |
| `studio/job_manager.py` | Create | In-memory job registry, per-UC locks |
| `studio/gpu_monitor.py` | Create | nvidia-smi parsing, vLLM process detection |
| `studio/pipeline_runner.py` | Create | Subprocess spawn + SSE log streaming |
| `studio/api.py` | Create | All `/studio/api/*` endpoints |
| `studio/routes.py` | Create | FastAPI router, page serving, UC CRUD, migration |
| `templates/studio/hub.html` | Create | Full studio hub page (sidebar, breadcrumb, UC cards, module panels) |
| `kvforge_portal.py` | Modify | Mount studio router (1 line change) |
| `kvforge_registry.json` | Auto-created | UC index (created by migration on first start) |
| `examples/*/uc_config.json` | Auto-created | Per-UC studio config (created by migration) |
| `tests/test_studio_job_manager.py` | Create | Unit tests for job_manager |
| `tests/test_studio_gpu_monitor.py` | Create | Unit tests for gpu_monitor |
| `tests/test_studio_routes.py` | Create | Integration tests for API routes |

---

## Task 1: studio package + job_manager

**Files:**
- Create: `studio/__init__.py`
- Create: `studio/job_manager.py`
- Create: `tests/test_studio_job_manager.py`

- [ ] **Step 1: Create package marker**

```python
# studio/__init__.py
# KVForge Studio — generalized pipeline UI package
```

- [ ] **Step 2: Write failing tests for job_manager**

```python
# tests/test_studio_job_manager.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from studio.job_manager import JobManager, JobStatus, DuplicateJobError


def test_create_job_returns_job_id():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    assert job_id is not None and len(job_id) > 0


def test_get_job_returns_created_job():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    job = jm.get(job_id)
    assert job["uc_id"] == "usecase3_squad"
    assert job["step"] == "train"
    assert job["status"] == JobStatus.RUNNING


def test_duplicate_uc_raises():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    with pytest.raises(DuplicateJobError):
        jm.create("usecase3_squad", "prs-eval")


def test_different_ucs_allowed_concurrently():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    job_id2 = jm.create("usecase1_customer_support", "train")
    assert job_id2 is not None


def test_complete_job_releases_lock():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.complete(job_id, exit_code=0)
    assert jm.get(job_id)["status"] == JobStatus.DONE
    # Now same UC can start a new job
    job_id2 = jm.create("usecase3_squad", "prs-eval")
    assert job_id2 is not None


def test_fail_job():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.fail(job_id, "CUDA out of memory")
    assert jm.get(job_id)["status"] == JobStatus.FAILED
    assert "CUDA" in jm.get(job_id)["error"]


def test_append_log_line():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.append_log(job_id, "epoch 1/3 complete")
    jm.append_log(job_id, "epoch 2/3 complete")
    assert len(jm.get(job_id)["last_lines"]) == 2


def test_log_capped_at_100_lines():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    for i in range(150):
        jm.append_log(job_id, f"line {i}")
    assert len(jm.get(job_id)["last_lines"]) == 100


def test_list_active_returns_running_jobs():
    jm = JobManager()
    jm.create("usecase3_squad", "train")
    active = jm.list_active()
    assert len(active) == 1
    assert active[0]["uc_id"] == "usecase3_squad"


def test_stop_job_marks_stopped():
    jm = JobManager()
    job_id = jm.create("usecase3_squad", "train")
    jm.stop(job_id)
    assert jm.get(job_id)["status"] == JobStatus.STOPPED
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/hemant/Downloads/RoPE/qdrant
python -m pytest tests/test_studio_job_manager.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'studio'`

- [ ] **Step 4: Implement job_manager**

```python
# studio/job_manager.py
"""In-memory job registry with per-UC locking for pipeline steps."""

import uuid
import time
from enum import Enum
from threading import Lock
from typing import Optional


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    STOPPED = "stopped"


class DuplicateJobError(Exception):
    """Raised when a UC already has a running job."""


_MAX_LOG_LINES = 100


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._uc_locks: dict[str, str] = {}  # uc_id -> job_id
        self._lock = Lock()

    def create(self, uc_id: str, step: str) -> str:
        with self._lock:
            if uc_id in self._uc_locks:
                raise DuplicateJobError(
                    f"{uc_id} already has a running job: {self._uc_locks[uc_id]}"
                )
            job_id = str(uuid.uuid4())[:8]
            self._jobs[job_id] = {
                "job_id":     job_id,
                "uc_id":      uc_id,
                "step":       step,
                "status":     JobStatus.RUNNING,
                "pid":        None,
                "start_time": time.time(),
                "last_lines": [],
                "error":      None,
            }
            self._uc_locks[uc_id] = job_id
            return job_id

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def set_pid(self, job_id: str, pid: int):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["pid"] = pid

    def append_log(self, job_id: str, line: str):
        with self._lock:
            if job_id in self._jobs:
                lines = self._jobs[job_id]["last_lines"]
                lines.append(line)
                if len(lines) > _MAX_LOG_LINES:
                    self._jobs[job_id]["last_lines"] = lines[-_MAX_LOG_LINES:]

    def complete(self, job_id: str, exit_code: int):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.DONE if exit_code == 0 else JobStatus.FAILED
                self._uc_locks.pop(job["uc_id"], None)

    def fail(self, job_id: str, error: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.FAILED
                job["error"] = error
                self._uc_locks.pop(job["uc_id"], None)

    def stop(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["status"] = JobStatus.STOPPED
                self._uc_locks.pop(job["uc_id"], None)

    def list_active(self) -> list[dict]:
        with self._lock:
            return [j for j in self._jobs.values() if j["status"] == JobStatus.RUNNING]


# Module-level singleton used by routes and pipeline_runner
_manager = JobManager()

def get_manager() -> JobManager:
    return _manager
```

- [ ] **Step 5: Run tests — all must pass**

```bash
python -m pytest tests/test_studio_job_manager.py -v
```
Expected: 10 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add studio/__init__.py studio/job_manager.py tests/test_studio_job_manager.py
git commit -m "feat: studio package + job manager with per-UC locking"
```

---

## Task 2: gpu_monitor

**Files:**
- Create: `studio/gpu_monitor.py`
- Create: `tests/test_studio_gpu_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_studio_gpu_monitor.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch
from studio.gpu_monitor import parse_nvidia_smi, find_vllm_processes, get_gpu_status


_SAMPLE_NVIDIA_SMI = """0, NVIDIA A10G, 24564, 2100
1, NVIDIA A10G, 24564, 20500
2, NVIDIA A10G, 24564, 20500
3, NVIDIA A10G, 24564, 1900"""

_SAMPLE_PS = """ubuntu 1956739  0.0  0.5 /home/ubuntu/qdrant/venv/bin/python3 -m vllm.entrypoints.openai.api_server --lora-modules uc3=examples/usecase3_squad/lora_checkpoints/v1/ --port 8093
ubuntu 1771247  0.0  0.5 /home/ubuntu/qdrant/venv/bin/python3 -m vllm.entrypoints.openai.api_server --lora-modules uc1=examples/usecase1_customer_support/lora_checkpoints/v8_dpo/final --port 8091"""


def test_parse_nvidia_smi_returns_four_gpus():
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert len(gpus) == 4


def test_parse_nvidia_smi_free_gpu():
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert gpus[0]["free_gb"] == pytest.approx((24564 - 2100) / 1024, abs=0.1)
    assert gpus[0]["status"] == "free"


def test_parse_nvidia_smi_busy_gpu():
    gpus = parse_nvidia_smi(_SAMPLE_NVIDIA_SMI)
    assert gpus[1]["status"] == "busy"
    assert gpus[1]["used_gb"] == pytest.approx(20500 / 1024, abs=0.1)


def test_find_vllm_processes_parses_uc_id():
    procs = find_vllm_processes(_SAMPLE_PS)
    assert len(procs) == 2
    pids = {p["pid"] for p in procs}
    assert 1956739 in pids


def test_find_vllm_processes_extracts_port():
    procs = find_vllm_processes(_SAMPLE_PS)
    ports = {p["port"] for p in procs}
    assert 8093 in ports
    assert 8091 in ports


def test_get_gpu_status_marks_busy_gpus():
    with patch("studio.gpu_monitor._run_nvidia_smi", return_value=_SAMPLE_NVIDIA_SMI), \
         patch("studio.gpu_monitor._run_ps", return_value=_SAMPLE_PS):
        status = get_gpu_status()
    assert status["has_free_gpu"] is True
    busy = [g for g in status["gpus"] if g["status"] == "busy"]
    assert len(busy) == 2


def test_get_gpu_status_no_free_gpu():
    all_busy = "0, NVIDIA A10G, 24564, 21000\n1, NVIDIA A10G, 24564, 21000\n2, NVIDIA A10G, 24564, 21000\n3, NVIDIA A10G, 24564, 21000"
    with patch("studio.gpu_monitor._run_nvidia_smi", return_value=all_busy), \
         patch("studio.gpu_monitor._run_ps", return_value=""):
        status = get_gpu_status()
    assert status["has_free_gpu"] is False


def test_get_gpu_status_nvidia_smi_unavailable():
    with patch("studio.gpu_monitor._run_nvidia_smi", side_effect=FileNotFoundError):
        status = get_gpu_status()
    assert status["error"] == "nvidia-smi not found"
    assert status["gpus"] == []

import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_gpu_monitor.py -v 2>&1 | head -10
```
Expected: `ModuleNotFoundError: No module named 'studio.gpu_monitor'`

- [ ] **Step 3: Implement gpu_monitor**

```python
# studio/gpu_monitor.py
"""GPU availability monitor: parses nvidia-smi and detects vLLM processes."""

import re
import subprocess
from typing import Optional


def _run_nvidia_smi() -> str:
    """Run nvidia-smi and return CSV output. Raises FileNotFoundError if unavailable."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
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
        cmd = " ".join(tokens[10:]) if len(tokens) > 10 else " ".join(tokens[2:])

        port_m = re.search(r"--port\s+(\d+)", cmd)
        lora_m = re.search(r"--lora-modules\s+(\S+)", cmd)

        port  = int(port_m.group(1)) if port_m else None
        label = lora_m.group(1).split("=")[0] if lora_m else "vllm"

        procs.append({"pid": pid, "port": port, "label": label, "cmd": cmd})
    return procs


def get_gpu_status() -> dict:
    try:
        smi_out = _run_nvidia_smi()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"error": "nvidia-smi not found", "gpus": [], "has_free_gpu": False}

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
    import signal, os, time
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
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_studio_gpu_monitor.py -v
```
Expected: 8 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add studio/gpu_monitor.py tests/test_studio_gpu_monitor.py
git commit -m "feat: gpu_monitor — nvidia-smi parsing and vLLM process detection"
```

---

## Task 3: Migration + registry

**Files:**
- Create: `studio/migration.py`
- Create: `tests/test_studio_migration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_studio_migration.py
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from studio.migration import migrate_existing_use_cases, load_registry


def _make_fake_examples(tmp: Path):
    """Create minimal fake examples/usecase3_squad/config.json."""
    uc = tmp / "examples" / "usecase3_squad"
    uc.mkdir(parents=True)
    cfg = {
        "collection": "squad-qa",
        "vector_store": "faiss",
        "vector_dim": 384,
        "chunk_size": 600,
        "chunk_overlap": 60,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "4bit",
        "vllm_url": "http://localhost:8093",
        "loader": "jsonl",
        "dashboard_port": 8083,
    }
    (uc / "config.json").write_text(json.dumps(cfg))
    return tmp


def test_migrate_creates_registry(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    registry_path = tmp_path / "kvforge_registry.json"
    assert registry_path.exists()
    data = json.loads(registry_path.read_text())
    assert any(uc["id"] == "usecase3_squad" for uc in data["use_cases"])


def test_migrate_creates_uc_config(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    cfg_path = tmp_path / "examples" / "usecase3_squad" / "uc_config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert cfg["vectordb"]["store"] == "faiss"
    assert cfg["vectordb"]["dimensions"] == 384
    assert cfg["llm"]["vllm_url"] == "http://localhost:8093"


def test_migrate_idempotent(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    migrate_existing_use_cases(root=tmp_path)  # Second run should not error
    data = json.loads((tmp_path / "kvforge_registry.json").read_text())
    ids = [uc["id"] for uc in data["use_cases"]]
    assert ids.count("usecase3_squad") == 1  # No duplicates


def test_load_registry_returns_list(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    ucs = load_registry(root=tmp_path)
    assert isinstance(ucs, list)
    assert len(ucs) == 1
    assert ucs[0]["id"] == "usecase3_squad"


def test_uc_config_type_is_example(tmp_path):
    _make_fake_examples(tmp_path)
    migrate_existing_use_cases(root=tmp_path)
    cfg = json.loads((tmp_path / "examples" / "usecase3_squad" / "uc_config.json").read_text())
    assert cfg["type"] == "example"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_migration.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement migration**

```python
# studio/migration.py
"""One-time migration: creates kvforge_registry.json and uc_config.json from config.json."""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent

_STORE_MAP = {"qdrant": "qdrant", "chromadb": "chromadb", "faiss": "faiss"}
_DISPLAY_NAMES = {
    "usecase1_customer_support": "Customer Support",
    "usecase2_pubmedqa":          "PubMedQA",
    "usecase3_squad":             "SQuAD",
    "usecase4_bedrock_userguide": "Bedrock User Guide",
}


def _config_to_uc_config(uc_id: str, cfg: dict) -> dict:
    loader = cfg.get("loader", "jsonl")
    source_type = "pdf" if loader == "pdf" else "huggingface"
    return {
        "id":           uc_id,
        "display_name": _DISPLAY_NAMES.get(uc_id, uc_id),
        "type":         "example",
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "data": {
            "source_type":  source_type,
            # pdf_path is not in config.json; for known UC4 set a default, else leave blank
        "source_path":  "examples/usecase4_bedrock_userguide/data/" if loader == "pdf" else "",
            "dataset_id":   cfg.get("dataset_id", ""),
            "split":        cfg.get("split", "train"),
            "text_column":  cfg.get("jsonl_text_key", "text"),
            "max_rows":     cfg.get("max_rows", 5000),
        },
        "vectordb": {
            "store":           _STORE_MAP.get(cfg.get("vector_store", "qdrant"), "qdrant"),
            "dimensions":      cfg.get("vector_dim", 384),
            "chunk_size":      cfg.get("chunk_size", 512),
            "chunk_overlap":   cfg.get("chunk_overlap", 64),
            "embedding_model": cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
            "index_type":      "hnsw",
        },
        "llm": {
            "local_model":          cfg.get("llm_model", "meta-llama/Llama-3.2-3B-Instruct"),
            "quantization":         cfg.get("quantization", "4bit"),
            "vllm_url":             cfg.get("vllm_url", ""),
            "comparison_provider":  "gemini",
            "comparison_model":     "gemini-1.5-flash",
        },
    }


def migrate_existing_use_cases(root: Path = ROOT):
    registry_path = root / "kvforge_registry.json"

    # Load existing registry or start fresh
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {"use_cases": []}

    existing_ids = {uc["id"] for uc in registry["use_cases"]}

    for config_path in sorted((root / "examples").glob("*/config.json")):
        uc_id = config_path.parent.name
        cfg   = json.loads(config_path.read_text())

        # Write uc_config.json (overwrite to keep fresh)
        uc_config = _config_to_uc_config(uc_id, cfg)
        uc_config_path = config_path.parent / "uc_config.json"
        uc_config_path.write_text(json.dumps(uc_config, indent=2))

        # Add to registry if not already present
        if uc_id not in existing_ids:
            registry["use_cases"].append({
                "id":           uc_id,
                "display_name": uc_config["display_name"],
                "type":         "example",
            })
            existing_ids.add(uc_id)

    registry_path.write_text(json.dumps(registry, indent=2))


def load_registry(root: Path = ROOT) -> list[dict]:
    registry_path = root / "kvforge_registry.json"
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text()).get("use_cases", [])


def add_to_registry(uc_id: str, display_name: str, root: Path = ROOT):
    registry_path = root / "kvforge_registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"use_cases": []}
    if not any(uc["id"] == uc_id for uc in registry["use_cases"]):
        registry["use_cases"].append({"id": uc_id, "display_name": display_name, "type": "custom"})
    registry_path.write_text(json.dumps(registry, indent=2))
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_studio_migration.py -v
```
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add studio/migration.py tests/test_studio_migration.py
git commit -m "feat: studio migration — auto-creates registry and uc_config.json from config.json"
```

---

## Task 4: pipeline_runner (subprocess + SSE)

**Files:**
- Create: `studio/pipeline_runner.py`

> Note: pipeline_runner is tested implicitly via the API integration tests in Task 6. Unit testing subprocess spawning directly is brittle; we test the observable behavior through the API instead.

- [ ] **Step 1: Implement pipeline_runner**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add studio/pipeline_runner.py
git commit -m "feat: pipeline_runner — async subprocess spawning with SSE log streaming"
```

---

## Task 5: studio/api.py — all API endpoints

**Files:**
- Create: `studio/api.py`
- Create: `tests/test_studio_routes.py`

- [ ] **Step 1: Write failing API tests**

```python
# tests/test_studio_routes.py
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def tmp_root(tmp_path):
    """Set up a minimal project root with one UC."""
    examples = tmp_path / "examples" / "usecase3_squad"
    examples.mkdir(parents=True)
    cfg = {
        "collection": "squad-qa", "vector_store": "faiss", "vector_dim": 384,
        "chunk_size": 600, "chunk_overlap": 60, "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct", "quantization": "4bit",
        "vllm_url": "http://localhost:8093", "loader": "jsonl", "dashboard_port": 8083,
    }
    (examples / "config.json").write_text(json.dumps(cfg))
    return tmp_path


@pytest.fixture
def client(tmp_root):
    with patch("studio.routes.ROOT", tmp_root), \
         patch("studio.api.ROOT", tmp_root), \
         patch("studio.migration.ROOT", tmp_root):
        import studio.routes as routes
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(routes.router)
        return TestClient(app)


def test_registry_returns_use_cases(client):
    r = client.get("/api/registry")
    assert r.status_code == 200
    data = r.json()
    assert "use_cases" in data


def test_get_uc_config(client, tmp_root):
    from studio.migration import migrate_existing_use_cases
    migrate_existing_use_cases(root=tmp_root)
    r = client.get("/api/uc/usecase3_squad/config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["vectordb"]["store"] == "faiss"


def test_save_uc_config(client, tmp_root):
    from studio.migration import migrate_existing_use_cases
    migrate_existing_use_cases(root=tmp_root)
    r = client.post("/api/uc/usecase3_squad/config",
                    json={"vectordb": {"store": "qdrant", "dimensions": 768,
                                       "chunk_size": 512, "chunk_overlap": 64,
                                       "embedding_model": "BAAI/bge-small-en-v1.5",
                                       "index_type": "hnsw"}})
    assert r.status_code == 200
    # Verify persisted
    cfg = json.loads((tmp_root / "examples" / "usecase3_squad" / "uc_config.json").read_text())
    assert cfg["vectordb"]["store"] == "qdrant"


def test_create_new_uc(client, tmp_root):
    r = client.post("/api/uc/new", json={"id": "my-test-uc", "display_name": "My Test"})
    assert r.status_code == 200
    registry = json.loads((tmp_root / "kvforge_registry.json").read_text())
    ids = [uc["id"] for uc in registry["use_cases"]]
    assert "my-test-uc" in ids


def test_gpu_check_returns_gpus():
    with patch("studio.gpu_monitor.get_gpu_status", return_value={
        "gpus": [{"id": 0, "status": "free", "free_gb": 20.0, "used_gb": 2.0, "total_gb": 24.0}],
        "has_free_gpu": True, "vllm_processes": []
    }):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import studio.routes as routes
        app = FastAPI()
        app.include_router(routes.router)
        c = TestClient(app)
        r = c.post("/api/gpu-check")
        assert r.status_code == 200
        assert r.json()["has_free_gpu"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_studio_routes.py -v 2>&1 | head -15
```

- [ ] **Step 3: Implement studio/api.py**

```python
# studio/api.py
"""All /studio/api/* endpoint handlers — imported by routes.py."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from studio.migration import migrate_existing_use_cases, load_registry, add_to_registry
from studio.gpu_monitor import get_gpu_status, stop_vllm_process
from studio.job_manager import get_manager, DuplicateJobError

ROOT = Path(__file__).resolve().parent.parent

api_router = APIRouter(prefix="/api")


# ── Registry ──────────────────────────────────────────────────────────────────

@api_router.get("/registry")
def get_registry():
    ucs = load_registry(root=ROOT)
    # Augment with phase/PRS from version.json where available
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
        # Add active job info
        jm = get_manager()
        active = next((j for j in jm.list_active() if j["uc_id"] == uc["id"]), None)
        uc_data["active_job"] = active
        result.append(uc_data)
    return {"use_cases": result}


# ── UC Config ─────────────────────────────────────────────────────────────────

@api_router.get("/uc/{uc_id}/config")
def get_uc_config(uc_id: str):
    path = ROOT / "examples" / uc_id / "uc_config.json"
    if not path.exists():
        raise HTTPException(404, f"uc_config.json not found for {uc_id}")
    return json.loads(path.read_text())


@api_router.post("/uc/{uc_id}/config")
async def save_uc_config(uc_id: str, request: Request):
    path = ROOT / "examples" / uc_id / "uc_config.json"
    if not path.exists():
        raise HTTPException(404, f"UC {uc_id} not found")
    existing = json.loads(path.read_text())
    updates = await request.json()
    # Merge top-level keys only (data/vectordb/llm sections)
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
    uc_dir = ROOT / "examples" / req.id
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
                    "comparison_provider": "gemini", "comparison_model": "gemini-1.5-flash"},
    }
    (uc_dir / "uc_config.json").write_text(json.dumps(uc_config, indent=2))
    add_to_registry(req.id, req.display_name, root=ROOT)
    return {"status": "created", "id": req.id}


# ── GPU ───────────────────────────────────────────────────────────────────────

@api_router.post("/gpu-check")
def gpu_check():
    return get_gpu_status()


class StopVllmRequest(BaseModel):
    port: int


@api_router.post("/gpu/stop-vllm")
def stop_vllm(req: StopVllmRequest):
    ok = stop_vllm_process(req.port)
    if not ok:
        raise HTTPException(500, f"Failed to stop vLLM on port {req.port}")
    return {"status": "stopped", "port": req.port}


# ── Pipeline Jobs ─────────────────────────────────────────────────────────────

class RunStepRequest(BaseModel):
    uc_id: str
    step: str  # "index" | "train" | "recompute" | "prs-eval" | "ab-eval"


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
    if job["pid"]:
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    jm.stop(job_id)
    return {"status": "stopped"}
```

- [ ] **Step 4: Run tests — all must pass**

```bash
python -m pytest tests/test_studio_routes.py -v
```
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add studio/api.py tests/test_studio_routes.py
git commit -m "feat: studio API — registry, UC config CRUD, GPU check, run-step endpoints"
```

---

## Task 6: studio/routes.py — router, SSE stream, page serving

**Files:**
- Create: `studio/routes.py`

- [ ] **Step 1: Implement routes.py**

```python
# studio/routes.py
"""FastAPI router for KVForge Studio — mounts at /studio in kvforge_portal.py."""

import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from studio.migration import migrate_existing_use_cases
from studio.api import api_router
from studio.job_manager import get_manager
from studio.pipeline_runner import run_step_streaming

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates" / "studio"

router = APIRouter()
router.include_router(api_router)


# ── Auto-migrate on import ────────────────────────────────────────────────────
_migrated = False

def _ensure_migrated():
    global _migrated
    if not _migrated:
        migrate_existing_use_cases(root=ROOT)
        _migrated = True


# ── Pages ─────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def studio_hub():
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()


@router.get("/uc/{uc_id}", response_class=HTMLResponse)
def uc_detail(uc_id: str):
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()


# ── SSE stream ────────────────────────────────────────────────────────────────

@router.get("/api/stream/{job_id}")
async def stream_job(job_id: str):
    jm = get_manager()
    job = jm.get(job_id)
    if not job:
        return StreamingResponse(
            iter([f"data: {json.dumps({'type':'error','message':'job not found'})}\n\n"]),
            media_type="text/event-stream"
        )

    async def event_generator():
        async for chunk in run_step_streaming(
            job["uc_id"], job["step"], job_id, jm
        ):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
```

- [ ] **Step 2: Mount router into kvforge_portal.py**

Find the line in `kvforge_portal.py` where `app = FastAPI(...)` is defined (around line 25), and add the studio router import + mount after `app` is created:

```python
# Add after: app = FastAPI(lifespan=lifespan)  (or wherever app is defined)
from studio.routes import router as _studio_router
app.include_router(_studio_router, prefix="/studio")
```

- [ ] **Step 3: Verify portal starts without errors**

```bash
cd /Users/hemant/Downloads/RoPE/qdrant
python -c "import kvforge_portal; print('OK')"
```
Expected: `OK` (no import errors)

- [ ] **Step 4: Commit**

```bash
git add studio/routes.py kvforge_portal.py
git commit -m "feat: studio router mounted at /studio in kvforge_portal"
```

---

## Task 7: Hub HTML page — shell, sidebar, topbar, breadcrumb

**Files:**
- Create: `templates/studio/hub.html`

This is a large self-contained HTML file. Build it in stages — this task covers the shell, sidebar, and topbar only (no UC card content yet; show a placeholder).

- [ ] **Step 1: Create templates directory**

```bash
mkdir -p /Users/hemant/Downloads/RoPE/qdrant/templates/studio
```

- [ ] **Step 2: Create hub.html shell**

```html
<!-- templates/studio/hub.html -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KVForge Studio</title>
<style>
/* === Reset + Base === */
*{box-sizing:border-box;margin:0;padding:0}
body{background:#020817;font-family:system-ui,-apple-system,sans-serif;color:#e2e8f0;font-size:13px;height:100vh;display:flex;overflow:hidden}

/* === Sidebar === */
#sidebar{width:220px;background:#080712;border-right:1px solid #1e1b4b;display:flex;flex-direction:column;transition:width 0.22s ease;flex-shrink:0;overflow:hidden}
#sidebar.collapsed{width:52px}

.sb-header-exp{padding:13px 12px;display:flex;align-items:center;gap:9px;border-bottom:1px solid #1e1b4b;flex-shrink:0}
#sidebar.collapsed .sb-header-exp{display:none}
.sb-header-col{display:none;padding:12px 0;justify-content:center;align-items:center;flex-direction:column;gap:8px;border-bottom:1px solid #1e1b4b;flex-shrink:0;cursor:pointer}
#sidebar.collapsed .sb-header-col{display:flex}
.sb-header-col:hover .logo-mark{box-shadow:0 0 12px #8b5cf688}

.logo-mark{width:28px;height:28px;background:linear-gradient(135deg,#8b5cf6,#06b6d4);border-radius:7px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:10px;flex-shrink:0;transition:box-shadow 0.2s}
.logo-text{background:linear-gradient(90deg,#a78bfa,#22d3ee);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;font-size:14px;white-space:nowrap}
.collapse-btn{margin-left:auto;color:#475569;cursor:pointer;display:flex;align-items:center;padding:4px;border-radius:5px;transition:color 0.15s,background 0.15s;flex-shrink:0}
.collapse-btn:hover{color:#a78bfa;background:#1e1b4b}
.expand-hint{color:#4c1d95;font-size:18px}

.sb-body{flex:1;overflow-y:auto;overflow-x:hidden;padding:10px 8px}
#sidebar.collapsed .nav-label,#sidebar.collapsed .section-title,#sidebar.collapsed .uc-name,#sidebar.collapsed .new-uc-label{display:none}
.section-title{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#334155;padding:0 6px;margin:4px 0 6px;white-space:nowrap}
.nav-item{display:flex;align-items:center;gap:9px;padding:7px 8px;border-radius:7px;cursor:pointer;color:#64748b;transition:background .15s,color .15s;white-space:nowrap;margin-bottom:2px}
#sidebar.collapsed .nav-item{justify-content:center;padding:8px 4px}
.nav-item:hover{background:#1e1b4b;color:#c4b5fd}
.nav-item.active{background:#1e1b4b;color:#a78bfa}
.nav-icon{font-size:15px;flex-shrink:0}
.nav-label{font-size:12px;font-weight:500}
.divider{height:1px;background:#1e1b4b;margin:8px 4px}
.uc-item{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;cursor:pointer;margin-bottom:1px;transition:background .15s}
#sidebar.collapsed .uc-item{justify-content:center;padding:6px 4px}
.uc-item:hover{background:#0f0f1a}
.uc-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.uc-name{font-size:11px;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.uc-item.active .uc-name{color:#22d3ee}
.new-uc-btn{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;cursor:pointer;border:1px dashed #4c1d95;color:#7c3aed;margin-top:6px;transition:background .15s;white-space:nowrap}
#sidebar.collapsed .new-uc-btn{justify-content:center}
.new-uc-btn:hover{background:#1a0a3a}

/* === Main === */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}

/* === Topbar === */
#topbar{background:#080712;border-bottom:1px solid #1e1b4b;padding:0 20px;height:48px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.breadcrumb{display:flex;align-items:center;gap:5px;font-size:12px}
.bc-seg{color:#475569;cursor:pointer;transition:color .15s;display:flex;align-items:center}
.bc-seg:hover{color:#a78bfa}
.bc-seg.current{color:#22d3ee;font-weight:600;cursor:default}
.bc-sep{color:#1e1b4b;font-size:16px;margin:0 2px}
.topbar-right{display:flex;align-items:center;gap:8px}
.gpu-pill{display:flex;align-items:center;gap:5px;background:#0f172a;border:1px solid #064e3b;border-radius:20px;padding:4px 10px;font-size:10px;color:#34d399;white-space:nowrap}
.gpu-dot{width:6px;height:6px;background:#34d399;border-radius:50%;animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.icon-btn{display:flex;align-items:center;gap:5px;background:#0f0f1a;border:1px solid #1e1b4b;border-radius:6px;padding:5px 10px;font-size:11px;color:#64748b;cursor:pointer;transition:border-color .15s,color .15s}
.icon-btn:hover{border-color:#4c1d95;color:#a78bfa}

/* === Content === */
#content{flex:1;overflow-y:auto;padding:20px}
.page-title{color:#a78bfa;font-size:18px;font-weight:700;margin-bottom:4px}
.page-sub{color:#475569;font-size:12px;margin-bottom:20px}

/* === New UC inline form === */
#new-uc-bar{display:none;background:#0c0a1e;border-bottom:1px solid #312e81;padding:8px 20px;align-items:center;gap:10px;flex-shrink:0}
#new-uc-bar.visible{display:flex}
#new-uc-bar input{background:#1e1b4b;border:1px solid #4338ca;border-radius:5px;padding:5px 10px;color:#e2e8f0;font-size:11px;outline:none}
#new-uc-bar input:focus{border-color:#7c3aed}
.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 14px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;border:none;transition:opacity .15s}
.btn-primary{background:linear-gradient(135deg,#7c3aed,#0891b2);color:#fff}
.btn-ghost{background:transparent;border:1px solid #312e81;color:#a78bfa}
.btn-danger{background:transparent;border:1px solid #7f1d1d;color:#f87171}
.btn:hover{opacity:.85}

/* === UC Cards === */
.uc-card{background:#0f0f1a;border:1px solid #1e1b4b;border-radius:10px;overflow:hidden;margin-bottom:10px}
.uc-card-header{padding:11px 14px;display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.uc-card-header:hover{background:#0a0a1e}
.uc-title{color:#a78bfa;font-weight:600;font-size:13px}
.phase-badge{padding:2px 8px;border-radius:10px;font-size:9px;font-weight:600}
.phase-1{background:#1e1b4b;color:#94a3b8}
.phase-2{background:#1c1917;color:#f59e0b}
.phase-3{background:#064e3b;color:#34d399}
.prs-val{font-size:10px;margin-left:4px}
.example-badge{background:#1e1b4b;color:#64748b;padding:2px 6px;border-radius:8px;font-size:9px}
.journey{display:flex;align-items:center;gap:3px;margin-left:auto}
.jd{width:8px;height:8px;border-radius:50%;background:#22d3ee}
.ja{width:8px;height:8px;border-radius:50%;background:#f59e0b;box-shadow:0 0 5px #f59e0b88}
.jl{width:8px;height:8px;border-radius:50%;border:1px solid #334155}
.jline{width:16px;height:1px;background:#1e1b4b}
.j-count{font-size:9px;margin-left:5px}

/* === Module Chips === */
.mod-strip{padding:6px 14px 8px;display:flex;gap:6px;border-top:1px solid #1e1b4b;background:#080a14;flex-wrap:wrap}
.mod-chip{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:6px;border:1px solid;cursor:pointer;font-size:10px;font-weight:500;transition:all .15s;white-space:nowrap;user-select:none}
.chip-arrow{font-size:9px;margin-left:2px;display:inline-block;transition:transform .2s}
.mod-chip.open .chip-arrow{transform:rotate(180deg)}
.chip-done{border-color:#064e3b;color:#34d399;background:#0a1a12}
.chip-done:hover{border-color:#0d9488;background:#0d1f17}
.chip-active{border-color:#7c3aed;color:#a78bfa;background:#12092a;box-shadow:0 0 8px #7c3aed44}
.chip-active:hover{border-color:#9d4edd}
.chip-locked{border-color:#1e293b;color:#334155;background:transparent;cursor:not-allowed}

/* === Module Panels === */
.mod-panel-wrap{max-height:0;overflow:hidden;transition:max-height .35s ease}
.mod-panel-wrap.open{max-height:1200px}
.mod-panel{border-top:1px solid #312e81;background:#0c0a1e;padding:14px}
.panel-top{display:flex;align-items:center;gap:8px;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #1e1b4b}
.panel-title{color:#a78bfa;font-weight:600;font-size:12px;flex:1}
.status-pill{padding:2px 8px;border-radius:8px;font-size:9px;font-weight:600}
.pill-done{background:#064e3b;color:#34d399}
.pill-running{background:#1c1917;color:#f59e0b}
.pill-error{background:#450a0a;color:#f87171}
.collapse-sm{display:inline-flex;align-items:center;gap:4px;font-size:10px;color:#475569;cursor:pointer;padding:3px 8px;border:1px solid #1e1b4b;border-radius:5px;transition:color .15s,border-color .15s}
.collapse-sm:hover{color:#a78bfa;border-color:#4c1d95}

/* === Forms === */
.field{margin-bottom:11px}
label{display:block;color:#64748b;font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;margin-bottom:5px}
input[type=text],input[type=number],select,textarea{width:100%;background:#080712;border:1px solid #1e1b4b;border-radius:6px;padding:7px 10px;color:#e2e8f0;font-size:11px;outline:none;transition:border-color .15s;font-family:inherit}
input:focus,select:focus,textarea:focus{border-color:#7c3aed}
textarea{resize:vertical;min-height:52px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.toggle-group{display:flex;border:1px solid #1e1b4b;border-radius:6px;overflow:hidden}
.topt{flex:1;padding:7px;text-align:center;color:#475569;cursor:pointer;font-size:10px;background:#080712;transition:all .15s}
.topt:not(:last-child){border-right:1px solid #1e1b4b}
.topt.sel{background:#1e1b4b;color:#a78bfa}
.hint{color:#475569;font-size:10px;margin-top:4px}
.btn-row{display:flex;justify-content:flex-end;gap:8px;margin-top:12px;padding-top:10px;border-top:1px solid #1e1b4b}
.section-lbl{color:#64748b;font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:7px}
.preview-card{background:#080712;border:1px solid #1e1b4b;border-radius:7px;padding:10px;margin-top:10px}
.preview-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #0f172a}
.preview-row:last-child{border-bottom:none}
.pk{color:#475569;font-size:10px}
.pv{color:#e2e8f0;font-size:10px;font-family:monospace}

/* === GPU Grid === */
.gpu-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:12px}
.gpu-card{background:#080712;border:1px solid;border-radius:7px;padding:8px 10px}
.gpu-free{border-color:#064e3b}
.gpu-busy{border-color:#7f1d1d}
.gpu-hd{display:flex;align-items:center;gap:5px;margin-bottom:5px;font-size:10px;font-weight:600}
.gpu-free .gpu-hd{color:#34d399}
.gpu-busy .gpu-hd{color:#f87171}
.vbar{height:4px;background:#1e1b4b;border-radius:3px;overflow:hidden;margin-bottom:3px}
.vfill{height:100%;border-radius:3px}
.gpu-free .vfill{background:linear-gradient(90deg,#059669,#34d399)}
.gpu-busy .vfill{background:linear-gradient(90deg,#dc2626,#f87171)}
.gpu-meta{font-size:9px;color:#475569}

/* === Pipeline Stepper === */
.pipeline{display:flex;flex-direction:column}
.pstep{display:flex;gap:10px;padding:7px 0;position:relative}
.pstep:not(:last-child)::after{content:'';position:absolute;left:10px;top:26px;bottom:-2px;width:1px;background:#1e1b4b}
.pico{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:11px;margin-top:1px}
.p-done{background:#064e3b;color:#34d399}
.p-running{background:#312e81;color:#a78bfa;box-shadow:0 0 7px #7c3aed66}
.p-pending{background:#0f172a;color:#334155;border:1px solid #1e293b}
.ptitle{font-size:11px;font-weight:600;margin-bottom:2px}
.pmeta{font-size:10px;color:#475569}
.pbar{height:4px;background:#1e1b4b;border-radius:3px;overflow:hidden;margin-top:5px}
.pfill{height:100%;background:linear-gradient(90deg,#7c3aed,#06b6d4);border-radius:3px;transition:width .5s}
.plog{background:#020817;border:1px solid #1e1b4b;border-radius:5px;padding:6px 9px;font-size:9px;font-family:monospace;color:#64748b;margin-top:5px;max-height:80px;overflow-y:auto;line-height:1.7}
.log-ok{color:#34d399}
.log-info{color:#60a5fa}
.log-warn{color:#f59e0b}
.log-err{color:#f87171}
.cur{display:inline-block;width:6px;height:10px;background:#a78bfa;animation:blink 1s infinite;vertical-align:middle}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

/* === Eval Stats === */
.eval-stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px}
.stat-card{background:#080712;border:1px solid #1e1b4b;border-radius:7px;padding:10px;text-align:center}
.stat-val{font-size:22px;font-weight:700;margin-bottom:3px}
.stat-label{font-size:9px;color:#475569;text-transform:uppercase;letter-spacing:.07em}
.stat-sub{font-size:9px;color:#334155;margin-top:2px}
.compare-row{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#080712;border:1px solid #1e1b4b;border-radius:6px;margin-bottom:5px}
.cmp-label{font-size:10px;color:#64748b;width:110px;flex-shrink:0}
.cmp-bars{flex:1;display:flex;flex-direction:column;gap:3px}
.cmp-bar{display:flex;align-items:center;gap:6px}
.bar-track{flex:1;height:6px;background:#1e1b4b;border-radius:3px;overflow:hidden}
.bar-a{height:100%;border-radius:3px;background:linear-gradient(90deg,#7c3aed,#a78bfa)}
.bar-b{height:100%;border-radius:3px;background:linear-gradient(90deg,#0891b2,#22d3ee)}
.bar-val{font-size:9px;color:#94a3b8;width:40px;text-align:right;flex-shrink:0}
.bar-name{font-size:9px;width:55px;flex-shrink:0}
.bar-name-a{color:#a78bfa}
.bar-name-b{color:#22d3ee}
</style>
</head>
<body>

<!-- ── Sidebar ── -->
<div id="sidebar">
  <div class="sb-header-exp">
    <div class="logo-mark">KV</div>
    <span class="logo-text">KVForge</span>
    <div class="collapse-btn" onclick="toggleSidebar()" title="Collapse">&#8249;</div>
  </div>
  <div class="sb-header-col" onclick="toggleSidebar()" title="Expand">
    <div class="logo-mark">KV</div>
    <div class="expand-hint">&#8250;</div>
  </div>
  <div class="sb-body">
    <div class="section-title">Workspace</div>
    <div class="nav-item active" onclick="navTo('hub')" title="Studio Hub">
      <span class="nav-icon">&#9783;</span>
      <span class="nav-label">Studio Hub</span>
    </div>
    <div class="nav-item" onclick="window.open('/kvq','_blank')" title="KVQ Live Stats">
      <span class="nav-icon">&#9096;</span>
      <span class="nav-label">KVQ Live Stats</span>
    </div>
    <div class="nav-item" onclick="window.open('/ab-eval/uc1','_blank')" title="A/B Reports">
      <span class="nav-icon">&#9638;</span>
      <span class="nav-label">A/B Reports</span>
    </div>
    <div class="nav-item" onclick="showSettings()" title="Settings">
      <span class="nav-icon">&#9881;</span>
      <span class="nav-label">Settings</span>
    </div>
    <div class="divider"></div>
    <div class="section-title">Use Cases</div>
    <div id="uc-list"></div>
    <div class="new-uc-btn" onclick="showNewUcForm()" title="New Use Case">
      <span style="font-size:14px">&#43;</span>
      <span class="new-uc-label nav-label">New Use Case</span>
    </div>
  </div>
</div>

<!-- ── Main ── -->
<div id="main">
  <div id="topbar">
    <div class="breadcrumb" id="breadcrumb">
      <div class="bc-seg" onclick="navTo('hub')">&#8962;</div>
      <span class="bc-sep">&#8250;</span>
      <div class="bc-seg current" id="bc-current">Studio Hub</div>
    </div>
    <div class="topbar-right">
      <div class="gpu-pill" id="gpu-pill" onclick="refreshGpu()">
        <div class="gpu-dot"></div>
        <span id="gpu-text">Checking GPUs...</span>
      </div>
      <div class="icon-btn" onclick="showApiKeys()">&#128273; API Keys</div>
    </div>
  </div>

  <!-- New UC form bar (hidden by default) -->
  <div id="new-uc-bar">
    <span style="color:#a78bfa;font-size:10px;font-weight:600">New use case:</span>
    <input id="new-uc-id" placeholder="id (e.g. medqa-2024)" style="width:150px">
    <input id="new-uc-name" placeholder="Display name" style="flex:1;max-width:240px">
    <div class="btn btn-primary" onclick="createUc()">Create</div>
    <div class="btn btn-ghost" onclick="hideNewUcForm()">Cancel</div>
  </div>

  <div id="content">
    <div class="page-title">Studio Hub</div>
    <div class="page-sub">Configure and run the KVForge pipeline on any dataset.</div>
    <div id="uc-cards"></div>
  </div>
</div>

<!-- ── API Keys Modal ── -->
<div id="keys-modal" style="display:none;position:fixed;inset:0;background:#000a;z-index:100;align-items:center;justify-content:center">
  <div style="background:#0f0f1a;border:1px solid #312e81;border-radius:12px;padding:24px;width:380px">
    <div style="color:#a78bfa;font-size:14px;font-weight:700;margin-bottom:16px">API Keys</div>
    <div class="field"><label>Gemini API Key</label>
      <input type="text" id="key-gemini" placeholder="AIza...">
      <div class="hint">Stored in browser only — never sent to server</div>
    </div>
    <div class="field"><label>OpenAI API Key</label>
      <input type="text" id="key-openai" placeholder="sk-...">
    </div>
    <div class="btn-row">
      <div class="btn btn-ghost" onclick="closeApiKeys()">Cancel</div>
      <div class="btn btn-primary" onclick="saveApiKeys()">Save Keys</div>
    </div>
  </div>
</div>

<script>
// ── State ───────────────────────────────────────────────────────────────────
const STATE = { ucs: [], openPanels: {}, pollingJob: null };

// ── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadApiKeysFromStorage();
  loadRegistry();
  refreshGpu();
});

// ── Sidebar ─────────────────────────────────────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('collapsed');
}

// ── Registry / UC list ──────────────────────────────────────────────────────
async function loadRegistry() {
  const res = await fetch('/studio/api/registry');
  const data = await res.json();
  STATE.ucs = data.use_cases || [];
  renderUcList();
  renderUcCards();
}

function renderUcList() {
  const el = document.getElementById('uc-list');
  el.innerHTML = STATE.ucs.map(uc => {
    const color = uc.phase === 3 ? '#22d3ee' : uc.phase === 2 ? '#f59e0b' : '#475569';
    return `<div class="uc-item" onclick="scrollToUc('${uc.id}')" title="${uc.display_name}">
      <div class="uc-dot" style="background:${color}"></div>
      <span class="uc-name">${uc.display_name || uc.id}</span>
    </div>`;
  }).join('');
}

// ── UC Cards ─────────────────────────────────────────────────────────────────
function renderUcCards() {
  const el = document.getElementById('uc-cards');
  el.innerHTML = STATE.ucs.map(uc => renderUcCard(uc)).join('');
}

function renderUcCard(uc) {
  const phase = uc.phase || 1;
  const prs = uc.prs ? uc.prs.toFixed(3) : '—';
  const prsColor = uc.prs >= 0.75 ? '#34d399' : uc.prs ? '#f59e0b' : '#475569';
  const dots = renderJourneyDots(uc);
  const typeTag = uc.type === 'example' ? '<span class="example-badge">Example</span>' : '';

  return `<div class="uc-card" id="uc-${uc.id}">
    <div class="uc-card-header" onclick="toggleUcCard('${uc.id}')">
      <span style="font-size:15px;color:#a78bfa">&#9638;</span>
      <span class="uc-title">${uc.display_name || uc.id}</span>
      ${typeTag}
      <span class="phase-badge phase-${phase}">Phase ${phase}</span>
      <span class="prs-val" style="color:${prsColor}">PRS ${prs}</span>
      ${dots}
    </div>
    <div class="mod-strip" id="chips-${uc.id}">
      ${renderChips(uc)}
    </div>
    ${renderPanels(uc)}
  </div>`;
}

function renderJourneyDots(uc) {
  // Determine completion of each module from uc_config + version
  const steps = getStepStatuses(uc);
  const dotHtml = steps.map((s, i) => {
    const dot = s === 'done' ? '<div class="jd"></div>'
               : s === 'active' ? '<div class="ja"></div>'
               : '<div class="jl"></div>';
    const line = i < steps.length - 1 ? '<div class="jline"></div>' : '';
    return dot + line;
  }).join('');
  const done = steps.filter(s => s === 'done').length;
  return `<div class="journey">${dotHtml}<span class="j-count" style="color:${done===5?'#34d399':'#64748b'}">${done}/5</span></div>`;
}

function getStepStatuses(uc) {
  // Returns ['done'|'active'|'locked'] for [data, vectordb, llm, train, eval]
  const cfg = STATE.configs && STATE.configs[uc.id];
  const hasData    = cfg && cfg.data && cfg.data.dataset_id;
  const hasVdb     = cfg && cfg.vectordb && cfg.vectordb.store;
  const hasLlm     = cfg && cfg.llm && cfg.llm.local_model;
  const hasTrain   = uc.phase >= 2;
  const hasEval    = uc.phase >= 2 && uc.prs;
  return [
    hasData ? 'done' : 'active',
    hasVdb  ? 'done' : hasData  ? 'active' : 'locked',
    hasLlm  ? 'done' : 'active',
    hasTrain? 'done' : hasLlm && hasData && hasVdb ? 'active' : 'locked',
    hasEval ? 'done' : hasTrain ? 'active' : 'locked',
  ];
}

// ── Module Chips ─────────────────────────────────────────────────────────────
function renderChips(uc) {
  const steps = getStepStatuses(uc);
  const labels = ['Data','VectorDB','LLM','Training','Evaluation'];
  const mods   = ['data','vdb','llm','train','eval'];
  return mods.map((m, i) => {
    const s = steps[i];
    const cls = s === 'done' ? 'chip-done' : s === 'active' ? 'chip-active' : 'chip-locked';
    const locked = s === 'locked';
    const onclick = locked ? '' : `onclick="togglePanel('${uc.id}','${m}')"`;
    const lockIcon = locked ? ' &#128274;' : '';
    return `<div class="mod-chip ${cls}" id="chip-${uc.id}-${m}" ${onclick} title="${locked ? getLockedReason(m) : ''}">
      ${labels[i]}${lockIcon}<span class="chip-arrow">&#9660;</span>
    </div>`;
  }).join('');
}

function getLockedReason(mod) {
  if (mod === 'vdb')   return 'Requires Data configured first';
  if (mod === 'train') return 'Requires Data, VectorDB and LLM configured';
  if (mod === 'eval')  return 'Requires at least one training round complete';
  return '';
}

// ── Module Panels ─────────────────────────────────────────────────────────────
function renderPanels(uc) {
  return ['data','vdb','llm','train','eval'].map(m =>
    `<div class="mod-panel-wrap" id="panel-${uc.id}-${m}">
       <div class="mod-panel" id="panel-body-${uc.id}-${m}">
         <div style="color:#475569;font-size:11px">Loading...</div>
       </div>
     </div>`
  ).join('');
}

// ── Panel Toggle ─────────────────────────────────────────────────────────────
async function togglePanel(ucId, mod) {
  const wrap  = document.getElementById(`panel-${ucId}-${mod}`);
  const chip  = document.getElementById(`chip-${ucId}-${mod}`);
  const isOpen = wrap.classList.contains('open');

  if (!isOpen) {
    // Load config if not cached
    if (!STATE.configs) STATE.configs = {};
    if (!STATE.configs[ucId]) {
      try {
        const r = await fetch(`/studio/api/uc/${ucId}/config`);
        STATE.configs[ucId] = await r.json();
      } catch(e) { STATE.configs[ucId] = {}; }
    }
    // Render panel content
    const body = document.getElementById(`panel-body-${ucId}-${mod}`);
    body.innerHTML = renderPanelContent(ucId, mod, STATE.configs[ucId]);
  }

  wrap.classList.toggle('open', !isOpen);
  if (chip) {
    chip.classList.toggle('open', !isOpen);
    const arrow = chip.querySelector('.chip-arrow');
    if (arrow) arrow.style.transform = isOpen ? '' : 'rotate(180deg)';
  }
}

function collapsePanel(ucId, mod) {
  togglePanel(ucId, mod); // Will close if open
}

// ── Panel Content Renderers ───────────────────────────────────────────────────
function renderPanelContent(ucId, mod, cfg) {
  switch(mod) {
    case 'data':  return renderDataPanel(ucId, cfg);
    case 'vdb':   return renderVdbPanel(ucId, cfg);
    case 'llm':   return renderLlmPanel(ucId, cfg);
    case 'train': return renderTrainPanel(ucId, cfg);
    case 'eval':  return renderEvalPanel(ucId, cfg);
    default: return '<div style="color:#475569">Unknown module</div>';
  }
}

function renderDataPanel(ucId, cfg) {
  const d = (cfg && cfg.data) || {};
  const isHf = !d.source_type || d.source_type === 'huggingface';
  return `<div class="panel-top">
    <span>&#9650;</span>
    <span class="panel-title">Data Source</span>
    ${d.dataset_id || d.source_path ? '<span class="status-pill pill-done">&#10003; Configured</span>' : ''}
    <div class="collapse-sm" onclick="togglePanel('${ucId}','data')">&#9650; Collapse</div>
  </div>
  <div class="field"><label>Source Type</label>
    <div class="toggle-group">
      <div class="topt ${isHf ? 'sel' : ''}" onclick="switchDataSource(this,'hf')">&#128194; HuggingFace</div>
      <div class="topt ${!isHf ? 'sel' : ''}" onclick="switchDataSource(this,'pdf')">&#128196; PDF / Local</div>
    </div>
  </div>
  <div id="data-hf-${ucId}" style="${isHf ? '' : 'display:none'}">
    <div class="row2">
      <div class="field"><label>Dataset ID</label><input type="text" id="d-dataset-${ucId}" value="${d.dataset_id || ''}"></div>
      <div class="field"><label>Split</label><select id="d-split-${ucId}">
        <option ${d.split==='train'?'selected':''}>train</option>
        <option ${d.split==='validation'?'selected':''}>validation</option>
        <option ${d.split==='test'?'selected':''}>test</option>
      </select></div>
    </div>
    <div class="row2">
      <div class="field"><label>Text Column</label><input type="text" id="d-col-${ucId}" value="${d.text_column || 'text'}"></div>
      <div class="field"><label>Max Rows</label><input type="number" id="d-rows-${ucId}" value="${d.max_rows || 5000}"></div>
    </div>
  </div>
  <div id="data-pdf-${ucId}" style="${!isHf ? '' : 'display:none'}">
    <div class="field"><label>Local Path (on server)</label>
      <input type="text" id="d-path-${ucId}" value="${d.source_path || ''}" placeholder="examples/myuc/data/">
      <div class="hint">Path relative to project root on the EC2 server</div>
    </div>
  </div>
  <div class="btn-row">
    <div class="btn btn-ghost" onclick="togglePanel('${ucId}','data')">&#9650; Collapse</div>
    <div class="btn btn-primary" onclick="saveDataConfig('${ucId}')">&#10003; Save</div>
  </div>`;
}

function renderVdbPanel(ucId, cfg) {
  const v = (cfg && cfg.vectordb) || {};
  const stores = ['qdrant','chromadb','faiss'];
  const storeTabs = stores.map(s =>
    `<div class="topt ${(v.store||'qdrant')===s ? 'sel' : ''}" onclick="selToggle(this)">${s.charAt(0).toUpperCase()+s.slice(1)}</div>`
  ).join('');
  return `<div class="panel-top">
    <span>&#9641;</span>
    <span class="panel-title">VectorDB Configuration</span>
    <div class="collapse-sm" onclick="togglePanel('${ucId}','vdb')">&#9650; Collapse</div>
  </div>
  <div class="field"><label>Store</label>
    <div class="toggle-group" id="vdb-store-${ucId}">${storeTabs}</div>
  </div>
  <div class="row3">
    <div class="field"><label>Dimensions</label><input type="number" id="vdb-dim-${ucId}" value="${v.dimensions||384}"></div>
    <div class="field"><label>Chunk Size</label><input type="number" id="vdb-chunk-${ucId}" value="${v.chunk_size||512}"></div>
    <div class="field"><label>Overlap</label><input type="number" id="vdb-overlap-${ucId}" value="${v.chunk_overlap||64}"></div>
  </div>
  <div class="row2">
    <div class="field"><label>Embedding Model</label>
      <select id="vdb-embed-${ucId}">
        <option ${(v.embedding_model||'').includes('MiniLM') ? 'selected' : ''}>sentence-transformers/all-MiniLM-L6-v2</option>
        <option ${(v.embedding_model||'').includes('bge-small') ? 'selected' : ''}>BAAI/bge-small-en-v1.5</option>
      </select>
    </div>
    <div class="field"><label>Index Type</label>
      <select id="vdb-idx-${ucId}">
        <option>HNSW (recommended)</option><option>Flat</option><option>IVF</option>
      </select>
    </div>
  </div>
  <div class="btn-row">
    <div class="btn btn-ghost" onclick="togglePanel('${ucId}','vdb')">&#9650; Collapse</div>
    <div class="btn btn-primary" onclick="saveVdbConfig('${ucId}')">&#10003; Save & Index</div>
  </div>`;
}

function renderLlmPanel(ucId, cfg) {
  const l = (cfg && cfg.llm) || {};
  const geminiKey = localStorage.getItem('kvf_gemini_key');
  return `<div class="panel-top">
    <span>&#9711;</span>
    <span class="panel-title">Model Selection</span>
    <div class="collapse-sm" onclick="togglePanel('${ucId}','llm')">&#9650; Collapse</div>
  </div>
  <div class="row2">
    <div>
      <div class="section-lbl">&#9881; Local Model (fine-tuned)</div>
      <div class="field"><label>Base Model</label>
        <select id="llm-model-${ucId}">
          <option ${(l.local_model||'').includes('Llama') ? 'selected':''}>${'meta-llama/Llama-3.2-3B-Instruct'}</option>
          <option ${(l.local_model||'').includes('Qwen')  ? 'selected':''}>Qwen/Qwen2.5-3B-Instruct</option>
        </select>
      </div>
      <div class="field"><label>Quantization</label>
        <div class="toggle-group">
          <div class="topt ${l.quantization==='fp16'?'sel':''}" onclick="selToggle(this)">fp16</div>
          <div class="topt ${(l.quantization||'4bit')==='4bit'?'sel':''}" onclick="selToggle(this)">4-bit</div>
          <div class="topt ${l.quantization==='8bit'?'sel':''}" onclick="selToggle(this)">8-bit</div>
        </div>
      </div>
    </div>
    <div>
      <div class="section-lbl">&#9729; Comparison Model</div>
      <div class="field"><label>Provider</label>
        <div class="toggle-group">
          <div class="topt ${(l.comparison_provider||'gemini')==='gemini'?'sel':''}" onclick="selToggle(this)">Gemini</div>
          <div class="topt ${l.comparison_provider==='openai'?'sel':''}" onclick="selToggle(this)">OpenAI</div>
        </div>
      </div>
      <div class="field"><label>Model</label>
        <select id="llm-cmp-${ucId}">
          <option>gemini-1.5-flash</option><option>gemini-1.5-pro</option>
        </select>
      </div>
      ${geminiKey
        ? '<div style="background:#0f172a;border:1px solid #312e81;border-radius:5px;padding:6px 10px;font-size:10px;color:#a78bfa;margin-top:4px">&#10003; Gemini key loaded</div>'
        : '<div style="background:#1c0a0a;border:1px solid #7f1d1d;border-radius:5px;padding:6px 10px;font-size:10px;color:#f87171;margin-top:4px;cursor:pointer" onclick="showApiKeys()">&#128273; Add Gemini key</div>'}
    </div>
  </div>
  <div class="btn-row">
    <div class="btn btn-ghost" onclick="togglePanel('${ucId}','llm')">&#9650; Collapse</div>
    <div class="btn btn-primary" onclick="saveLlmConfig('${ucId}')">&#10003; Save</div>
  </div>`;
}

function renderTrainPanel(ucId, cfg) {
  return `<div class="panel-top">
    <span>&#8635;</span>
    <span class="panel-title">Training Pipeline</span>
    <span class="status-pill" id="train-status-${ucId}"></span>
    <div class="collapse-sm" onclick="togglePanel('${ucId}','train')">&#9650; Collapse</div>
  </div>
  <div class="section-lbl">GPU Status <span style="color:#475569;font-weight:400;cursor:pointer" onclick="refreshGpuPanel('${ucId}')">(refresh)</span></div>
  <div class="gpu-grid" id="gpu-grid-${ucId}">
    <div style="color:#475569;font-size:10px">Loading GPU info...</div>
  </div>
  <div class="section-lbl">Pipeline Steps</div>
  <div class="pipeline" id="pipeline-${ucId}">
    ${renderPipelineSteps(ucId, cfg)}
  </div>
  <div class="btn-row">
    <div class="btn btn-danger" id="stop-btn-${ucId}" style="display:none" onclick="stopJob('${ucId}')">&#9632; Stop</div>
    <div class="btn btn-ghost" onclick="togglePanel('${ucId}','train')">&#9650; Collapse</div>
    <div class="btn btn-primary" id="run-btn-${ucId}" onclick="runNextStep('${ucId}')">&#9654; Run Pipeline</div>
  </div>`;
}

function renderPipelineSteps(ucId, cfg) {
  const uc = STATE.ucs.find(u => u.id === ucId) || {};
  const phase = uc.phase || 1;
  const history = uc.prs_history || [];
  const steps = [
    {label: 'KV Indexing', done: phase >= 1, meta: ''},
    {label: 'LoRA Training — Round 1', done: phase >= 2, meta: history[0] ? `PRS ${history[0].prs.toFixed(3)}` : ''},
    {label: 'KV Recompute', done: phase >= 2, meta: ''},
    {label: 'PRS Evaluation', done: phase >= 2, meta: ''},
    {label: 'LoRA Training — Round 2', done: phase >= 3, meta: history[1] ? `PRS ${history[1].prs.toFixed(3)}` : ''},
  ];
  return steps.map(s => `
    <div class="pstep">
      <div class="pico ${s.done ? 'p-done' : 'p-pending'}">${s.done ? '&#10003;' : '&#9675;'}</div>
      <div>
        <div class="ptitle" style="color:${s.done ? '#34d399' : '#334155'}">${s.label}</div>
        <div class="pmeta">${s.meta}</div>
      </div>
    </div>`).join('');
}

function renderEvalPanel(ucId, cfg) {
  const uc = STATE.ucs.find(u => u.id === ucId) || {};
  return `<div class="panel-top">
    <span>&#9638;</span>
    <span class="panel-title">Evaluation Results</span>
    <div class="collapse-sm" onclick="togglePanel('${ucId}','eval')">&#9650; Collapse</div>
  </div>
  <div class="eval-stats">
    <div class="stat-card"><div class="stat-val" style="color:#22d3ee" id="eval-win-${ucId}">—</div>
      <div class="stat-label">Win Rate</div><div class="stat-sub">PRS &#8805; 0.75</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#a78bfa" id="eval-prs-${ucId}">${uc.prs ? uc.prs.toFixed(3) : '—'}</div>
      <div class="stat-label">Avg PRS</div><div class="stat-sub">Phase ${uc.phase||1}</div></div>
    <div class="stat-card"><div class="stat-val" style="color:#34d399" id="eval-lat-${ucId}">—</div>
      <div class="stat-label">Speed Gain</div><div class="stat-sub">vs comparison model</div></div>
  </div>
  <div class="btn-row">
    <div class="btn btn-ghost" onclick="runAbEval('${ucId}')">&#8635; Re-run Eval</div>
    <div class="btn btn-primary" onclick="window.open('/ab-eval/${ucId}','_blank')">&#10138; Full A/B Report</div>
  </div>`;
}

// ── Toggle helpers ────────────────────────────────────────────────────────────
function selToggle(el) {
  const group = el.parentElement;
  group.querySelectorAll('.topt').forEach(e => e.classList.remove('sel'));
  el.classList.add('sel');
}

function switchDataSource(el, type) {
  selToggle(el);
  // Scope to the current panel only — find ucId from nearest mod-panel ancestor
  const panel = el.closest('.mod-panel');
  if (!panel) return;
  const ucId = panel.id.replace('panel-body-', '').replace(/-data$/, '');
  const hfEl  = document.getElementById(`data-hf-${ucId}`);
  const pdfEl = document.getElementById(`data-pdf-${ucId}`);
  if (hfEl)  hfEl.style.display  = type === 'hf'  ? '' : 'none';
  if (pdfEl) pdfEl.style.display = type === 'pdf' ? '' : 'none';
}

// ── Config save helpers ───────────────────────────────────────────────────────
async function saveDataConfig(ucId) {
  const src = document.getElementById(`d-dataset-${ucId}`) ? 'huggingface' : 'pdf';
  const payload = { data: {
    source_type:  'huggingface',
    dataset_id:   document.getElementById(`d-dataset-${ucId}`)?.value || '',
    split:        document.getElementById(`d-split-${ucId}`)?.value || 'train',
    text_column:  document.getElementById(`d-col-${ucId}`)?.value || 'text',
    max_rows:     parseInt(document.getElementById(`d-rows-${ucId}`)?.value || '5000'),
  }};
  await fetch(`/studio/api/uc/${ucId}/config`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (STATE.configs) delete STATE.configs[ucId];
  await loadRegistry();
}

async function saveVdbConfig(ucId) {
  const storeEl = document.querySelector(`#vdb-store-${ucId} .sel`);
  const payload = { vectordb: {
    store:           storeEl ? storeEl.textContent.trim().toLowerCase() : 'qdrant',
    dimensions:      parseInt(document.getElementById(`vdb-dim-${ucId}`)?.value || '384'),
    chunk_size:      parseInt(document.getElementById(`vdb-chunk-${ucId}`)?.value || '512'),
    chunk_overlap:   parseInt(document.getElementById(`vdb-overlap-${ucId}`)?.value || '64'),
    embedding_model: document.getElementById(`vdb-embed-${ucId}`)?.value || '',
    index_type:      'hnsw',
  }};
  await fetch(`/studio/api/uc/${ucId}/config`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (STATE.configs) delete STATE.configs[ucId];
  await loadRegistry();
}

async function saveLlmConfig(ucId) {
  const payload = { llm: {
    local_model: document.getElementById(`llm-model-${ucId}`)?.value || '',
    vllm_url: '',
    comparison_model: document.getElementById(`llm-cmp-${ucId}`)?.value || 'gemini-1.5-flash',
  }};
  await fetch(`/studio/api/uc/${ucId}/config`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (STATE.configs) delete STATE.configs[ucId];
}

// ── GPU ───────────────────────────────────────────────────────────────────────
async function refreshGpu() {
  try {
    const r = await fetch('/studio/api/gpu-check', {method:'POST'});
    const data = await r.json();
    const pill = document.getElementById('gpu-pill');
    const txt  = document.getElementById('gpu-text');
    if (data.error) {
      pill.style.borderColor = '#7f1d1d'; txt.textContent = 'GPU unavailable';
      return;
    }
    const free = data.gpus.filter(g => g.status === 'free').length;
    pill.style.borderColor = free > 0 ? '#064e3b' : '#7f1d1d';
    txt.textContent = `${free}/${data.gpus.length} GPUs free`;
  } catch(e) { document.getElementById('gpu-text').textContent = 'GPU check failed'; }
}

async function refreshGpuPanel(ucId) {
  const grid = document.getElementById(`gpu-grid-${ucId}`);
  if (!grid) return;
  try {
    const r = await fetch('/studio/api/gpu-check', {method:'POST'});
    const data = await r.json();
    if (data.error) { grid.innerHTML = `<div style="color:#f87171;font-size:10px">${data.error}</div>`; return; }
    grid.innerHTML = data.gpus.map(g => `
      <div class="gpu-card ${g.status === 'free' ? 'gpu-free' : 'gpu-busy'}">
        <div class="gpu-hd">&#9641; GPU ${g.id} — ${g.status === 'free' ? 'Free' : g.process || 'Busy'}</div>
        <div class="vbar"><div class="vfill" style="width:${Math.round(g.used_gb/g.total_gb*100)}%"></div></div>
        <div class="gpu-meta">${g.used_gb} / ${g.total_gb} GB used</div>
      </div>`).join('');
  } catch(e) { grid.innerHTML = '<div style="color:#f87171;font-size:10px">Failed to fetch GPU status</div>'; }
}

// ── Pipeline run ──────────────────────────────────────────────────────────────
async function runNextStep(ucId) {
  const uc = STATE.ucs.find(u => u.id === ucId);
  const phase = uc ? (uc.phase || 1) : 1;
  // Determine next step based on phase
  const step = phase < 1 ? 'index' : phase < 2 ? 'train' : 'train';

  const r = await fetch('/studio/api/run-step', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({uc_id: ucId, step})
  });
  if (!r.ok) {
    const err = await r.json();
    alert(err.detail || 'Failed to start pipeline');
    return;
  }
  const {job_id} = await r.json();
  startStreaming(ucId, job_id);
}

async function runAbEval(ucId) {
  const r = await fetch('/studio/api/run-step', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({uc_id: ucId, step: 'ab-eval'})
  });
  if (!r.ok) { alert('Failed to start A/B eval'); return; }
  const {job_id} = await r.json();
  startStreaming(ucId, job_id);
}

function startStreaming(ucId, jobId) {
  const logEl = document.getElementById(`pipeline-${ucId}`);
  const stopBtn = document.getElementById(`stop-btn-${ucId}`);
  const runBtn  = document.getElementById(`run-btn-${ucId}`);
  if (stopBtn) stopBtn.style.display = '';
  if (runBtn)  runBtn.style.display  = 'none';

  // Append live log area
  if (logEl) {
    logEl.innerHTML += `<div class="plog" id="log-${ucId}"></div>`;
  }

  const es = new EventSource(`/studio/api/stream/${jobId}`);
  STATE.pollingJob = jobId;

  // Start polling registry for status updates
  const pollInterval = setInterval(() => loadRegistry(), 5000);

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    const logDiv = document.getElementById(`log-${ucId}`);
    if (msg.type === 'log' && logDiv) {
      logDiv.innerHTML += `<div>${escHtml(msg.line)}</div>`;
      logDiv.scrollTop = logDiv.scrollHeight;
    }
    if (msg.type === 'done' || msg.type === 'error') {
      es.close();
      clearInterval(pollInterval);
      STATE.pollingJob = null;
      if (stopBtn) stopBtn.style.display = 'none';
      if (runBtn)  runBtn.style.display  = '';
      loadRegistry();
    }
  };
  es.onerror = () => { es.close(); clearInterval(pollInterval); };
}

async function stopJob(ucId) {
  if (!STATE.pollingJob) return;
  await fetch(`/studio/api/job/${STATE.pollingJob}`, {method:'DELETE'});
  STATE.pollingJob = null;
}

// ── New UC ────────────────────────────────────────────────────────────────────
function showNewUcForm() { document.getElementById('new-uc-bar').classList.add('visible'); }
function hideNewUcForm() { document.getElementById('new-uc-bar').classList.remove('visible'); }

async function createUc() {
  const id   = document.getElementById('new-uc-id').value.trim();
  const name = document.getElementById('new-uc-name').value.trim() || id;
  if (!id) { alert('ID is required'); return; }
  const r = await fetch('/studio/api/uc/new', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id, display_name: name})
  });
  if (!r.ok) { alert('Failed to create use case'); return; }
  hideNewUcForm();
  document.getElementById('new-uc-id').value = '';
  document.getElementById('new-uc-name').value = '';
  await loadRegistry();
}

// ── API Keys ──────────────────────────────────────────────────────────────────
function showApiKeys() {
  document.getElementById('key-gemini').value = localStorage.getItem('kvf_gemini_key') || '';
  document.getElementById('key-openai').value = localStorage.getItem('kvf_openai_key') || '';
  document.getElementById('keys-modal').style.display = 'flex';
}
function closeApiKeys() { document.getElementById('keys-modal').style.display = 'none'; }
function saveApiKeys() {
  localStorage.setItem('kvf_gemini_key', document.getElementById('key-gemini').value);
  localStorage.setItem('kvf_openai_key', document.getElementById('key-openai').value);
  closeApiKeys();
}
function loadApiKeysFromStorage() {
  // Keys are available via localStorage in JS; no server round-trip needed
}

// ── Nav helpers ───────────────────────────────────────────────────────────────
function navTo(where) {
  if (where === 'hub') {
    document.getElementById('bc-current').textContent = 'Studio Hub';
    loadRegistry();
  }
}
function scrollToUc(ucId) {
  const el = document.getElementById(`uc-${ucId}`);
  if (el) el.scrollIntoView({behavior:'smooth', block:'start'});
}
function toggleUcCard(ucId) {
  const chips = document.getElementById(`chips-${ucId}`);
  if (chips) chips.style.display = chips.style.display === 'none' ? '' : chips.style.display;
}
function showSettings() { alert('Settings coming soon.'); }
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>
```

- [ ] **Step 3: Verify the page loads at /studio**

Start the portal locally (or on EC2) and check:
```bash
cd /Users/hemant/Downloads/RoPE/qdrant
python kvforge_portal.py &
sleep 3
curl -s http://localhost:8080/studio | head -5
kill %1
```
Expected: `<!DOCTYPE html>` in response

- [ ] **Step 4: Commit**

```bash
git add templates/studio/hub.html
git commit -m "feat: studio hub HTML — sidebar, topbar, breadcrumb, UC cards, all 5 module panels"
```

---

## Task 8: Wire together + end-to-end smoke test

**Files:**
- Modify: `tests/test_studio_routes.py` (add end-to-end test)

- [ ] **Step 1: Add smoke test**

Add to `tests/test_studio_routes.py`:

```python
def test_hub_page_returns_html(tmp_root):
    """Smoke test: /studio serves the hub HTML page after migration."""
    with patch("studio.routes.ROOT", tmp_root), \
         patch("studio.api.ROOT", tmp_root), \
         patch("studio.migration.ROOT", tmp_root):
        import importlib, studio.routes as routes
        importlib.reload(routes)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(routes.router)
        # Create templates dir
        tpl_dir = tmp_root / "templates" / "studio"
        tpl_dir.mkdir(parents=True, exist_ok=True)
        (tpl_dir / "hub.html").write_text("<html><body>KVForge Studio</body></html>")
        # Patch TEMPLATES path in routes
        import studio.routes as r
        r.TEMPLATES = tpl_dir
        c = TestClient(app)
        resp = c.get("/")
        assert resp.status_code == 200
        assert "KVForge Studio" in resp.text
```

- [ ] **Step 2: Run all studio tests**

```bash
python -m pytest tests/test_studio_job_manager.py tests/test_studio_gpu_monitor.py tests/test_studio_migration.py tests/test_studio_routes.py -v
```
Expected: All tests PASS

- [ ] **Step 3: Run existing test suite to confirm no regressions**

```bash
python -m pytest tests/ -v --ignore=tests/test_studio_job_manager.py --ignore=tests/test_studio_gpu_monitor.py --ignore=tests/test_studio_migration.py --ignore=tests/test_studio_routes.py -x 2>&1 | tail -20
```
Expected: No new failures

- [ ] **Step 4: Final commit**

```bash
git add tests/test_studio_routes.py
git commit -m "test: add studio smoke test and verify no regressions"
```

---

## Deployment on EC2

After all tasks are complete, deploy to EC2:

```bash
# On EC2: pull latest and restart portal
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
cd /home/ubuntu/qdrant
git pull
# Restart portal (auto-runs migration on first /studio request)
pkill -f 'kvforge_portal.py' || true
sleep 2
tmux new-session -d -s portal 'cd /home/ubuntu/qdrant && /home/ubuntu/qdrant/venv/bin/python kvforge_portal.py --port 8080 > logs/portal.log 2>&1'
echo 'Portal restarted'
"
```

Verify at: `http://13.221.47.200:8080/studio`
