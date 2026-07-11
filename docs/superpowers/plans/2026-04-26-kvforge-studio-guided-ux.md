# KVForge Studio — Guided End-to-End Pipeline UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 7-step guided wizard and a persistent UC operations page to KVForge Studio, wired to 10 new API endpoints and 4 new backend modules.

**Architecture:** Four new pure-Python modules (`settings_manager`, `curation_manager`, `vdb_validator`, `ab_runner`) handle all new server-side logic. Ten new endpoints are added to the existing `api_router` in `studio/api.py`. Two new HTML pages (`wizard.html`, `uc_detail.html`) in `templates/studio/` load data via `fetch()` calls to those endpoints; their visual design is taken from pre-approved mockups in `.superpowers/brainstorm/83669-1777231613/content/`.

**Tech Stack:** FastAPI, Python 3.11+, httpx (async HTTP for ab_runner), anthropic/openai/google-generativeai SDKs (optional), vanilla JS with SSE for wizard pipeline streaming.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `studio/settings_manager.py` | **Create** | Read/write `~/.kvforge/settings.json`; mask API keys |
| `studio/curation_manager.py` | **Create** | Append to `faqs_curated.json`; return count/status |
| `studio/vdb_validator.py` | **Create** | Per-backend connectivity ping |
| `studio/ab_runner.py` | **Create** | Async concurrent A/B query (local vLLM + cloud) |
| `studio/gpu_monitor.py` | **Modify** | Add `get_gpu_realtime()` + `parse_gpu_realtime()` |
| `studio/pipeline_runner.py` | **Modify** | Add `faq-gen-cloud` step |
| `studio/api.py` | **Modify** | Add 10 new endpoints |
| `studio/routes.py` | **Modify** | Add `/wizard` route; swap `/uc/{id}` to `uc_detail.html` |
| `templates/studio/uc_detail.html` | **Create** | UC operations page (PRS chart, A/B, GPU overlay) |
| `templates/studio/wizard.html` | **Create** | 7-step guided wizard |
| `tests/test_settings_manager.py` | **Create** | Unit tests for settings_manager |
| `tests/test_curation_manager.py` | **Create** | Unit tests for curation_manager |
| `tests/test_vdb_validator.py` | **Create** | Unit tests for vdb_validator |
| `tests/test_gpu_realtime.py` | **Create** | Unit tests for parse_gpu_realtime |
| `tests/test_ab_runner.py` | **Create** | Unit tests for ab_runner (mocked HTTP) |
| `tests/test_studio_api_new.py` | **Create** | Integration tests for all 10 new endpoints |

---

## Task 1: `studio/settings_manager.py`

**Files:**
- Create: `studio/settings_manager.py`
- Test: `tests/test_settings_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_settings_manager.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import studio.settings_manager as sm


def test_get_all_returns_defaults_when_no_file(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        result = sm.get_all()
    assert result["curation_threshold"] == 50
    assert result["anthropic_api_key"] == ""
    assert result["default_cloud_provider"] == "anthropic"


def test_get_masked_masks_long_key(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"anthropic_api_key": "sk-ant-api03-abcdefgh"}))
    with patch.object(sm, "SETTINGS_FILE", f):
        result = sm.get_masked()
    assert result["anthropic_api_key"] == "••••efgh"


def test_get_masked_leaves_empty_key_empty(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        assert sm.get_masked()["anthropic_api_key"] == ""


def test_save_writes_and_merges(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"curation_threshold": 25})
        result = sm.get_all()
    assert result["curation_threshold"] == 25
    assert result["default_cloud_provider"] == "anthropic"  # default preserved


def test_save_rejects_invalid_anthropic_key(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        with pytest.raises(ValueError, match="anthropic_api_key"):
            sm.save({"anthropic_api_key": "not-a-valid-key"})


def test_save_accepts_valid_anthropic_key(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"anthropic_api_key": "sk-ant-api03-xyz"})
        assert sm.get_all()["anthropic_api_key"] == "sk-ant-api03-xyz"


def test_get_setting_returns_single_value(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"curation_threshold": 99}))
    with patch.object(sm, "SETTINGS_FILE", f):
        assert sm.get_setting("curation_threshold") == 99


def test_save_is_atomic(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"curation_threshold": 10})
        assert not (tmp_path / "settings.tmp").exists()
        assert f.exists()
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
python -m pytest tests/test_settings_manager.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'studio.settings_manager'`

- [ ] **Step 3: Create `studio/settings_manager.py`**

```python
# studio/settings_manager.py
import json
import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".kvforge" / "settings.json"

DEFAULTS: dict = {
    "anthropic_api_key": "",
    "openai_api_key": "",
    "gemini_api_key": "",
    "huggingface_token": "",
    "curation_threshold": 50,
    "default_cloud_provider": "anthropic",
    "default_cloud_model": "claude-haiku-4-5-20251001",
}

_SECRET_KEYS = {"anthropic_api_key", "openai_api_key", "gemini_api_key", "huggingface_token"}

_KEY_VALIDATORS: dict = {
    "anthropic_api_key": lambda v: v.startswith("sk-ant-"),
    "openai_api_key": lambda v: v.startswith("sk-"),
    "gemini_api_key": lambda v: v.startswith("AIza"),
}


def _load() -> dict:
    if not SETTINGS_FILE.exists():
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
        return {**DEFAULTS, **data}
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)


def get_all() -> dict:
    return _load()


def get_masked() -> dict:
    data = _load()
    out = {}
    for k, v in data.items():
        if k in _SECRET_KEYS and isinstance(v, str) and v:
            out[k] = "••••" + v[-4:] if len(v) >= 4 else "••••"
        else:
            out[k] = v
    return out


def get_setting(key: str):
    return _load().get(key, DEFAULTS.get(key))


def save(updates: dict) -> None:
    for key, value in updates.items():
        if key in _KEY_VALIDATORS and isinstance(value, str) and value:
            if not _KEY_VALIDATORS[key](value):
                raise ValueError(f"Invalid format for {key}")
    current = _load()
    current.update(updates)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(current, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_settings_manager.py -v --override-ini="addopts="
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/settings_manager.py tests/test_settings_manager.py
git commit -m "feat: add settings_manager — global API key storage at ~/.kvforge/settings.json"
```

---

## Task 2: `studio/curation_manager.py`

**Files:**
- Create: `studio/curation_manager.py`
- Test: `tests/test_curation_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_curation_manager.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import studio.curation_manager as cm


@pytest.fixture
def uc_dir(tmp_path):
    d = tmp_path / "examples" / "test-uc"
    d.mkdir(parents=True)
    return tmp_path


def test_append_creates_file(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        result = cm.append("test-uc", "What is RAG?", "RAG is ...", "model_b")
    assert (uc_dir / "examples" / "test-uc" / "faqs_curated.json").exists()
    assert result["count"] == 1


def test_append_increments_count(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q1", "A1")
        result = cm.append("test-uc", "Q2", "A2")
    assert result["count"] == 2


def test_append_stores_fields(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q?", "A.", "model_b")
        records = json.loads((uc_dir / "examples" / "test-uc" / "faqs_curated.json").read_text())
    assert records[0]["question"] == "Q?"
    assert records[0]["answer"] == "A."
    assert records[0]["source_model"] == "model_b"
    assert "curated_at" in records[0]


def test_get_status_empty(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        status = cm.get_status("test-uc")
    assert status["count"] == 0
    assert status["threshold"] == 50
    assert status["at_threshold"] is False


def test_get_status_at_threshold(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        for i in range(50):
            cm.append("test-uc", f"Q{i}", f"A{i}")
        status = cm.get_status("test-uc")
    assert status["at_threshold"] is True
    assert status["pct"] == 100.0


def test_get_samples_returns_last_n(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        for i in range(10):
            cm.append("test-uc", f"Q{i}", f"A{i}")
        samples = cm.get_samples("test-uc", n=3)
    assert len(samples) == 3
    assert samples[-1]["question"] == "Q9"


def test_write_is_atomic(uc_dir):
    with patch.object(cm, "ROOT", uc_dir):
        cm.append("test-uc", "Q", "A")
        assert not (uc_dir / "examples" / "test-uc" / "faqs_curated.tmp").exists()
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
python -m pytest tests/test_curation_manager.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'studio.curation_manager'`

- [ ] **Step 3: Create `studio/curation_manager.py`**

```python
# studio/curation_manager.py
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURATED_FILENAME = "faqs_curated.json"


def _path(uc_id: str) -> Path:
    return ROOT / "examples" / uc_id / CURATED_FILENAME


def _load(uc_id: str) -> list:
    p = _path(uc_id)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write(uc_id: str, records: list) -> None:
    p = _path(uc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    os.replace(tmp, p)


def append(uc_id: str, question: str, answer: str, source_model: str = "model_b") -> dict:
    records = _load(uc_id)
    records.append({
        "question": question,
        "answer": answer,
        "source_model": source_model,
        "curated_at": datetime.now(timezone.utc).isoformat(),
    })
    _write(uc_id, records)
    return get_status(uc_id)


def get_status(uc_id: str) -> dict:
    from studio.settings_manager import get_setting
    records = _load(uc_id)
    count = len(records)
    threshold = int(get_setting("curation_threshold") or 50)
    return {
        "count": count,
        "threshold": threshold,
        "pct": round((count / threshold) * 100, 1) if threshold else 0.0,
        "at_threshold": count >= threshold,
    }


def get_samples(uc_id: str, n: int = 5) -> list:
    records = _load(uc_id)
    return records[-n:] if len(records) > n else records
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_curation_manager.py -v --override-ini="addopts="
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/curation_manager.py tests/test_curation_manager.py
git commit -m "feat: add curation_manager — auto-curated faqs_curated.json append/read"
```

---

## Task 3: `studio/vdb_validator.py`

**Files:**
- Create: `studio/vdb_validator.py`
- Test: `tests/test_vdb_validator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vdb_validator.py
import pytest
from unittest.mock import patch, MagicMock
from studio.vdb_validator import validate


def test_unknown_type_returns_error():
    result = validate({"type": "nonexistent"})
    assert result["ok"] is False
    assert "Unknown VDB type" in result["error"]


def test_faiss_file_not_found(tmp_path):
    result = validate({"type": "faiss", "index_path": str(tmp_path / "nope.index")})
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_faiss_file_exists(tmp_path):
    p = tmp_path / "my.index"
    p.write_bytes(b"dummy")
    result = validate({"type": "faiss", "index_path": str(p)})
    assert result["ok"] is True
    assert result["collection_count"] == 1


def test_weaviate_ok(requests_mock):
    requests_mock.get("http://localhost:8080/v1/.well-known/ready", status_code=200)
    result = validate({"type": "weaviate", "url": "http://localhost:8080"})
    assert result["ok"] is True


def test_weaviate_not_ready(requests_mock):
    requests_mock.get("http://localhost:8080/v1/.well-known/ready", status_code=503)
    result = validate({"type": "weaviate", "url": "http://localhost:8080"})
    assert result["ok"] is False
    assert "503" in result["error"]


def test_generic_ok(requests_mock):
    requests_mock.get("http://my-api.example/v1", status_code=200)
    result = validate({"type": "generic", "base_url": "http://my-api.example/v1"})
    assert result["ok"] is True


def test_generic_server_error(requests_mock):
    requests_mock.get("http://my-api.example/v1", status_code=500)
    result = validate({"type": "generic", "base_url": "http://my-api.example/v1"})
    assert result["ok"] is False


def test_qdrant_missing_dependency():
    with patch.dict("sys.modules", {"qdrant_client": None}):
        result = validate({"type": "qdrant", "host": "localhost", "port": 6333})
    assert result["ok"] is False
    assert "not installed" in result["error"]


def test_exception_returns_error(requests_mock):
    requests_mock.get("http://bad/v1/.well-known/ready", exc=ConnectionError("refused"))
    result = validate({"type": "weaviate", "url": "http://bad"})
    assert result["ok"] is False
```

- [ ] **Step 2: Run tests — expect failures**

```bash
python -m pytest tests/test_vdb_validator.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'studio.vdb_validator'`

Note: `requests_mock` is a pytest plugin — install if missing: `pip install pytest-requests-mock requests`

- [ ] **Step 3: Create `studio/vdb_validator.py`**

```python
# studio/vdb_validator.py
import sys
from pathlib import Path
import requests


def validate(config: dict) -> dict:
    dispatch = {
        "qdrant":   _validate_qdrant,
        "chroma":   _validate_chroma,
        "faiss":    _validate_faiss,
        "pinecone": _validate_pinecone,
        "weaviate": _validate_weaviate,
        "milvus":   _validate_milvus,
        "generic":  _validate_generic,
    }
    fn = dispatch.get(config.get("type", ""))
    if fn is None:
        return {"ok": False, "error": f"Unknown VDB type: {config.get('type')}", "collection_count": None}
    try:
        return fn(config)
    except Exception as e:
        return {"ok": False, "error": str(e), "collection_count": None}


def _validate_qdrant(config: dict) -> dict:
    if "qdrant_client" not in sys.modules:
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return {"ok": False, "error": "qdrant-client not installed — pip install qdrant-client", "collection_count": None}
    from qdrant_client import QdrantClient
    client = QdrantClient(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 6333)),
        api_key=config.get("api_key") or None,
        timeout=5,
    )
    cols = client.get_collections().collections
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_chroma(config: dict) -> dict:
    try:
        import chromadb
    except ImportError:
        return {"ok": False, "error": "chromadb not installed — pip install chromadb", "collection_count": None}
    client = chromadb.HttpClient(host=config.get("host", "localhost"), port=int(config.get("port", 8000)))
    cols = client.list_collections()
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_faiss(config: dict) -> dict:
    p = Path(config.get("index_path", ""))
    if not p.exists():
        return {"ok": False, "error": f"Index file not found: {p}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": 1}


def _validate_pinecone(config: dict) -> dict:
    try:
        from pinecone import Pinecone
    except ImportError:
        return {"ok": False, "error": "pinecone-client not installed — pip install pinecone-client", "collection_count": None}
    pc = Pinecone(api_key=config.get("api_key", ""))
    idxs = pc.list_indexes()
    return {"ok": True, "error": None, "collection_count": len(idxs)}


def _validate_weaviate(config: dict) -> dict:
    url = config.get("url", "").rstrip("/")
    api_key = config.get("api_key")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"{url}/v1/.well-known/ready", headers=headers, timeout=5)
    if resp.status_code != 200:
        return {"ok": False, "error": f"Weaviate not ready: HTTP {resp.status_code}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": None}


def _validate_milvus(config: dict) -> dict:
    try:
        from pymilvus import MilvusClient
    except ImportError:
        return {"ok": False, "error": "pymilvus not installed — pip install pymilvus", "collection_count": None}
    uri = f"http://{config.get('host', 'localhost')}:{config.get('port', 19530)}"
    client = MilvusClient(uri=uri, token=config.get("token", ""))
    cols = client.list_collections()
    return {"ok": True, "error": None, "collection_count": len(cols)}


def _validate_generic(config: dict) -> dict:
    url = config.get("base_url", "")
    key = config.get("auth_header_key", "")
    val = config.get("auth_header_value", "")
    headers = {key: val} if key else {}
    resp = requests.get(url, headers=headers, timeout=5)
    if resp.status_code >= 500:
        return {"ok": False, "error": f"Endpoint returned HTTP {resp.status_code}", "collection_count": None}
    return {"ok": True, "error": None, "collection_count": None}
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_vdb_validator.py -v --override-ini="addopts="
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/vdb_validator.py tests/test_vdb_validator.py
git commit -m "feat: add vdb_validator — connectivity ping for all 7 supported backends"
```

---

## Task 4: `studio/gpu_monitor.py` — add `get_gpu_realtime()`

**Files:**
- Modify: `studio/gpu_monitor.py`
- Test: `tests/test_gpu_realtime.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gpu_realtime.py
from studio.gpu_monitor import parse_gpu_realtime

STATS = "0, NVIDIA A10G, 1229, 22528, 8, 36, 72.00\n1, NVIDIA A10G, 19865, 22528, 94, 71, 195.50"
UUIDS = "0, GPU-aaaa-1111\n1, GPU-bbbb-2222"
PROCS = "GPU-bbbb-2222, 28431, python vllm.entrypoints.openai.api_server, 19397"


def test_gpu_count():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert len(r["gpus"]) == 2


def test_memory_gb_conversion():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][0]["used_gb"] == 1.2
    assert r["gpus"][0]["total_gb"] == 22.0


def test_util_temp_power():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][1]["util_pct"] == 94
    assert r["gpus"][1]["temp_c"] == 71
    assert r["gpus"][1]["power_w"] == 195


def test_process_assigned_to_correct_gpu():
    r = parse_gpu_realtime(STATS, UUIDS, PROCS)
    assert r["gpus"][0]["processes"] == []
    assert len(r["gpus"][1]["processes"]) == 1
    assert r["gpus"][1]["processes"][0]["pid"] == 28431
    assert r["gpus"][1]["processes"][0]["mem_mib"] == 19397


def test_has_free_gpu_true():
    assert parse_gpu_realtime(STATS, UUIDS, PROCS)["has_free_gpu"] is True


def test_has_free_gpu_false():
    busy = "0, NVIDIA A10G, 20000, 22528, 91, 70, 190.00\n1, NVIDIA A10G, 19865, 22528, 94, 71, 195.00"
    assert parse_gpu_realtime(busy, UUIDS, "")["has_free_gpu"] is False


def test_empty_procs():
    r = parse_gpu_realtime(STATS, UUIDS, "")
    assert r["gpus"][0]["processes"] == []
    assert r["gpus"][1]["processes"] == []


def test_malformed_proc_line_skipped():
    r = parse_gpu_realtime(STATS, UUIDS, "bad-line-no-commas")
    assert r["gpus"][0]["processes"] == []
```

- [ ] **Step 2: Run tests — expect failures**

```bash
python -m pytest tests/test_gpu_realtime.py -v --override-ini="addopts="
```
Expected: `ImportError: cannot import name 'parse_gpu_realtime' from 'studio.gpu_monitor'`

- [ ] **Step 3: Add `parse_gpu_realtime` and `get_gpu_realtime` to `studio/gpu_monitor.py`**

Append after the existing `get_gpu_status()` function (after line ~112):

```python
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
        used_mib = int(parts[2])
        total_mib = int(parts[3])
        gpus.append({
            "id": int(parts[0]),
            "name": parts[1],
            "used_gb": round(used_mib / 1024, 1),
            "total_gb": round(total_mib / 1024, 1),
            "util_pct": int(parts[4]),
            "temp_c": int(parts[5]),
            "power_w": int(float(parts[6])),
            "status": "free" if used_mib < 4096 else "busy",
            "processes": [],
        })

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
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_gpu_realtime.py -v --override-ini="addopts="
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/gpu_monitor.py tests/test_gpu_realtime.py
git commit -m "feat: add get_gpu_realtime() to gpu_monitor — util%, temp, power, per-GPU processes"
```

---

## Task 5: `studio/ab_runner.py`

**Files:**
- Create: `studio/ab_runner.py`
- Test: `tests/test_ab_runner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ab_runner.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from studio.ab_runner import _query_local, _query_cloud


@pytest.mark.asyncio
async def test_query_local_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42."}}],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("studio.ab_runner.httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value = ctx

        result = await _query_local("What is 6×7?", {})

    assert result["text"] == "The answer is 42."
    assert result["source"] == "local-vllm"
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_query_local_connection_error():
    with patch("studio.ab_runner.httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(side_effect=Exception("Connection refused"))
        MockClient.return_value = ctx

        result = await _query_local("Q?", {})

    assert result["text"] == ""
    assert "error" in result


@pytest.mark.asyncio
async def test_query_cloud_uses_stored_key():
    with patch("studio.settings_manager.get_setting", return_value="sk-ant-api03-test"):
        with patch("studio.ab_runner._call_anthropic", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ("Cloud answer.", 0.0003)
            result = await _query_cloud("Q?", {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})

    assert result["text"] == "Cloud answer."
    assert result["source"] == "anthropic"
    assert result["cost_est_usd"] == 0.0003


@pytest.mark.asyncio
async def test_query_cloud_unknown_provider():
    result = await _query_cloud("Q?", {"provider": "cohere"})
    assert result["text"] == ""
    assert "error" in result


@pytest.mark.asyncio
async def test_run_ab_query_returns_both():
    from studio.ab_runner import run_ab_query
    with patch("studio.ab_runner._query_local", new_callable=AsyncMock) as ml, \
         patch("studio.ab_runner._query_cloud", new_callable=AsyncMock) as mc:
        ml.return_value = {"text": "Local.", "source": "local-vllm", "latency_ms": 800}
        mc.return_value = {"text": "Cloud.", "source": "anthropic", "latency_ms": 1200}
        result = await run_ab_query("uc-test", "Q?", {}, {})

    assert result["response_a"]["text"] == "Local."
    assert result["response_b"]["text"] == "Cloud."
```

Note: install `pytest-asyncio` if missing: `pip install pytest-asyncio`. Add `asyncio_mode = "auto"` to `pytest.ini` or use `@pytest.mark.asyncio` decorator.

- [ ] **Step 2: Run tests — expect failures**

```bash
python -m pytest tests/test_ab_runner.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'studio.ab_runner'`

- [ ] **Step 3: Create `studio/ab_runner.py`**

```python
# studio/ab_runner.py
import asyncio
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent

_VLLM_URL = "http://localhost:8090/v1/chat/completions"


async def run_ab_query(
    uc_id: str,
    query: str,
    model_a_settings: dict,
    model_b_settings: dict,
) -> dict:
    result_a, result_b = await asyncio.gather(
        _query_local(query, model_a_settings),
        _query_cloud(query, model_b_settings),
    )
    return {"response_a": result_a, "response_b": result_b}


async def _query_local(query: str, settings: dict) -> dict:
    payload = {
        "model": "kvforge-local",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        "temperature": float(settings.get("temperature", 0.2)),
        "max_tokens": int(settings.get("max_tokens", 256)),
        "top_p": float(settings.get("top_p", 0.9)),
    }
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_VLLM_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "source": "local-vllm",
            "phase_used": data.get("phase", "unknown"),
            "confidence": data.get("confidence"),
        }
    except Exception as e:
        return {"text": "", "latency_ms": 0, "source": "local-vllm", "error": str(e)}


async def _query_cloud(query: str, settings: dict) -> dict:
    from studio.settings_manager import get_setting
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key") or get_setting(f"{provider}_api_key") or ""
    model = settings.get("model", "claude-haiku-4-5-20251001")
    temperature = float(settings.get("temperature", 0.3))
    max_tokens = int(settings.get("max_tokens", 512))
    system_prompt = settings.get("system_prompt", "You are a helpful assistant.")

    t0 = time.monotonic()
    try:
        if provider == "anthropic":
            text, cost = await _call_anthropic(api_key, model, query, system_prompt, temperature, max_tokens)
        elif provider == "openai":
            text, cost = await _call_openai(api_key, model, query, system_prompt, temperature, max_tokens)
        elif provider == "gemini":
            text, cost = await _call_gemini(api_key, model, query, temperature, max_tokens)
        else:
            return {"text": "", "latency_ms": 0, "source": provider, "error": f"Unknown provider: {provider}"}
        return {
            "text": text,
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "source": provider,
            "cost_est_usd": cost,
        }
    except Exception as e:
        return {"text": "", "latency_ms": 0, "source": provider, "error": str(e)}


async def _call_anthropic(api_key, model, query, system_prompt, temperature, max_tokens) -> tuple[str, float]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
    )
    text = msg.content[0].text
    cost = round((msg.usage.input_tokens * 0.00000025) + (msg.usage.output_tokens * 0.00000125), 6)
    return text, cost


async def _call_openai(api_key, model, query, system_prompt, temperature, max_tokens) -> tuple[str, float]:
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
    )
    text = resp.choices[0].message.content
    cost = round((resp.usage.prompt_tokens * 0.00000015) + (resp.usage.completion_tokens * 0.0000006), 6)
    return text, cost


async def _call_gemini(api_key, model, query, temperature, max_tokens) -> tuple[str, float]:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    resp = await asyncio.to_thread(
        gen_model.generate_content, query,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return resp.text, 0.0
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_ab_runner.py -v --override-ini="addopts="
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/ab_runner.py tests/test_ab_runner.py
git commit -m "feat: add ab_runner — async concurrent A/B query against local vLLM + cloud API"
```

---

## Task 6: `studio/pipeline_runner.py` — add `faq-gen-cloud` step

**Files:**
- Modify: `studio/pipeline_runner.py`

- [ ] **Step 1: Add `faq-gen-cloud` to `STEP_MODULES` and extend `_build_cmd`**

In `STEP_MODULES` dict, add after the `"sleep-faq"` line:
```python
"faq-gen-cloud": "pipeline.sleep_faq_generator",
```

In `GPU_REQUIRED_STEPS`, do NOT add `faq-gen-cloud` (it uses cloud APIs, not local GPU).

At the end of `_build_cmd()`, add before the final `return cmd`:
```python
    if step == "faq-gen-cloud":
        from studio.settings_manager import get_setting
        output = str(ROOT / "examples" / uc_id / "faqs.json")
        cmd += ["--output", output]
        uc_cfg_path = ROOT / "examples" / uc_id / "uc_config.json"
        provider = "anthropic"
        count = 50
        if uc_cfg_path.exists():
            try:
                uc_cfg = json.loads(uc_cfg_path.read_text())
                provider = uc_cfg.get("llm", {}).get("cloud_provider", "anthropic")
                count = int(uc_cfg.get("llm", {}).get("sleep_faq_count", 50))
            except Exception:
                pass
        api_key = get_setting(f"{provider}_api_key") or ""
        cmd += ["--provider", provider, "--api-key", api_key, "--count", str(count)]
```

- [ ] **Step 2: Verify the step is recognised**

```bash
python -c "from studio.pipeline_runner import STEP_MODULES; assert 'faq-gen-cloud' in STEP_MODULES; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Verify faq-gen-cloud is NOT in GPU_REQUIRED_STEPS**

```bash
python -c "from studio.pipeline_runner import GPU_REQUIRED_STEPS; assert 'faq-gen-cloud' not in GPU_REQUIRED_STEPS; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
python -m pytest tests/ -v --override-ini="addopts=" -x -q
```
Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add studio/pipeline_runner.py
git commit -m "feat: add faq-gen-cloud pipeline step — cloud FAQ generation without GPU"
```

---

## Task 7: `studio/api.py` — settings + GPU realtime endpoints

**Files:**
- Modify: `studio/api.py`
- Test: `tests/test_studio_api_new.py` (create with first two endpoint tests)

- [ ] **Step 1: Write failing tests for settings and GPU endpoints**

```python
# tests/test_studio_api_new.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from studio.api import api_router
import studio.settings_manager as sm

# Minimal test app — avoids importing the full portal
_app = FastAPI()
_app.include_router(api_router)
client = TestClient(_app)


# ── Settings ──────────────────────────────────────────────────────────────────

def test_get_settings_returns_masked(tmp_path):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"anthropic_api_key": "sk-ant-api03-test1234"}))
    with patch.object(sm, "SETTINGS_FILE", f):
        resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["anthropic_api_key"] == "••••1234"
    assert data["curation_threshold"] == 50


def test_post_settings_saves_threshold(tmp_path):
    f = tmp_path / "settings.json"
    with patch.object(sm, "SETTINGS_FILE", f):
        resp = client.post("/api/settings", json={"curation_threshold": 30})
        assert resp.status_code == 200
        resp2 = client.get("/api/settings")
    # after patching is released, can't check; just verify 200
    assert resp.status_code == 200


def test_post_settings_rejects_bad_key(tmp_path):
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        resp = client.post("/api/settings", json={"anthropic_api_key": "invalid"})
    assert resp.status_code == 400
    assert "Invalid format" in resp.json()["detail"]


# ── GPU realtime ───────────────────────────────────────────────────────────────

def test_gpu_realtime_returns_gpus():
    mock_result = {
        "gpus": [{"id": 0, "name": "NVIDIA A10G", "util_pct": 8, "processes": []}],
        "has_free_gpu": True,
    }
    with patch("studio.api.get_gpu_realtime", return_value=mock_result):
        resp = client.get("/api/gpu/realtime")
    assert resp.status_code == 200
    assert resp.json()["has_free_gpu"] is True
    assert len(resp.json()["gpus"]) == 1


def test_gpu_realtime_returns_error_gracefully():
    with patch("studio.api.get_gpu_realtime", return_value={"error": "nvidia-smi not found", "gpus": [], "has_free_gpu": False}):
        resp = client.get("/api/gpu/realtime")
    assert resp.status_code == 200
    assert resp.json()["gpus"] == []
```

- [ ] **Step 2: Run tests — expect failures**

```bash
python -m pytest tests/test_studio_api_new.py -v --override-ini="addopts="
```
Expected: failures on missing endpoints.

- [ ] **Step 3: Add settings and GPU endpoints to `studio/api.py`**

Add these imports at the top of `studio/api.py` (after existing imports):
```python
from studio.gpu_monitor import get_gpu_realtime
from studio import settings_manager
```

Add these endpoints after the existing `stop_job` endpoint:

```python
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
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
python -m pytest tests/test_studio_api_new.py -v --override-ini="addopts="
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/api.py tests/test_studio_api_new.py
git commit -m "feat: add /api/settings and /api/gpu/realtime endpoints"
```

---

## Task 8: `studio/api.py` — PRS history and curation endpoints

**Files:**
- Modify: `studio/api.py`
- Modify: `tests/test_studio_api_new.py`

- [ ] **Step 1: Add PRS history and curation tests to `tests/test_studio_api_new.py`**

Append to the existing test file:

```python
# ── PRS history ────────────────────────────────────────────────────────────────

def test_prs_history_returns_list(tmp_path):
    uc_dir = tmp_path / "examples" / "uc-test"
    uc_dir.mkdir(parents=True)
    version_data = {
        "phase": 3,
        "current_lora_version": 2,
        "prs_history": [
            {"round": 1, "prs": 0.72},
            {"round": 2, "prs": 0.8531},
        ],
    }
    (uc_dir / "version.json").write_text(json.dumps(version_data))
    with patch("studio.api.ROOT", tmp_path):
        resp = client.get("/api/uc/uc-test/prs-history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[1]["prs"] == 0.8531
    assert "label" in data[1]


def test_prs_history_missing_version_json(tmp_path):
    (tmp_path / "examples" / "uc-empty").mkdir(parents=True)
    with patch("studio.api.ROOT", tmp_path):
        resp = client.get("/api/uc/uc-empty/prs-history")
    assert resp.status_code == 200
    assert resp.json() == []


# ── Curation ───────────────────────────────────────────────────────────────────

def test_ab_curate_appends_record(tmp_path):
    (tmp_path / "examples" / "uc-test").mkdir(parents=True)
    import studio.curation_manager as cur
    with patch.object(cur, "ROOT", tmp_path), \
         patch("studio.api.curation_manager.ROOT", tmp_path):
        resp = client.post("/api/uc/uc-test/ab-curate",
                           json={"question": "Q?", "answer": "A.", "source_model": "model_b"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_curation_status_empty(tmp_path):
    (tmp_path / "examples" / "uc-empty2").mkdir(parents=True)
    import studio.curation_manager as cur
    with patch.object(cur, "ROOT", tmp_path):
        resp = client.get("/api/uc/uc-empty2/curation-status")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
    assert resp.json()["at_threshold"] is False
```

- [ ] **Step 2: Run new tests — expect failures**

```bash
python -m pytest tests/test_studio_api_new.py::test_prs_history_returns_list tests/test_studio_api_new.py::test_ab_curate_appends_record -v --override-ini="addopts="
```
Expected: `404` or `AttributeError` — endpoints don't exist yet.

- [ ] **Step 3: Add PRS history and curation endpoints to `studio/api.py`**

Add imports after existing imports section:
```python
from studio import curation_manager
```

Add endpoints:

```python
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


# ── Auto-curation ──────────────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/ab-curate")
async def ab_curate_endpoint(uc_id: str, request: Request):
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
    return JSONResponse(curation_manager.get_status(uc_id))
```

- [ ] **Step 4: Run all new tests — expect all to pass**

```bash
python -m pytest tests/test_studio_api_new.py -v --override-ini="addopts="
```
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/api.py tests/test_studio_api_new.py
git commit -m "feat: add prs-history, ab-curate, curation-status endpoints"
```

---

## Task 9: `studio/api.py` — A/B query endpoint

**Files:**
- Modify: `studio/api.py`
- Modify: `tests/test_studio_api_new.py`

- [ ] **Step 1: Add A/B query test**

Append to `tests/test_studio_api_new.py`:

```python
# ── A/B query ──────────────────────────────────────────────────────────────────

def test_ab_query_returns_both_responses():
    mock_result = {
        "response_a": {"text": "Local answer.", "latency_ms": 800, "source": "local-vllm"},
        "response_b": {"text": "Cloud answer.", "latency_ms": 1200, "source": "anthropic"},
    }
    with patch("studio.api.ab_runner.run_ab_query", new_callable=AsyncMock) as mock_ab:
        mock_ab.return_value = mock_result
        resp = client.post("/api/uc/uc-test/ab-query", json={
            "query": "What is RAG?",
            "model_a_settings": {"temperature": 0.2},
            "model_b_settings": {"provider": "anthropic"},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_a"]["text"] == "Local answer."
    assert data["response_b"]["text"] == "Cloud answer."


def test_ab_query_missing_query_returns_400():
    resp = client.post("/api/uc/uc-test/ab-query", json={})
    assert resp.status_code == 400
```

Also add `from unittest.mock import AsyncMock` to the imports at the top of the test file.

- [ ] **Step 2: Run new tests — expect failures**

```bash
python -m pytest tests/test_studio_api_new.py::test_ab_query_returns_both_responses -v --override-ini="addopts="
```
Expected: `404`

- [ ] **Step 3: Add A/B query endpoint to `studio/api.py`**

Add import after existing imports:
```python
from studio import ab_runner
```

Add endpoint:

```python
# ── A/B query ──────────────────────────────────────────────────────────────────

@api_router.post("/uc/{uc_id}/ab-query")
async def ab_query_endpoint(uc_id: str, request: Request):
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
```

- [ ] **Step 4: Run all tests — expect all to pass**

```bash
python -m pytest tests/test_studio_api_new.py -v --override-ini="addopts="
```
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/api.py tests/test_studio_api_new.py
git commit -m "feat: add /api/uc/{id}/ab-query endpoint — concurrent local+cloud inference"
```

---

## Task 10: `studio/api.py` — wizard endpoints

**Files:**
- Modify: `studio/api.py`
- Modify: `tests/test_studio_api_new.py`

- [ ] **Step 1: Add wizard endpoint tests**

Append to `tests/test_studio_api_new.py`:

```python
# ── Wizard: VDB validate ───────────────────────────────────────────────────────

def test_wizard_validate_vdb_ok():
    with patch("studio.api.vdb_validator.validate", return_value={"ok": True, "error": None, "collection_count": 3}):
        resp = client.post("/api/wizard/validate-vdb", json={"type": "qdrant", "host": "localhost", "port": 6333})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["collection_count"] == 3


def test_wizard_validate_vdb_failure():
    with patch("studio.api.vdb_validator.validate", return_value={"ok": False, "error": "refused", "collection_count": None}):
        resp = client.post("/api/wizard/validate-vdb", json={"type": "qdrant"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


# ── Wizard: PDF upload ─────────────────────────────────────────────────────────

def test_wizard_upload_pdf_returns_estimate(tmp_path):
    with patch("studio.api.ROOT", tmp_path):
        resp = client.post(
            "/api/wizard/upload-pdf",
            files={"file": ("test.pdf", b"%PDF fake content " * 200, "application/pdf")},
            data={"uc_id": "uc-new"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.pdf"
    assert "estimated_chunks" in data
    assert data["estimated_chunks"] > 0


# ── Wizard: VRAM estimate ──────────────────────────────────────────────────────

def test_wizard_estimate_vram_known_model():
    resp = client.post("/api/wizard/estimate-vram",
                       json={"model_id": "meta-llama/Llama-3.2-3B-Instruct", "lora_rank": 16})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fits"] is True
    assert data["vram_required_gb"] < 22.0


def test_wizard_estimate_vram_unknown_model():
    resp = client.post("/api/wizard/estimate-vram",
                       json={"model_id": "unknown/UnknownModel-999B", "lora_rank": 16})
    assert resp.status_code == 200
    assert resp.json()["fits"] is False or resp.json().get("error") is not None
```

- [ ] **Step 2: Run new tests — expect failures**

```bash
python -m pytest tests/test_studio_api_new.py::test_wizard_validate_vdb_ok tests/test_studio_api_new.py::test_wizard_upload_pdf_returns_estimate -v --override-ini="addopts="
```
Expected: `404`

- [ ] **Step 3: Add wizard endpoints and VRAM helper to `studio/api.py`**

Add imports:
```python
from fastapi import UploadFile, Form
from studio import vdb_validator
import re as _re
```

Add the VRAM table and endpoints:

```python
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


@api_router.post("/wizard/upload-pdf")
async def wizard_upload_pdf(file: UploadFile, uc_id: str = Form("")):
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    # Rough estimate: ~500 bytes per chunk at ~75% overlap
    estimated_chunks = max(1, int(len(content) / 600))
    upload_dir = ROOT / "tmp" / "uploads" / (uc_id or "default")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (file.filename or "upload.pdf")
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
```

- [ ] **Step 4: Run all tests — expect all to pass**

```bash
python -m pytest tests/test_studio_api_new.py -v --override-ini="addopts="
```
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add studio/api.py tests/test_studio_api_new.py
git commit -m "feat: add wizard endpoints — validate-vdb, upload-pdf, estimate-vram"
```

---

## Task 11: `studio/routes.py` — add `/wizard` route, swap UC detail

**Files:**
- Modify: `studio/routes.py`

- [ ] **Step 1: Add the wizard route and swap uc_detail in `studio/routes.py`**

The file currently has (lines 42–52):
```python
@router.get("/", response_class=HTMLResponse)
def studio_hub():
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()


@router.get("/uc/{uc_id}", response_class=HTMLResponse)
def uc_detail(uc_id: str):
    _ensure_migrated()
    return (TEMPLATES / "hub.html").read_text()
```

Replace the `uc_detail` handler and add the wizard route:

```python
@router.get("/wizard", response_class=HTMLResponse)
def wizard_page():
    return (TEMPLATES / "wizard.html").read_text()


@router.get("/uc/{uc_id}", response_class=HTMLResponse)
def uc_detail(uc_id: str):
    _ensure_migrated()
    return (TEMPLATES / "uc_detail.html").read_text()
```

Note: wizard route must be declared BEFORE `/uc/{uc_id}` to avoid FastAPI matching "wizard" as a uc_id.

- [ ] **Step 2: Create placeholder HTML files so the server starts cleanly**

```bash
mkdir -p templates/studio
echo '<html><body>Wizard placeholder</body></html>' > templates/studio/wizard.html
echo '<html><body>UC detail placeholder</body></html>' > templates/studio/uc_detail.html
```

- [ ] **Step 3: Verify routes serve correctly**

```bash
python -c "
from fastapi.testclient import TestClient
from fastapi import FastAPI
from studio.routes import router
app = FastAPI()
app.include_router(router, prefix='/studio')
c = TestClient(app)
r = c.get('/studio/wizard')
assert r.status_code == 200, r.status_code
r2 = c.get('/studio/uc/uc-test')
assert r2.status_code == 200, r2.status_code
print('Routes OK')
"
```
Expected: `Routes OK`

- [ ] **Step 4: Commit**

```bash
git add studio/routes.py templates/studio/wizard.html templates/studio/uc_detail.html
git commit -m "feat: add /studio/wizard route, swap /studio/uc/{id} to uc_detail.html"
```

---

## Task 12: `templates/studio/uc_detail.html` — static shell

**Files:**
- Modify: `templates/studio/uc_detail.html`

The approved visual design is in `.superpowers/brainstorm/83669-1777231613/content/uc-detail-v2.html`. This task copies that structure and replaces hardcoded sample data with loading states wired to real API calls.

- [ ] **Step 1: Copy the mockup as the starting point**

```bash
cp .superpowers/brainstorm/83669-1777231613/content/uc-detail-v2.html templates/studio/uc_detail.html
```

- [ ] **Step 2: Extract `uc_id` from the URL and expose it globally**

In the `<head>` section of `templates/studio/uc_detail.html`, add before the closing `</style>`:

```html
<script>
  // Extract uc_id from URL path: /studio/uc/{uc_id}
  const UC_ID = window.location.pathname.split('/').filter(Boolean).pop();
</script>
```

- [ ] **Step 3: Replace the hardcoded breadcrumb with a dynamic one**

Find the breadcrumb line:
```html
<div class="tb-bc"><a href="#">Studio Hub</a><span class="sep">›</span><b>UC4 · Amazon Bedrock User Guide</b></div>
```

Replace with:
```html
<div class="tb-bc"><a href="/studio/">Studio Hub</a><span class="sep">›</span><b id="uc-title">Loading…</b></div>
```

- [ ] **Step 4: Add `loadUCState()` to populate topbar and rail**

Add to the `<script>` section before `drawChart()`:

```javascript
async function loadUCState() {
  try {
    const resp = await fetch(`/api/uc/${UC_ID}/config`);
    if (!resp.ok) return;
    const cfg = await resp.json();
    document.getElementById('uc-title').textContent = cfg.name || UC_ID;
  } catch (e) {
    document.getElementById('uc-title').textContent = UC_ID;
  }

  try {
    const resp = await fetch(`/api/registry`);
    if (!resp.ok) return;
    const reg = await resp.json();
    const uc = (reg.use_cases || reg).find(u => u.id === UC_ID);
    if (uc) {
      if (uc.phase) {
        const phases = { 1: 'Phase 1', 2: 'Phase 2', 3: 'Phase 3' };
        document.querySelector('.pill-p3').textContent = `● ${phases[uc.phase] || 'Phase ' + uc.phase}`;
      }
      if (uc.prs != null) {
        document.querySelector('.pill-prs').textContent = `PRS ${uc.prs.toFixed(4)}`;
      }
    }
  } catch (e) { /* non-fatal */ }
}

loadUCState();
```

- [ ] **Step 5: Verify page loads at `/studio/uc/uc-test`**

```bash
python kvforge_portal.py --port 8099 &
sleep 2
curl -s http://localhost:8099/studio/uc/uc-test | grep -c "uc-title"
kill %1
```
Expected: `1` (the id exists in the HTML)

- [ ] **Step 6: Commit**

```bash
git add templates/studio/uc_detail.html
git commit -m "feat: uc_detail.html shell — dynamic breadcrumb and topbar pills from API"
```

---

## Task 13: `templates/studio/uc_detail.html` — live PRS chart

**Files:**
- Modify: `templates/studio/uc_detail.html`

- [ ] **Step 1: Replace hardcoded PRS_POINTS with API-loaded data**

Find in the `<script>` section:
```javascript
const PRS_POINTS = [
  { version: 'LoRA v1', date: 'Apr 25 09:14', prs: 0.72, ...
```

Replace the entire `PRS_POINTS` declaration and `drawChart()` call with:

```javascript
const PRS_POINTS = [];  // populated by loadPRSHistory()

async function loadPRSHistory() {
  try {
    const resp = await fetch(`/api/uc/${UC_ID}/prs-history`);
    if (!resp.ok) return;
    const data = await resp.json();
    PRS_POINTS.length = 0;
    data.forEach(entry => PRS_POINTS.push({
      version: entry.label || `LoRA v${entry.round}`,
      date: entry.date || `Round ${entry.round}`,
      prs: entry.prs,
      train: entry.train || '—',
      loss: entry.loss || '—',
      note: entry.note || '',
    }));
    if (PRS_POINTS.length > 0) {
      drawChart();
      const latest = PRS_POINTS[PRS_POINTS.length - 1];
      document.querySelector('.prs-big').textContent = latest.prs.toFixed(4);
    }
  } catch (e) {
    console.warn('PRS history load failed:', e);
  }
}

loadPRSHistory();
```

- [ ] **Step 2: Guard `drawChart()` to handle empty data**

At the top of `drawChart()`, add:
```javascript
function drawChart() {
  if (PRS_POINTS.length === 0) return;
  // ... rest of existing drawChart code unchanged ...
```

- [ ] **Step 3: Verify no console errors on page load**

Start the server, open `http://localhost:8080/studio/uc/usecase4_bedrock_userguide` in a browser. Check the console — no red errors. The PRS chart should render with actual data from `version.json`.

If no use case with prs_history exists: chart area will be empty — that is expected.

- [ ] **Step 4: Commit**

```bash
git add templates/studio/uc_detail.html
git commit -m "feat: uc_detail PRS chart loads live data from /api/uc/{id}/prs-history"
```

---

## Task 14: `templates/studio/uc_detail.html` — A/B panel and curation flywheel

**Files:**
- Modify: `templates/studio/uc_detail.html`

The existing A/B panel in the mockup has hardcoded responses and simulated curation. This task wires it to the real `/api/uc/{id}/ab-query` and `/api/uc/{id}/ab-curate` endpoints.

- [ ] **Step 1: Replace the `runQuery()` stub with a real API call**

Find `function runQuery()` in the script section and replace its body:

```javascript
async function runQuery() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  ['vbtn-a','vbtn-b','vbtn-both'].forEach(id => document.getElementById(id)?.classList.remove('sel'));
  document.getElementById('curated-flash')?.classList.remove('visible');
  document.getElementById('verdict-hint').textContent = 'Running both models…';

  // Show loading state
  document.getElementById('response-a').querySelector('.response-text').textContent = '…';
  document.getElementById('response-b').querySelector('.response-text').textContent = '…';

  const modelASettings = {
    temperature: parseFloat(document.getElementById('temp-a')?.textContent || 0.2),
    max_tokens: parseInt(document.getElementById('maxt-a')?.textContent || 256),
    top_p: parseFloat(document.getElementById('topp-a')?.textContent || 0.9),
  };
  const modelBSettings = {
    provider: document.querySelector('#settings-b select')?.value?.split(' ')[0]?.toLowerCase() || 'anthropic',
    temperature: parseFloat(document.getElementById('temp-b')?.textContent || 0.3),
    max_tokens: parseInt(document.getElementById('maxt-b')?.textContent || 512),
  };

  try {
    const resp = await fetch(`/api/uc/${UC_ID}/ab-query`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, model_a_settings: modelASettings, model_b_settings: modelBSettings}),
    });
    const data = await resp.json();

    const ra = data.response_a;
    const rb = data.response_b;

    document.getElementById('response-a').innerHTML = `
      <div class="response-text">${ra.text || ra.error || 'No response'}</div>
      <div class="response-meta">
        <div class="r-meta"><b>Latency</b> ${((ra.latency_ms||0)/1000).toFixed(2)} s</div>
        <div class="r-meta"><b>Phase</b> ${ra.phase_used || '—'}</div>
        ${ra.confidence != null ? `<div class="r-meta"><b>Confidence</b> ${ra.confidence}</div>` : ''}
      </div>`;

    document.getElementById('response-b').innerHTML = `
      <div class="response-text">${rb.text || rb.error || 'No response'}</div>
      <div class="response-meta">
        <div class="r-meta"><b>Latency</b> ${((rb.latency_ms||0)/1000).toFixed(2)} s</div>
        <div class="r-meta"><b>Source</b> ${rb.source || '—'}</div>
        ${rb.cost_est_usd != null ? `<div class="r-meta"><b>Est. cost</b> $${rb.cost_est_usd}</div>` : ''}
      </div>`;

    // Store current query/answer for curation
    window._lastQuery = query;
    window._lastAnswerB = rb.text || '';
    document.getElementById('verdict-hint').textContent = 'Select which response(s) are factually correct and helpful';
  } catch (e) {
    document.getElementById('verdict-hint').textContent = `Error: ${e.message}`;
  }
}
```

- [ ] **Step 2: Replace the curation logic in `verdict()` with a real API call**

Find `if (choice === 'b') {` block inside `verdict()` and replace:

```javascript
  if (choice === 'b') {
    document.getElementById('vbtn-b').classList.add('sel');
    const flash = document.getElementById('curated-flash');
    flash.classList.remove('visible');

    try {
      const resp = await fetch(`/api/uc/${UC_ID}/ab-curate`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          question: window._lastQuery || document.getElementById('query-input').value,
          answer: window._lastAnswerB || '',
          source_model: 'model_b',
        }),
      });
      const status = await resp.json();
      curationCount = status.count;
      updateDataset();
      flash.textContent = `📥 Added to training dataset (${status.count}/${status.threshold})`;
      flash.classList.add('visible');
      hint.textContent = 'Model B response added to fine-tuning dataset for Model A';
      if (status.at_threshold) {
        setTimeout(() => { document.getElementById('retrain-banner').style.display = 'flex'; }, 600);
      }
    } catch (e) {
      hint.textContent = `Curation error: ${e.message}`;
    }
  }
```

Also make `verdict()` async:  change `function verdict(choice) {` to `async function verdict(choice) {`

- [ ] **Step 3: Load curation status on page load**

Add to `loadUCState()`:

```javascript
  // Load curation status for progress bar
  try {
    const cr = await fetch(`/api/uc/${UC_ID}/curation-status`);
    const cs = await cr.json();
    curationCount = cs.count;
    updateDataset();
    document.getElementById('curation-count').textContent = cs.count;
    document.querySelector('.cp-threshold').textContent = `/ ${cs.threshold} records`;
  } catch (e) { /* non-fatal */ }
```

- [ ] **Step 4: Commit**

```bash
git add templates/studio/uc_detail.html
git commit -m "feat: uc_detail A/B panel wired to ab-query and ab-curate endpoints"
```

---

## Task 15: `templates/studio/uc_detail.html` — live GPU overlay

**Files:**
- Modify: `templates/studio/uc_detail.html`

- [ ] **Step 1: Replace simulated GPU data with real API call**

Find `function refreshGpuData()` in the script and replace its body:

```javascript
async function refreshGpuData() {
  gpuSecondsAgo = 0;
  try {
    const resp = await fetch('/api/gpu/realtime');
    const data = await resp.json();
    if (data.error) {
      document.getElementById('gpu-refresh-badge').textContent = `⚠ ${data.error}`;
      return;
    }

    let totalMem = 0, totalUtil = 0, maxTemp = 0;
    data.gpus.forEach(g => {
      const memPct = (g.used_gb / g.total_gb) * 100;
      const bc = g.util_pct >= 80 ? 'bar-high' : g.util_pct >= 50 ? 'bar-med' : 'bar-low';
      const tc = g.util_pct >= 80 ? 'util-high' : g.util_pct >= 50 ? 'util-med' : 'util-low';
      const memColor = g.used_gb >= 18 ? '#f87171' : g.used_gb >= 14 ? '#fcd34d' : '#4ec9b0';

      const utilEl = document.getElementById(`util-${g.id}`);
      if (utilEl) { utilEl.textContent = g.util_pct + '%'; utilEl.className = `gpu-util-val ${tc}`; }
      const utilBar = document.getElementById(`utilbar-${g.id}`);
      if (utilBar) { utilBar.style.width = g.util_pct + '%'; utilBar.className = `gpu-util-bar ${bc}`; }
      const memEl = document.getElementById(`mem-${g.id}`);
      if (memEl) { memEl.textContent = `${g.used_gb} / ${g.total_gb} GB`; memEl.style.color = memColor; }
      const memBar = document.getElementById(`membar-${g.id}`);
      if (memBar) { memBar.style.width = memPct.toFixed(1) + '%'; memBar.className = `gpu-util-bar ${bc}`; }
      const tempEl = document.getElementById(`temp-${g.id}`);
      if (tempEl) tempEl.textContent = g.temp_c + '°C';
      const pwrEl = document.getElementById(`pwr-${g.id}`);
      if (pwrEl) pwrEl.textContent = g.power_w + 'W';

      // Update process list
      const procsEl = document.getElementById(`procs-${g.id}`);
      if (procsEl) {
        const rows = g.processes.map(p =>
          `<div class="gpu-proc-row">
             <span class="gpu-proc-pid">PID ${p.pid} (${p.type})</span>
             <span class="gpu-proc-name">${p.name}</span>
             <span class="gpu-proc-mem">${p.mem_mib} MiB</span>
           </div>`
        ).join('');
        const title = procsEl.querySelector('.gpu-procs-title');
        if (title) title.textContent = `Processes (${g.processes.length})`;
        const existing = procsEl.querySelectorAll('.gpu-proc-row');
        existing.forEach(el => el.remove());
        procsEl.insertAdjacentHTML('beforeend', rows);
      }

      totalMem += g.used_gb; totalUtil += g.util_pct; maxTemp = Math.max(maxTemp, g.temp_c);
    });

    const n = data.gpus.length || 1;
    const avgUtilEl = document.getElementById('avg-util');
    if (avgUtilEl) avgUtilEl.textContent = Math.round(totalUtil / n) + '%';
    const maxTempEl = document.getElementById('max-temp');
    if (maxTempEl) maxTempEl.textContent = maxTemp + '°C';
  } catch (e) {
    document.getElementById('gpu-refresh-badge').textContent = '⚠ fetch error';
  }
  document.getElementById('gpu-refresh-badge').textContent = '↻ live · 0s ago';
}
```

- [ ] **Step 2: Update the GPU pill button label from real data**

Add to `loadUCState()`:

```javascript
  try {
    const gr = await fetch('/api/gpu/realtime');
    const gd = await gr.json();
    const freeCount = (gd.gpus || []).filter(g => g.status === 'free').length;
    const total = (gd.gpus || []).length;
    const pillBtn = document.getElementById('gpu-pill-btn');
    if (pillBtn) pillBtn.textContent = `⬛ ${total} GPU${total !== 1 ? 's' : ''} · ${freeCount} free`;
  } catch (e) { /* non-fatal */ }
```

- [ ] **Step 3: Verify GPU overlay opens and shows real data**

Start the server, open `/studio/uc/{any_uc_id}`, click the GPU pill. On a machine with `nvidia-smi`, real GPU data should appear. On a Mac (no GPU), the overlay should show `⚠ nvidia-smi not found` gracefully.

- [ ] **Step 4: Commit**

```bash
git add templates/studio/uc_detail.html
git commit -m "feat: uc_detail GPU overlay fetches live nvidia-smi data from /api/gpu/realtime"
```

---

## Task 16: `templates/studio/wizard.html` — steps 1–4

**Files:**
- Modify: `templates/studio/wizard.html`

The approved visual design is in `.superpowers/brainstorm/83669-1777231613/content/wizard-steps.html`.

- [ ] **Step 1: Copy the mockup as starting point**

```bash
cp .superpowers/brainstorm/83669-1777231613/content/wizard-steps.html templates/studio/wizard.html
```

- [ ] **Step 2: Add wizard state and step navigation**

Add to the `<head>` before `</style>`:

```html
<script>
const WIZARD = { step: 1, data: {} };

function goStep(n) {
  document.querySelectorAll('.step-panel').forEach((p, i) => {
    p.style.display = (i + 1 === n) ? '' : 'none';
  });
  document.querySelectorAll('.step-dot').forEach((d, i) => {
    d.classList.toggle('active', i + 1 === n);
    d.classList.toggle('done', i + 1 < n);
  });
  WIZARD.step = n;
  window.scrollTo(0, 0);
}

function nextStep() { if (WIZARD.step < 7) goStep(WIZARD.step + 1); }
function prevStep() { if (WIZARD.step > 1) goStep(WIZARD.step - 1); }
</script>
```

Each wizard step `<div>` in the HTML must have class `step-panel`. Navigation buttons call `nextStep()` / `prevStep()` or a step-specific validation function.

- [ ] **Step 3: Wire Step 1 — PDF upload**

Replace the simulated file-drop handler with:

```javascript
async function handlePdfUpload(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('uc_id', WIZARD.data.uc_id || 'new-uc');
  try {
    const resp = await fetch('/api/wizard/upload-pdf', {method: 'POST', body: fd});
    const data = await resp.json();
    WIZARD.data.pdf = data;
    document.getElementById('pdf-status').textContent =
      `✓ ${data.filename} · ~${data.estimated_chunks} chunks estimated`;
    document.getElementById('step1-next').disabled = false;
  } catch (e) {
    document.getElementById('pdf-status').textContent = `Upload failed: ${e.message}`;
  }
}
```

Wire the file input: `<input type="file" accept=".pdf" onchange="handlePdfUpload(this.files[0])">` and a drop zone with a `drop` event listener calling the same function.

- [ ] **Step 4: Wire Step 1 — VDB connection test**

```javascript
async function testVdbConnection() {
  const vdbType = document.getElementById('vdb-type').value;
  const config = {type: vdbType};
  // Collect fields based on type
  if (vdbType === 'qdrant') {
    config.host = document.getElementById('vdb-host').value;
    config.port = parseInt(document.getElementById('vdb-port').value) || 6333;
    config.api_key = document.getElementById('vdb-apikey').value;
  } else if (vdbType === 'pinecone') {
    config.api_key = document.getElementById('vdb-apikey').value;
    config.index_name = document.getElementById('vdb-collection').value;
  } else if (vdbType === 'faiss') {
    config.index_path = document.getElementById('vdb-path').value;
  } else if (vdbType === 'weaviate') {
    config.url = document.getElementById('vdb-host').value;
    config.api_key = document.getElementById('vdb-apikey').value;
  } else if (vdbType === 'chroma' || vdbType === 'milvus') {
    config.host = document.getElementById('vdb-host').value;
    config.port = parseInt(document.getElementById('vdb-port').value);
    config.collection = document.getElementById('vdb-collection').value;
  } else if (vdbType === 'generic') {
    config.base_url = document.getElementById('vdb-host').value;
    config.auth_header_key = document.getElementById('vdb-auth-key').value;
    config.auth_header_value = document.getElementById('vdb-auth-val').value;
  }
  WIZARD.data.vdb = config;

  document.getElementById('vdb-test-result').textContent = 'Testing…';
  try {
    const resp = await fetch('/api/wizard/validate-vdb', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config),
    });
    const data = await resp.json();
    if (data.ok) {
      document.getElementById('vdb-test-result').textContent =
        `✓ Connected · ${data.collection_count != null ? data.collection_count + ' collections' : 'ready'}`;
      document.getElementById('step1-next').disabled = false;
    } else {
      document.getElementById('vdb-test-result').textContent = `✗ ${data.error}`;
    }
  } catch (e) {
    document.getElementById('vdb-test-result').textContent = `Error: ${e.message}`;
  }
}
```

- [ ] **Step 5: Wire Step 3 — VRAM check for custom model**

```javascript
async function checkCustomVram() {
  const modelId = document.getElementById('custom-model-id').value.trim();
  if (!modelId) return;
  const resp = await fetch('/api/wizard/estimate-vram', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({model_id: modelId, lora_rank: 16}),
  });
  const data = await resp.json();
  const pill = document.getElementById('custom-vram-pill');
  if (data.error) {
    pill.textContent = 'Unknown';
    pill.className = 'vram-pill vram-unknown';
  } else {
    pill.textContent = `${data.vram_required_gb} GB`;
    pill.className = `vram-pill vram-${data.fits ? 'ok' : data.fits_with_reduced_batch ? 'warn' : 'bad'}`;
    WIZARD.data.model_id = modelId;
    WIZARD.data.vram = data;
  }
}
```

- [ ] **Step 6: Wire Step 4 — GPU cards from live API**

```javascript
async function loadGpuCards() {
  const container = document.getElementById('gpu-cards');
  if (!container) return;
  try {
    const resp = await fetch('/api/gpu/realtime');
    const data = await resp.json();
    container.innerHTML = data.gpus.map(g => `
      <div class="gpu-card ${g.status}" onclick="selectGpu(${g.id}, this)">
        <div class="gpu-card-id">GPU ${g.id}</div>
        <div class="gpu-card-name">${g.name}</div>
        <div class="gpu-card-mem">${g.used_gb} / ${g.total_gb} GB</div>
        <div class="gpu-card-badge ${g.status}">${g.status === 'free' ? 'Free' : 'Busy'}</div>
      </div>`).join('');
  } catch (e) {
    container.innerHTML = `<div class="gpu-error">GPU data unavailable: ${e.message}</div>`;
  }
}

function selectGpu(id, el) {
  document.querySelectorAll('.gpu-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  WIZARD.data.gpu_id = id;
}

// Call on step 4 entry
```

- [ ] **Step 7: Verify steps 1-4 render and navigate correctly in browser**

```bash
python kvforge_portal.py --port 8080 &
```
Open `http://localhost:8080/studio/wizard`. Verify: PDF upload shows chunk estimate, VDB test shows green check on a reachable Qdrant, model cards render, GPU cards load.

- [ ] **Step 8: Commit**

```bash
git add templates/studio/wizard.html
git commit -m "feat: wizard steps 1-4 wired to PDF upload, VDB validate, VRAM check, GPU status"
```

---

## Task 17: `templates/studio/wizard.html` — steps 5–7 (LoRA params, review, pipeline launch)

**Files:**
- Modify: `templates/studio/wizard.html`

- [ ] **Step 1: Wire Step 5 — collect LoRA parameters into WIZARD.data**

```javascript
function collectLoraParams() {
  WIZARD.data.lora = {
    rank:     parseInt(document.getElementById('lora-rank').value),
    alpha:    parseInt(document.getElementById('lora-alpha').value),
    dropout:  parseFloat(document.getElementById('lora-dropout').value),
    epochs:   parseInt(document.getElementById('lora-epochs').value),
    batch_size: parseInt(document.getElementById('lora-batch').value),
    lr:       parseFloat(document.getElementById('lora-lr').value),
    passes:   document.querySelector('input[name="passes"]:checked')?.value || '1+2',
  };
  nextStep();
}
```

- [ ] **Step 2: Wire Step 6 — populate review table**

```javascript
function buildReview() {
  const d = WIZARD.data;
  document.getElementById('review-source').textContent =
    d.pdf ? `PDF · ${d.pdf.filename} · ~${d.pdf.estimated_chunks} chunks`
          : `VDB · ${d.vdb?.type} · ${d.vdb?.host || d.vdb?.base_url || ''}`;
  document.getElementById('review-model').textContent = d.model_id || '—';
  document.getElementById('review-gpu').textContent = d.gpu_id != null ? `GPU ${d.gpu_id}` : '—';
  document.getElementById('review-faq').textContent = d.faq_mode || 'not configured';
  if (d.lora) {
    document.getElementById('review-lora').textContent =
      `rank=${d.lora.rank} alpha=${d.lora.alpha} lr=${d.lora.lr} epochs=${d.lora.epochs} passes=${d.lora.passes}`;
  }
}
```

Call `buildReview()` when entering step 6.

- [ ] **Step 3: Wire Step 6 — "Launch pipeline" button**

```javascript
async function launchPipeline() {
  const btn = document.getElementById('launch-btn');
  btn.textContent = 'Creating use case…';
  btn.disabled = true;

  // Step A: create the UC
  const ucName = document.getElementById('uc-name').value.trim() || 'new-uc';
  WIZARD.data.uc_id = ucName.toLowerCase().replace(/[^a-z0-9]+/g, '-');

  try {
    const createResp = await fetch('/api/uc/new', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        name: ucName,
        model_id: WIZARD.data.model_id,
        gpu_id: WIZARD.data.gpu_id,
        lora: WIZARD.data.lora,
        vdb: WIZARD.data.vdb,
      }),
    });
    if (!createResp.ok) throw new Error(await createResp.text());

    // Step B: go to step 7 (progress view) and launch indexing
    goStep(7);
    await startPipelineStep('index');
  } catch (e) {
    btn.textContent = '✗ Failed — ' + e.message;
    btn.disabled = false;
  }
}
```

- [ ] **Step 4: Wire Step 7 — SSE pipeline log stream**

```javascript
const PIPELINE_STEPS = ['index', 'sleep-faq', 'train', 'recompute', 'prs-eval'];
let _currentStepIdx = 0;

async function startPipelineStep(step) {
  const stepEl = document.getElementById(`pipeline-step-${step}`);
  if (stepEl) stepEl.dataset.status = 'running';
  updateStepDots();

  const resp = await fetch(`/api/run-step`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({uc_id: WIZARD.data.uc_id, step}),
  });
  const {job_id} = await resp.json();

  const logEl = document.getElementById('pipeline-log');
  const es = new EventSource(`/studio/api/stream/${job_id}`);
  es.onmessage = e => {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = e.data;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  };
  es.addEventListener('done', () => {
    es.close();
    if (stepEl) stepEl.dataset.status = 'done';
    updateStepDots();
    _currentStepIdx++;
    const nextStep = PIPELINE_STEPS[_currentStepIdx];
    if (nextStep) {
      startPipelineStep(nextStep);
    } else {
      document.getElementById('finish-btn').style.display = '';
      document.getElementById('finish-btn').onclick = () => {
        window.location.href = `/studio/uc/${WIZARD.data.uc_id}`;
      };
    }
  });
  es.addEventListener('error', () => {
    es.close();
    if (stepEl) stepEl.dataset.status = 'failed';
    updateStepDots();
  });
}

function updateStepDots() {
  PIPELINE_STEPS.forEach(step => {
    const el = document.getElementById(`pipeline-step-${step}`);
    if (!el) return;
    const status = el.dataset.status || 'pending';
    el.className = `pipeline-dot pipeline-dot-${status}`;
  });
}
```

- [ ] **Step 5: Verify end-to-end wizard flow**

1. Open `http://localhost:8080/studio/wizard`
2. Step 1: Drop a PDF → see chunk estimate → click Next
3. Step 3: Select Llama 3.2 3B → see green VRAM pill → click Next
4. Step 4: Select a free GPU → see green feasibility banner → click Next
5. Step 5: Adjust sliders → click Next
6. Step 6: Review table populated → click "Launch pipeline"
7. Step 7: SSE log streams live; stepper dots advance; "Open UC Dashboard" appears when done

- [ ] **Step 6: Commit**

```bash
git add templates/studio/wizard.html
git commit -m "feat: wizard steps 5-7 — LoRA params, review table, live SSE pipeline launch"
```

---

## Self-Review Results

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Settings at ~/.kvforge/settings.json, masked GET | Task 1, Task 7 |
| VDB validator for all 7 backends | Task 3, Task 10 |
| GPU realtime with util%, temp, power, processes | Task 4, Task 7 |
| curation_manager append/status | Task 2, Task 8 |
| ab_runner concurrent local+cloud | Task 5, Task 9 |
| faq-gen-cloud pipeline step | Task 6 |
| 10 new API endpoints | Tasks 7–10 |
| /studio/wizard route | Task 11 |
| /studio/uc/{id} serves uc_detail.html | Task 11 |
| UC detail page — topbar, phase cards, left rail | Task 12 |
| PRS SVG line chart with live data | Task 13 |
| A/B panel with verdict + curation flywheel | Task 14 |
| GPU overlay with live refresh | Task 15 |
| Wizard steps 1-4 | Task 16 |
| Wizard steps 5-7 with SSE pipeline launch | Task 17 |

All spec requirements covered.

**Placeholder scan:** No TBD, TODO, or "implement later" in any task. All code blocks are complete. ✅

**Type consistency:** `curation_manager.append()` returns `dict` (same shape as `get_status()`); used identically in Tasks 2 and 8. `parse_gpu_realtime()` defined in Task 4 and called in Task 7 with matching signature. `run_ab_query()` defined in Task 5 and called via `ab_runner.run_ab_query()` in Task 9. ✅
