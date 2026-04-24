import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_client(tmp_path):
    import pipeline.monitoring_dashboard as dash_mod
    cfg = {
        "collection": "test_col",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "vector_store": "qdrant",
        "vector_dim": 384,
        "qdrant_url": "http://localhost:9999",  # intentionally unreachable
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    dash_mod._config_path = str(cfg_path)
    dash_mod._cfg = {}  # reset cache
    from fastapi.testclient import TestClient
    return TestClient(dash_mod.app, raise_server_exceptions=False)


def test_connectivity_returns_keys(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/connectivity")
    assert resp.status_code == 200
    data = resp.json()
    assert "qdrant" in data
    assert "gpu" in data
    assert "llm" in data
    # Qdrant unreachable → ok=False
    assert data["qdrant"]["ok"] is False


def test_error_hint_known_pattern(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/error-hint?msg=No+module+named+pypdf")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hint"] is not None
    assert "pypdf" in data["hint"]


def test_error_hint_unknown(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/error-hint?msg=something+random+unknown+xyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hint"] is None


def test_error_hint_cuda_oom(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/api/error-hint?msg=CUDA+out+of+memory+when+allocating+tensor")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hint"] is not None
    assert data["severity"] == "error"
