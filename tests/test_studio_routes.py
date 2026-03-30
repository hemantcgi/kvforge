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
    import importlib
    import studio.api as api
    import studio.routes as routes
    with patch("studio.migration.ROOT", tmp_root):
        importlib.reload(api)
        importlib.reload(routes)
    with patch("studio.api.ROOT", tmp_root), \
         patch("studio.routes.ROOT", tmp_root), \
         patch("studio.migration.ROOT", tmp_root):
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(routes.router)
        yield TestClient(app)


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
        import importlib
        import studio.routes as routes
        importlib.reload(routes)
        app = FastAPI()
        app.include_router(routes.router)
        c = TestClient(app)
        r = c.post("/api/gpu-check")
        assert r.status_code == 200
        assert r.json()["has_free_gpu"] is True


def test_create_new_uc_has_custom_type(client, tmp_root):
    r = client.post("/api/uc/new", json={"id": "my-custom-uc", "display_name": "Custom"})
    assert r.status_code == 200
    cfg = json.loads((tmp_root / "examples" / "my-custom-uc" / "uc_config.json").read_text())
    assert cfg["type"] == "custom"


def test_run_step_returns_409_on_duplicate(client, tmp_root):
    from studio.migration import migrate_existing_use_cases
    migrate_existing_use_cases(root=tmp_root)
    # First run-step succeeds
    r1 = client.post("/api/run-step", json={"uc_id": "usecase3_squad", "step": "train"})
    assert r1.status_code == 200
    # Second run-step for same UC returns 409
    r2 = client.post("/api/run-step", json={"uc_id": "usecase3_squad", "step": "prs-eval"})
    assert r2.status_code == 409


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
