import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_no_config(tmp_path):
    """Dashboard with no config file — shows setup wizard."""
    from dashboard.app import create_app
    app = create_app(config_path=str(tmp_path / "nonexistent.json"))
    return TestClient(app)


@pytest.fixture
def client_with_config(tmp_path):
    """Dashboard with a valid KVForgeConfig — shows manage page."""
    cfg = {
        "use_case_name": "Test UC",
        "collection": "test-col",
        "version_file": str(tmp_path / "v.json"),
        "addons": ["indexing", "inference"],
        "addon_config": {
            "indexing": {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384},
            "inference": {"llm_model": "meta-llama/Llama-3.2-3B-Instruct"},
        },
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    from dashboard.app import create_app
    app = create_app(config_path=str(p))
    return TestClient(app)


def test_health_returns_ok(client_no_config):
    resp = client_no_config.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_no_config_root_returns_setup_html(client_no_config):
    resp = client_no_config.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert b"setup" in resp.content.lower()


def test_with_config_root_returns_manage_html(client_with_config):
    resp = client_with_config.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert b"addon" in resp.content.lower()


def test_api_addons_returns_all_registered(client_with_config):
    from addons.registry import AddonRegistry
    AddonRegistry.load_builtins()
    resp = client_with_config.get("/api/addons")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    names = [a["name"] for a in data["available"]]
    assert "indexing" in names
    assert "inference" in names
    assert "mcp" in names


def test_api_addons_marks_active_correctly(client_with_config):
    from addons.registry import AddonRegistry
    AddonRegistry.load_builtins()
    resp = client_with_config.get("/api/addons")
    data = resp.json()
    active = {a["name"] for a in data["available"] if a["active"]}
    assert "indexing" in active
    assert "inference" in active
    assert "training" not in active  # not in this config's addons list


def test_api_status_no_config(client_no_config):
    resp = client_no_config.get("/api/status")
    assert resp.status_code == 200
    d = resp.json()
    assert d["configured"] is False


def test_api_status_with_config(client_with_config):
    resp = client_with_config.get("/api/status")
    assert resp.status_code == 200
    d = resp.json()
    assert d["configured"] is True
    assert d["use_case_name"] == "Test UC"
    assert d["collection"] == "test-col"
    assert "indexing" in d["addons"]
