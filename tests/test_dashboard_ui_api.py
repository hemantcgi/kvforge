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


def test_dashboard_html_has_chartjs(tmp_path):
    import pipeline.monitoring_dashboard as dm
    assert "chart.js" in dm.DASHBOARD_HTML.lower()


def test_dashboard_html_has_phase_stepper(tmp_path):
    import pipeline.monitoring_dashboard as dm
    assert "phase-stepper" in dm.DASHBOARD_HTML.lower()


def test_dashboard_html_has_prs_chart_canvas(tmp_path):
    import pipeline.monitoring_dashboard as dm
    assert "prs-chart" in dm.DASHBOARD_HTML


def test_dashboard_html_has_flywheel_section(tmp_path):
    import pipeline.monitoring_dashboard as dm
    assert "flywheel-chart" in dm.DASHBOARD_HTML
    assert "fw-rounds" in dm.DASHBOARD_HTML
    assert "fw-tbody" in dm.DASHBOARD_HTML


def test_set_model_b_config_with_base_url(tmp_path):
    """POST /api/set_model_b_config accepts and stores base_url."""
    import pipeline.monitoring_dashboard as dm
    client = _make_client(tmp_path)
    resp = client.post("/api/set_model_b_config", json={
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "sk-test-1234",
        "base_url": "http://localhost:8090/v1",
    })
    assert resp.status_code == 200
    assert dm._model_b_config["base_url"] == "http://localhost:8090/v1"
    assert dm._model_b_config["provider"] == "openai"


def test_set_model_b_config_base_url_optional(tmp_path):
    """base_url defaults to empty string when omitted."""
    import pipeline.monitoring_dashboard as dm
    client = _make_client(tmp_path)
    resp = client.post("/api/set_model_b_config", json={
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "api_key": "my-key",
    })
    assert resp.status_code == 200
    assert dm._model_b_config.get("base_url", "") == ""


def test_dashboard_html_has_base_url_input(tmp_path):
    """Dashboard HTML contains the base_url input field for OpenAI-compatible endpoints."""
    import pipeline.monitoring_dashboard as dm
    assert "b_base_url" in dm.DASHBOARD_HTML
    assert "Base URL" in dm.DASHBOARD_HTML
