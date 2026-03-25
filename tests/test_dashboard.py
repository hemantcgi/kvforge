# tests/test_dashboard.py
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def _make_client():
    import pipeline.monitoring_dashboard as md
    # Patch _cfg to avoid reading my_config.json
    with patch.object(md, "_cfg", {"qdrant_host": "localhost", "qdrant_port": 6333,
                                    "collection": "test", "dashboard_port": 8080}):
        return TestClient(md.app)


def test_health_returns_ok():
    client = _make_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version_returns_phase():
    client = _make_client()
    with patch("pipeline.monitoring_dashboard.ver.load", return_value={"phase": 1,
               "current_lora_version": 0, "prs_history": [], "known_good_queries": []}):
        r = client.get("/api/version")
        assert r.status_code == 200
        assert "phase" in r.json()


def test_dashboard_html_contains_script():
    client = _make_client()
    r = client.get("/")
    assert r.status_code == 200
    assert "<script>" in r.text
    assert "api/stats" in r.text
