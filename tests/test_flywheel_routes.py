import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_app():
    from studio.flywheel_routes import flywheel_router
    app = FastAPI()
    app.include_router(flywheel_router)
    return app


def test_flywheel_all_ucs_returns_list(tmp_path):
    import core.analytics as an

    cfg1 = {"collection": "uc1",
            "analytics_db": str(tmp_path / "uc1_analytics.db"),
            "cost_per_1k_tokens": 5.0, "tokens_per_ms_baseline": 0.8}
    an.init_db(cfg1)
    cs = {"0": {"label": "auth", "phase": 2, "prs": 0.72,
                "faq_coverage": 0.8, "vdb_coverage": 0.7, "realtime_coverage": 0.6,
                "learned_weights": {}, "threshold": 0.75,
                "prs_history": [0.55, 0.63, 0.72], "query_count": 50}}
    an.record_round(cfg1, lora_version=1, cluster_state=cs, tier_distribution={})

    mock_uc = MagicMock()
    mock_uc.id = "uc1"
    mock_uc.config = cfg1

    with patch("studio.flywheel_routes._get_all_uc_configs", return_value=[mock_uc]):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/flywheel/all")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["uc_id"] == "uc1"


def test_flywheel_all_ucs_empty(tmp_path):
    with patch("studio.flywheel_routes._get_all_uc_configs", return_value=[]):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/flywheel/all")
        assert resp.status_code == 200
        assert resp.json() == []
