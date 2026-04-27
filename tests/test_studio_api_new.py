# tests/test_studio_api_new.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from unittest.mock import patch, MagicMock
import studio.settings_manager as sm
from studio.gpu_monitor import parse_gpu_realtime


# ── Settings ──────────────────────────────────────────────────────────────────

def test_get_settings_returns_masked(tmp_path):
    """Test that GET /api/settings masks API keys."""
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"anthropic_api_key": "sk-ant-api03-test1234"}))

    with patch.object(sm, "SETTINGS_FILE", f):
        result = sm.get_masked()

    assert result["anthropic_api_key"] == "••••1234"
    assert result["curation_threshold"] == 50


def test_post_settings_saves_threshold(tmp_path):
    """Test that POST /api/settings saves curation_threshold."""
    f = tmp_path / "settings.json"

    with patch.object(sm, "SETTINGS_FILE", f):
        sm.save({"curation_threshold": 30})
        result = sm.get_setting("curation_threshold")

    assert result == 30


def test_post_settings_rejects_bad_key(tmp_path):
    """Test that POST /api/settings rejects invalid API key format."""
    with patch.object(sm, "SETTINGS_FILE", tmp_path / "settings.json"):
        try:
            sm.save({"anthropic_api_key": "invalid"})
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Invalid format" in str(e)


# ── GPU realtime ───────────────────────────────────────────────────────────────

def test_gpu_realtime_parses_single_gpu():
    """Test that parse_gpu_realtime correctly handles a single GPU."""
    stats = "0, NVIDIA A10G, 2048, 24564, 8, 45, 35"
    uuids = "0, GPU-12345678"
    procs = ""

    result = parse_gpu_realtime(stats, uuids, procs)

    assert result["has_free_gpu"] is True
    assert len(result["gpus"]) == 1
    assert result["gpus"][0]["id"] == 0
    assert result["gpus"][0]["util_pct"] == 8
    assert result["gpus"][0]["processes"] == []


def test_gpu_realtime_handles_no_nvidia_smi():
    """Test that get_gpu_realtime gracefully handles missing nvidia-smi."""
    from studio.gpu_monitor import get_gpu_realtime

    with patch("studio.gpu_monitor._run_smi_realtime", side_effect=FileNotFoundError):
        result = get_gpu_realtime()

    assert result["error"] == "nvidia-smi not found"
    assert result["gpus"] == []
    assert result["has_free_gpu"] is False


# ── PRS history ────────────────────────────────────────────────────────────────

def test_prs_history_returns_list(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
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
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-test/prs-history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[1]["prs"] == 0.8531
    assert "label" in data[1]


def test_prs_history_missing_version_json(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-empty").mkdir(parents=True)
    with patch("studio.api.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-empty/prs-history")
    assert resp.status_code == 200
    assert resp.json() == []


# ── Curation ───────────────────────────────────────────────────────────────────

def test_ab_curate_appends_record(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-test").mkdir(parents=True)
    import studio.curation_manager as cur
    with patch.object(cur, "ROOT", tmp_path), \
         patch("studio.api.curation_manager.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.post("/api/uc/uc-test/ab-curate",
                           json={"question": "Q?", "answer": "A.", "source_model": "model_b"})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_curation_status_empty(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-empty2").mkdir(parents=True)
    import studio.curation_manager as cur
    with patch.object(cur, "ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-empty2/curation-status")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
    assert resp.json()["at_threshold"] is False
