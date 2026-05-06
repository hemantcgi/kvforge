# tests/test_studio_api_new.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from unittest.mock import patch, MagicMock, AsyncMock
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
    with patch("studio.api.ROOT", tmp_path), \
         patch.object(cur, "ROOT", tmp_path), \
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


def test_ab_curate_rejects_traversal(tmp_path):
    """Test that ab-curate rejects path traversal attempts."""
    from studio.api import _uc_path
    from fastapi import HTTPException
    with patch("studio.api.ROOT", tmp_path):
        # Test that _uc_path raises HTTPException 400 on traversal
        try:
            _uc_path("../etc/passwd")
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            assert e.status_code == 400


# ── A/B query ──────────────────────────────────────────────────────────────────

def test_ab_query_returns_both_responses():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    mock_result = {
        "response_a": {"text": "Local answer.", "latency_ms": 800, "source": "local-vllm"},
        "response_b": {"text": "Cloud answer.", "latency_ms": 1200, "source": "anthropic"},
    }
    with patch("studio.api.ab_runner.run_ab_query", new_callable=AsyncMock) as mock_ab:
        mock_ab.return_value = mock_result
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.post("/api/uc/uc-test/ab-query", json={
            "query": "What is RAG?",
            "model_a_settings": {"temperature": 0.2},
            "model_b_settings": {"provider": "anthropic"},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_a"]["text"] == "Local answer."
    assert data["response_b"]["text"] == "Cloud answer."
    mock_ab.assert_called_once_with(
        uc_id="uc-test",
        query="What is RAG?",
        model_a_settings={"temperature": 0.2},
        model_b_settings={"provider": "anthropic"},
    )


def test_ab_query_missing_query_returns_400():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/uc/uc-test/ab-query", json={})
    assert resp.status_code == 400


# ── Wizard: VDB validate ───────────────────────────────────────────────────────

def test_wizard_validate_vdb_ok():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    with patch("studio.api.vdb_validator.validate", return_value={"ok": True, "error": None, "collection_count": 3}):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.post("/api/wizard/validate-vdb", json={"type": "qdrant", "host": "localhost", "port": 6333})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["collection_count"] == 3


def test_wizard_validate_vdb_failure():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    with patch("studio.api.vdb_validator.validate", return_value={"ok": False, "error": "refused", "collection_count": None}):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.post("/api/wizard/validate-vdb", json={"type": "qdrant"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


# ── Wizard: PDF upload ─────────────────────────────────────────────────────────

def test_wizard_upload_pdf_returns_estimate(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    with patch("studio.api.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
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
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/wizard/estimate-vram",
                       json={"model_id": "meta-llama/Llama-3.2-3B-Instruct", "lora_rank": 16})
    assert resp.status_code == 200
    data = resp.json()
    assert data["fits"] is True
    assert data["vram_required_gb"] < 22.0


def test_wizard_estimate_vram_unknown_model():
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(api_router)
    client = TestClient(app)
    resp = client.post("/api/wizard/estimate-vram",
                       json={"model_id": "unknown/UnknownModel-999B", "lora_rank": 16})
    assert resp.status_code == 200
    assert resp.json()["fits"] is False or resp.json().get("error") is not None


# ── eval-summary ──────────────────────────────────────────────────────────────

def _eval_client(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-eval").mkdir(parents=True)
    app = FastAPI()
    app.include_router(api_router)
    return TestClient(app), tmp_path


def test_eval_summary_no_file(tmp_path):
    """Returns has_results=False when ab_eval_results.json is absent."""
    client, root = _eval_client(tmp_path)
    with patch("studio.api.ROOT", root):
        resp = client.get("/api/uc/uc-eval/eval-summary")
    assert resp.status_code == 200
    assert resp.json()["has_results"] is False


def test_eval_summary_empty_list(tmp_path):
    """Returns has_results=False when file exists but is an empty list."""
    client, root = _eval_client(tmp_path)
    (root / "examples" / "uc-eval" / "ab_eval_results.json").write_text("[]")
    with patch("studio.api.ROOT", root):
        resp = client.get("/api/uc/uc-eval/eval-summary")
    assert resp.json()["has_results"] is False


def test_eval_summary_bad_json(tmp_path):
    """Returns has_results=False on parse error."""
    client, root = _eval_client(tmp_path)
    (root / "examples" / "uc-eval" / "ab_eval_results.json").write_text("{bad json")
    with patch("studio.api.ROOT", root):
        resp = client.get("/api/uc/uc-eval/eval-summary")
    assert resp.json()["has_results"] is False


def test_eval_summary_happy_path(tmp_path):
    """Aggregates metrics correctly from a list of eval records."""
    client, root = _eval_client(tmp_path)
    records = [
        {"latency_a_ms": 200, "latency_b_ms": 400, "sem_sim_a": 0.8, "sem_sim_b": 0.7, "prs_score": 0.80},
        {"latency_a_ms": 300, "latency_b_ms": 600, "sem_sim_a": 0.6, "sem_sim_b": 0.5, "prs_score": 0.70},
    ]
    (root / "examples" / "uc-eval" / "ab_eval_results.json").write_text(json.dumps(records))
    with patch("studio.api.ROOT", root):
        resp = client.get("/api/uc/uc-eval/eval-summary")
    data = resp.json()
    assert data["has_results"] is True
    assert data["total"] == 2
    assert data["wins"] == 1          # only prs_score >= 0.75
    assert data["win_rate"] == 50.0
    assert data["avg_lat_a_ms"] == 250
    assert data["avg_lat_b_ms"] == 500
    # KVForge (A) is faster → speed_gain_pct should be positive
    assert data["speed_gain_pct"] > 0
    assert data["avg_sem_a"] == 0.7
    assert data["avg_sem_b"] == 0.6
