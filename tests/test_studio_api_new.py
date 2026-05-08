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


# ── uc_logs disk fallback ─────────────────────────────────────────────────────

def test_uc_logs_from_job_manager(tmp_path):
    """When job exists in job manager, returns its data directly."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-logs").mkdir(parents=True)
    fake_job = {"last_lines": ["line1", "line2"], "status": "running", "step": "train", "job_id": "job-abc"}
    mock_jm = MagicMock()
    mock_jm.last_for_uc.return_value = fake_job
    with patch("studio.api.ROOT", tmp_path), patch("studio.api.get_manager", return_value=mock_jm):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-logs/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lines"] == ["line1", "line2"]
    assert data["status"] == "running"
    assert data["step"] == "train"
    assert data["job_id"] == "job-abc"


def test_uc_logs_disk_fallback_done(tmp_path):
    """Falls back to last_run.log when job manager has no record, parses step, status=done."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    uc_dir = tmp_path / "examples" / "uc-logs2"
    uc_dir.mkdir(parents=True)
    log_content = "[studio] step=prs-eval job=job-xyz\nRunning evaluation…\n[studio] done (exit 0)"
    (uc_dir / "last_run.log").write_text(log_content)
    mock_jm = MagicMock()
    mock_jm.last_for_uc.return_value = None
    with patch("studio.api.ROOT", tmp_path), patch("studio.api.get_manager", return_value=mock_jm), \
         patch("studio.pipeline_runner.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-logs2/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["step"] == "prs-eval"
    assert data["status"] == "done"
    assert data["job_id"] is None
    assert "[studio] step=prs-eval job=job-xyz" in data["lines"]


def test_uc_logs_disk_fallback_failed(tmp_path):
    """Disk fallback sets status=failed when last line contains [studio] failed."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    uc_dir = tmp_path / "examples" / "uc-fail"
    uc_dir.mkdir(parents=True)
    log_content = "[studio] step=train job=job-fail\nTraining…\n[studio] failed (exit 1)"
    (uc_dir / "last_run.log").write_text(log_content)
    mock_jm = MagicMock()
    mock_jm.last_for_uc.return_value = None
    with patch("studio.api.ROOT", tmp_path), patch("studio.api.get_manager", return_value=mock_jm), \
         patch("studio.pipeline_runner.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-fail/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert data["step"] == "train"


def test_uc_logs_empty_when_no_job_and_no_disk(tmp_path):
    """Returns empty response when no job in memory and no last_run.log on disk."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-none").mkdir(parents=True)
    mock_jm = MagicMock()
    mock_jm.last_for_uc.return_value = None
    with patch("studio.api.ROOT", tmp_path), patch("studio.api.get_manager", return_value=mock_jm), \
         patch("studio.pipeline_runner.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-none/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["lines"] == []
    assert data["status"] is None
    assert data["step"] is None
    assert data["job_id"] is None


# ── registry has_index ────────────────────────────────────────────────────────

def test_registry_has_index_true_when_version_json_exists(tmp_path):
    """has_index is True when version.json is present (kv_indexer creates it)."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    uc_dir = tmp_path / "examples" / "uc-indexed"
    uc_dir.mkdir(parents=True)
    (uc_dir / "version.json").write_text(json.dumps({"phase": 1, "current_lora_version": 0, "prs_history": []}))
    with patch("studio.api.ROOT", tmp_path), \
         patch("studio.api.load_registry", return_value=[{"id": "uc-indexed", "display_name": "Test"}]), \
         patch("studio.api.get_manager") as mock_jm:
        mock_jm.return_value.list_active.return_value = []
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/registry")
    assert resp.status_code == 200
    ucs = resp.json()["use_cases"]
    assert ucs[0]["has_index"] is True


def test_registry_has_index_false_when_no_version_json(tmp_path):
    """has_index is False when version.json is absent (UC not yet indexed)."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    uc_dir = tmp_path / "examples" / "uc-new"
    uc_dir.mkdir(parents=True)
    with patch("studio.api.ROOT", tmp_path), \
         patch("studio.api.load_registry", return_value=[{"id": "uc-new", "display_name": "New"}]), \
         patch("studio.api.get_manager") as mock_jm:
        mock_jm.return_value.list_active.return_value = []
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/registry")
    assert resp.status_code == 200
    ucs = resp.json()["use_cases"]
    assert ucs[0]["has_index"] is False


def test_sync_history_endpoint_no_db(tmp_path):
    """Returns empty list when no sync DB exists yet."""
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    (tmp_path / "examples" / "uc-sync").mkdir(parents=True)
    with patch("studio.api.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-sync/sync-history")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


def test_sync_history_endpoint_with_runs(tmp_path):
    """Returns sync run records from the sync DB."""
    import sqlite3
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    uc_dir = tmp_path / "examples" / "uc-hist"
    uc_dir.mkdir(parents=True)
    db_path = uc_dir / "sync.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE sync_runs (
        id INTEGER PRIMARY KEY, uc_name TEXT, started_at TEXT,
        finished_at TEXT, files_checked INTEGER, files_changed INTEGER,
        chunks_added INTEGER, chunks_superseded INTEGER,
        pii_detections INTEGER, errors TEXT
    )""")
    conn.execute(
        "INSERT INTO sync_runs VALUES (1,'uc-hist','2026-05-01T10:00:00+00:00','2026-05-01T10:01:00+00:00',100,5,12,3,0,'')"
    )
    conn.commit()
    conn.close()
    with patch("studio.api.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/uc-hist/sync-history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runs"][0]["files_checked"] == 100


def test_sync_history_rejects_path_traversal(tmp_path):
    from fastapi.testclient import TestClient
    from studio.api import api_router
    from fastapi import FastAPI
    with patch("studio.api.ROOT", tmp_path):
        app = FastAPI()
        app.include_router(api_router)
        client = TestClient(app)
        resp = client.get("/api/uc/../../../etc/passwd/sync-history")
    assert resp.status_code in (400, 403, 404)
