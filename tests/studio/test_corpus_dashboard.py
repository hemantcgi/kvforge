import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


def _make_app():
    from dashboard.app import create_app
    return create_app("tests/fixtures/demo_config.json")


def test_archival_candidates_endpoint():
    candidates = [
        {"chunk_id": "c1", "action": "archive",
         "reason": "0.97 sim to chunk c2 · 0 FAQ appearances · 45 days cold",
         "text_preview": "Redundant content...", "estimated_savings_kb": 131},
    ]
    with patch("dashboard.routes.get_archival_candidates", return_value=candidates):
        client = TestClient(_make_app())
        resp = client.get("/api/corpus/archival-candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["chunk_id"] == "c1"


def test_confirm_archive_endpoint():
    with patch("dashboard.routes.execute_archive") as mock_exec:
        client = TestClient(_make_app())
        resp = client.post("/api/corpus/confirm-archive", json={"chunk_id": "c1"})
    assert resp.status_code == 200
    mock_exec.assert_called_once_with("c1")


def test_reinstatement_candidates_endpoint():
    candidates = [
        {"chunk_id": "c2", "action": "reinstate",
         "reason": "Retrieved 6 times (threshold: 5).",
         "text_preview": "Important archived content...",
         "retrieval_count": 6},
    ]
    with patch("dashboard.routes.get_reinstatement_candidates", return_value=candidates):
        client = TestClient(_make_app())
        resp = client.get("/api/corpus/reinstatement-candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"][0]["retrieval_count"] == 6


def test_confirm_reinstate_endpoint():
    with patch("dashboard.routes.execute_reinstate") as mock_exec:
        client = TestClient(_make_app())
        resp = client.post("/api/corpus/confirm-reinstate", json={"chunk_id": "c2"})
    assert resp.status_code == 200
    mock_exec.assert_called_once_with("c2")
