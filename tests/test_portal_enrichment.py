# tests/test_portal_enrichment.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


def _make_portal_client():
    import kvforge_portal as portal
    return TestClient(portal.app)


def test_use_cases_have_required_fields():
    import kvforge_portal as portal
    required = {"id", "title", "subtitle", "description", "port", "color",
                "vectordb", "vectordb_url", "embed_model", "llm_model", "ab_eval_dir"}
    for uc in portal.USE_CASES:
        missing = required - uc.keys()
        assert not missing, f"{uc['id']} missing fields: {missing}"


def test_use_cases_vectordb_values():
    import kvforge_portal as portal
    uc_map = {uc["id"]: uc for uc in portal.USE_CASES}
    assert uc_map["uc1"]["vectordb"] == "Qdrant"
    assert uc_map["uc2"]["vectordb"] == "ChromaDB"
    assert uc_map["uc3"]["vectordb"] == "FAISS"
    assert uc_map["uc4"]["vectordb"] == "Qdrant"
    # Qdrant UCs use sentinel
    assert uc_map["uc1"]["vectordb_url"] == "qdrant"
    assert uc_map["uc4"]["vectordb_url"] == "qdrant"


def test_status_includes_prs():
    """GET /api/status returns prs field per UC."""
    import kvforge_portal as portal

    async def fake_get(url, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/health"):
            resp.json.return_value = {"status": "ok"}
        elif url.endswith("/api/version"):
            resp.json.return_value = {
                "phase": 3,
                "prs_history": [{"round": 1, "prs": 0.77}],
            }
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    with patch("kvforge_portal.httpx.AsyncClient", return_value=mock_client):
        client = _make_portal_client()
        r = client.get("/api/status")

    assert r.status_code == 200
    data = r.json()
    for uc_id, info in data.items():
        assert "prs" in info, f"{uc_id} missing prs field"
    assert data["uc1"]["prs"] == 0.77


def test_status_prs_null_when_no_history():
    """prs is null when prs_history is empty."""
    import kvforge_portal as portal

    async def fake_get(url, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/health"):
            resp.json.return_value = {"status": "ok"}
        elif url.endswith("/api/version"):
            resp.json.return_value = {"phase": 3, "prs_history": []}
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    with patch("kvforge_portal.httpx.AsyncClient", return_value=mock_client):
        client = _make_portal_client()
        r = client.get("/api/status")

    data = r.json()
    assert data["uc1"]["prs"] is None
