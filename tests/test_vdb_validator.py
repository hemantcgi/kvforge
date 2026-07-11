# tests/test_vdb_validator.py
import pytest
from unittest.mock import patch, MagicMock
from studio.vdb_validator import validate


def test_unknown_type_returns_error():
    result = validate({"type": "nonexistent"})
    assert result["ok"] is False
    assert "Unknown VDB type" in result["error"]


def test_faiss_file_not_found(tmp_path):
    result = validate({"type": "faiss", "index_path": str(tmp_path / "nope.index")})
    assert result["ok"] is False
    assert "not found" in result["error"]


def test_faiss_file_exists(tmp_path):
    p = tmp_path / "my.index"
    p.write_bytes(b"dummy")
    result = validate({"type": "faiss", "index_path": str(p)})
    assert result["ok"] is True
    assert result["collection_count"] == 1


def test_weaviate_ok(requests_mock):
    requests_mock.get("http://localhost:8080/v1/.well-known/ready", status_code=200)
    result = validate({"type": "weaviate", "url": "http://localhost:8080"})
    assert result["ok"] is True


def test_weaviate_not_ready(requests_mock):
    requests_mock.get("http://localhost:8080/v1/.well-known/ready", status_code=503)
    result = validate({"type": "weaviate", "url": "http://localhost:8080"})
    assert result["ok"] is False
    assert "503" in result["error"]


def test_generic_ok(requests_mock):
    requests_mock.get("http://my-api.example/v1", status_code=200)
    result = validate({"type": "generic", "base_url": "http://my-api.example/v1"})
    assert result["ok"] is True


def test_generic_server_error(requests_mock):
    requests_mock.get("http://my-api.example/v1", status_code=500)
    result = validate({"type": "generic", "base_url": "http://my-api.example/v1"})
    assert result["ok"] is False


def test_qdrant_missing_dependency():
    with patch.dict("sys.modules", {"qdrant_client": None}):
        result = validate({"type": "qdrant", "host": "localhost", "port": 6333})
    assert result["ok"] is False
    assert "not installed" in result["error"]


def test_exception_returns_error(requests_mock):
    requests_mock.get("http://bad/v1/.well-known/ready", exc=ConnectionError("refused"))
    result = validate({"type": "weaviate", "url": "http://bad"})
    assert result["ok"] is False
