import pytest
from unittest.mock import MagicMock


def test_archive_chunk_clears_kv_blob(tmp_path):
    from pipeline.archive_manager import archive_chunk
    from addons.corpus_intelligence.archive import LocalArchiveBackend

    mock_vs = MagicMock()
    backend = LocalArchiveBackend(str(tmp_path))
    chunk = {
        "id": "chunk_arc",
        "payload": {
            "text": "The chunk text to archive.",
            "kv_cache": "AAAA",
            "kv_token_path": None,
            "status": "active",
        }
    }
    archive_chunk(chunk, backend=backend, vector_store=mock_vs, collection="col")

    assert backend.read("chunk_arc") == "The chunk text to archive."

    mock_vs.update_payload.assert_called_once()
    payload = mock_vs.update_payload.call_args[1]["payload"]
    assert payload.get("kv_cache") is None or payload.get("kv_cache") == ""
    assert payload["status"] == "archived"
    assert "archive_path" in payload
    assert payload["archive_retrieval_count"] == 0


def test_archive_chunk_removes_disk_kv(tmp_path):
    from pipeline.archive_manager import archive_chunk
    from addons.corpus_intelligence.archive import LocalArchiveBackend

    kv_path = tmp_path / "chunk_kv.npz"
    kv_path.write_bytes(b"fake npz")

    mock_vs = MagicMock()
    backend = LocalArchiveBackend(str(tmp_path / "archive"))
    chunk = {
        "id": "chunk_kv",
        "payload": {
            "text": "text",
            "kv_cache": "AAAA",
            "kv_token_path": str(kv_path),
            "status": "active",
        }
    }
    archive_chunk(chunk, backend=backend, vector_store=mock_vs, collection="col")
    assert not kv_path.exists(), "Disk KV file should be deleted on archive"
