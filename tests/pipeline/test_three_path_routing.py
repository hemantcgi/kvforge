import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def _mock_chunk(status="active", kv_token_path=None):
    return {
        "id": "chunk1",
        "payload": {
            "text": "Sample chunk text.",
            "kv_cache": "AAAA",
            "kv_token_path": kv_token_path,
            "status": status,
            "archive_path": "/archive/chunk1.txt" if status == "archived" else None,
        }
    }


def test_routes_enhanced_chunk(tmp_path):
    from pipeline.kv_inference import route_chunk_injection
    chunk = _mock_chunk(kv_token_path=str(tmp_path / "chunk1.npz"))
    arr = np.zeros((4, 2, 2, 8, 64), dtype=np.float16)
    with patch("pipeline.kv_inference.load_token_kv", return_value=arr) as mock_load:
        result = route_chunk_injection(chunk, cfg={}, tq_config=None)
    assert result["path"] == "enhanced"
    mock_load.assert_called_once()


def test_routes_active_chunk():
    from pipeline.kv_inference import route_chunk_injection
    chunk = _mock_chunk(status="active", kv_token_path=None)
    with patch("pipeline.kv_inference.deserialize_kv") as mock_deser:
        mock_deser.return_value = np.zeros((4, 2, 2, 64), dtype=np.float16)
        result = route_chunk_injection(chunk, cfg={"kv_num_layers": 4,
            "kv_num_heads": 2, "kv_head_dim": 64}, tq_config=None)
    assert result["path"] == "active"


def test_routes_archive_chunk():
    from pipeline.kv_inference import route_chunk_injection
    chunk = _mock_chunk(status="archived")
    with patch("pipeline.kv_inference._fetch_archive_text", return_value="archived text"):
        result = route_chunk_injection(chunk, cfg={}, tq_config=None)
    assert result["path"] == "archive"
    assert result["text"] == "archived text"


def test_archive_retrieval_count_incremented():
    from pipeline.kv_inference import route_chunk_injection
    mock_vs = MagicMock()
    chunk = _mock_chunk(status="archived")
    chunk["payload"]["archive_retrieval_count"] = 2
    with patch("pipeline.kv_inference._fetch_archive_text", return_value="text"):
        route_chunk_injection(chunk, cfg={}, tq_config=None, vector_store=mock_vs)
    mock_vs.update_payload.assert_called_once()
    updated_payload = mock_vs.update_payload.call_args[1]["payload"]
    assert updated_payload["archive_retrieval_count"] == 3
