"""Tests for RemoteComputeBackend."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

import core.kv_utils as kv_utils
from addons.compute.protocol import ComputeBackend
from addons.compute.remote_backend import RemoteComputeBackend


def _make_response(data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_compute_kv_batch_posts_correctly():
    mock_arr = np.zeros((28, 2, 8, 128), dtype=np.float16)
    tensors = [kv_utils.serialize_kv(mock_arr), kv_utils.serialize_kv(mock_arr)]
    response_data = {"tensors": tensors, "shape": [28, 2, 8, 128], "count": 2, "elapsed_ms": 10.0}

    mock_client = MagicMock()
    mock_client.post.return_value = _make_response(response_data)

    with patch("httpx.Client", return_value=mock_client):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")

    backend.compute_kv_batch(["text1", "text2"], 28, 8, 128)

    mock_client.post.assert_called_once_with(
        "http://localhost:9999/compute_kv",
        json={
            "texts": ["text1", "text2"],
            "num_layers": 28,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    )


def test_compute_kv_batch_empty_returns_empty():
    mock_client = MagicMock()

    with patch("httpx.Client", return_value=mock_client):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")

    result = backend.compute_kv_batch([], 28, 8, 128)
    assert result == []
    mock_client.post.assert_not_called()


def test_compute_kv_batch_deserializes_tensors():
    mock_arr = np.zeros((28, 2, 8, 128), dtype=np.float16)
    tensors = [kv_utils.serialize_kv(mock_arr), kv_utils.serialize_kv(mock_arr)]
    response_data = {"tensors": tensors, "shape": [28, 2, 8, 128], "count": 2, "elapsed_ms": 5.0}

    mock_client = MagicMock()
    mock_client.post.return_value = _make_response(response_data)

    with patch("httpx.Client", return_value=mock_client):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")

    result = backend.compute_kv_batch(["a", "b"], 28, 8, 128)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (28, 2, 8, 128)
        assert arr.dtype == np.float16


def test_health_calls_get():
    mock_client = MagicMock()
    mock_client.get.return_value = _make_response({"status": "ok"})

    with patch("httpx.Client", return_value=mock_client):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")

    result = backend.health()
    mock_client.get.assert_called_once_with("http://localhost:9999/health", timeout=10.0)
    assert result["status"] == "ok"


def test_api_key_in_header():
    with patch("httpx.Client") as mock_client_cls:
        RemoteComputeBackend(worker_url="http://localhost:9999", api_key="secret")
    # Verify X-API-Key is in the headers passed to httpx.Client constructor
    _, kwargs = mock_client_cls.call_args
    headers = kwargs.get("headers", {})
    assert headers.get("X-API-Key") == "secret"


def test_satisfies_protocol():
    with patch("httpx.Client"):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")
    assert isinstance(backend, ComputeBackend)
