"""Tests for LocalComputeBackend."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from addons.compute.protocol import ComputeBackend


def _make_backend():
    """Construct LocalComputeBackend with all heavy dependencies mocked."""
    with patch("core.model_loader.load", return_value=(MagicMock(), MagicMock())), \
         patch("core.version.load", return_value={}):
        from addons.compute.local_backend import LocalComputeBackend
        return LocalComputeBackend({})


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_compute_kv_batch_returns_correct_shapes():
    mock_arr = np.zeros((28, 2, 8, 128), dtype=np.float16)
    backend = _make_backend()
    # Patch at the location where local_backend imported compute_kv_for_chunk
    with patch("addons.compute.local_backend.compute_kv_for_chunk", return_value=mock_arr):
        result = backend.compute_kv_batch(["text1", "text2"], 28, 8, 128)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (28, 2, 8, 128)


def test_compute_kv_batch_empty():
    backend = _make_backend()
    result = backend.compute_kv_batch([], 28, 8, 128)
    assert result == []


def test_health_returns_local():
    backend = _make_backend()
    # mock model.parameters() to avoid real torch call
    backend._model.parameters.return_value = iter([MagicMock(device="cpu")])
    h = backend.health()
    assert h["backend"] == "local"


def test_satisfies_protocol():
    backend = _make_backend()
    assert isinstance(backend, ComputeBackend)
