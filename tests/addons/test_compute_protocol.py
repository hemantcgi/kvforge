"""Tests for ComputeBackend protocol and get_backend() factory."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from addons.compute.protocol import ComputeBackend


# ── Protocol structural checks ────────────────────────────────────────────────

class _FullImpl:
    def compute_kv_batch(self, texts, num_layers, num_kv_heads, head_dim):
        return []

    def health(self):
        return {}


class _MissingComputeKvBatch:
    def health(self):
        return {}


def test_minimal_class_satisfies_protocol():
    obj = _FullImpl()
    assert isinstance(obj, ComputeBackend)


def test_missing_compute_kv_batch_fails_protocol():
    obj = _MissingComputeKvBatch()
    assert not isinstance(obj, ComputeBackend)


def test_local_backend_satisfies_protocol():
    from addons.compute.local_backend import LocalComputeBackend
    assert issubclass(LocalComputeBackend, ComputeBackend) or True  # structural check via isinstance below
    with patch("core.model_loader.load", return_value=(MagicMock(), MagicMock())), \
         patch("core.version.load", return_value={}):
        backend = LocalComputeBackend({})
    assert isinstance(backend, ComputeBackend)


def test_remote_backend_satisfies_protocol():
    from addons.compute.remote_backend import RemoteComputeBackend
    with patch("httpx.Client"):
        backend = RemoteComputeBackend(worker_url="http://localhost:9999")
    assert isinstance(backend, ComputeBackend)


# ── get_backend() factory ─────────────────────────────────────────────────────

def test_get_backend_default_returns_local():
    from addons.compute import get_backend
    from addons.compute.local_backend import LocalComputeBackend
    with patch("addons.compute.local_backend.LocalComputeBackend.__init__", return_value=None):
        backend = get_backend({})
    assert isinstance(backend, LocalComputeBackend)


def test_get_backend_remote_returns_remote():
    from addons.compute import get_backend
    from addons.compute.remote_backend import RemoteComputeBackend
    cfg = {
        "addon_config": {
            "compute": {
                "backend": "remote",
                "worker_url": "http://localhost:9999",
            }
        }
    }
    with patch("httpx.Client"):
        backend = get_backend(cfg)
    assert isinstance(backend, RemoteComputeBackend)
