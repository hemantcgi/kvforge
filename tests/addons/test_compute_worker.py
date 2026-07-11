"""Tests for the KVForge compute worker FastAPI app."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import addons.compute.worker as worker_mod
import core.kv_utils as kv_utils


@pytest.fixture(autouse=True)
def reset_worker_state():
    """Reset global worker state before each test to avoid state bleed."""
    worker_mod._model = None
    worker_mod._tokenizer = None
    worker_mod._model_id = ""
    worker_mod._api_key = ""
    yield
    worker_mod._model = None
    worker_mod._tokenizer = None
    worker_mod._model_id = ""
    worker_mod._api_key = ""


# ── Health endpoint tests ─────────────────────────────────────────────────────

def test_health_no_model():
    client = TestClient(worker_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["model_loaded"] is False


def test_health_fields():
    client = TestClient(worker_mod.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "device" in data
    assert "model_id" in data
    assert "model_loaded" in data
    assert "gpu_info" in data


# ── compute_kv endpoint tests ─────────────────────────────────────────────────

def test_compute_kv_no_model_returns_503():
    client = TestClient(worker_mod.app)
    r = client.post("/compute_kv", json={
        "texts": ["hello"],
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    })
    assert r.status_code == 503


def test_compute_kv_empty_texts_returns_200():
    client = TestClient(worker_mod.app)
    r = client.post("/compute_kv", json={
        "texts": [],
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["tensors"] == []


def test_compute_kv_with_model(monkeypatch):
    mock_arr = np.zeros((28, 2, 8, 128), dtype=np.float16)
    monkeypatch.setattr("core.compute.compute_kv_for_chunk", lambda *a, **kw: mock_arr)
    monkeypatch.setattr("core.model_loader.load", lambda *a, **kw: (MagicMock(), MagicMock()))
    monkeypatch.setattr("core.version.load", lambda: {})

    worker_mod._model = None
    worker_mod.load_model("test-model")

    client = TestClient(worker_mod.app)
    r = client.post("/compute_kv", json={
        "texts": ["a", "b"],
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert len(data["tensors"]) == 2
    assert data["shape"] == [28, 2, 8, 128]


def test_compute_kv_wrong_api_key_returns_403():
    worker_mod._api_key = "secret"
    client = TestClient(worker_mod.app)
    r = client.post(
        "/compute_kv",
        json={"texts": [], "num_layers": 28, "num_kv_heads": 8, "head_dim": 128},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 403


def test_compute_kv_correct_api_key_passes():
    worker_mod._api_key = "secret"
    client = TestClient(worker_mod.app)
    r = client.post(
        "/compute_kv",
        json={"texts": [], "num_layers": 28, "num_kv_heads": 8, "head_dim": 128},
        headers={"X-API-Key": "secret"},
    )
    assert r.status_code == 200


def test_infer_returns_501():
    client = TestClient(worker_mod.app)
    r = client.post("/infer")
    assert r.status_code == 501
