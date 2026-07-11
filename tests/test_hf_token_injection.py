# tests/test_hf_token_injection.py
import pytest
from unittest.mock import patch


def test_hf_token_injected_for_index_step(monkeypatch):
    """_build_env must set HF_TOKEN when huggingface_token is configured."""
    from studio import settings_manager
    monkeypatch.setattr(settings_manager, "get_setting",
                        lambda key: "hf_test_tok_abcd1234" if key == "huggingface_token" else "")

    from studio.pipeline_runner import _build_env
    env = _build_env("usecase1_customer_support", "index")
    assert env.get("HF_TOKEN") == "hf_test_tok_abcd1234"


def test_hf_token_injected_for_train_step(monkeypatch):
    from studio import settings_manager
    monkeypatch.setattr(settings_manager, "get_setting",
                        lambda key: "hf_tok_xyz" if key == "huggingface_token" else "")

    from studio.pipeline_runner import _build_env
    env = _build_env("usecase1_customer_support", "train")
    assert env.get("HF_TOKEN") == "hf_tok_xyz"


def test_hf_token_not_injected_when_empty(monkeypatch):
    from studio import settings_manager
    monkeypatch.setattr(settings_manager, "get_setting", lambda key: "")

    from studio.pipeline_runner import _build_env
    env = _build_env("usecase1_customer_support", "index")
    assert "HF_TOKEN" not in env or not env["HF_TOKEN"]


def test_settings_route_exists():
    """GET /studio/settings must return 200 (not 404)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.routes import router
    app = FastAPI()
    app.include_router(router, prefix="/studio")
    client = TestClient(app)
    resp = client.get("/studio/settings")
    assert resp.status_code == 200
