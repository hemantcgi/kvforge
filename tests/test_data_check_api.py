# tests/test_data_check_api.py
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


def _make_app():
    from fastapi import FastAPI
    from studio.api import api_router
    app = FastAPI()
    app.include_router(api_router, prefix="/api")
    return app


def test_data_check_corpus_exists(tmp_path, monkeypatch):
    uc_id = "usecase1_customer_support"
    corpus = tmp_path / "examples" / uc_id / "data" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"text":"hello"}\n')
    cfg_file = tmp_path / "examples" / uc_id / "config.json"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps({"addon_config": {"indexing": {"loader": "jsonl"}}}))

    import studio.pipeline_runner as pr
    monkeypatch.setattr(pr, "ROOT", tmp_path)
    import studio.api as api_mod
    monkeypatch.setattr(api_mod, "ROOT", tmp_path, raising=False)

    client = TestClient(_make_app())
    resp = client.get(f"/api/check-data/{uc_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_exists"] is True


def test_data_check_corpus_missing(tmp_path, monkeypatch):
    uc_id = "usecase1_customer_support"
    cfg_file = tmp_path / "examples" / uc_id / "config.json"
    cfg_file.parent.mkdir(parents=True)
    cfg_file.write_text(json.dumps({"addon_config": {"indexing": {"loader": "jsonl"}}}))

    import studio.api as api_mod
    monkeypatch.setattr(api_mod, "ROOT", tmp_path, raising=False)

    client = TestClient(_make_app())
    resp = client.get(f"/api/check-data/{uc_id}")
    assert resp.status_code == 200
    assert resp.json()["corpus_exists"] is False
