# tests/test_sync_webhook.py
import os, uuid, hmac, hashlib, json, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

import db.store as store
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sync.webhook import make_webhook_router

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _reset_store():
    yield
    store.close()

def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "wh.db")
    store._local.__dict__.clear()
    store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id, "a@b.com", "admin", "local"))
    cid = str(uuid.uuid4())
    secret = "webhook-secret-xyz"
    store.execute(
        "INSERT INTO connector_configs(id,type,name,credentials_json,webhook_secret,created_by) VALUES(?,?,?,?,?,?)",
        (cid, "gdrive", "GD", '{"k":"v"}', secret, admin_id)
    )
    store.execute("INSERT INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
                  (cid, "uc1", '{}'))
    store.commit()
    return cid, secret

def _make_client(tmp_path, monkeypatch, triggered_calls):
    cid, secret = _setup(tmp_path, monkeypatch)
    async def fake_run(cid, uc_id, trigger):
        triggered_calls.append((cid, uc_id, trigger))
    app = FastAPI()
    app.include_router(make_webhook_router(run_fn=fake_run))
    return TestClient(app, raise_server_exceptions=False), cid, secret

def _sign(body: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"

def test_valid_signature_accepted(tmp_path, monkeypatch):
    triggered = []
    client, cid, secret = _make_client(tmp_path, monkeypatch, triggered)
    body = json.dumps({"change": "new-file"}).encode()
    r = client.post(f"/webhooks/{cid}",
                    content=body,
                    headers={"X-Hub-Signature-256": _sign(body, secret),
                             "Content-Type": "application/json"})
    assert r.status_code == 200

def test_invalid_signature_rejected(tmp_path, monkeypatch):
    client, cid, _ = _make_client(tmp_path, monkeypatch, [])
    body = b'{"change":"x"}'
    r = client.post(f"/webhooks/{cid}",
                    content=body,
                    headers={"X-Hub-Signature-256": "sha256=invalidsig",
                             "Content-Type": "application/json"})
    assert r.status_code == 401

def test_unknown_connector_returns_404(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    async def fake_run(*a): pass
    app = FastAPI()
    app.include_router(make_webhook_router(run_fn=fake_run))
    client = TestClient(app, raise_server_exceptions=False)
    body = b"{}"
    r = client.post("/webhooks/nonexistent",
                    content=body,
                    headers={"X-Hub-Signature-256": _sign(body, "x"),
                             "Content-Type": "application/json"})
    assert r.status_code == 404
