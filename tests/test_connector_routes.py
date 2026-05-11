# tests/test_connector_routes.py
import os, uuid, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

from fastapi import FastAPI
from fastapi.testclient import TestClient
import db.store as store
from auth.middleware import AuthMiddleware
from connectors.routes import connector_router

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

def _make_client(tmp_path, role="admin"):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    import bcrypt, jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    uid = str(uuid.uuid4())
    h = bcrypt.hashpw(b"pw", bcrypt.gensalt(rounds=4)).decode()
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid, "admin@x.com", h, role, "local"))
    exp = datetime.now(timezone.utc) + timedelta(hours=1)
    tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()), uid, tok, exp.isoformat()))
    store.commit()
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(connector_router)
    client = TestClient(app, raise_server_exceptions=False)
    return client, tok

def test_create_connector(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type": "gdrive", "name": "My GDrive", "credentials": {"key": "val"}},
        cookies={"kvforge_session": tok})
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "My GDrive"
    assert d["credentials"] == "●●●●●●"

def test_list_connectors(tmp_path):
    client, tok = _make_client(tmp_path)
    client.post("/studio/api/connectors",
        json={"type": "s3", "name": "My S3", "credentials": {"bucket": "b"}},
        cookies={"kvforge_session": tok})
    r = client.get("/studio/api/connectors", cookies={"kvforge_session": tok})
    assert r.status_code == 200
    assert len(r.json()) == 1

def test_viewer_cannot_create(tmp_path):
    client, tok = _make_client(tmp_path, role="viewer")
    r = client.post("/studio/api/connectors",
        json={"type": "s3", "name": "X", "credentials": {}},
        cookies={"kvforge_session": tok})
    assert r.status_code == 403

def test_add_scope(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type": "gdrive", "name": "GD", "credentials": {"k": "v"}},
        cookies={"kvforge_session": tok})
    cid = r.json()["id"]
    r2 = client.post(f"/studio/api/connectors/{cid}/scopes",
        json={"uc_id": "uc1", "scope_config": {"folder_id": "abc"}},
        cookies={"kvforge_session": tok})
    assert r2.status_code == 200
    r3 = client.get(f"/studio/api/connectors/{cid}/scopes", cookies={"kvforge_session": tok})
    assert len(r3.json()) == 1

def test_delete_connector(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type": "s3", "name": "S3", "credentials": {"k": "v"}},
        cookies={"kvforge_session": tok})
    cid = r.json()["id"]
    r2 = client.delete(f"/studio/api/connectors/{cid}", cookies={"kvforge_session": tok})
    assert r2.status_code == 200
    assert client.get("/studio/api/connectors", cookies={"kvforge_session": tok}).json() == []
