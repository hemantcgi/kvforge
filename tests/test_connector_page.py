# tests/test_connector_page.py
import os, uuid, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _reset_store():
    import db.store as store
    yield
    store.close()

def _make_portal_client(tmp_path, role, monkeypatch):
    import db.store as store
    db_path = tmp_path / f"{role}.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)
    store._local.__dict__.clear()
    store.migrate()
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (uid, f"{role}@x.com", role, "local"))
    exp = datetime.now(timezone.utc) + timedelta(hours=1)
    tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()), uid, tok, exp.isoformat()))
    store.commit()
    import kvforge_portal
    from fastapi.testclient import TestClient
    client = TestClient(kvforge_portal.app, raise_server_exceptions=False)
    return client, tok

def test_connectors_page_requires_editor(tmp_path, monkeypatch):
    client, tok = _make_portal_client(tmp_path, "viewer", monkeypatch)
    r = client.get("/studio/connectors", cookies={"kvforge_session": tok})
    assert r.status_code == 403

def test_connectors_page_loads_for_admin(tmp_path, monkeypatch):
    client, tok = _make_portal_client(tmp_path, "admin", monkeypatch)
    r = client.get("/studio/connectors", cookies={"kvforge_session": tok})
    assert r.status_code == 200
    assert b"Connectors" in r.content
