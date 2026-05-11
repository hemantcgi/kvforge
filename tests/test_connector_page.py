# tests/test_connector_page.py
import os, uuid, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

def _make_portal_client(tmp_path, role):
    import db.store as store
    store.DB_PATH = tmp_path / f"{role}.db"
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

def test_connectors_page_requires_editor(tmp_path):
    client, tok = _make_portal_client(tmp_path, "viewer")
    r = client.get("/studio/connectors", cookies={"kvforge_session": tok})
    assert r.status_code == 403

def test_connectors_page_loads_for_admin(tmp_path):
    client, tok = _make_portal_client(tmp_path, "admin")
    r = client.get("/studio/connectors", cookies={"kvforge_session": tok})
    assert r.status_code == 200
    assert b"Connectors" in r.content
