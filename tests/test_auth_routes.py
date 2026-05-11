# tests/test_auth_routes.py
import os, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import db.store as store
from auth.middleware import AuthMiddleware

def _make_client(tmp_path):
    store.DB_PATH = tmp_path / "test.db"
    store.close()
    store.migrate()
    from auth.routes import router
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    return TestClient(app, follow_redirects=False)

def test_login_valid(tmp_path):
    import bcrypt, uuid
    client = _make_client(tmp_path)
    uid = str(uuid.uuid4())
    h = bcrypt.hashpw(b"secret", bcrypt.gensalt(rounds=4)).decode()
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid,"u@test.com",h,"admin","local"))
    store.commit()
    r = client.post("/auth/login", data={"email":"u@test.com","password":"secret"})
    assert r.status_code == 302
    assert "kvforge_session" in r.cookies

def test_login_wrong_password(tmp_path):
    client = _make_client(tmp_path)
    r = client.post("/auth/login", data={"email":"nobody@x.com","password":"wrong"})
    assert r.status_code == 302
    assert "kvforge_session" not in r.cookies

def test_signup_via_invite(tmp_path):
    import uuid; from datetime import datetime, timedelta, timezone
    client = _make_client(tmp_path)
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"admin@x.com","admin","local"))
    exp = (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()
    store.execute("INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
                  ("tok123","new@x.com","editor",admin_id,exp))
    store.commit()
    r = client.post("/auth/signup", data={"token":"tok123","password":"newpass123"})
    assert r.status_code == 302
    row = store.fetchone("SELECT * FROM users WHERE email=?", ("new@x.com",))
    assert row is not None
    assert row["role"] == "editor"
    used = store.fetchone("SELECT used_at FROM invite_tokens WHERE token=?", ("tok123",))
    assert used["used_at"] is not None

def test_invite_token_single_use(tmp_path):
    import uuid; from datetime import datetime, timedelta, timezone
    client = _make_client(tmp_path)
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"admin@x.com","admin","local"))
    exp = (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()
    store.execute("INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
                  ("tok456","b@x.com","viewer",admin_id,exp))
    store.commit()
    client.post("/auth/signup", data={"token":"tok456","password":"pw123456"})
    r2 = client.post("/auth/signup", data={"token":"tok456","password":"pw999999"})
    assert r2.status_code in (400, 422)
    count = store.fetchone("SELECT COUNT(*) as n FROM users WHERE email=?",("b@x.com",))
    assert count["n"] == 1
