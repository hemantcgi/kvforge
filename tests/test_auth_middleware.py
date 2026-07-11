import os, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _make_app(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "test.db"
    store.close()
    store.migrate()
    # Insert a user and session
    import uuid, bcrypt
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid, "a@b.com", bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode(), "admin", "local"))
    exp = datetime.now(timezone.utc) + timedelta(hours=1)
    tok = pyjwt.encode({"sub": uid, "role": "admin", "exp": exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    sid = str(uuid.uuid4())
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (sid, uid, tok, exp.isoformat()))
    store.commit()
    from auth.middleware import AuthMiddleware
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/studio/api/test")
    def protected(request: Request):
        return {"user": request.state.user.email}

    return app, tok


def test_no_cookie_returns_401(tmp_path):
    app, _ = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/studio/api/test", cookies={})
    assert r.status_code == 401


def test_valid_cookie_passes(tmp_path):
    app, tok = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/studio/api/test", cookies={"kvforge_session": tok})
    assert r.status_code == 200
    assert r.json()["user"] == "a@b.com"


def test_auth_route_passes_without_cookie(tmp_path):
    app, _ = _make_app(tmp_path)

    @app.get("/auth/login")
    def login(): return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/auth/login")
    assert r.status_code == 200


def test_unauthenticated_page_redirects_to_login(tmp_path):
    app, _ = _make_app(tmp_path)
    @app.get("/studio/")
    def studio(): return {"ok": True}
    client = TestClient(app, raise_server_exceptions=False, follow_redirects=False)
    r = client.get("/studio/")
    assert r.status_code == 302
    assert "/auth/login" in r.headers.get("location", "")
    assert "next" in r.headers.get("location", "")


def test_invalid_token_returns_401(tmp_path):
    app, _ = _make_app(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/studio/api/test", cookies={"kvforge_session": "garbage.token.value"})
    assert r.status_code == 401


def test_expired_session_returns_401(tmp_path):
    import db.store as store
    import uuid
    store.DB_PATH = tmp_path / "exp.db"
    store.close()
    store.migrate()
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid,"exp@b.com","hashed","admin","local"))
    # Insert session with expires_at in the past
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    past = datetime.now(timezone.utc) - timedelta(hours=2)
    tok = pyjwt.encode({"sub": uid, "role": "admin", "exp": past},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()), uid, tok, past.isoformat()))
    store.commit()
    from auth.middleware import AuthMiddleware
    from fastapi import FastAPI, Request
    app2 = FastAPI()
    app2.add_middleware(AuthMiddleware)
    @app2.get("/studio/api/test")
    def protected(request: Request):
        return {"user": request.state.user.email}
    client = TestClient(app2, raise_server_exceptions=False)
    r = client.get("/studio/api/test", cookies={"kvforge_session": tok})
    assert r.status_code == 401
