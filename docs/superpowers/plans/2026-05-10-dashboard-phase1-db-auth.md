# Dashboard Phase 1 — DB Foundation & Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the SQLite database layer and complete auth system (email/password + Google/Microsoft/AWS OAuth, invite-only signup, JWT cookies, role middleware).

**Architecture:** New `db/` package is a thread-safe sqlite3 wrapper; `auth/` package adds models, middleware, routes, OAuth clients. Both are wired into `kvforge_portal.py` via FastAPI lifespan and router include. Phases 2–4 build on top of these.

**Tech Stack:** FastAPI, sqlite3, bcrypt, PyJWT, authlib, msal, cryptography (Fernet), python-dotenv

**Spec:** `docs/superpowers/specs/2026-05-10-dashboard-auth-connectors-design.md` §1

**Depends on:** nothing (foundation)

**Next plan:** `2026-05-10-dashboard-phase2-connectors.md`

---

### Task 1: db/ package

**Files:**
- Create: `db/__init__.py`
- Create: `db/schema.sql`
- Create: `db/store.py`
- Create: `tests/test_db_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_store.py
import os, tempfile, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-key-32bytesXXXXXXXXXXXXXX")

def test_migrate_creates_tables(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store._local.__dict__.clear()
    store.migrate()
    rows = store.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r["name"] for r in rows}
    assert {"users","sessions","invite_tokens","connector_configs",
            "connector_uc_scopes","sync_runs"} <= names

def test_execute_and_fetchone(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test2.db")
    store._local.__dict__.clear()
    store.migrate()
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  ("u1","a@b.com","admin","local"))
    store.commit()
    row = store.fetchone("SELECT * FROM users WHERE id=?", ("u1",))
    assert row["email"] == "a@b.com"
    assert row["role"] == "admin"
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_db_store.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Create db/__init__.py**

```python
# db/__init__.py
```

- [ ] **Step 4: Create db/schema.sql**

```sql
-- db/schema.sql
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    hashed_pw TEXT,
    role TEXT NOT NULL CHECK(role IN ('admin','editor','viewer')),
    provider TEXT NOT NULL DEFAULT 'local',
    provider_id TEXT,
    invited_by TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    jwt_token TEXT NOT NULL,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS invite_tokens (
    token TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    role TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES users(id),
    expires_at DATETIME NOT NULL,
    used_at DATETIME
);
CREATE TABLE IF NOT EXISTS connector_configs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('gdrive','s3','sharepoint')),
    name TEXT NOT NULL,
    credentials_json TEXT NOT NULL,
    schedule_cron TEXT,
    webhook_secret TEXT,
    created_by TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS connector_uc_scopes (
    connector_id TEXT NOT NULL REFERENCES connector_configs(id) ON DELETE CASCADE,
    uc_id TEXT NOT NULL,
    scope_config_json TEXT NOT NULL,
    last_sync_at DATETIME,
    last_delta_token TEXT,
    PRIMARY KEY (connector_id, uc_id)
);
CREATE TABLE IF NOT EXISTS sync_runs (
    id TEXT PRIMARY KEY,
    connector_id TEXT NOT NULL REFERENCES connector_configs(id),
    uc_id TEXT NOT NULL,
    trigger TEXT NOT NULL CHECK(trigger IN ('manual','scheduled','webhook')),
    status TEXT NOT NULL CHECK(status IN ('running','ok','error')),
    files_total INTEGER DEFAULT 0,
    files_done INTEGER DEFAULT 0,
    error TEXT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME
);
```

- [ ] **Step 5: Create db/store.py**

```python
# db/store.py
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path.home() / ".kvforge" / "studio.db"
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


def execute(sql: str, params=()) -> sqlite3.Cursor:
    return _conn().execute(sql, params)


def fetchall(sql: str, params=()) -> list:
    return _conn().execute(sql, params).fetchall()


def fetchone(sql: str, params=()):
    return _conn().execute(sql, params).fetchone()


def commit() -> None:
    _conn().commit()


def migrate() -> None:
    schema = (Path(__file__).parent / "schema.sql").read_text()
    _conn().executescript(schema)
    _conn().commit()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_db_store.py -v --override-ini="addopts="
```
Expected: `2 passed`

- [ ] **Step 7: Install dependencies**

```bash
pip install bcrypt PyJWT authlib msal cryptography httpx
```

- [ ] **Step 8: Commit**

```bash
git add db/ tests/test_db_store.py
git commit -m "feat: add db/ package — SQLite store with schema migration"
```

---

### Task 2: auth/models.py

**Files:**
- Create: `auth/__init__.py`
- Create: `auth/models.py`
- Create: `tests/test_auth_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth_models.py
import sqlite3
from datetime import datetime
from auth.models import User

def test_user_from_row():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE users(id,email,hashed_pw,role,provider,
                    provider_id,invited_by,created_at)""")
    conn.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?,?)",
                 ("u1","x@y.com",None,"admin","google","gid",None,"2026-01-01"))
    row = conn.execute("SELECT * FROM users").fetchone()
    u = User.from_row(row)
    assert u.id == "u1"
    assert u.role == "admin"
    assert u.provider == "google"
    assert u.hashed_pw is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_auth_models.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'auth'`

- [ ] **Step 3: Create auth/__init__.py**

```python
# auth/__init__.py
```

- [ ] **Step 4: Create auth/models.py**

```python
# auth/models.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class User:
    id: str
    email: str
    hashed_pw: str | None
    role: Literal["admin", "editor", "viewer"]
    provider: Literal["local", "google", "microsoft", "aws", "saml"]
    provider_id: str | None
    invited_by: str | None
    created_at: str  # ISO string from SQLite

    @classmethod
    def from_row(cls, row) -> "User":
        return cls(
            id=row["id"], email=row["email"], hashed_pw=row["hashed_pw"],
            role=row["role"], provider=row["provider"],
            provider_id=row["provider_id"], invited_by=row["invited_by"],
            created_at=row["created_at"],
        )


@dataclass
class InviteToken:
    token: str
    email: str
    role: str
    created_by: str
    expires_at: str
    used_at: str | None


@dataclass
class Session:
    id: str
    user_id: str
    jwt_token: str
    expires_at: str
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_auth_models.py -v --override-ini="addopts="
```
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add auth/ tests/test_auth_models.py
git commit -m "feat: add auth/models.py — User, InviteToken, Session dataclasses"
```

---

### Task 3: auth/middleware.py

**Files:**
- Create: `auth/middleware.py`
- Create: `tests/test_auth_middleware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth_middleware.py
import os, tempfile, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

def _make_app(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    # Insert a user and session
    import uuid, bcrypt
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid,"a@b.com",bcrypt.hashpw(b"pw",bcrypt.gensalt()).decode(),"admin","local"))
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
    def protected(request):
        from fastapi import Request
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_auth_middleware.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'auth.middleware'`

- [ ] **Step 3: Create auth/middleware.py**

```python
# auth/middleware.py
import os
from datetime import datetime, timezone
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse, RedirectResponse
import jwt as pyjwt
import db.store as store
from auth.models import User

SECRET = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")

_PUBLIC_PREFIXES = ("/auth/", "/webhooks/", "/static/")
_API_PREFIXES = ("/studio/api/", "/sync/", "/api/")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        token = request.cookies.get("kvforge_session")
        user = _validate_token(token) if token else None

        if user is None:
            if any(path.startswith(p) for p in _API_PREFIXES):
                return JSONResponse({"detail": "not authenticated"}, status_code=401)
            return RedirectResponse(f"/auth/login?next={path}", status_code=302)

        request.state.user = user
        return await call_next(request)


def _validate_token(token: str) -> User | None:
    try:
        payload = pyjwt.decode(token, SECRET, algorithms=["HS256"])
        row = store.fetchone(
            "SELECT user_id FROM sessions WHERE jwt_token=? AND expires_at > ?",
            (token, datetime.now(timezone.utc).isoformat())
        )
        if row is None:
            return None
        u = store.fetchone("SELECT * FROM users WHERE id=?", (payload["sub"],))
        return User.from_row(u) if u else None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_auth_middleware.py -v --override-ini="addopts="
```
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add auth/middleware.py tests/test_auth_middleware.py
git commit -m "feat: add AuthMiddleware — JWT cookie validation, role on request.state"
```

---

### Task 4: auth/routes.py (email/password + invite + signup)

**Files:**
- Create: `auth/routes.py`
- Create: `tests/test_auth_routes.py`
- Create: `templates/studio/auth/login.html`
- Create: `templates/studio/auth/signup.html`
- Create: `templates/studio/auth/admin_users.html`

- [ ] **Step 1: Create templates**

```html
<!-- templates/studio/auth/login.html -->
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>KVForge Studio — Sign In</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#e0e0e0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;padding:32px;width:340px}
h1{font-size:16px;color:#4ec9b0;margin-bottom:4px}
.sub{font-size:12px;color:#555;margin-bottom:20px}
input{width:100%;background:#252525;border:1px solid #333;border-radius:6px;padding:8px 12px;font-size:13px;color:#ccc;margin-bottom:10px}
button{width:100%;background:#4ec9b0;border:none;border-radius:6px;padding:9px;font-size:13px;font-weight:700;color:#111;cursor:pointer;margin-bottom:14px}
.err{color:#ce9178;font-size:12px;margin-bottom:10px}
.divider{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.divider-line{flex:1;height:1px;background:#2a2a2a}
.divider-text{font-size:11px;color:#555}
.oauth-btn{display:block;width:100%;background:#252525;border:1px solid #333;border-radius:6px;padding:8px 12px;font-size:12px;color:#ccc;margin-bottom:8px;text-align:center;text-decoration:none;cursor:pointer}
</style></head><body>
<div class="card">
  <h1>KVForge Studio</h1>
  <div class="sub">Sign in to your workspace</div>
  {{#if error}}<div class="err">{{error}}</div>{{/if}}
  <form method="post" action="/auth/login">
    <input name="email" type="email" placeholder="Email address" required>
    <input name="password" type="password" placeholder="Password" required>
    <button type="submit">Sign In</button>
  </form>
  <div class="divider"><div class="divider-line"></div><div class="divider-text">or continue with</div><div class="divider-line"></div></div>
  {{#if google_enabled}}<a class="oauth-btn" href="/auth/oauth/google">Sign in with Google</a>{{/if}}
  {{#if microsoft_enabled}}<a class="oauth-btn" href="/auth/oauth/microsoft">Sign in with Microsoft</a>{{/if}}
  {{#if aws_enabled}}<a class="oauth-btn" href="/auth/oauth/aws">Sign in with AWS</a>{{/if}}
</div>
</body></html>
```

```html
<!-- templates/studio/auth/signup.html -->
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>KVForge Studio — Complete Signup</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#e0e0e0;font-family:sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh}
.card{background:#1e1e1e;border:1px solid #2a2a2a;border-radius:10px;padding:32px;width:340px}
h1{font-size:16px;color:#4ec9b0;margin-bottom:4px}
.sub{font-size:12px;color:#555;margin-bottom:20px}
.field-label{font-size:11px;color:#777;margin-bottom:4px}
input{width:100%;background:#252525;border:1px solid #333;border-radius:6px;padding:8px 12px;font-size:13px;color:#ccc;margin-bottom:10px}
input[readonly]{color:#555}
button{width:100%;background:#4ec9b0;border:none;border-radius:6px;padding:9px;font-size:13px;font-weight:700;color:#111;cursor:pointer}
</style></head><body>
<div class="card">
  <h1>Complete your account</h1>
  <div class="sub">You were invited as <strong>{{role}}</strong></div>
  <form method="post" action="/auth/signup">
    <input type="hidden" name="token" value="{{token}}">
    <div class="field-label">Email</div>
    <input name="email" type="email" value="{{email}}" readonly>
    <div class="field-label">Set Password</div>
    <input name="password" type="password" placeholder="Min 8 characters" required>
    <button type="submit">Create Account</button>
  </form>
</div>
</body></html>
```

```html
<!-- templates/studio/auth/admin_users.html -->
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>KVForge Studio — User Management</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#e0e0e0;font-family:sans-serif;padding:32px}
h1{color:#4ec9b0;font-size:18px;margin-bottom:24px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:#555;font-size:11px;text-transform:uppercase;letter-spacing:.06em;padding:8px;border-bottom:1px solid #2a2a2a}
td{padding:8px;border-bottom:1px solid #1e1e1e;color:#ccc}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px}
.admin{background:rgba(206,145,120,.15);color:#ce9178}
.editor{background:rgba(78,201,176,.12);color:#4ec9b0}
.viewer{background:rgba(156,220,254,.1);color:#9cdcfe}
.invite-btn{background:rgba(78,201,176,.12);border:1px solid rgba(78,201,176,.3);color:#4ec9b0;border-radius:6px;padding:6px 14px;font-size:12px;font-weight:700;cursor:pointer;margin-bottom:16px}
</style></head><body>
<h1>User Management</h1>
<button class="invite-btn" onclick="inviteUser()">+ Invite User</button>
<table>
<thead><tr><th>Email</th><th>Role</th><th>Provider</th><th>Joined</th></tr></thead>
<tbody id="users-tbody"></tbody>
</table>
<script>
async function load(){
  const r=await fetch('/studio/api/users');
  const users=await r.json();
  document.getElementById('users-tbody').innerHTML=users.map(u=>`
    <tr><td>${u.email}</td>
    <td><span class="badge ${u.role}">${u.role}</span></td>
    <td>${u.provider}</td><td>${u.created_at||''}</td></tr>`).join('');
}
async function inviteUser(){
  const email=prompt('Email to invite:');if(!email)return;
  const role=prompt('Role (admin/editor/viewer):','viewer');
  const r=await fetch('/auth/invite',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,role})});
  const d=await r.json();
  if(d.signup_url)alert('Invite link: '+d.signup_url);
}
load();
</script>
</body></html>
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_auth_routes.py
import os, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import db.store as store
from auth.middleware import AuthMiddleware

def _make_client(tmp_path):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    from auth.routes import router
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(router)
    @app.get("/studio/api/me-check")
    def me_check(request: Request):
        return {"role": request.state.user.role}
    return TestClient(app, follow_redirects=False)

def test_login_valid(tmp_path):
    import bcrypt, uuid
    store.DB_PATH = tmp_path / "x.db"; store._local.__dict__.clear(); store.migrate()
    uid = str(uuid.uuid4())
    h = bcrypt.hashpw(b"secret", bcrypt.gensalt(rounds=4)).decode()
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid,"u@test.com",h,"admin","local"))
    store.commit()
    client = _make_client(tmp_path)
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
    store.DB_PATH = tmp_path / "y.db"; store._local.__dict__.clear(); store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"admin@x.com","admin","local"))
    exp = (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()
    store.execute("INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
                  ("tok123","new@x.com","editor",admin_id,exp))
    store.commit()
    client = _make_client(tmp_path)
    r = client.post("/auth/signup", data={"token":"tok123","password":"newpass123"})
    assert r.status_code == 302
    row = store.fetchone("SELECT * FROM users WHERE email=?", ("new@x.com",))
    assert row is not None
    assert row["role"] == "editor"
    used = store.fetchone("SELECT used_at FROM invite_tokens WHERE token=?", ("tok123",))
    assert used["used_at"] is not None

def test_invite_token_single_use(tmp_path):
    import uuid; from datetime import datetime, timedelta, timezone
    store.DB_PATH = tmp_path / "z.db"; store._local.__dict__.clear(); store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"admin@x.com","admin","local"))
    exp = (datetime.now(timezone.utc)+timedelta(hours=24)).isoformat()
    store.execute("INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
                  ("tok456","b@x.com","viewer",admin_id,exp))
    store.commit()
    client = _make_client(tmp_path)
    client.post("/auth/signup", data={"token":"tok456","password":"pw123456"})
    r2 = client.post("/auth/signup", data={"token":"tok456","password":"pw999999"})
    assert r2.status_code in (400, 422, 302)
    count = store.fetchone("SELECT COUNT(*) as n FROM users WHERE email=?",("b@x.com",))
    assert count["n"] == 1
```

- [ ] **Step 3: Run to verify it fails**

```bash
python -m pytest tests/test_auth_routes.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'auth.routes'`

- [ ] **Step 4: Create auth/routes.py**

```python
# auth/routes.py
import os, uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import bcrypt
import jwt as pyjwt
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import db.store as store
from auth.models import User

SECRET = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")
TEMPLATES = Path(__file__).resolve().parent.parent / "templates" / "studio" / "auth"
router = APIRouter(prefix="/auth", tags=["auth"])


def _make_jwt(uid: str, role: str) -> tuple[str, datetime]:
    exp = datetime.now(timezone.utc) + timedelta(days=7)
    tok = pyjwt.encode({"sub": uid, "role": role, "exp": exp}, SECRET, algorithm="HS256")
    return tok, exp


def _create_session(user: User) -> tuple[str, datetime]:
    tok, exp = _make_jwt(user.id, user.role)
    store.execute(
        "INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
        (str(uuid.uuid4()), user.id, tok, exp.isoformat())
    )
    store.commit()
    return tok, exp


def _count_users() -> int:
    row = store.fetchone("SELECT COUNT(*) as n FROM users")
    return row["n"] if row else 0


def _render(name: str, **ctx) -> str:
    html = (TEMPLATES / name).read_text()
    for k, v in ctx.items():
        html = html.replace("{{" + k + "}}", str(v) if v else "")
    return html


@router.get("/login", response_class=HTMLResponse)
async def login_page(error: str = ""):
    env = os.environ
    return HTMLResponse(_render("login.html",
        error=f'<div class="err">{error}</div>' if error else "",
        google_enabled="" if env.get("GOOGLE_CLIENT_ID") else "<!--",
        microsoft_enabled="" if env.get("MICROSOFT_CLIENT_ID") else "<!--",
        aws_enabled="" if env.get("AWS_COGNITO_CLIENT_ID") else "<!--",
    ))


@router.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    row = store.fetchone("SELECT * FROM users WHERE email=? AND provider='local'", (email,))
    if not row or not bcrypt.checkpw(password.encode(), row["hashed_pw"].encode()):
        return RedirectResponse("/auth/login?error=Invalid+credentials", status_code=302)
    user = User.from_row(row)
    tok, exp = _create_session(user)
    next_url = request.query_params.get("next", "/studio/")
    resp = RedirectResponse(next_url, status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp


@router.get("/logout")
async def logout(request: Request):
    tok = request.cookies.get("kvforge_session")
    if tok:
        store.execute("DELETE FROM sessions WHERE jwt_token=?", (tok,))
        store.commit()
    resp = RedirectResponse("/auth/login", status_code=302)
    resp.delete_cookie("kvforge_session")
    return resp


@router.get("/me")
async def me(request: Request):
    u = getattr(request.state, "user", None)
    if not u:
        return JSONResponse({"detail": "not authenticated"}, status_code=401)
    return {"id": u.id, "email": u.email, "role": u.role, "provider": u.provider}


@router.post("/invite")
async def invite(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    body = await request.json()
    email, role = body["email"], body.get("role", "viewer")
    tok = str(uuid.uuid4())
    exp = datetime.now(timezone.utc) + timedelta(hours=48)
    store.execute(
        "INSERT INTO invite_tokens(token,email,role,created_by,expires_at) VALUES(?,?,?,?,?)",
        (tok, email, role, u.id, exp.isoformat())
    )
    store.commit()
    base = str(request.base_url).rstrip("/")
    return {"signup_url": f"{base}/auth/signup?token={tok}", "expires_at": exp.isoformat()}


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(token: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    inv = store.fetchone(
        "SELECT * FROM invite_tokens WHERE token=? AND used_at IS NULL AND expires_at > ?",
        (token, now)
    ) if token else None
    if not inv:
        return HTMLResponse("<h1 style='color:#ce9178;font-family:sans-serif;padding:40px'>Invalid or expired invite link</h1>", status_code=400)
    return HTMLResponse(_render("signup.html", email=inv["email"], role=inv["role"], token=token))


@router.post("/signup")
async def signup(token: str = Form(...), password: str = Form(...)):
    now = datetime.now(timezone.utc).isoformat()
    inv = store.fetchone(
        "SELECT * FROM invite_tokens WHERE token=? AND used_at IS NULL AND expires_at > ?",
        (token, now)
    )
    if not inv:
        return JSONResponse({"detail": "invalid or expired token"}, status_code=400)
    role = "admin" if _count_users() == 0 else inv["role"]
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
    uid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO users(id,email,hashed_pw,role,provider,invited_by) VALUES(?,?,?,?,?,?)",
        (uid, inv["email"], hashed, role, "local", inv["created_by"])
    )
    store.execute("UPDATE invite_tokens SET used_at=? WHERE token=?", (now, token))
    store.commit()
    user_row = store.fetchone("SELECT * FROM users WHERE id=?", (uid,))
    tok, _ = _create_session(User.from_row(user_row))
    resp = RedirectResponse("/studio/", status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_auth_routes.py -v --override-ini="addopts="
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add auth/routes.py templates/studio/auth/ tests/test_auth_routes.py
git commit -m "feat: add auth routes — login, logout, invite, signup with JWT cookies"
```

---

### Task 5: auth/oauth.py + auth/saml.py

**Files:**
- Create: `auth/oauth.py`
- Create: `auth/saml.py`

- [ ] **Step 1: Create auth/oauth.py**

```python
# auth/oauth.py
"""OAuth2 / OIDC integration. Each provider is only active if its env vars are set."""
import os, uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
import db.store as store
from auth.models import User
from auth.routes import _create_session

router = APIRouter(prefix="/auth/oauth", tags=["oauth"])

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
MS_CLIENT_ID = os.environ.get("MICROSOFT_CLIENT_ID", "")
MS_CLIENT_SECRET = os.environ.get("MICROSOFT_CLIENT_SECRET", "")
MS_TENANT_ID = os.environ.get("MICROSOFT_TENANT_ID", "common")
AWS_POOL_ID = os.environ.get("AWS_COGNITO_POOL_ID", "")
AWS_CLIENT_ID = os.environ.get("AWS_COGNITO_CLIENT_ID", "")
AWS_REGION = os.environ.get("AWS_COGNITO_REGION", "us-east-1")


def _upsert_oauth_user(email: str, provider: str, provider_id: str) -> User:
    """Find or create a user for an OAuth login. First-ever user gets admin."""
    from auth.routes import _count_users
    row = store.fetchone("SELECT * FROM users WHERE provider=? AND provider_id=?",
                         (provider, provider_id))
    if row:
        return User.from_row(row)
    # Also check by email (invite may have pre-created them)
    row = store.fetchone("SELECT * FROM users WHERE email=?", (email,))
    if row:
        store.execute("UPDATE users SET provider=?, provider_id=? WHERE id=?",
                      (provider, provider_id, row["id"]))
        store.commit()
        return User.from_row(store.fetchone("SELECT * FROM users WHERE id=?", (row["id"],)))
    role = "admin" if _count_users() == 0 else "viewer"
    uid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO users(id,email,role,provider,provider_id) VALUES(?,?,?,?,?)",
        (uid, email, role, provider, provider_id)
    )
    store.commit()
    return User.from_row(store.fetchone("SELECT * FROM users WHERE id=?", (uid,)))


def _oauth_redirect(request: Request, user: User) -> RedirectResponse:
    tok, _ = _create_session(user)
    resp = RedirectResponse("/studio/", status_code=302)
    resp.set_cookie("kvforge_session", tok, httponly=True, secure=False, samesite="lax")
    return resp


# ── Google ────────────────────────────────────────────────────────────────────

@router.get("/google")
async def google_login(request: Request):
    if not GOOGLE_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Google+OAuth+not+configured")
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register("google",
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", name="google_callback")
async def google_callback(request: Request):
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register("google",
        client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo") or await oauth.google.userinfo(token=token)
    user = _upsert_oauth_user(userinfo["email"], "google", userinfo["sub"])
    return _oauth_redirect(request, user)


# ── Microsoft ─────────────────────────────────────────────────────────────────

@router.get("/microsoft")
async def microsoft_login(request: Request):
    if not MS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=Microsoft+OAuth+not+configured")
    import msal
    app = msal.ConfidentialClientApplication(
        MS_CLIENT_ID, authority=f"https://login.microsoftonline.com/{MS_TENANT_ID}",
        client_credential=MS_CLIENT_SECRET,
    )
    redirect_uri = str(request.url_for("microsoft_callback"))
    flow = app.initiate_auth_code_flow(["openid", "email", "profile"], redirect_uri=redirect_uri)
    request.session["msal_flow"] = flow
    return RedirectResponse(flow["auth_uri"])


@router.get("/microsoft/callback", name="microsoft_callback")
async def microsoft_callback(request: Request):
    import msal
    flow = request.session.pop("msal_flow", {})
    msal_app = msal.ConfidentialClientApplication(
        MS_CLIENT_ID, authority=f"https://login.microsoftonline.com/{MS_TENANT_ID}",
        client_credential=MS_CLIENT_SECRET,
    )
    result = msal_app.acquire_token_by_auth_code_flow(flow, dict(request.query_params))
    if "error" in result:
        return RedirectResponse(f"/auth/login?error={result['error']}")
    claims = result.get("id_token_claims", {})
    user = _upsert_oauth_user(claims.get("email",""), "microsoft", claims.get("oid",""))
    return _oauth_redirect(request, user)


# ── AWS Cognito ───────────────────────────────────────────────────────────────

@router.get("/aws")
async def aws_login(request: Request):
    if not AWS_CLIENT_ID:
        return RedirectResponse("/auth/login?error=AWS+OAuth+not+configured")
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    domain = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{AWS_POOL_ID}"
    oauth.register("aws",
        client_id=AWS_CLIENT_ID,
        server_metadata_url=f"{domain}/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    redirect_uri = str(request.url_for("aws_callback"))
    return await oauth.aws.authorize_redirect(request, redirect_uri)


@router.get("/aws/callback", name="aws_callback")
async def aws_callback(request: Request):
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    domain = f"https://cognito-idp.{AWS_REGION}.amazonaws.com/{AWS_POOL_ID}"
    oauth.register("aws",
        client_id=AWS_CLIENT_ID,
        server_metadata_url=f"{domain}/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    token = await oauth.aws.authorize_access_token(request)
    userinfo = token.get("userinfo") or await oauth.aws.userinfo(token=token)
    user = _upsert_oauth_user(userinfo["email"], "aws", userinfo["sub"])
    return _oauth_redirect(request, user)
```

- [ ] **Step 2: Create auth/saml.py (stub)**

```python
# auth/saml.py
"""SAML 2.0 stub — defined now, implemented later.

Role mapping: SAML group attribute → Admin / Editor / Viewer.
No invite token required for SAML users — assertion is the invite.
"""
from __future__ import annotations
from fastapi import Request
from auth.models import User


class SAMLProvider:
    """Stub — raise NotImplementedError until SAML is implemented."""

    def handle_callback(self, request: Request) -> User:
        raise NotImplementedError("SAML not yet implemented")
```

- [ ] **Step 3: Add /studio/api/users endpoint to auth/routes.py**

Append to `auth/routes.py`:

```python
@router.get("/studio-users", tags=["admin"])
async def list_users(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    rows = store.fetchall("SELECT id,email,role,provider,created_at FROM users ORDER BY created_at")
    return [dict(r) for r in rows]


@router.put("/studio-users/{uid}/role", tags=["admin"])
async def change_role(uid: str, request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role != "admin":
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    body = await request.json()
    new_role = body.get("role")
    if new_role not in ("admin", "editor", "viewer"):
        return JSONResponse({"detail": "invalid role"}, status_code=422)
    store.execute("UPDATE users SET role=? WHERE id=?", (new_role, uid))
    store.commit()
    return {"ok": True}
```

- [ ] **Step 4: Commit**

```bash
git add auth/oauth.py auth/saml.py auth/routes.py
git commit -m "feat: add OAuth clients (Google/Microsoft/AWS Cognito), SAML stub, user management endpoints"
```

---

### Task 6: Wire into kvforge_portal.py

**Files:**
- Modify: `kvforge_portal.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_portal_auth_wire.py
import os
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

def test_portal_has_auth_router(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    from fastapi.testclient import TestClient
    import kvforge_portal
    kvforge_portal.app.state.db_path = tmp_path / "test.db"
    client = TestClient(kvforge_portal.app, raise_server_exceptions=False)
    r = client.get("/auth/login")
    assert r.status_code == 200
    assert b"KVForge Studio" in r.content

def test_unauthenticated_studio_redirects_to_login(tmp_path):
    import db.store as store
    store.DB_PATH = tmp_path / "p.db"
    store._local.__dict__.clear()
    from fastapi.testclient import TestClient
    import kvforge_portal
    client = TestClient(kvforge_portal.app, follow_redirects=False, raise_server_exceptions=False)
    r = client.get("/studio/")
    assert r.status_code == 302
    assert "/auth/login" in r.headers.get("location","")
```

- [ ] **Step 2: Modify kvforge_portal.py — add lifespan DB migration, auth router, and middleware**

Find the lifespan function in `kvforge_portal.py` and replace it, then add imports and router/middleware:

```python
# Add after the existing imports block:
from auth.middleware import AuthMiddleware
from auth.routes import router as _auth_router
from auth.oauth import router as _oauth_router

# Replace the existing _lifespan with:
@asynccontextmanager
async def _lifespan(app: FastAPI):
    import db.store as store
    store.migrate()
    yield

# After app = FastAPI(...), add middleware and routers:
app.add_middleware(AuthMiddleware)
app.include_router(_auth_router)
app.include_router(_oauth_router)
# Also expose /studio/api/users pointing to auth list:
```

Full diff to apply:

```python
# In kvforge_portal.py, replace:
#   from fastapi import FastAPI, HTTPException
# with:
from fastapi import FastAPI, HTTPException
from auth.middleware import AuthMiddleware
from auth.routes import router as _auth_router
from auth.oauth import router as _oauth_router

# Replace _lifespan:
@asynccontextmanager
async def _lifespan(app: FastAPI):
    import db.store as store
    store.migrate()
    yield

# After app = FastAPI(...):
app.add_middleware(AuthMiddleware)
app.include_router(_auth_router)
app.include_router(_oauth_router)
```

Apply these changes to `kvforge_portal.py` by editing the file directly.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_portal_auth_wire.py -v --override-ini="addopts="
```
Expected: `2 passed`

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
python -m pytest tests/ -v --override-ini="addopts=" -x --ignore=tests/ui
```
Expected: no new failures

- [ ] **Step 5: Commit**

```bash
git add kvforge_portal.py tests/test_portal_auth_wire.py
git commit -m "feat: wire auth into kvforge_portal — AuthMiddleware, DB migration on startup, auth routes mounted"
```

---

## Verification

```bash
python kvforge_portal.py --port 8080
# open http://localhost:8080/auth/login  → sign-in page loads
# open http://localhost:8080/studio/     → redirects to /auth/login
```

**Run all Phase 1 tests:**
```bash
python -m pytest tests/test_db_store.py tests/test_auth_models.py tests/test_auth_middleware.py tests/test_auth_routes.py tests/test_portal_auth_wire.py -v --override-ini="addopts="
```
Expected: all pass.

**Proceed to:** `docs/superpowers/plans/2026-05-10-dashboard-phase2-connectors.md`
