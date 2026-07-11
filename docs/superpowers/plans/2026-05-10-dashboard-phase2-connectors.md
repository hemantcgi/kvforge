# Dashboard Phase 2 — Connector Registry & Management UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `connectors/registry.py` (CRUD for connector configs with Fernet encryption), `connectors/routes.py` (REST API), and the `/studio/connectors` dashboard page.

**Architecture:** Registry wraps `db/store.py` with Fernet encryption/decryption for credentials. Routes enforce role checks. Template renders a live connector list with progress and history.

**Tech Stack:** FastAPI, cryptography (Fernet), sqlite3, Jinja2-free HTML templates

**Spec:** `docs/superpowers/specs/2026-05-10-dashboard-auth-connectors-design.md` §2

**Depends on:** Phase 1 (db/ + auth/ packages)

**Next plan:** `2026-05-10-dashboard-phase3-sync.md`

---

### Task 1: connectors/registry.py

**Files:**
- Create: `connectors/registry.py`
- Create: `tests/test_connector_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_connector_registry.py
import os, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import db.store as store
from connectors.registry import ConnectorRegistry

def _setup(tmp_path):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    return ConnectorRegistry()

def test_create_and_list(tmp_path):
    reg = _setup(tmp_path)
    cfg = reg.create(
        connector_type="gdrive",
        name="My Drive",
        credentials={"service_account_json": "fake-json"},
        created_by="user1",
    )
    assert cfg["id"]
    assert cfg["type"] == "gdrive"
    assert cfg["name"] == "My Drive"
    assert "credentials" not in cfg  # never returned in full

    rows = reg.list_all()
    assert len(rows) == 1
    assert rows[0]["name"] == "My Drive"
    assert rows[0]["credentials"] == "●●●●●●"  # masked

def test_credentials_encrypted_at_rest(tmp_path):
    reg = _setup(tmp_path)
    reg.create("s3", "My S3", {"access_key": "AKIAIOSFODNN7EXAMPLE"}, "user1")
    raw = store.fetchone("SELECT credentials_json FROM connector_configs")
    assert raw is not None
    assert "AKIAIOSFODNN7EXAMPLE" not in raw["credentials_json"]

def test_get_decrypted_credentials(tmp_path):
    reg = _setup(tmp_path)
    cfg = reg.create("s3", "S3 Test", {"bucket": "my-bucket", "prefix": "docs/"}, "user1")
    creds = reg.get_credentials(cfg["id"])
    assert creds["bucket"] == "my-bucket"
    assert creds["prefix"] == "docs/"

def test_update_schedule(tmp_path):
    reg = _setup(tmp_path)
    cfg = reg.create("sharepoint", "SP", {"tenant": "t"}, "user1")
    reg.update(cfg["id"], schedule_cron="*/30 * * * *")
    row = store.fetchone("SELECT schedule_cron FROM connector_configs WHERE id=?", (cfg["id"],))
    assert row["schedule_cron"] == "*/30 * * * *"

def test_delete_connector(tmp_path):
    reg = _setup(tmp_path)
    cfg = reg.create("gdrive", "GD", {"k": "v"}, "user1")
    reg.delete(cfg["id"])
    assert store.fetchone("SELECT id FROM connector_configs WHERE id=?", (cfg["id"],)) is None

def test_add_and_list_scope(tmp_path):
    reg = _setup(tmp_path)
    cfg = reg.create("s3", "S3", {"bucket": "b"}, "user1")
    reg.upsert_scope(cfg["id"], "uc1", {"bucket": "b", "prefix": "docs/"})
    scopes = reg.list_scopes(cfg["id"])
    assert len(scopes) == 1
    assert scopes[0]["uc_id"] == "uc1"
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_connector_registry.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'connectors.registry'`

- [ ] **Step 3: Create connectors/registry.py**

```python
# connectors/registry.py
import json, os, uuid
from base64 import urlsafe_b64encode
from pathlib import Path
from cryptography.fernet import Fernet
import db.store as store

_MASK = "●●●●●●"


def _fernet() -> Fernet:
    raw = os.environ.get("KVFORGE_SECRET_KEY", "dev-secret-change-me")
    # Fernet needs exactly 32 url-safe base64 bytes; pad/truncate as needed
    key_bytes = raw.encode()[:32].ljust(32, b"0")
    return Fernet(urlsafe_b64encode(key_bytes))


def _encrypt(data: dict) -> str:
    return _fernet().encrypt(json.dumps(data).encode()).decode()


def _decrypt(token: str) -> dict:
    return json.loads(_fernet().decrypt(token.encode()))


class ConnectorRegistry:

    def create(self, connector_type: str, name: str,
               credentials: dict, created_by: str,
               schedule_cron: str | None = None,
               webhook_secret: str | None = None) -> dict:
        cid = str(uuid.uuid4())
        enc = _encrypt(credentials)
        store.execute(
            "INSERT INTO connector_configs(id,type,name,credentials_json,"
            "schedule_cron,webhook_secret,created_by) VALUES(?,?,?,?,?,?,?)",
            (cid, connector_type, name, enc, schedule_cron, webhook_secret, created_by)
        )
        store.commit()
        return self._safe_row(store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,)))

    def list_all(self) -> list[dict]:
        rows = store.fetchall("SELECT * FROM connector_configs ORDER BY created_at DESC")
        return [self._safe_row(r) for r in rows]

    def get(self, cid: str) -> dict | None:
        row = store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,))
        return self._safe_row(row) if row else None

    def get_credentials(self, cid: str) -> dict:
        row = store.fetchone("SELECT credentials_json FROM connector_configs WHERE id=?", (cid,))
        if not row:
            raise KeyError(f"connector {cid} not found")
        return _decrypt(row["credentials_json"])

    def update(self, cid: str,
               credentials: dict | None = None,
               schedule_cron: str | None = ...,
               webhook_secret: str | None = ...,
               name: str | None = None) -> dict:
        row = store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,))
        if not row:
            raise KeyError(f"connector {cid} not found")
        new_enc = _encrypt(credentials) if credentials else row["credentials_json"]
        new_cron = row["schedule_cron"] if schedule_cron is ... else schedule_cron
        new_ws = row["webhook_secret"] if webhook_secret is ... else webhook_secret
        new_name = name or row["name"]
        store.execute(
            "UPDATE connector_configs SET name=?,credentials_json=?,schedule_cron=?,webhook_secret=? WHERE id=?",
            (new_name, new_enc, new_cron, new_ws, cid)
        )
        store.commit()
        return self._safe_row(store.fetchone("SELECT * FROM connector_configs WHERE id=?", (cid,)))

    def delete(self, cid: str) -> None:
        store.execute("DELETE FROM connector_configs WHERE id=?", (cid,))
        store.commit()

    def upsert_scope(self, connector_id: str, uc_id: str, scope_config: dict) -> None:
        enc = json.dumps(scope_config)
        store.execute(
            "INSERT OR REPLACE INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
            (connector_id, uc_id, enc)
        )
        store.commit()

    def list_scopes(self, connector_id: str) -> list[dict]:
        rows = store.fetchall("SELECT * FROM connector_uc_scopes WHERE connector_id=?", (connector_id,))
        return [{"connector_id": r["connector_id"], "uc_id": r["uc_id"],
                 "scope_config": json.loads(r["scope_config_json"]),
                 "last_sync_at": r["last_sync_at"],
                 "last_delta_token": r["last_delta_token"]} for r in rows]

    def delete_scope(self, connector_id: str, uc_id: str) -> None:
        store.execute("DELETE FROM connector_uc_scopes WHERE connector_id=? AND uc_id=?",
                      (connector_id, uc_id))
        store.commit()

    @staticmethod
    def _safe_row(row) -> dict:
        d = dict(row)
        d["credentials"] = _MASK
        d.pop("credentials_json", None)
        return d
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_connector_registry.py -v --override-ini="addopts="
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add connectors/registry.py tests/test_connector_registry.py
git commit -m "feat: add ConnectorRegistry — Fernet-encrypted credentials, CRUD + scope management"
```

---

### Task 2: connectors/routes.py (REST API)

**Files:**
- Create: `connectors/routes.py`
- Create: `tests/test_connector_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_connector_routes.py
import os, uuid, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import db.store as store
from auth.middleware import AuthMiddleware
from connectors.routes import connector_router

def _make_client(tmp_path, role="admin"):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    # Create user + session
    import bcrypt, jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    uid = str(uuid.uuid4())
    h = bcrypt.hashpw(b"pw", bcrypt.gensalt(rounds=4)).decode()
    store.execute("INSERT INTO users(id,email,hashed_pw,role,provider) VALUES(?,?,?,?,?)",
                  (uid,"admin@x.com",h,role,"local"))
    exp = datetime.now(timezone.utc) + timedelta(hours=1)
    tok = pyjwt.encode({"sub":uid,"role":role,"exp":exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()),uid,tok,exp.isoformat()))
    store.commit()
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(connector_router)
    client = TestClient(app, raise_server_exceptions=False)
    return client, tok

def test_create_connector(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type":"gdrive","name":"My GDrive","credentials":{"key":"val"}},
        cookies={"kvforge_session":tok})
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "My GDrive"
    assert d["credentials"] == "●●●●●●"

def test_list_connectors(tmp_path):
    client, tok = _make_client(tmp_path)
    client.post("/studio/api/connectors",
        json={"type":"s3","name":"My S3","credentials":{"bucket":"b"}},
        cookies={"kvforge_session":tok})
    r = client.get("/studio/api/connectors", cookies={"kvforge_session":tok})
    assert r.status_code == 200
    assert len(r.json()) == 1

def test_viewer_cannot_create(tmp_path):
    client, tok = _make_client(tmp_path, role="viewer")
    r = client.post("/studio/api/connectors",
        json={"type":"s3","name":"X","credentials":{}},
        cookies={"kvforge_session":tok})
    assert r.status_code == 403

def test_add_scope(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type":"gdrive","name":"GD","credentials":{"k":"v"}},
        cookies={"kvforge_session":tok})
    cid = r.json()["id"]
    r2 = client.post(f"/studio/api/connectors/{cid}/scopes",
        json={"uc_id":"uc1","scope_config":{"folder_id":"abc"}},
        cookies={"kvforge_session":tok})
    assert r2.status_code == 200
    r3 = client.get(f"/studio/api/connectors/{cid}/scopes", cookies={"kvforge_session":tok})
    assert len(r3.json()) == 1

def test_delete_connector(tmp_path):
    client, tok = _make_client(tmp_path)
    r = client.post("/studio/api/connectors",
        json={"type":"s3","name":"S3","credentials":{"k":"v"}},
        cookies={"kvforge_session":tok})
    cid = r.json()["id"]
    r2 = client.delete(f"/studio/api/connectors/{cid}", cookies={"kvforge_session":tok})
    assert r2.status_code == 200
    assert client.get("/studio/api/connectors", cookies={"kvforge_session":tok}).json() == []
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_connector_routes.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'connectors.routes'`

- [ ] **Step 3: Create connectors/routes.py**

```python
# connectors/routes.py
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from connectors.registry import ConnectorRegistry

connector_router = APIRouter(prefix="/studio/api/connectors", tags=["connectors"])
_registry = ConnectorRegistry()

ADMIN_ONLY = ("admin",)
EDITOR_UP = ("admin", "editor")
ANY_AUTH = ("admin", "editor", "viewer")


def _require_role(request: Request, roles: tuple) -> JSONResponse | None:
    u = getattr(request.state, "user", None)
    if not u or u.role not in roles:
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return None


@connector_router.get("")
async def list_connectors(request: Request):
    if err := _require_role(request, ANY_AUTH): return err
    return _registry.list_all()


@connector_router.post("")
async def create_connector(request: Request):
    if err := _require_role(request, ADMIN_ONLY): return err
    body = await request.json()
    cfg = _registry.create(
        connector_type=body["type"],
        name=body["name"],
        credentials=body.get("credentials", {}),
        created_by=request.state.user.id,
        schedule_cron=body.get("schedule_cron"),
        webhook_secret=body.get("webhook_secret"),
    )
    return cfg


@connector_router.put("/{cid}")
async def update_connector(cid: str, request: Request):
    if err := _require_role(request, ADMIN_ONLY): return err
    body = await request.json()
    try:
        return _registry.update(cid,
            credentials=body.get("credentials"),
            schedule_cron=body.get("schedule_cron", ...),
            name=body.get("name"),
        )
    except KeyError:
        return JSONResponse({"detail": "not found"}, status_code=404)


@connector_router.delete("/{cid}")
async def delete_connector(cid: str, request: Request):
    if err := _require_role(request, ADMIN_ONLY): return err
    try:
        _registry.delete(cid)
        return {"ok": True}
    except KeyError:
        return JSONResponse({"detail": "not found"}, status_code=404)


@connector_router.post("/{cid}/test")
async def test_connector(cid: str, request: Request):
    if err := _require_role(request, ADMIN_ONLY): return err
    try:
        creds = _registry.get_credentials(cid)
        cfg = _registry.get(cid)
        result = await asyncio.wait_for(_run_test(cfg["type"], creds), timeout=10.0)
        return result
    except asyncio.TimeoutError:
        return JSONResponse({"ok": False, "error": "timeout after 10s"}, status_code=200)
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _run_test(connector_type: str, creds: dict) -> dict:
    if connector_type == "gdrive":
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        import json
        info = json.loads(creds.get("service_account_json", "{}"))
        sa_creds = Credentials.from_service_account_info(info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"])
        svc = build("drive", "v3", credentials=sa_creds, cache_discovery=False)
        files = svc.files().list(pageSize=1).execute()
        return {"ok": True, "detail": f"Connected — {len(files.get('files',[]))} files visible"}

    elif connector_type == "s3":
        import boto3, botocore
        s3 = boto3.client("s3",
            aws_access_key_id=creds.get("access_key_id"),
            aws_secret_access_key=creds.get("secret_access_key"),
            region_name=creds.get("region", "us-east-1"),
        )
        s3.head_bucket(Bucket=creds.get("bucket", ""))
        return {"ok": True, "detail": "S3 bucket reachable"}

    elif connector_type == "sharepoint":
        import msal, httpx
        msal_app = msal.ConfidentialClientApplication(
            creds["client_id"],
            authority=f"https://login.microsoftonline.com/{creds['tenant_id']}",
            client_credential=creds["client_secret"],
        )
        result = msal_app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"])
        if "error" in result:
            return {"ok": False, "error": result.get("error_description", result["error"])}
        async with httpx.AsyncClient() as hc:
            r = await hc.get(
                f"https://graph.microsoft.com/v1.0/sites/{creds.get('site_url','')}",
                headers={"Authorization": f"Bearer {result['access_token']}"},
            )
        return {"ok": r.status_code == 200, "detail": f"HTTP {r.status_code}"}
    return {"ok": False, "error": f"unknown type {connector_type}"}


@connector_router.get("/{cid}/scopes")
async def list_scopes(cid: str, request: Request):
    if err := _require_role(request, ANY_AUTH): return err
    return _registry.list_scopes(cid)


@connector_router.post("/{cid}/scopes")
async def add_scope(cid: str, request: Request):
    if err := _require_role(request, EDITOR_UP): return err
    body = await request.json()
    _registry.upsert_scope(cid, body["uc_id"], body.get("scope_config", {}))
    return {"ok": True}


@connector_router.delete("/{cid}/scopes/{uc_id}")
async def delete_scope(cid: str, uc_id: str, request: Request):
    if err := _require_role(request, EDITOR_UP): return err
    _registry.delete_scope(cid, uc_id)
    return {"ok": True}
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_connector_routes.py -v --override-ini="addopts="
```
Expected: `5 passed`

- [ ] **Step 5: Add connector_router to kvforge_portal.py**

In `kvforge_portal.py`, add:
```python
from connectors.routes import connector_router
# after app.include_router(_oauth_router):
app.include_router(connector_router)
```

- [ ] **Step 6: Commit**

```bash
git add connectors/routes.py tests/test_connector_routes.py kvforge_portal.py
git commit -m "feat: add connector REST API — CRUD, scope management, test-connection endpoint"
```

---

### Task 3: /studio/connectors HTML page

**Files:**
- Create: `templates/studio/connectors.html`
- Modify: `studio/routes.py` (add page route)
- Create: `tests/test_connector_page.py`

- [ ] **Step 1: Create templates/studio/connectors.html**

```html
<!-- templates/studio/connectors.html -->
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>KVForge Studio — Connectors</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#e0e0e0;font-family:sans-serif;padding:24px}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}
h1{color:#4ec9b0;font-size:18px}
.btn-primary{background:rgba(78,201,176,.15);border:1px solid rgba(78,201,176,.3);color:#4ec9b0;border-radius:6px;padding:7px 14px;font-size:12px;font-weight:700;cursor:pointer}
.conn-list{display:grid;gap:10px;margin-bottom:24px}
.conn-row{display:flex;align-items:center;gap:14px;background:#1e1e1e;border:1px solid #2a2a2a;border-radius:8px;padding:12px 16px}
.conn-icon{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#fff;flex-shrink:0}
.conn-info{flex:1}
.conn-name{font-size:13px;font-weight:700;color:#e0e0e0}
.conn-meta{font-size:11px;color:#555;margin-top:2px}
.status-dot{width:7px;height:7px;border-radius:50%;background:#444}
.status-ok{background:#4ec9b0;box-shadow:0 0 5px rgba(78,201,176,.5)}
.status-err{background:#ce9178}
.status-text{font-size:11px;color:#777}
.conn-actions{display:flex;gap:6px;margin-left:10px}
.btn-sm{font-size:11px;font-weight:600;padding:4px 10px;border-radius:5px;cursor:pointer;border:1px solid #333;background:transparent;color:#888}
.btn-sm:hover{border-color:#4ec9b0;color:#4ec9b0}
.btn-err{border-color:#ce9178;color:#ce9178}
.btn-add{background:rgba(78,201,176,.05);border:1.5px dashed rgba(78,201,176,.3);color:#4ec9b0;border-radius:8px;padding:12px;text-align:center;font-size:12px;font-weight:600;cursor:pointer;width:100%}
.section-title{font-size:11px;font-weight:700;color:#555;letter-spacing:.07em;text-transform:uppercase;margin-bottom:10px}
.history-table{width:100%;border-collapse:collapse;font-size:12px}
.history-table th{text-align:left;font-size:10px;color:#444;text-transform:uppercase;letter-spacing:.06em;padding:6px 8px;border-bottom:1px solid #252525}
.history-table td{padding:7px 8px;border-bottom:1px solid #1e1e1e;color:#999}
.run-ok{color:#4ec9b0;font-weight:700}
.run-err{color:#ce9178;font-weight:700}
.run-running{color:#9cdcfe;font-weight:700}
.tag{display:inline-block;font-size:10px;font-weight:600;padding:2px 7px;border-radius:4px}
.tag-manual{background:rgba(156,220,254,.1);color:#9cdcfe}
.tag-scheduled{background:rgba(78,201,176,.1);color:#4ec9b0}
.tag-webhook{background:rgba(206,145,120,.1);color:#ce9178}
</style>
</head><body>
<div class="header">
  <h1>Connectors</h1>
  <button class="btn-primary" onclick="showAddForm()">+ Add Connector</button>
</div>

<div id="conn-list" class="conn-list"></div>

<div style="margin-top:24px">
  <div class="section-title">Sync Run History</div>
  <table class="history-table">
    <thead><tr><th>Connector</th><th>Trigger</th><th>Result</th><th>Files</th><th>Duration</th></tr></thead>
    <tbody id="history-tbody"></tbody>
  </table>
</div>

<script>
const ICONS = {gdrive:{abbr:'GD',bg:'#0F9D58'}, s3:{abbr:'S3',bg:'#FF9900'}, sharepoint:{abbr:'SP',bg:'#0078D4'}};

async function load() {
  const r = await fetch('/studio/api/connectors', {credentials:'include'});
  const connectors = await r.json();
  const list = document.getElementById('conn-list');
  if (!connectors.length) { list.innerHTML='<div style="color:#555;font-size:13px">No connectors configured yet.</div>'; return; }
  list.innerHTML = connectors.map(c => {
    const ic = ICONS[c.type] || {abbr:c.type.toUpperCase().slice(0,2), bg:'#555'};
    const dot = c.last_error ? 'status-err' : 'status-ok';
    const statusText = c.last_error ? 'Error' : 'Healthy';
    return `<div class="conn-row" id="conn-${c.id}">
      <div class="conn-icon" style="background:${ic.bg}">${ic.abbr}</div>
      <div class="conn-info">
        <div class="conn-name">${c.name}</div>
        <div class="conn-meta">${c.type} · ${c.schedule_cron || 'manual only'}</div>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <div class="status-dot ${dot}"></div>
        <div class="status-text">${statusText}</div>
      </div>
      <div class="conn-actions">
        <button class="btn-sm" onclick="syncNow('${c.id}')">↺ Sync Now</button>
        <button class="btn-sm" onclick="deleteConn('${c.id}')">Delete</button>
      </div>
    </div>`;
  }).join('') + '<button class="btn-add" onclick="showAddForm()">+ Add Connector</button>';

  loadHistory();
}

async function loadHistory() {
  const r = await fetch('/studio/api/sync-runs', {credentials:'include'});
  if (!r.ok) return;
  const runs = await r.json();
  document.getElementById('history-tbody').innerHTML = runs.slice(0,50).map(run => {
    const cls = run.status==='ok'?'run-ok':run.status==='error'?'run-err':'run-running';
    const sym = run.status==='ok'?'✓ ok':run.status==='error'?'✗ error':'● running';
    const dur = run.finished_at
      ? ((new Date(run.finished_at)-new Date(run.started_at))/1000).toFixed(1)+'s'
      : ((Date.now()-new Date(run.started_at))/1000).toFixed(0)+'s…';
    return `<tr>
      <td>${run.connector_name||run.connector_id}</td>
      <td><span class="tag tag-${run.trigger}">${run.trigger}</span></td>
      <td><span class="${cls}">${sym}</span></td>
      <td>${run.files_done} new</td>
      <td style="font-family:monospace">${dur}</td>
    </tr>`;
  }).join('');
}

async function syncNow(cid) {
  const r = await fetch(`/studio/api/connectors/${cid}/sync`, {method:'POST', credentials:'include'});
  const d = await r.json();
  if (d.run_id) alert('Sync started: ' + d.run_id);
  setTimeout(load, 2000);
}

async function deleteConn(cid) {
  if (!confirm('Delete this connector and all its scopes?')) return;
  await fetch(`/studio/api/connectors/${cid}`, {method:'DELETE', credentials:'include'});
  load();
}

function showAddForm() {
  const t = prompt('Connector type (gdrive/s3/sharepoint):');
  if (!t) return;
  const n = prompt('Connector name:');
  if (!n) return;
  const creds = prompt('Credentials JSON (paste):');
  if (!creds) return;
  fetch('/studio/api/connectors', {
    method:'POST', credentials:'include',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({type:t, name:n, credentials: JSON.parse(creds)})
  }).then(()=>load());
}

load();
setInterval(loadHistory, 5000);
</script>
</body></html>
```

- [ ] **Step 2: Add /studio/connectors page route to studio/routes.py**

In `studio/routes.py`, add after existing page routes:

```python
@router.get("/connectors", response_class=HTMLResponse)
def connectors_page(request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role not in ("admin", "editor"):
        from fastapi.responses import Response
        return Response("<h1>403 Forbidden</h1>", status_code=403, media_type="text/html")
    return (TEMPLATES / "connectors.html").read_text()
```

- [ ] **Step 3: Add /studio/api/sync-runs endpoint to connectors/routes.py**

Append to `connectors/routes.py`:

```python
from fastapi import APIRouter as _AR
_sync_runs_router = APIRouter(prefix="/studio/api", tags=["sync-runs"])

@_sync_runs_router.get("/sync-runs")
async def list_sync_runs(request: Request):
    u = getattr(request.state, "user", None)
    if not u:
        return JSONResponse({"detail": "not authenticated"}, status_code=401)
    rows = store.fetchall(
        "SELECT sr.*, cc.name as connector_name FROM sync_runs sr "
        "LEFT JOIN connector_configs cc ON sr.connector_id=cc.id "
        "ORDER BY sr.started_at DESC LIMIT 50"
    )
    return [dict(r) for r in rows]
```

Then add `_sync_runs_router` to `kvforge_portal.py`:
```python
from connectors.routes import _sync_runs_router
app.include_router(_sync_runs_router)
```

- [ ] **Step 4: Write test**

```python
# tests/test_connector_page.py
import os
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

def test_connectors_page_requires_editor(tmp_path):
    import uuid, bcrypt, jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    import db.store as store
    store.DB_PATH = tmp_path / "test.db"; store._local.__dict__.clear(); store.migrate()
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (uid,"v@x.com","viewer","local"))
    exp = datetime.now(timezone.utc)+timedelta(hours=1)
    tok = pyjwt.encode({"sub":uid,"role":"viewer","exp":exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()),uid,tok,exp.isoformat()))
    store.commit()
    import kvforge_portal
    from fastapi.testclient import TestClient
    client = TestClient(kvforge_portal.app, raise_server_exceptions=False)
    r = client.get("/studio/connectors", cookies={"kvforge_session":tok})
    assert r.status_code == 403

def test_connectors_page_loads_for_admin(tmp_path):
    import uuid, bcrypt, jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    import db.store as store
    store.DB_PATH = tmp_path / "cp.db"; store._local.__dict__.clear(); store.migrate()
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (uid,"a@x.com","admin","local"))
    exp = datetime.now(timezone.utc)+timedelta(hours=1)
    tok = pyjwt.encode({"sub":uid,"role":"admin","exp":exp},
                       "test-secret-32bytesXXXXXXXXXXXX", algorithm="HS256")
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()),uid,tok,exp.isoformat()))
    store.commit()
    import kvforge_portal
    from fastapi.testclient import TestClient
    client = TestClient(kvforge_portal.app, raise_server_exceptions=False)
    r = client.get("/studio/connectors", cookies={"kvforge_session":tok})
    assert r.status_code == 200
    assert b"Connectors" in r.content
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_connector_page.py -v --override-ini="addopts="
```
Expected: `2 passed`

- [ ] **Step 6: Run full suite for regressions**

```bash
python -m pytest tests/ -v --override-ini="addopts=" -x --ignore=tests/ui
```

- [ ] **Step 7: Commit**

```bash
git add templates/studio/connectors.html studio/routes.py connectors/routes.py kvforge_portal.py tests/test_connector_page.py
git commit -m "feat: add /studio/connectors dashboard page — connector list, sync history, role-gated"
```

---

## Verification

```bash
python kvforge_portal.py --port 8080
# GET http://localhost:8080/studio/connectors → connector dashboard renders
# POST http://localhost:8080/studio/api/connectors (with admin cookie) → creates connector
# GET http://localhost:8080/studio/api/connectors → masked credentials
```

**Run all Phase 2 tests:**
```bash
python -m pytest tests/test_connector_registry.py tests/test_connector_routes.py tests/test_connector_page.py -v --override-ini="addopts="
```
Expected: all pass.

**Proceed to:** `docs/superpowers/plans/2026-05-10-dashboard-phase3-sync.md`
