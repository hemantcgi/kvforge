# Dashboard Phase 3 — Sync Engine, Scheduler & Webhooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `sync/progress.py` (SSE bus), `connectors/sync_engine.py` (file-sync orchestration), `sync/scheduler.py` (APScheduler cron jobs), and `sync/webhook.py` (HMAC-validated push receiver). Wire scheduler startup into portal lifespan.

**Architecture:** `SyncEngine.run()` always executes in a background thread (run_in_executor) and pushes events to an `asyncio.Queue` in `progress.py`. SSE endpoint drains that queue. Scheduler and webhook both call `SyncEngine.run()`. Every run writes a `sync_runs` record on start and updates on completion.

**Tech Stack:** APScheduler, asyncio, hmac, hashlib, FastAPI StreamingResponse

**Spec:** `docs/superpowers/specs/2026-05-10-dashboard-auth-connectors-design.md` §3

**Depends on:** Phase 1 (db/), Phase 2 (connectors/registry.py)

**Next plan:** `2026-05-10-dashboard-phase4-ui-tests.md`

---

### Task 1: sync/progress.py — SSE progress bus

**Files:**
- Create: `sync/__init__.py`
- Create: `sync/progress.py`
- Create: `tests/test_sync_progress.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_progress.py
import asyncio, pytest
from sync.progress import ProgressBus

@pytest.mark.asyncio
async def test_publish_and_receive():
    bus = ProgressBus()
    await bus.publish("conn1", {"event":"progress","files_done":5})
    event = await asyncio.wait_for(bus.get("conn1"), timeout=1.0)
    assert event["files_done"] == 5

@pytest.mark.asyncio
async def test_complete_event_clears_queue():
    bus = ProgressBus()
    await bus.publish("conn2", {"event":"complete","files_done":10})
    event = await asyncio.wait_for(bus.get("conn2"), timeout=1.0)
    assert event["event"] == "complete"

@pytest.mark.asyncio
async def test_separate_connectors_isolated():
    bus = ProgressBus()
    await bus.publish("conn-a", {"event":"progress","msg":"a"})
    await bus.publish("conn-b", {"event":"progress","msg":"b"})
    a = await asyncio.wait_for(bus.get("conn-a"), timeout=1.0)
    b = await asyncio.wait_for(bus.get("conn-b"), timeout=1.0)
    assert a["msg"] == "a"
    assert b["msg"] == "b"
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_sync_progress.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'sync'`

- [ ] **Step 3: Install pytest-asyncio**

```bash
pip install pytest-asyncio apscheduler
```

Add to `pytest.ini` (or create it):
```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 4: Create sync/__init__.py**

```python
# sync/__init__.py
```

- [ ] **Step 5: Create sync/progress.py**

```python
# sync/progress.py
"""Per-connector asyncio.Queue SSE bus.

SyncEngine publishes dicts; GET /sync/stream/{connector_id} drains them.
"""
import asyncio
import json
from collections import defaultdict
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

progress_router = APIRouter(tags=["sync-progress"])


class ProgressBus:
    """Thin wrapper so tests can instantiate an isolated bus."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

    async def publish(self, connector_id: str, event: dict) -> None:
        await self._queues[connector_id].put(event)

    async def get(self, connector_id: str) -> dict:
        return await self._queues[connector_id].get()


# Module-level singleton used by SyncEngine and the SSE route
_bus = ProgressBus()


async def publish(connector_id: str, event: dict) -> None:
    await _bus.publish(connector_id, event)


@progress_router.get("/sync/stream/{connector_id}")
async def sync_stream(connector_id: str, request: Request):
    u = getattr(request.state, "user", None)
    if not u or u.role not in ("admin", "editor"):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "forbidden"}, status_code=403)

    async def event_gen():
        q = _bus._queues[connector_id]
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("event") in ("complete", "error"):
                    break
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_sync_progress.py -v --override-ini="addopts="
```
Expected: `3 passed`

- [ ] **Step 7: Add progress_router to portal**

In `kvforge_portal.py`:
```python
from sync.progress import progress_router
app.include_router(progress_router)
```

- [ ] **Step 8: Commit**

```bash
git add sync/__init__.py sync/progress.py tests/test_sync_progress.py kvforge_portal.py
git commit -m "feat: add sync/progress.py — asyncio.Queue SSE bus + GET /sync/stream/{id}"
```

---

### Task 2: connectors/sync_engine.py

**Files:**
- Create: `connectors/sync_engine.py`
- Create: `tests/test_sync_engine.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_engine.py
import os, asyncio, uuid, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import db.store as store
from connectors.registry import ConnectorRegistry
from connectors.sync_engine import SyncEngine
from connectors.base import SourceFile
from datetime import datetime

def _setup(tmp_path):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    reg = ConnectorRegistry()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"a@b.com","admin","local"))
    store.commit()
    cfg = reg.create("s3","Test S3",{"bucket":"b","prefix":"docs/"},admin_id)
    reg.upsert_scope(cfg["id"],"uc1",{"bucket":"b","prefix":"docs/"})
    return cfg["id"]

class FakeConnector:
    def __init__(self):
        self.files = [
            SourceFile("f1","file1.txt","docs/file1.txt",100,datetime.now(),"text/plain"),
        ]
    def list_files(self): return self.files
    def download(self, f): return b"hello world content"
    def get_modified_at(self, f): return f.modified_at
    def supports_delta(self): return False
    def get_delta(self, token): return self.files, "new-token"

@pytest.mark.asyncio
async def test_sync_creates_run_record(tmp_path):
    cid = _setup(tmp_path)
    engine = SyncEngine(connector_factory=lambda cfg_type, creds: FakeConnector(),
                        embed_fn=lambda chunks: [[0.1]*4 for _ in chunks],
                        upsert_fn=lambda uc_id, chunks, embeddings, kvs: None,
                        kv_fn=lambda chunks: [None]*len(chunks))
    await engine.run(cid, "uc1", "manual")
    run = store.fetchone("SELECT * FROM sync_runs WHERE connector_id=?", (cid,))
    assert run is not None
    assert run["status"] == "ok"
    assert run["files_done"] >= 0

@pytest.mark.asyncio
async def test_sync_emits_progress_events(tmp_path):
    from sync.progress import _bus
    cid = _setup(tmp_path)
    events = []
    orig_publish = _bus.publish
    async def capture(conn_id, ev):
        events.append(ev)
        await orig_publish(conn_id, ev)
    _bus.publish = capture
    try:
        engine = SyncEngine(connector_factory=lambda cfg_type, creds: FakeConnector(),
                            embed_fn=lambda chunks: [[0.1]*4 for _ in chunks],
                            upsert_fn=lambda uc_id, chunks, embeddings, kvs: None,
                            kv_fn=lambda chunks: [None]*len(chunks))
        await engine.run(cid, "uc1", "manual")
    finally:
        _bus.publish = orig_publish
    assert any(e.get("stage") == "discover" for e in events)
    assert any(e.get("event") == "complete" for e in events)

@pytest.mark.asyncio
async def test_sync_error_recorded(tmp_path):
    cid = _setup(tmp_path)
    def bad_factory(t, c): raise RuntimeError("auth failed")
    engine = SyncEngine(connector_factory=bad_factory,
                        embed_fn=lambda x:[],
                        upsert_fn=lambda *a:None,
                        kv_fn=lambda x:[])
    await engine.run(cid, "uc1", "manual")
    run = store.fetchone("SELECT * FROM sync_runs WHERE connector_id=?", (cid,))
    assert run["status"] == "error"
    assert "auth failed" in run["error"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_sync_engine.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'connectors.sync_engine'`

- [ ] **Step 3: Create connectors/sync_engine.py**

```python
# connectors/sync_engine.py
"""Orchestrates list→diff→download→ingest→upsert per connector+UC scope."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import db.store as store
from connectors.base import SourceFile
from connectors.registry import ConnectorRegistry
from sync.progress import publish

_registry = ConnectorRegistry()


class SyncEngine:
    def __init__(
        self,
        connector_factory: Callable,   # (type: str, creds: dict) -> SourceConnector
        embed_fn: Callable,             # (chunks: list[str]) -> list[list[float]]
        upsert_fn: Callable,            # (uc_id, chunks, embeddings, kv_tensors) -> None
        kv_fn: Callable,                # (chunks: list[str]) -> list[tensor|None]
    ):
        self._factory = connector_factory
        self._embed = embed_fn
        self._upsert = upsert_fn
        self._kv = kv_fn

    async def run(self, connector_id: str, uc_id: str, trigger: str) -> str:
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        store.execute(
            "INSERT INTO sync_runs(id,connector_id,uc_id,trigger,status,started_at) VALUES(?,?,?,?,?,?)",
            (run_id, connector_id, uc_id, trigger, "running", now)
        )
        store.commit()

        try:
            await self._run_inner(run_id, connector_id, uc_id, trigger)
            store.execute(
                "UPDATE sync_runs SET status='ok',finished_at=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "complete", "connector_id": connector_id,
                                         "uc_id": uc_id, "run_id": run_id})
        except Exception as exc:
            store.execute(
                "UPDATE sync_runs SET status='error',error=?,finished_at=? WHERE id=?",
                (str(exc), datetime.now(timezone.utc).isoformat(), run_id)
            )
            store.commit()
            await publish(connector_id, {"event": "error", "connector_id": connector_id,
                                         "uc_id": uc_id, "error": str(exc)})
        return run_id

    async def _run_inner(self, run_id: str, connector_id: str, uc_id: str, trigger: str) -> None:
        creds = _registry.get_credentials(connector_id)
        cfg = _registry.get(connector_id)
        scopes = _registry.list_scopes(connector_id)
        scope = next((s for s in scopes if s["uc_id"] == uc_id), None)
        if scope is None:
            raise ValueError(f"no scope configured for connector {connector_id} / UC {uc_id}")

        connector = self._factory(cfg["type"], creds)

        await publish(connector_id, {"event": "progress", "stage": "discover",
                                     "connector_id": connector_id, "uc_id": uc_id,
                                     "files_total": 0, "files_done": 0,
                                     "message": "discovering files…"})

        # Discover files
        loop = asyncio.get_running_loop()
        if connector.supports_delta():
            last_token = scope.get("last_delta_token")
            files, new_token = await loop.run_in_executor(
                None, connector.get_delta, last_token)
        else:
            all_files: list[SourceFile] = await loop.run_in_executor(None, connector.list_files)
            # Delta via modification time
            scope_row = store.fetchone(
                "SELECT last_sync_at FROM connector_uc_scopes WHERE connector_id=? AND uc_id=?",
                (connector_id, uc_id)
            )
            last_sync = scope_row["last_sync_at"] if scope_row and scope_row["last_sync_at"] else None
            if last_sync:
                from datetime import datetime as _dt
                last_dt = _dt.fromisoformat(last_sync.replace("Z",""))
                files = [f for f in all_files if f.modified_at.replace(tzinfo=None) > last_dt]
            else:
                files = all_files
            new_token = None

        store.execute("UPDATE sync_runs SET files_total=? WHERE id=?", (len(files), run_id))
        store.commit()
        await publish(connector_id, {"event": "progress", "stage": "discover",
                                     "connector_id": connector_id, "uc_id": uc_id,
                                     "files_total": len(files), "files_done": 0,
                                     "message": f"{len(files)} files to process"})

        # Process each file
        for i, sf in enumerate(files):
            raw: bytes = await loop.run_in_executor(None, connector.download, sf)
            chunks = _chunk_bytes(sf, raw)

            await publish(connector_id, {"event": "progress", "stage": "index",
                                         "connector_id": connector_id, "uc_id": uc_id,
                                         "files_total": len(files), "files_done": i,
                                         "message": f"embedding {sf.name}"})

            if chunks:
                embeddings = self._embed(chunks)
                kv_tensors = self._kv(chunks)
                self._upsert(uc_id, chunks, embeddings, kv_tensors)

            store.execute("UPDATE sync_runs SET files_done=? WHERE id=?", (i + 1, run_id))
            store.commit()

        # Persist delta token and last sync time
        now = datetime.now(timezone.utc).isoformat()
        store.execute(
            "UPDATE connector_uc_scopes SET last_sync_at=?,last_delta_token=? "
            "WHERE connector_id=? AND uc_id=?",
            (now, new_token, connector_id, uc_id)
        )
        store.commit()


def _chunk_bytes(sf: SourceFile, raw: bytes) -> list[str]:
    """Simple line-based chunking for now; replace with loader dispatch."""
    text = raw.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Group into ~512-char chunks
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) > 512:
            if buf:
                chunks.append(buf)
            buf = line
        else:
            buf = buf + " " + line if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def _default_connector_factory(connector_type: str, creds: dict):
    """Build a real SourceConnector from type + decrypted credentials."""
    if connector_type == "gdrive":
        from connectors.gdrive_connector import GDriveConnector
        return GDriveConnector(creds)
    if connector_type == "s3":
        from connectors.s3_connector import S3Connector
        return S3Connector(creds)
    if connector_type == "sharepoint":
        from connectors.sharepoint_connector import SharePointConnector
        return SharePointConnector(creds)
    raise ValueError(f"unknown connector type: {connector_type}")


def make_default_engine() -> SyncEngine:
    """Create a SyncEngine wired to real connectors (no-op embed/upsert/kv for now)."""
    return SyncEngine(
        connector_factory=_default_connector_factory,
        embed_fn=lambda chunks: [[0.0] * 384 for _ in chunks],
        upsert_fn=lambda uc_id, chunks, embs, kvs: None,
        kv_fn=lambda chunks: [None] * len(chunks),
    )
```

- [ ] **Step 4: Add POST /studio/api/connectors/{cid}/sync endpoint to connectors/routes.py**

Append to `connectors/routes.py`:

```python
@connector_router.post("/{cid}/sync")
async def trigger_sync(cid: str, request: Request):
    if err := _require_role(request, EDITOR_UP): return err
    scopes = _registry.list_scopes(cid)
    if not scopes:
        return JSONResponse({"detail": "no scopes configured for this connector"}, status_code=400)
    from connectors.sync_engine import make_default_engine
    engine = make_default_engine()
    import asyncio
    run_ids = []
    for scope in scopes:
        run_id = asyncio.create_task(engine.run(cid, scope["uc_id"], "manual"))
        run_ids.append(scope["uc_id"])
    return {"ok": True, "triggered_scopes": run_ids}
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_sync_engine.py -v --override-ini="addopts="
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add connectors/sync_engine.py connectors/routes.py tests/test_sync_engine.py
git commit -m "feat: add SyncEngine — file discovery, diff, download, ingest, SSE progress, sync_runs persistence"
```

---

### Task 3: sync/scheduler.py

**Files:**
- Create: `sync/scheduler.py`
- Create: `tests/test_sync_scheduler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_scheduler.py
import os, uuid, asyncio, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import db.store as store
from sync.scheduler import SyncScheduler

def _setup(tmp_path):
    store.DB_PATH = tmp_path / "sched.db"
    store._local.__dict__.clear()
    store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"a@b.com","admin","local"))
    cid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO connector_configs(id,type,name,credentials_json,schedule_cron,created_by) VALUES(?,?,?,?,?,?)",
        (cid,"s3","S3",'{"k":"v"}', "*/5 * * * *", admin_id)
    )
    store.execute("INSERT INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
                  (cid,"uc1",'{"bucket":"b"}'))
    store.commit()
    return cid

def test_scheduler_registers_jobs(tmp_path):
    cid = _setup(tmp_path)
    triggered = []
    async def fake_run(connector_id, uc_id, trigger):
        triggered.append((connector_id, uc_id, trigger))
    sched = SyncScheduler(run_fn=fake_run)
    sched.load_from_db()
    jobs = sched.list_jobs()
    assert len(jobs) >= 1
    assert any(j["connector_id"] == cid for j in jobs)

def test_scheduler_reschedule(tmp_path):
    cid = _setup(tmp_path)
    sched = SyncScheduler(run_fn=lambda *a: None)
    sched.load_from_db()
    sched.reschedule(cid, "0 * * * *")
    jobs = sched.list_jobs()
    assert any(j["connector_id"] == cid for j in jobs)

def test_scheduler_remove(tmp_path):
    cid = _setup(tmp_path)
    sched = SyncScheduler(run_fn=lambda *a: None)
    sched.load_from_db()
    sched.remove(cid)
    jobs = sched.list_jobs()
    assert not any(j["connector_id"] == cid for j in jobs)
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_sync_scheduler.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'sync.scheduler'`

- [ ] **Step 3: Create sync/scheduler.py**

```python
# sync/scheduler.py
"""APScheduler-based cron scheduler for connector syncs.

Started in FastAPI lifespan; reads connector_configs.schedule_cron on startup.
"""
from __future__ import annotations
import asyncio
from typing import Callable
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import db.store as store


class SyncScheduler:
    def __init__(self, run_fn: Callable):
        """run_fn(connector_id, uc_id, trigger) — coroutine or plain callable."""
        self._run_fn = run_fn
        self._scheduler = AsyncIOScheduler()
        self._job_ids: dict[str, list[str]] = {}  # connector_id → [job_id, ...]

    def load_from_db(self) -> None:
        rows = store.fetchall(
            "SELECT cc.id as cid, cc.schedule_cron, cus.uc_id "
            "FROM connector_configs cc "
            "JOIN connector_uc_scopes cus ON cus.connector_id=cc.id "
            "WHERE cc.schedule_cron IS NOT NULL"
        )
        for row in rows:
            self._add_job(row["cid"], row["uc_id"], row["schedule_cron"])

    def _add_job(self, connector_id: str, uc_id: str, cron_expr: str) -> str:
        parts = cron_expr.split()
        if len(parts) == 5:
            minute, hour, day, month, dow = parts
        else:
            minute, hour, day, month, dow = "*", "*", "*", "*", "*"
        trigger = CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=dow)
        job_id = f"sync_{connector_id}_{uc_id}"
        self._scheduler.add_job(
            self._run_fn, trigger, args=[connector_id, uc_id, "scheduled"],
            id=job_id, replace_existing=True,
        )
        self._job_ids.setdefault(connector_id, [])
        if job_id not in self._job_ids[connector_id]:
            self._job_ids[connector_id].append(job_id)
        return job_id

    def reschedule(self, connector_id: str, cron_expr: str) -> None:
        self.remove(connector_id)
        scopes = store.fetchall(
            "SELECT uc_id FROM connector_uc_scopes WHERE connector_id=?", (connector_id,))
        for row in scopes:
            self._add_job(connector_id, row["uc_id"], cron_expr)

    def remove(self, connector_id: str) -> None:
        for jid in self._job_ids.pop(connector_id, []):
            try:
                self._scheduler.remove_job(jid)
            except Exception:
                pass

    def list_jobs(self) -> list[dict]:
        result = []
        for cid, jids in self._job_ids.items():
            for jid in jids:
                job = self._scheduler.get_job(jid)
                result.append({"connector_id": cid, "job_id": jid,
                                "next_run": str(job.next_run_time) if job else None})
        return result

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
```

- [ ] **Step 4: Wire scheduler into kvforge_portal.py lifespan**

Replace the current `_lifespan` in `kvforge_portal.py`:

```python
# Add import at top of kvforge_portal.py:
from sync.scheduler import SyncScheduler
from connectors.sync_engine import make_default_engine

_scheduler: SyncScheduler | None = None

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _scheduler
    import db.store as store
    store.migrate()
    engine = make_default_engine()
    _scheduler = SyncScheduler(run_fn=engine.run)
    _scheduler.load_from_db()
    _scheduler.start()
    yield
    if _scheduler:
        _scheduler.shutdown()
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_sync_scheduler.py -v --override-ini="addopts="
```
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add sync/scheduler.py tests/test_sync_scheduler.py kvforge_portal.py
git commit -m "feat: add SyncScheduler — APScheduler cron jobs loaded from DB, wired into portal lifespan"
```

---

### Task 4: sync/webhook.py

**Files:**
- Create: `sync/webhook.py`
- Create: `tests/test_sync_webhook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sync_webhook.py
import os, uuid, hmac, hashlib, json, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import db.store as store
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sync.webhook import webhook_router

def _setup(tmp_path):
    store.DB_PATH = tmp_path / "wh.db"
    store._local.__dict__.clear()
    store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id,"a@b.com","admin","local"))
    cid = str(uuid.uuid4())
    secret = "webhook-secret-xyz"
    store.execute(
        "INSERT INTO connector_configs(id,type,name,credentials_json,webhook_secret,created_by) VALUES(?,?,?,?,?,?)",
        (cid,"gdrive","GD",'{"k":"v"}',secret,admin_id)
    )
    store.execute("INSERT INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
                  (cid,"uc1",'{}'))
    store.commit()
    return cid, secret

def _make_client(triggered_calls):
    async def fake_run(cid, uc_id, trigger):
        triggered_calls.append((cid, uc_id, trigger))
    app = FastAPI()
    from sync.webhook import make_webhook_router
    app.include_router(make_webhook_router(run_fn=fake_run))
    return TestClient(app, raise_server_exceptions=False)

def _sign(body: bytes, secret: str) -> str:
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"

def test_valid_signature_accepted(tmp_path):
    cid, secret = _setup(tmp_path)
    triggered = []
    client = _make_client(triggered)
    body = json.dumps({"change": "new-file"}).encode()
    r = client.post(f"/webhooks/{cid}",
                    content=body,
                    headers={"X-Hub-Signature-256": _sign(body, secret),
                             "Content-Type": "application/json"})
    assert r.status_code == 200

def test_invalid_signature_rejected(tmp_path):
    cid, secret = _setup(tmp_path)
    client = _make_client([])
    body = b'{"change":"x"}'
    r = client.post(f"/webhooks/{cid}",
                    content=body,
                    headers={"X-Hub-Signature-256": "sha256=invalidsig",
                             "Content-Type": "application/json"})
    assert r.status_code == 401

def test_unknown_connector_returns_404(tmp_path):
    _setup(tmp_path)
    client = _make_client([])
    body = b"{}"
    r = client.post("/webhooks/nonexistent",
                    content=body,
                    headers={"X-Hub-Signature-256": _sign(body, "x"),
                             "Content-Type": "application/json"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_sync_webhook.py -v --override-ini="addopts="
```
Expected: `ModuleNotFoundError: No module named 'sync.webhook'`

- [ ] **Step 3: Create sync/webhook.py**

```python
# sync/webhook.py
"""HMAC-SHA256 validated webhook receiver for push change notifications.

Accepts POST /webhooks/{connector_id} — no auth cookie required.
Validates X-Hub-Signature-256 header; enqueues sync as background task.
"""
import asyncio
import hashlib
import hmac
from typing import Callable

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import db.store as store

webhook_router = APIRouter(tags=["webhooks"])


def make_webhook_router(run_fn: Callable) -> APIRouter:
    """Factory so tests can inject a fake run_fn."""
    router = APIRouter(tags=["webhooks"])

    @router.post("/webhooks/{connector_id}")
    async def receive_webhook(connector_id: str, request: Request,
                              background_tasks: BackgroundTasks):
        row = store.fetchone(
            "SELECT webhook_secret FROM connector_configs WHERE id=?", (connector_id,))
        if not row:
            return JSONResponse({"detail": "connector not found"}, status_code=404)

        secret = row["webhook_secret"] or ""
        body = await request.body()
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig_header, expected):
            return JSONResponse({"detail": "invalid signature"}, status_code=401)

        scopes = store.fetchall(
            "SELECT uc_id FROM connector_uc_scopes WHERE connector_id=?", (connector_id,))
        for scope in scopes:
            background_tasks.add_task(run_fn, connector_id, scope["uc_id"], "webhook")

        return {"ok": True, "queued_scopes": len(scopes)}

    return router
```

- [ ] **Step 4: Add webhook router to kvforge_portal.py**

```python
# In kvforge_portal.py _lifespan, after creating engine:
from sync.webhook import make_webhook_router
app.include_router(make_webhook_router(run_fn=engine.run))
```

Actually add the router inside lifespan isn't ideal. Better approach — create it at module level after `_scheduler` setup:

```python
# In kvforge_portal.py, after app = FastAPI(...):
from sync.webhook import make_webhook_router as _make_wh
# We'll register the webhook router after engine is created in lifespan.
# For simplicity, use the default engine in module scope:
from connectors.sync_engine import make_default_engine as _make_engine
_engine_for_wh = _make_engine()
app.include_router(_make_wh(run_fn=_engine_for_wh.run))
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_sync_webhook.py -v --override-ini="addopts="
```
Expected: `3 passed`

- [ ] **Step 6: Run full suite**

```bash
python -m pytest tests/ -v --override-ini="addopts=" -x --ignore=tests/ui
```

- [ ] **Step 7: Commit**

```bash
git add sync/webhook.py tests/test_sync_webhook.py kvforge_portal.py
git commit -m "feat: add webhook receiver — HMAC-SHA256 validation, background sync trigger"
```

---

## Verification

```bash
python kvforge_portal.py --port 8080

# SSE stream (with valid session cookie):
curl -N -H "Cookie: kvforge_session=<tok>" http://localhost:8080/sync/stream/<connector_id>

# Manual sync:
curl -X POST -H "Cookie: kvforge_session=<tok>" \
  http://localhost:8080/studio/api/connectors/<id>/sync

# Webhook (signed):
BODY='{"kind":"drive#change"}'; SECRET="your-secret"
SIG="sha256=$(echo -n $BODY | openssl dgst -sha256 -hmac $SECRET | awk '{print $2}')"
curl -X POST http://localhost:8080/webhooks/<connector_id> \
  -H "X-Hub-Signature-256: $SIG" -H "Content-Type: application/json" -d "$BODY"
```

**Run all Phase 3 tests:**
```bash
python -m pytest tests/test_sync_progress.py tests/test_sync_engine.py tests/test_sync_scheduler.py tests/test_sync_webhook.py -v --override-ini="addopts="
```
Expected: all pass.

**Proceed to:** `docs/superpowers/plans/2026-05-10-dashboard-phase4-ui-tests.md`
