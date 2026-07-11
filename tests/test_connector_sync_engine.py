# tests/test_connector_sync_engine.py
import os, asyncio, uuid, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

import db.store as store
from connectors.registry import ConnectorRegistry
from connectors.sync_engine import SyncEngine
from connectors.base import SourceFile
from datetime import datetime

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _reset_store():
    import db.store as store
    yield
    store.close()

def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.close()
    store.migrate()
    reg = ConnectorRegistry()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id, "a@b.com", "admin", "local"))
    store.commit()
    cfg = reg.create("s3", "Test S3", {"bucket": "b", "prefix": "docs/"}, admin_id)
    reg.upsert_scope(cfg["id"], "uc1", {"bucket": "b", "prefix": "docs/"})
    return cfg["id"]

class FakeConnector:
    def __init__(self):
        self.files = [
            SourceFile("f1", "file1.txt", "docs/file1.txt", 100, datetime.now(), "text/plain"),
        ]
    def list_files(self): return self.files
    def download(self, f): return b"hello world content"
    def get_modified_at(self, f): return f.modified_at
    def supports_delta(self): return False
    def get_delta(self, token): return self.files, "new-token"

@pytest.mark.asyncio
async def test_sync_creates_run_record(tmp_path, monkeypatch):
    cid = _setup(tmp_path, monkeypatch)
    engine = SyncEngine(
        connector_factory=lambda cfg_type, creds: FakeConnector(),
        embed_fn=lambda chunks: [[0.1] * 4 for _ in chunks],
        upsert_fn=lambda uc_id, chunks, embeddings, kvs: None,
        kv_fn=lambda chunks: [None] * len(chunks),
    )
    await engine.run(cid, "uc1", "manual")
    run = store.fetchone("SELECT * FROM sync_runs WHERE connector_id=?", (cid,))
    assert run is not None
    assert run["status"] == "ok"
    assert run["files_done"] >= 0

@pytest.mark.asyncio
async def test_sync_emits_progress_events(tmp_path, monkeypatch):
    from sync.progress import _bus
    cid = _setup(tmp_path, monkeypatch)
    events = []
    orig_publish = _bus.publish
    async def capture(conn_id, ev):
        events.append(ev)
        await orig_publish(conn_id, ev)
    _bus.publish = capture
    try:
        engine = SyncEngine(
            connector_factory=lambda cfg_type, creds: FakeConnector(),
            embed_fn=lambda chunks: [[0.1] * 4 for _ in chunks],
            upsert_fn=lambda uc_id, chunks, embeddings, kvs: None,
            kv_fn=lambda chunks: [None] * len(chunks),
        )
        await engine.run(cid, "uc1", "manual")
    finally:
        _bus.publish = orig_publish
    assert any(e.get("stage") == "discover" for e in events)
    assert any(e.get("event") == "complete" for e in events)

@pytest.mark.asyncio
async def test_sync_error_recorded(tmp_path, monkeypatch):
    cid = _setup(tmp_path, monkeypatch)
    def bad_factory(t, c): raise RuntimeError("auth failed")
    engine = SyncEngine(
        connector_factory=bad_factory,
        embed_fn=lambda x: [],
        upsert_fn=lambda *a: None,
        kv_fn=lambda x: [],
    )
    await engine.run(cid, "uc1", "manual")
    run = store.fetchone("SELECT * FROM sync_runs WHERE connector_id=?", (cid,))
    assert run["status"] == "error"
    assert "auth failed" in run["error"]
