# tests/test_sync_scheduler.py
import os, uuid, pytest
os.environ.setdefault("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

import db.store as store
from sync.scheduler import SyncScheduler

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("KVFORGE_SECRET_KEY", "test-secret-32bytesXXXXXXXXXXXX")

@pytest.fixture(autouse=True)
def _reset_store():
    yield
    store.close()

def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sched.db")
    store._local.__dict__.clear()
    store.migrate()
    admin_id = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (admin_id, "a@b.com", "admin", "local"))
    cid = str(uuid.uuid4())
    store.execute(
        "INSERT INTO connector_configs(id,type,name,credentials_json,schedule_cron,created_by) VALUES(?,?,?,?,?,?)",
        (cid, "s3", "S3", '{"k":"v"}', "*/5 * * * *", admin_id)
    )
    store.execute("INSERT INTO connector_uc_scopes(connector_id,uc_id,scope_config_json) VALUES(?,?,?)",
                  (cid, "uc1", '{"bucket":"b"}'))
    store.commit()
    return cid

def test_scheduler_registers_jobs(tmp_path, monkeypatch):
    cid = _setup(tmp_path, monkeypatch)
    async def fake_run(connector_id, uc_id, trigger):
        pass
    sched = SyncScheduler(run_fn=fake_run)
    sched.load_from_db()
    jobs = sched.list_jobs()
    assert len(jobs) >= 1
    assert any(j["connector_id"] == cid for j in jobs)

def test_scheduler_reschedule(tmp_path, monkeypatch):
    cid = _setup(tmp_path, monkeypatch)
    sched = SyncScheduler(run_fn=lambda *a: None)
    sched.load_from_db()
    sched.reschedule(cid, "0 * * * *")
    jobs = sched.list_jobs()
    assert any(j["connector_id"] == cid for j in jobs)

def test_scheduler_remove(tmp_path, monkeypatch):
    cid = _setup(tmp_path, monkeypatch)
    sched = SyncScheduler(run_fn=lambda *a: None)
    sched.load_from_db()
    sched.remove(cid)
    jobs = sched.list_jobs()
    assert not any(j["connector_id"] == cid for j in jobs)
