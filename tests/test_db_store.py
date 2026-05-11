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
