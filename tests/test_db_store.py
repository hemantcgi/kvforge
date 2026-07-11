import pytest

def test_migrate_creates_tables(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.close()
    store.migrate()
    rows = store.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r["name"] for r in rows}
    assert {"users","sessions","invite_tokens","connector_configs",
            "connector_uc_scopes","sync_runs"} <= names

def test_execute_and_fetchone(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test2.db")
    store.close()
    store.migrate()
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  ("u1","a@b.com","admin","local"))
    store.commit()
    row = store.fetchone("SELECT * FROM users WHERE id=?", ("u1",))
    assert row["email"] == "a@b.com"
    assert row["role"] == "admin"

def test_invalid_role_rejected(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "c1.db")
    store.close()
    store.migrate()
    import pytest
    with pytest.raises(Exception):
        store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                      ("u2","b@b.com","hacker","local"))
        store.commit()
    store.close()

def test_duplicate_email_rejected(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "c2.db")
    store.close()
    store.migrate()
    import pytest
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  ("u3","dup@b.com","admin","local"))
    store.commit()
    with pytest.raises(Exception):
        store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                      ("u4","dup@b.com","editor","local"))
        store.commit()
    store.close()

def test_session_cascades_on_user_delete(tmp_path, monkeypatch):
    import db.store as store
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "c3.db")
    store.close()
    store.migrate()
    import uuid
    uid = str(uuid.uuid4())
    store.execute("INSERT INTO users(id,email,role,provider) VALUES(?,?,?,?)",
                  (uid,"cas@b.com","admin","local"))
    store.execute("INSERT INTO sessions(id,user_id,jwt_token,expires_at) VALUES(?,?,?,?)",
                  (str(uuid.uuid4()),uid,"tok","2099-01-01"))
    store.commit()
    assert store.fetchone("SELECT id FROM sessions WHERE user_id=?", (uid,)) is not None
    store.execute("DELETE FROM users WHERE id=?", (uid,))
    store.commit()
    assert store.fetchone("SELECT id FROM sessions WHERE user_id=?", (uid,)) is None
    store.close()
