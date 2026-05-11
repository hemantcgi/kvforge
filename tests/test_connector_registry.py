# tests/test_connector_registry.py
import os, pytest
os.environ["KVFORGE_SECRET_KEY"] = "test-secret-32bytesXXXXXXXXXXXX"

import db.store as store
from connectors.registry import ConnectorRegistry

def _setup(tmp_path):
    store.DB_PATH = tmp_path / "test.db"
    store._local.__dict__.clear()
    store.migrate()
    # Insert a test user
    store.execute(
        "INSERT INTO users(id,email,role) VALUES(?,?,?)",
        ("user1", "user1@test.com", "admin")
    )
    store.commit()
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
    assert cfg["credentials"] == "●●●●●●"  # masked, never full
    assert "credentials_json" not in cfg

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
