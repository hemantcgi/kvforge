import sys, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_sync_db_created_on_init(tmp_path):
    from core.sync_engine import SyncStateDB
    db = SyncStateDB(str(tmp_path / "sync.db"))
    assert (tmp_path / "sync.db").exists()


def test_sync_db_tables_exist(tmp_path):
    from core.sync_engine import SyncStateDB
    db = SyncStateDB(str(tmp_path / "sync.db"))
    conn = sqlite3.connect(str(tmp_path / "sync.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "document_hashes" in tables
    assert "section_hashes" in tables
    assert "deleted_docs" in tables
    assert "sync_runs" in tables


def test_upsert_and_get_doc_hash(tmp_path):
    from core.sync_engine import SyncStateDB
    db = SyncStateDB(str(tmp_path / "sync.db"))
    db.upsert_doc_hash("uc1", "file_001", "abc123", "2026-01-01T00:00:00+00:00")
    row = db.get_doc_hash("uc1", "file_001")
    assert row["doc_hash"] == "abc123"


def test_upsert_and_get_section_hash(tmp_path):
    from core.sync_engine import SyncStateDB
    db = SyncStateDB(str(tmp_path / "sync.db"))
    db.upsert_section_hash("uc1", "file_001", "slide:1", "deadbeef", '["chunk_a"]', "2026-01-01T00:00:00+00:00")
    rows = db.get_section_hashes("uc1", "file_001")
    assert rows[0]["section_id"] == "slide:1"
    assert rows[0]["content_hash"] == "deadbeef"


def test_list_source_ids(tmp_path):
    from core.sync_engine import SyncStateDB
    db = SyncStateDB(str(tmp_path / "sync.db"))
    db.upsert_doc_hash("uc1", "file_a", "h1", "2026-01-01T00:00:00+00:00")
    db.upsert_doc_hash("uc1", "file_b", "h2", "2026-01-01T00:00:00+00:00")
    ids = db.list_source_ids("uc1")
    assert "file_a" in ids
    assert "file_b" in ids
