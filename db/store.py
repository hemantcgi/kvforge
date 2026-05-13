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
    _migrate_connector_types()


def _migrate_connector_types() -> None:
    """Expand connector_configs.type CHECK constraint to include open-source connector types.

    SQLite does not support ALTER COLUMN, so we recreate the table when the old
    constraint (gdrive/s3/sharepoint only) is still in place.
    """
    row = _conn().execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='connector_configs'"
    ).fetchone()
    if not row:
        return
    # If all 7 types are already listed, nothing to do
    if "'espn'" in row[0] and "'wikipedia'" in row[0]:
        return
    _conn().executescript("""
        PRAGMA foreign_keys=OFF;
        ALTER TABLE connector_configs RENAME TO _connector_configs_old;
        CREATE TABLE connector_configs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL CHECK(type IN ('gdrive','s3','sharepoint','wikipedia','fda','edgar','espn')),
            name TEXT NOT NULL,
            credentials_json TEXT NOT NULL,
            schedule_cron TEXT,
            webhook_secret TEXT,
            created_by TEXT REFERENCES users(id),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO connector_configs SELECT * FROM _connector_configs_old;
        DROP TABLE _connector_configs_old;
        PRAGMA foreign_keys=ON;
    """)
    _conn().commit()


def close() -> None:
    conn = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None
