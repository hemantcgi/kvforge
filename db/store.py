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
