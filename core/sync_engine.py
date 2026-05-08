"""Sync engine: section-hash diffing and deletion detection.

SyncStateDB manages the SQLite state for the sync engine.
SyncEngine orchestrates polling, diffing, and re-indexing.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SyncStateDB:
    """SQLite-backed store for document and section hashes."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS document_hashes (
                    uc_name     TEXT NOT NULL,
                    source_id   TEXT NOT NULL,
                    doc_hash    TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    PRIMARY KEY (uc_name, source_id)
                );
                CREATE TABLE IF NOT EXISTS section_hashes (
                    uc_name      TEXT NOT NULL,
                    source_id    TEXT NOT NULL,
                    section_id   TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_ids    TEXT NOT NULL,
                    indexed_at   TEXT NOT NULL,
                    PRIMARY KEY (uc_name, source_id, section_id)
                );
                CREATE TABLE IF NOT EXISTS deleted_docs (
                    uc_name    TEXT NOT NULL,
                    source_id  TEXT NOT NULL,
                    deleted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sync_runs (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    uc_name      TEXT NOT NULL,
                    started_at   TEXT NOT NULL,
                    finished_at  TEXT,
                    files_checked  INTEGER DEFAULT 0,
                    files_changed  INTEGER DEFAULT 0,
                    chunks_added   INTEGER DEFAULT 0,
                    chunks_superseded INTEGER DEFAULT 0,
                    pii_detections INTEGER DEFAULT 0,
                    errors       TEXT DEFAULT ''
                );
            """)

    def upsert_doc_hash(self, uc_name: str, source_id: str, doc_hash: str, modified_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO document_hashes VALUES (?,?,?,?)",
                (uc_name, source_id, doc_hash, modified_at),
            )

    def get_doc_hash(self, uc_name: str, source_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM document_hashes WHERE uc_name=? AND source_id=?",
                (uc_name, source_id),
            ).fetchone()
            return dict(row) if row else None

    def upsert_section_hash(self, uc_name: str, source_id: str, section_id: str,
                             content_hash: str, chunk_ids: str, indexed_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO section_hashes VALUES (?,?,?,?,?,?)",
                (uc_name, source_id, section_id, content_hash, chunk_ids, indexed_at),
            )

    def get_section_hashes(self, uc_name: str, source_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM section_hashes WHERE uc_name=? AND source_id=?",
                (uc_name, source_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_source_ids(self, uc_name: str) -> set[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT source_id FROM document_hashes WHERE uc_name=?", (uc_name,)
            ).fetchall()
            return {r["source_id"] for r in rows}

    def record_sync_run(self, uc_name: str, **stats) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO sync_runs
                   (uc_name, started_at, finished_at, files_checked, files_changed,
                    chunks_added, chunks_superseded, pii_detections, errors)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    uc_name,
                    stats.get("started_at", datetime.now(timezone.utc).isoformat()),
                    stats.get("finished_at"),
                    stats.get("files_checked", 0),
                    stats.get("files_changed", 0),
                    stats.get("chunks_added", 0),
                    stats.get("chunks_superseded", 0),
                    stats.get("pii_detections", 0),
                    stats.get("errors", ""),
                ),
            )
            return cur.lastrowid

    def get_sync_runs(self, uc_name: str, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sync_runs WHERE uc_name=? ORDER BY id DESC LIMIT ?",
                (uc_name, limit),
            ).fetchall()
            return [dict(r) for r in rows]
