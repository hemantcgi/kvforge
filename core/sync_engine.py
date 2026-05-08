"""Sync engine: section-hash diffing and deletion detection.

SyncStateDB manages the SQLite state for the sync engine.
SyncEngine (added in a subsequent commit) orchestrates polling, diffing, and re-indexing.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from connectors.base import SourceConnector, SourceFile
from ingestion.directory_loader import EXTENSION_MAP


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
                    files_deleted  INTEGER DEFAULT 0,
                    chunks_added   INTEGER DEFAULT 0,
                    chunks_superseded INTEGER DEFAULT 0,
                    pii_detections INTEGER DEFAULT 0,
                    errors       TEXT DEFAULT ''
                );
            """)
            try:
                conn.execute("ALTER TABLE sync_runs ADD COLUMN files_deleted INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # column already exists

    def upsert_doc_hash(self, uc_name: str, source_id: str, doc_hash: str, modified_at: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO document_hashes (uc_name, source_id, doc_hash, modified_at) VALUES (?,?,?,?)",
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
                "INSERT OR REPLACE INTO section_hashes (uc_name, source_id, section_id, content_hash, chunk_ids, indexed_at) VALUES (?,?,?,?,?,?)",
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
                    files_deleted, chunks_added, chunks_superseded, pii_detections, errors)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    uc_name,
                    stats.get("started_at", datetime.now(timezone.utc).isoformat()),
                    stats.get("finished_at"),
                    stats.get("files_checked", 0),
                    stats.get("files_changed", 0),
                    stats.get("files_deleted", 0),
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

    def record_deleted_doc(self, uc_name: str, source_id: str, deleted_at: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM document_hashes WHERE uc_name=? AND source_id=?", (uc_name, source_id))
            conn.execute("DELETE FROM section_hashes WHERE uc_name=? AND source_id=?", (uc_name, source_id))
            conn.execute(
                "INSERT INTO deleted_docs (uc_name, source_id, deleted_at) VALUES (?,?,?)",
                (uc_name, source_id, deleted_at),
            )

    def get_deleted_docs(self, uc_name: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM deleted_docs WHERE uc_name=? ORDER BY deleted_at DESC",
                (uc_name,),
            ).fetchall()
            return [dict(r) for r in rows]


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_sections(file_name: str, data: bytes, tmp_dir: str) -> list[tuple[str, str, list[dict]]]:
    """Return list of (section_id, section_hash, chunks) for a file."""
    from ingestion.registry import get_loader
    suffix = Path(file_name).suffix.lower()
    loader_name = EXTENSION_MAP.get(suffix)
    if not loader_name:
        return []

    tmp_path = Path(tmp_dir) / file_name
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(data)

    loader = get_loader({"loader": loader_name})
    chunks = loader.load(str(tmp_path))

    section_map: dict[str, list[dict]] = {}
    for chunk in chunks:
        sh = chunk.get("metadata", {}).get("section_hash", _file_hash(data))
        if sh not in section_map:
            section_map[sh] = []
        section_map[sh].append(chunk)

    results = []
    heading_counts: dict[str, int] = {}
    for section_hash, section_chunks in section_map.items():
        meta = section_chunks[0].get("metadata", {})
        if "slide_number" in meta:
            section_id = f"slide:{meta['slide_number']}"
        elif "heading_text" in meta:
            h = meta["heading_text"]
            heading_counts[h] = heading_counts.get(h, 0) + 1
            section_id = f"heading:{h}:{heading_counts[h]}"
        elif "sheet_name" in meta:
            section_id = f"sheet:{meta['sheet_name']}:row:{meta.get('row_range', {}).get('start', 0)}"
        else:
            section_id = f"chunk:{section_hash[:8]}"
        results.append((section_id, section_hash, section_chunks))

    return results


class SyncEngine:
    """Orchestrate polling, section-hash diffing, and re-indexing for one UC."""

    def __init__(self, uc_name: str, db: SyncStateDB, connector: SourceConnector,
                 cfg, indexer, superseder=None):
        self.uc_name = uc_name
        self.db = db
        self.connector = connector
        self.cfg = cfg
        self.indexer = indexer  # callable(cfg, chunks)
        self.superseder = superseder or (lambda chunk_ids: None)

    def run(self) -> dict:
        started_at = datetime.now(timezone.utc).isoformat()
        stats = {
            "files_checked": 0, "files_changed": 0, "files_deleted": 0,
            "chunks_added": 0, "chunks_superseded": 0, "errors": "",
        }

        files = self.connector.list_files()
        current_ids = {f.id for f in files}
        previous_ids = self.db.list_source_ids(self.uc_name)

        deleted_ids = previous_ids - current_ids
        for source_id in deleted_ids:
            self._handle_deletion(source_id)
            stats["files_deleted"] += 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            for source_file in files:
                stats["files_checked"] += 1
                try:
                    changed = self._process_file(source_file, tmp_dir, stats)
                    if changed:
                        stats["files_changed"] += 1
                except Exception as e:
                    stats["errors"] += f"{source_file.name}: {e}\n"

        self.db.record_sync_run(self.uc_name, started_at=started_at,
                                finished_at=datetime.now(timezone.utc).isoformat(), **stats)
        return stats

    def _process_file(self, source_file: SourceFile, tmp_dir: str, stats: dict) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        data = self.connector.download(source_file)
        doc_hash = _file_hash(data)

        stored = self.db.get_doc_hash(self.uc_name, source_file.id)
        if stored and stored["doc_hash"] == doc_hash:
            return False

        self.db.upsert_doc_hash(self.uc_name, source_file.id, doc_hash,
                                source_file.modified_at.isoformat())

        sections = _get_sections(source_file.name, data, tmp_dir)
        if not sections:
            return False

        stored_sections = {
            row["section_id"]: row
            for row in self.db.get_section_hashes(self.uc_name, source_file.id)
        }

        for section_id, section_hash, chunks in sections:
            stored_sec = stored_sections.get(section_id)
            if stored_sec and stored_sec["content_hash"] == section_hash:
                continue

            if stored_sec:
                old_chunk_ids = json.loads(stored_sec["chunk_ids"])
                self._supersede_chunks(old_chunk_ids)
                stats["chunks_superseded"] += len(old_chunk_ids)

            self.indexer(self.cfg, chunks)
            new_chunk_ids = [c.get("id", "") for c in chunks]
            stats["chunks_added"] += len(chunks)
            self.db.upsert_section_hash(
                self.uc_name, source_file.id, section_id, section_hash,
                json.dumps(new_chunk_ids), now,
            )

        return True

    def _supersede_chunks(self, chunk_ids: list[str]) -> None:
        self.superseder(chunk_ids)

    def _handle_deletion(self, source_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.db.record_deleted_doc(self.uc_name, source_id, now)
