"""SQLite-backed real-time query logger for KVForge dynamic PRS.

Every query routed through KVForge is recorded here, enabling:

* Real-time coverage signal for per-cluster PRS computation.
* Dissatisfaction detection via re-query tracking.
* Training pair export for LoRA fine-tuning.

Schema
------
``query_log`` table columns:

* ``id`` — auto-increment primary key.
* ``timestamp`` — Unix epoch (float).
* ``query_text`` — original query string.
* ``answer_text`` — answer produced by the router.
* ``cluster_id`` — cluster the query was routed to (nullable).
* ``chunk_id`` — specific chunk used for retrieval (nullable).
* ``routed_to`` — ``'retrieval'`` or ``'parametric'``.
* ``requeried`` — 1 if a subsequent identical query was detected (dissatisfaction).
* ``embedding`` — optional JSON-encoded embedding blob.

WAL mode is used for concurrent read/write access from multiple threads/processes.

Public API
----------
* ``init_db(db_path)`` — create the schema; idempotent.
* ``log_query(...)`` → row id (int).
* ``mark_requeried(db_path, original_query, window_minutes)`` — flag dissatisfaction.
* ``get_cluster_stats(db_path, cluster_id, window_minutes)`` → dict.
* ``get_training_pairs(db_path, cluster_id, limit)`` → list of dicts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Optional

_lock = threading.Lock()

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS query_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL    NOT NULL,
    query_text  TEXT  NOT NULL,
    answer_text TEXT  NOT NULL,
    cluster_id  TEXT,
    chunk_id    TEXT,
    routed_to   TEXT  NOT NULL,
    requeried   INTEGER DEFAULT 0,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_cluster   ON query_log(cluster_id);
CREATE INDEX IF NOT EXISTS idx_timestamp ON query_log(timestamp);
"""


def init_db(db_path: str) -> None:
    """Create the ``query_log`` table and indices if they do not already exist.

    Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.

    Args:
        db_path: File-system path to the SQLite database.
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def log_query(
    db_path: str,
    query_text: str,
    answer_text: str,
    routed_to: str,
    cluster_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    embedding: Optional[list[float]] = None,
) -> int:
    """Insert a query record and return the new row id.

    Args:
        db_path: Path to the SQLite database.
        query_text: Raw query string.
        answer_text: Answer returned to the user.
        routed_to: ``'retrieval'`` or ``'parametric'``.
        cluster_id: Cluster the query was routed to (optional).
        chunk_id: Specific retrieved chunk id (optional).
        embedding: Query embedding as a Python list (optional; stored as JSON blob).

    Returns:
        Integer row id of the inserted record.
    """
    emb_blob = json.dumps(embedding).encode() if embedding else None
    with _lock, sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO query_log
               (timestamp, query_text, answer_text, cluster_id, chunk_id, routed_to, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), query_text, answer_text, cluster_id, chunk_id, routed_to, emb_blob),
        )
        conn.commit()
        return cur.lastrowid


def mark_requeried(
    db_path: str, original_query: str, window_minutes: int = 10
) -> None:
    """Flag parametric answers for *original_query* as re-queried (dissatisfaction signal).

    Only records in the parametric routing category within the time window are updated.

    Args:
        db_path: Path to the SQLite database.
        original_query: The query text to match.
        window_minutes: How far back to look for the original parametric answer.
    """
    cutoff = time.time() - window_minutes * 60
    with _lock, sqlite3.connect(db_path) as conn:
        conn.execute(
            """UPDATE query_log SET requeried = 1
               WHERE routed_to = 'parametric' AND timestamp > ? AND query_text = ?""",
            (cutoff, original_query),
        )
        conn.commit()


def get_cluster_stats(
    db_path: str, cluster_id: str, window_minutes: int = 10
) -> dict:
    """Return real-time coverage stats for a cluster within the recent time window.

    Real-time coverage is the fraction of parametric answers that were NOT
    re-queried (i.e. the user accepted the answer).

    Args:
        db_path: Path to the SQLite database.
        cluster_id: Cluster identifier to filter by.
        window_minutes: Recency window in minutes.

    Returns:
        Dict with keys:

        * ``'realtime_coverage'`` — float in [0, 1].
        * ``'query_count'`` — total queries in window.
    """
    cutoff = time.time() - window_minutes * 60
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN routed_to='parametric' AND requeried=0 THEN 1 ELSE 0 END)
               FROM query_log WHERE cluster_id = ? AND timestamp > ?""",
            (cluster_id, cutoff),
        ).fetchone()
    total, good = row[0], row[1] or 0
    if not total:
        return {"realtime_coverage": 0.0, "query_count": 0}
    return {"realtime_coverage": good / total, "query_count": total}


def get_training_pairs(
    db_path: str,
    cluster_id: Optional[str] = None,
    limit: int = 1000,
) -> list[dict]:
    """Return retrieval-routed Q&A pairs suitable for LoRA fine-tuning.

    Only ``routed_to='retrieval'`` records are returned — these are the cases
    where the model fell back to RAG, so they represent the training frontier.

    Args:
        db_path: Path to the SQLite database.
        cluster_id: If provided, filter to this cluster only.
        limit: Maximum number of records to return.

    Returns:
        List of dicts with keys ``'question'``, ``'answer'``, ``'cluster_id'``.
    """
    with sqlite3.connect(db_path) as conn:
        if cluster_id is not None:
            rows = conn.execute(
                """SELECT query_text, answer_text, cluster_id FROM query_log
                   WHERE routed_to = 'retrieval' AND cluster_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (cluster_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT query_text, answer_text, cluster_id FROM query_log
                   WHERE routed_to = 'retrieval' ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [{"question": r[0], "answer": r[1], "cluster_id": r[2]} for r in rows]
