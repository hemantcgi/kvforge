# vectorstore/pgvector_store.py
"""PostgreSQL + pgvector backend for KVForge."""
import json
import re
from typing import Any
from vectorstore.base import Point, ScoredPoint

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None  # type: ignore
    register_vector = None  # type: ignore


def _sanitise_table(name: str) -> str:
    """Replace non-alphanumeric/underscore characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class PGVectorStore:
    """VectorStore backed by PostgreSQL with the pgvector extension.

    One table per collection. Uses cosine distance operator (<=>).

    Args:
        cfg: Datasource config dict with pgvector_dsn, collection, vector_dim,
             and optional pgvector_table.

    Raises:
        ImportError: If psycopg2 or pgvector packages are not installed.
    """

    def __init__(self, cfg: dict) -> None:
        if psycopg2 is None:
            raise ImportError(
                "PGVectorStore requires: pip install psycopg2-binary pgvector"
            )
        self._conn = psycopg2.connect(cfg["pgvector_dsn"])
        self._conn.autocommit = True
        register_vector(self._conn)
        self._dim = cfg.get("vector_dim", 384)
        self._default_table = _sanitise_table(cfg.get("collection", "kvforge"))
        self._table_override = _sanitise_table(cfg.get("pgvector_table", "")) or ""

    def _table(self, collection: str) -> str:
        return self._table_override or _sanitise_table(collection)

    def create_collection(self, name: str, dim: int) -> None:
        t = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id      BIGINT PRIMARY KEY,
                    embedding vector({dim}),
                    payload JSONB NOT NULL DEFAULT '{{}}'
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {t}_embedding_idx
                    ON {t} USING ivfflat (embedding vector_cosine_ops)
            """)

    def collection_exists(self, name: str) -> bool:
        t = self._table(name)
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s LIMIT 1", (t,)
            )
            return cur.fetchone() is not None

    def delete_collection(self, name: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table(name)}")

    def upsert(self, collection: str, points: list[Point]) -> None:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            for p in points:
                cur.execute(
                    f"INSERT INTO {t} (id, embedding, payload) VALUES (%s, %s, %s) "
                    f"ON CONFLICT (id) DO UPDATE SET embedding=EXCLUDED.embedding, "
                    f"payload=EXCLUDED.payload",
                    (p.id, p.vector, json.dumps(p.payload)),
                )

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold: float | None = None) -> list[ScoredPoint]:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload, 1 - (embedding <=> %s::vector) AS score "
                f"FROM {t} ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, vector, top_k),
            )
            rows = cur.fetchall()
        results = []
        for row_id, payload_str, score in rows:
            if score_threshold is not None and score < score_threshold:
                continue
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            results.append(ScoredPoint(id=row_id, score=score, payload=payload))
        return results

    def scroll(self, collection: str, limit: int = 100,
               with_payload: bool = True, with_vectors: bool = False,
               offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]:
        t = self._table(collection)
        off = int(offset) if offset is not None else 0
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, payload FROM {t} ORDER BY id LIMIT %s OFFSET %s",
                (limit, off),
            )
            rows = cur.fetchall()
        points = []
        for row_id, payload_str in rows:
            payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            points.append(ScoredPoint(id=row_id, score=0.0,
                                      payload=payload if with_payload else {}))
        next_offset = off + len(rows) if len(rows) == limit else None
        return points, next_offset

    def set_payload(self, collection: str, point_id: int | str,
                    payload: dict) -> None:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(
                f"UPDATE {t} SET payload = payload || %s::jsonb WHERE id = %s",
                (json.dumps(payload), point_id),
            )

    def count(self, collection: str) -> int:
        t = self._table(collection)
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            row = cur.fetchone()
        return row[0] if row else 0
