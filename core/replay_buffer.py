"""SQLite-backed replay buffer for experience replay during LoRA fine-tuning.

Stores document chunks with tier labels and provides weighted-random sampling
so that frequently accessed (hot) chunks are more likely to appear in each
training batch.  This prevents catastrophic forgetting of well-learned content.

Tier sampling weights:

* ``hot``    → 8  (most recently and frequently accessed)
* ``warm``   → 4
* ``cold``   → 2
* ``frozen`` → 1  (never retrieved)

The SQLite database is safe for multi-threaded access via the
``check_same_thread=False`` flag.
"""

import random
import sqlite3
from pathlib import Path

TIER_WEIGHTS = {"hot": 8, "warm": 4, "cold": 2, "frozen": 1}
DEFAULT_DB = str(Path(__file__).parent / "replay_buffer.db")


class ReplayBuffer:
    """Weighted-random replay buffer persisted in a SQLite database.

    Chunks are stored with a ``tier`` label (``hot``, ``warm``, ``cold``, or
    ``frozen``).  Sampling is weighted by tier so that high-priority chunks
    appear more often during LoRA training.

    Args:
        db_path: Path to the SQLite database file.  Created automatically if
            it does not exist.

    Attributes:
        TIER_WEIGHTS: Module-level dict mapping tier names to sampling weights.
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                text     TEXT NOT NULL,
                tier     TEXT NOT NULL DEFAULT 'frozen'
            )
        """)
        # Per-chunk Knowledge Differentiation Score (KDS), imported from the
        # vector store to close the training feedback loop. Kept in its own
        # table so a KDS can be recorded before the chunk row itself exists.
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS chunk_kds (
                chunk_id INTEGER PRIMARY KEY,
                kds      REAL NOT NULL
            )
        """)
        self._con.commit()

    def add_kds(self, chunk_id: int, kds: float) -> None:
        """Record (upsert) the KDS for a chunk.

        Independent of whether the chunk itself is in the buffer — the trainer
        imports KDS from the vector store before the chunk rows are added.
        """
        self._con.execute(
            "INSERT OR REPLACE INTO chunk_kds (chunk_id, kds) VALUES (?, ?)",
            (int(chunk_id), float(kds)),
        )
        self._con.commit()

    def get_kds_map(self) -> dict:
        """Return ``{chunk_id: kds}`` for every chunk with a recorded KDS."""
        rows = self._con.execute("SELECT chunk_id, kds FROM chunk_kds").fetchall()
        return {r[0]: r[1] for r in rows}

    def add_chunks(self, chunks: list[dict], max_size: int = 5000) -> None:
        """Insert or replace chunks; evict lowest-value chunks if buffer exceeds max_size."""
        self._con.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, text, tier) VALUES (?,?,?)",
            [(c["chunk_id"], c["text"], c.get("tier", "frozen")) for c in chunks],
        )
        self._con.commit()
        self.evict_to_cap(max_size)

    def update_tier(self, chunk_id: int, tier: str) -> None:
        """Update the tier label for a single chunk.

        Args:
            chunk_id: Identifier of the chunk to update.
            tier: New tier label — one of ``'hot'``, ``'warm'``, ``'cold'``,
                ``'frozen'``.
        """
        self._con.execute("UPDATE chunks SET tier=? WHERE chunk_id=?",
                          (tier, chunk_id))
        self._con.commit()

    def update_tiers_bulk(self, updates: list[tuple[int, str]]) -> None:
        """Batch-update tier labels for multiple chunks in a single transaction.

        Args:
            updates: List of ``(chunk_id, tier)`` tuples.
        """
        self._con.executemany("UPDATE chunks SET tier=? WHERE chunk_id=?",
                              [(t, cid) for cid, t in updates])
        self._con.commit()

    @staticmethod
    def _sample_weight(row, kds_map: dict | None) -> float:
        """Tier weight, boosted for under-differentiated (low-KDS) chunks.

        A chunk with a recorded KDS is boosted by ``1 + (1 - kds)`` (KDS
        clamped to ``[0, 1]``): low-KDS (poorly learned) chunks are sampled
        more so the next LoRA round reinforces them; chunks with no KDS are
        left at their plain tier weight.
        """
        w = float(TIER_WEIGHTS.get(row[2], 1))
        if kds_map:
            kds = kds_map.get(row[0])
            if kds is not None:
                w *= 1.0 + (1.0 - max(0.0, min(1.0, float(kds))))
        return w

    def sample(self, n: int, weight_by_tier: bool = True,
               kds_map: dict | None = None) -> list[dict]:
        """Draw up to *n* chunks from the buffer with optional tier weighting.

        When *weight_by_tier* is ``True``, chunks are drawn using
        ``random.choices`` with weights proportional to their tier (see
        ``TIER_WEIGHTS``), then deduplicated while preserving the approximate
        distribution.

        Args:
            n: Maximum number of chunks to return.
            weight_by_tier: If ``True`` (default), sample proportional to tier
                weight.  If ``False``, sample uniformly at random.
            kds_map: Optional ``{chunk_id: kds}`` mapping. When provided (and
                *weight_by_tier* is ``True``), under-differentiated (low-KDS)
                chunks are up-weighted; see ``_sample_weight``. Ignored for
                uniform sampling.

        Returns:
            List of dicts with keys ``chunk_id``, ``text``, and ``tier``.
            May be shorter than *n* if the buffer has fewer than *n* entries.
        """
        rows = self._con.execute(
            "SELECT chunk_id, text, tier FROM chunks"
        ).fetchall()
        if not rows:
            return []
        if weight_by_tier:
            weights = [self._sample_weight(r, kds_map) for r in rows]
            k = min(n, len(rows))
            chosen = random.choices(rows, weights=weights, k=k)
            # deduplicate while preserving approximate distribution
            seen, result = set(), []
            for row in chosen:
                if row[0] not in seen:
                    seen.add(row[0])
                    result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
            # if dedup reduced below n, top up using the same weighting
            remaining = [r for r in rows if r[0] not in seen]
            while len(result) < k and remaining:
                rem_weights = [self._sample_weight(r, kds_map) for r in remaining]
                (row,) = random.choices(remaining, weights=rem_weights, k=1)
                result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
                remaining.remove(row)
                seen.add(row[0])
        else:
            chosen = random.sample(rows, min(n, len(rows)))
            result = [{"chunk_id": r[0], "text": r[1], "tier": r[2]} for r in chosen]
        return result

    def count(self) -> int:
        """Return the total number of chunks currently stored in the buffer."""
        return self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def evict_to_cap(self, max_size: int = 5000) -> int:
        """Evict oldest + lowest-tier chunks if buffer exceeds max_size. Returns count removed."""
        current = self.count()
        if current <= max_size:
            return 0
        # Order: frozen first (tier), then by rowid (oldest) — evict the least valuable
        tier_order = "CASE tier WHEN 'frozen' THEN 0 WHEN 'cold' THEN 1 WHEN 'warm' THEN 2 ELSE 3 END"
        to_delete = current - max_size
        self._con.execute(f"""
            DELETE FROM chunks WHERE chunk_id IN (
                SELECT chunk_id FROM chunks
                ORDER BY {tier_order} ASC, rowid ASC
                LIMIT ?
            )
        """, (to_delete,))
        self._con.commit()
        return to_delete
