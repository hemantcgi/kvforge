"""
replay_buffer.py — SQLite-backed replay buffer for LoRA training.

Tier weights for sampling:
  hot    → 8
  warm   → 4
  cold   → 2
  frozen → 1
"""

import random
import sqlite3
from pathlib import Path

TIER_WEIGHTS = {"hot": 8, "warm": 4, "cold": 2, "frozen": 1}
DEFAULT_DB = str(Path(__file__).parent / "replay_buffer.db")


class ReplayBuffer:
    def __init__(self, db_path: str = DEFAULT_DB):
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                text     TEXT NOT NULL,
                tier     TEXT NOT NULL DEFAULT 'frozen'
            )
        """)
        self._con.commit()

    def add_chunks(self, chunks: list[dict], max_size: int = 5000) -> None:
        """Insert or replace chunks; evict lowest-value chunks if buffer exceeds max_size."""
        self._con.executemany(
            "INSERT OR REPLACE INTO chunks (chunk_id, text, tier) VALUES (?,?,?)",
            [(c["chunk_id"], c["text"], c.get("tier", "frozen")) for c in chunks],
        )
        self._con.commit()
        self.evict_to_cap(max_size)

    def update_tier(self, chunk_id: int, tier: str) -> None:
        self._con.execute("UPDATE chunks SET tier=? WHERE chunk_id=?",
                          (tier, chunk_id))
        self._con.commit()

    def update_tiers_bulk(self, updates: list[tuple[int, str]]) -> None:
        """updates: list of (chunk_id, tier)"""
        self._con.executemany("UPDATE chunks SET tier=? WHERE chunk_id=?",
                              [(t, cid) for cid, t in updates])
        self._con.commit()

    def sample(self, n: int, weight_by_tier: bool = True) -> list[dict]:
        rows = self._con.execute(
            "SELECT chunk_id, text, tier FROM chunks"
        ).fetchall()
        if not rows:
            return []
        if weight_by_tier:
            weights = [TIER_WEIGHTS.get(r[2], 1) for r in rows]
            k = min(n, len(rows))
            chosen = random.choices(rows, weights=weights, k=k)
            # deduplicate while preserving approximate distribution
            seen, result = set(), []
            for row in chosen:
                if row[0] not in seen:
                    seen.add(row[0])
                    result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
            # if dedup reduced below n, top up using tier weights
            remaining = [r for r in rows if r[0] not in seen]
            while len(result) < k and remaining:
                rem_weights = [TIER_WEIGHTS.get(r[2], 1) for r in remaining]
                (row,) = random.choices(remaining, weights=rem_weights, k=1)
                result.append({"chunk_id": row[0], "text": row[1], "tier": row[2]})
                remaining.remove(row)
                seen.add(row[0])
        else:
            chosen = random.sample(rows, min(n, len(rows)))
            result = [{"chunk_id": r[0], "text": r[1], "tier": r[2]} for r in chosen]
        return result

    def count(self) -> int:
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
