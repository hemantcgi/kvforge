"""Thread-safe in-memory access counter and tier classifier for document chunks.

``AccessTracker`` accumulates retrieval events in memory so that the
background flush loop (``kv_background.py``) can batch-write them to the
vector store rather than making one round-trip per query.

``compute_tiers`` classifies a list of chunk metadata dicts into four
performance tiers based on access frequency and recency.

``generate_report`` produces a JSON summary report for the monitoring
dashboard.
"""

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class _Counter:
    count: int = 0
    rank_sum: float = 0.0
    last_ts: int = 0
    parametric_hits: int = 0


class AccessTracker:
    """Thread-safe in-memory store for chunk access counts and retrieval ranks.

    All mutations are protected by an internal ``threading.Lock``.
    Call ``snapshot_and_clear`` from the background flush thread to atomically
    retrieve and reset the accumulated data.
    """

    def __init__(self):
        self._data: dict[int, _Counter] = {}
        self._lock = threading.Lock()

    def record(self, chunk_id: int, rank: int) -> None:
        """Record a retrieval event for *chunk_id* at position *rank*.

        Args:
            chunk_id: Identifier of the retrieved chunk.
            rank: 1-based position in the retrieval result list (1 = top hit).
        """
        with self._lock:
            if chunk_id not in self._data:
                self._data[chunk_id] = _Counter()
            c = self._data[chunk_id]
            c.count += 1
            c.rank_sum += rank
            c.last_ts = int(time.time())

    def record_parametric_hit(self, chunk_ids: list[int]) -> None:
        """Increment the parametric-hit counter for a list of chunk IDs.

        Called by the confidence gate when a query is answered directly from
        model weights (no retrieval), but the chunks that *would have* been
        retrieved are tracked for statistics.

        Args:
            chunk_ids: List of chunk identifiers to credit with a parametric hit.
        """
        with self._lock:
            for cid in chunk_ids:
                if cid not in self._data:
                    self._data[cid] = _Counter()
                self._data[cid].parametric_hits += 1

    def snapshot_and_clear(self) -> dict[int, dict]:
        """Atomically snapshot and clear all accumulated access data.

        Returns:
            Dict mapping ``chunk_id`` to a dict with keys ``count``,
            ``rank_sum``, ``last_ts``, and ``parametric_hits``.
            The internal buffer is emptied before returning.
        """
        with self._lock:
            snap = {
                cid: {"count": c.count, "rank_sum": c.rank_sum,
                       "last_ts": c.last_ts, "parametric_hits": c.parametric_hits}
                for cid, c in self._data.items()
            }
            self._data.clear()
        return snap

    def query_count(self) -> int:
        """Return the total number of retrieval events recorded since the last clear.

        Returns:
            Sum of ``count`` across all tracked chunks.
        """
        with self._lock:
            return sum(c.count for c in self._data.values())


def compute_tiers(chunks: list[dict]) -> dict[int, str]:
    """
    Classify each chunk into hot/warm/cold/frozen.
    Rules applied in order (first match wins):
      frozen : access_count == 0
      hot    : top 15% of non-frozen AND accessed within 7 days
      warm   : next 50% of non-frozen AND accessed within 30 days
      cold   : all remaining non-frozen
    """
    now = int(time.time())
    result: dict[int, str] = {}

    frozen_ids = {c["chunk_id"] for c in chunks if c.get("access_count", 0) == 0}
    for c in chunks:
        if c["chunk_id"] in frozen_ids:
            result[c["chunk_id"]] = "frozen"

    non_frozen = sorted(
        [c for c in chunks if c["chunk_id"] not in frozen_ids],
        key=lambda c: c.get("access_count", 0),
        reverse=True,
    )
    n = len(non_frozen)
    hot_cutoff  = max(1, int(n * 0.15))
    warm_cutoff = max(1, int(n * 0.65))  # top 15% + next 50%

    for i, c in enumerate(non_frozen):
        last_ts = c.get("last_accessed_ts") or 0
        age_days = (now - last_ts) / 86400 if last_ts else 999

        if i < hot_cutoff and age_days <= 7:
            result[c["chunk_id"]] = "hot"
        elif i < warm_cutoff and age_days <= 30:
            result[c["chunk_id"]] = "warm"
        else:
            result[c["chunk_id"]] = "cold"

    return result


def generate_report(chunks: list[dict], parametric_rate: float,
                     output_path: str = "access_report.json") -> None:
    """Generate a JSON access report and write it to *output_path*.

    The report contains tier distribution counts, the parametric answer rate,
    the most-accessed pages, and a sample of frozen chunk IDs.

    Args:
        chunks: List of chunk metadata dicts (from the vector store payload),
            each expected to contain ``chunk_id``, ``access_count``,
            ``last_accessed_ts``, and optionally ``page``.
        parametric_rate: Fraction of queries answered parametrically in [0, 1].
        output_path: Destination path for the JSON report file.
    """
    tiers = compute_tiers(chunks)
    counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}
    frozen_ids = []
    for cid, tier in tiers.items():
        counts[tier] += 1
        if tier == "frozen":
            frozen_ids.append(cid)

    # Most accessed pages
    page_counts: dict[int, int] = {}
    for c in chunks:
        page = c.get("page", 0)
        page_counts[page] = page_counts.get(page, 0) + c.get("access_count", 0)
    top_pages = sorted(page_counts, key=page_counts.get, reverse=True)[:5]

    report = {
        "generated_at": int(time.time()),
        "summary": {**counts, "total": len(chunks)},
        "parametric_answer_rate": round(parametric_rate, 4),
        "most_accessed_pages": top_pages,
        "frozen_chunk_ids": frozen_ids[:50],  # cap to first 50
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Access report written to {output_path}")
