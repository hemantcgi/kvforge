"""
access_tracker.py — Thread-safe in-memory access counter + tier classifier.

Used directly by kv_background.py for the flush loop.
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
    def __init__(self):
        self._data: dict[int, _Counter] = {}
        self._lock = threading.Lock()

    def record(self, chunk_id: int, rank: int) -> None:
        with self._lock:
            if chunk_id not in self._data:
                self._data[chunk_id] = _Counter()
            c = self._data[chunk_id]
            c.count += 1
            c.rank_sum += rank
            c.last_ts = int(time.time())

    def record_parametric_hit(self, chunk_ids: list[int]) -> None:
        with self._lock:
            for cid in chunk_ids:
                if cid not in self._data:
                    self._data[cid] = _Counter()
                self._data[cid].parametric_hits += 1

    def snapshot_and_clear(self) -> dict[int, dict]:
        with self._lock:
            snap = {
                cid: {"count": c.count, "rank_sum": c.rank_sum,
                       "last_ts": c.last_ts, "parametric_hits": c.parametric_hits}
                for cid, c in self._data.items()
            }
            self._data.clear()
        return snap

    def query_count(self) -> int:
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
