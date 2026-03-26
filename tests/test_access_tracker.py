# tests/test_access_tracker.py
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.access_tracker import compute_tiers, AccessTracker


def test_compute_tiers_frozen_first():
    now = int(time.time())
    chunks = [
        {"chunk_id": 1, "access_count": 0,  "last_accessed_ts": None},
        {"chunk_id": 2, "access_count": 10, "last_accessed_ts": now - 86400},
        {"chunk_id": 3, "access_count": 5,  "last_accessed_ts": now - 86400 * 40},
        {"chunk_id": 4, "access_count": 1,  "last_accessed_ts": now - 86400 * 40},
    ]
    tiers = compute_tiers(chunks)
    assert tiers[1] == "frozen"   # access_count == 0
    assert tiers[2] in ("hot", "warm")  # recently accessed, high count
    assert tiers[3] in ("cold", "warm")
    # no chunk should be both frozen and another tier
    assert len(set(tiers.values())) >= 2


def test_tracker_record_and_snapshot():
    tracker = AccessTracker()
    tracker.record(chunk_id=42, rank=1)
    tracker.record(chunk_id=42, rank=2)
    tracker.record(chunk_id=99, rank=3)
    snap = tracker.snapshot_and_clear()
    assert snap[42]["count"] == 2
    assert snap[99]["count"] == 1
    assert tracker.snapshot_and_clear() == {}  # cleared
