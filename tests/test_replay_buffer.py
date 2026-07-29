"""Tests for the KDS-aware replay buffer."""
import random

from core.replay_buffer import ReplayBuffer, TIER_WEIGHTS


def test_kds_boost_low_kds_sampled_more(tmp_path):
    """Low-KDS chunks receive a higher sampling weight than high-KDS chunks."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    # Same tier so only the KDS multiplier drives the difference.
    rb.add_chunks([
        {"chunk_id": 1, "text": "high kds", "tier": "hot"},
        {"chunk_id": 2, "text": "low kds", "tier": "hot"},
    ])
    kds_map = {1: 0.9, 2: 0.2}
    random.seed(42)
    counts = {1: 0, 2: 0}
    for _ in range(1000):
        sample = rb.sample(n=1, weight_by_tier=True, kds_map=kds_map)
        counts[sample[0]["chunk_id"]] += 1
    # hot weight=8; chunk 2 gets 4x boost vs chunk 1's 1x -> ~4:1 ratio.
    assert counts[2] > counts[1] * 3


def test_kds_missing_leaves_tier_weight_unchanged(tmp_path):
    """Chunks absent from kds_map keep their tier weight."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "hot", "tier": "hot"},
        {"chunk_id": 2, "text": "warm", "tier": "warm"},
    ])
    # Only chunk 1 has a KDS entry; chunk 2 should use warm=4 unchanged.
    kds_map = {1: 0.9}
    random.seed(42)
    counts = {1: 0, 2: 0}
    for _ in range(1000):
        sample = rb.sample(n=1, weight_by_tier=True, kds_map=kds_map)
        counts[sample[0]["chunk_id"]] += 1
    # Expected ratio hot:warm = 8:4 = 2:1.
    assert counts[1] > counts[2]


def test_kds_weight_by_tier_false_unaffected(tmp_path):
    """Uniform sampling ignores the kds_map entirely."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "hot", "tier": "hot"},
        {"chunk_id": 2, "text": "warm", "tier": "warm"},
    ])
    # Even with extreme KDS weights, uniform sampling should split ~50/50.
    kds_map = {1: 0.2, 2: 0.2}
    random.seed(42)
    counts = {1: 0, 2: 0}
    for _ in range(1000):
        sample = rb.sample(n=1, weight_by_tier=False, kds_map=kds_map)
        counts[sample[0]["chunk_id"]] += 1
    assert 400 < counts[1] < 600
    assert 400 < counts[2] < 600


def test_kds_defaults_to_no_change_when_kds_map_missing(tmp_path):
    """Without kds_map the existing tier-weighted behavior is preserved."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "hot", "tier": "hot"},
        {"chunk_id": 2, "text": "warm", "tier": "warm"},
    ])
    random.seed(42)
    counts = {1: 0, 2: 0}
    for _ in range(1000):
        sample = rb.sample(n=1, weight_by_tier=True)
        counts[sample[0]["chunk_id"]] += 1
    # Expected ratio hot:warm = 8:4 = 2:1.
    assert counts[1] > counts[2]


def test_add_kds_and_get_kds_map(tmp_path):
    """KDS scores can be stored and retrieved as a chunk_id->kds mapping."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "hot", "tier": "hot"},
        {"chunk_id": 2, "text": "warm", "tier": "warm"},
    ])
    rb.add_kds(1, 0.2)
    rb.add_kds(2, 0.8)
    assert rb.get_kds_map() == {1: 0.2, 2: 0.8}


def test_add_kds_preserved_on_add_chunks(tmp_path):
    """Re-inserting a chunk must not wipe its KDS score."""
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([{"chunk_id": 1, "text": "hot", "tier": "hot"}])
    rb.add_kds(1, 0.2)
    rb.add_chunks([{"chunk_id": 1, "text": "hot updated", "tier": "warm"}])
    assert rb.get_kds_map()[1] == 0.2
    row = rb._con.execute("SELECT text, tier FROM chunks WHERE chunk_id=1").fetchone()
    assert row == ("hot updated", "warm")


def test_kds_multiplier_defaults():
    """Default multiplier thresholds match the spec: <0.4 -> 4x, <0.7 -> 2x, else 1x."""
    rb = ReplayBuffer(db_path=":memory:")
    assert rb._kds_multiplier(None) == 1.0
    assert rb._kds_multiplier(0.9) == 1.0
    assert rb._kds_multiplier(0.7) == 1.0
    assert rb._kds_multiplier(0.69) == 2.0
    assert rb._kds_multiplier(0.4) == 2.0
    assert rb._kds_multiplier(0.39) == 4.0
    assert rb._kds_multiplier(0.0) == 4.0


def test_kds_thresholds_configurable():
    """Custom threshold ranges are honored."""
    rb = ReplayBuffer(db_path=":memory:", kds_thresholds=[
        (0.0, 0.5, 3.0),
        (0.5, 1.0, 1.5),
    ])
    assert rb._kds_multiplier(0.49) == 3.0
    assert rb._kds_multiplier(0.5) == 1.5
    assert rb._kds_multiplier(0.9) == 1.5


def test_kds_migration_adds_column_to_existing_db(tmp_path):
    """Opening an existing buffer without a kds column adds it transparently."""
    db = tmp_path / "replay.db"
    import sqlite3
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT, tier TEXT)")
    con.commit()
    con.close()

    rb = ReplayBuffer(db_path=str(db))
    rb.add_kds(1, 0.5)
    assert rb.get_kds_map()[1] == 0.5


def test_kds_row_weight_formula(tmp_path):
    """Effective weight is tier weight multiplied by KDS multiplier."""
    rb = ReplayBuffer(db_path=":memory:")
    assert rb._row_weight((1, "text", "hot"), {1: 0.2}) == TIER_WEIGHTS["hot"] * 4.0
    assert rb._row_weight((1, "text", "hot"), {1: 0.5}) == TIER_WEIGHTS["hot"] * 2.0
    assert rb._row_weight((1, "text", "hot"), {1: 0.9}) == TIER_WEIGHTS["hot"] * 1.0
    assert rb._row_weight((2, "text", "hot"), {1: 0.2}) == TIER_WEIGHTS["hot"] * 1.0
