"""Tests for the ReplayBuffer KDS feedback API (add_kds / get_kds_map / kds-weighted sample).

These cover the API that pipeline/lora_trainer.py's KDS feedback loop calls
(get_kds_map, add_kds, sample(..., kds_map=...)). Before this fix those methods
did not exist and the trainer crashed with AttributeError on every run.
"""
import random

from core.replay_buffer import ReplayBuffer


def _buf(tmp_path, name="rb.db"):
    return ReplayBuffer(db_path=str(tmp_path / name))


def test_get_kds_map_empty_by_default(tmp_path):
    rb = _buf(tmp_path)
    assert rb.get_kds_map() == {}


def test_add_kds_then_get_kds_map_roundtrip(tmp_path):
    rb = _buf(tmp_path)
    rb.add_kds(1, 0.25)
    rb.add_kds(2, 0.80)
    assert rb.get_kds_map() == {1: 0.25, 2: 0.80}


def test_add_kds_is_upsert(tmp_path):
    rb = _buf(tmp_path)
    rb.add_kds(1, 0.1)
    rb.add_kds(1, 0.9)  # overwrite
    assert rb.get_kds_map() == {1: 0.9}


def test_add_kds_independent_of_chunk_presence(tmp_path):
    # The trainer imports KDS before the chunk rows exist; add_kds must not require the chunk.
    rb = _buf(tmp_path)
    rb.add_kds(42, 0.5)
    assert rb.get_kds_map() == {42: 0.5}
    assert rb.count() == 0  # no chunk rows created


def test_sample_accepts_kds_map_kwarg(tmp_path):
    rb = _buf(tmp_path)
    rb.add_chunks([{"chunk_id": i, "text": f"c{i}", "tier": "warm"} for i in range(5)])
    out = rb.sample(n=3, weight_by_tier=True, kds_map={0: 0.1, 1: 0.9})
    assert len(out) == 3
    assert all("chunk_id" in c for c in out)


def test_sample_kds_map_none_is_backward_compatible(tmp_path):
    rb = _buf(tmp_path)
    rb.add_chunks([{"chunk_id": i, "text": f"c{i}", "tier": "warm"} for i in range(4)])
    random.seed(0)
    a = rb.sample(n=4, weight_by_tier=True)
    random.seed(0)
    b = rb.sample(n=4, weight_by_tier=True, kds_map=None)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_sample_boosts_low_kds_chunks(tmp_path):
    # Two same-tier chunks: one under-differentiated (low KDS) should be drawn
    # more often than the well-differentiated (high KDS) one.
    rb = _buf(tmp_path)
    rb.add_chunks([
        {"chunk_id": 1, "text": "low", "tier": "warm"},
        {"chunk_id": 2, "text": "high", "tier": "warm"},
    ])
    kds_map = {1: 0.05, 2: 0.95}
    random.seed(1234)
    low = high = 0
    for _ in range(400):
        picked = rb.sample(n=1, weight_by_tier=True, kds_map=kds_map)
        if picked and picked[0]["chunk_id"] == 1:
            low += 1
        elif picked:
            high += 1
    assert low > high, f"low-KDS chunk should be sampled more (low={low}, high={high})"
