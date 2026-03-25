# tests/test_lora_trainer.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_buffer import ReplayBuffer


def test_add_and_sample(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([
        {"chunk_id": 1, "text": "alpha", "tier": "hot"},
        {"chunk_id": 2, "text": "beta",  "tier": "warm"},
        {"chunk_id": 3, "text": "gamma", "tier": "cold"},
        {"chunk_id": 4, "text": "delta", "tier": "frozen"},
    ])
    samples = rb.sample(n=3, weight_by_tier=True)
    assert len(samples) == 3
    assert all("text" in s for s in samples)


def test_sample_respects_available_count(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([{"chunk_id": 1, "text": "only one", "tier": "hot"}])
    samples = rb.sample(n=10, weight_by_tier=True)
    assert len(samples) == 1  # can't return more than available


def test_update_tier(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    rb.add_chunks([{"chunk_id": 99, "text": "test", "tier": "cold"}])
    rb.update_tier(99, "hot")
    row = rb._con.execute("SELECT tier FROM chunks WHERE chunk_id=99").fetchone()
    assert row[0] == "hot"


def test_evict_to_cap(tmp_path):
    db = tmp_path / "replay.db"
    rb = ReplayBuffer(db_path=str(db))
    # Add 6 chunks with mixed tiers; cap at 4
    rb.add_chunks([
        {"chunk_id": i, "text": f"text {i}", "tier": "frozen"} for i in range(4)
    ] + [
        {"chunk_id": 10, "text": "hot", "tier": "hot"},
        {"chunk_id": 11, "text": "warm", "tier": "warm"},
    ], max_size=4)
    # Cap applied: 2 frozen (lowest-value) should be evicted
    assert rb.count() == 4
    # hot and warm chunks must survive
    tiers = {row[0]: row[1] for row in
             rb._con.execute("SELECT chunk_id, tier FROM chunks").fetchall()}
    assert tiers.get(10) == "hot"
    assert tiers.get(11) == "warm"


def test_fetch_chunks_for_source_filter():
    """fetch_chunks_for_source builds the correct Qdrant filter."""
    from unittest.mock import MagicMock, patch
    from pipeline.lora_trainer import fetch_chunks_for_source

    mock_client = MagicMock()
    mock_client.scroll.return_value = ([], None)   # empty collection is fine
    chunks = fetch_chunks_for_source(mock_client, "my_coll", "guide.pdf")
    assert isinstance(chunks, list)
    # Verify the filter was applied
    call_kwargs = mock_client.scroll.call_args.kwargs
    assert call_kwargs["collection_name"] == "my_coll"
    must_cond = call_kwargs["scroll_filter"].must[0]
    assert must_cond.key == "source_file"


def test_main_help():
    """lora_trainer --help exits cleanly."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.lora_trainer", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--source-file" in result.stdout
