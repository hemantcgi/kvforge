"""
tests/test_integration_smoke.py — Integration smoke tests (Task 15).

All 5 tests run locally without a GPU.
"""

import json
import random
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import core.version as ver
import core.kv_utils as kv_utils
from core.replay_buffer import ReplayBuffer
from pipeline.kv_inference import decide_inference_mode
from core.confidence_gate import compute_hedging_score, decide_gate


# ── Test 1: version round-trip ────────────────────────────────────────────

def test_version_roundtrip(tmp_path):
    """Write a version dict to a temp JSON file, verify it round-trips correctly."""
    tmpfile = tmp_path / "version_test.json"

    # Pre-write initial data so the file exists
    initial = {
        "current_lora_version": 7,
        "checkpoint_path": "/checkpoints/lora_v7",
        "phase": 2,
        "prs_history": [{"round": 1, "prs": 0.82}],
        "known_good_queries": ["what is qdrant?"],
    }
    tmpfile.write_text(json.dumps(initial))

    original_version_file = ver.VERSION_FILE
    try:
        # Point version module at temp file
        ver.init({"version_file": str(tmpfile)})

        # Save new data
        data_to_save = {
            "current_lora_version": 42,
            "checkpoint_path": "/tmp/lora_v42",
            "phase": 3,
            "prs_history": [{"round": 5, "prs": 0.95}],
            "known_good_queries": ["hello world"],
        }
        ver.save(data_to_save)

        # Load and assert round-trip
        loaded = ver.load()
        assert loaded["current_lora_version"] == 42
        assert loaded["checkpoint_path"] == "/tmp/lora_v42"
        assert loaded["phase"] == 3
        assert loaded["prs_history"] == [{"round": 5, "prs": 0.95}]
        assert loaded["known_good_queries"] == ["hello world"]
    finally:
        # Always restore VERSION_FILE to its original value
        ver.VERSION_FILE = original_version_file


# ── Test 2: kv_utils full round-trip ─────────────────────────────────────

def test_kv_utils_full_roundtrip():
    """Serialize and deserialize a float16 numpy array; values and dtype must survive."""
    rng = np.random.default_rng(seed=42)
    original = rng.standard_normal((2, 2, 4, 8)).astype(np.float16)

    b64 = kv_utils.serialize_kv(original)
    assert isinstance(b64, str), "serialize_kv should return a string"

    result = kv_utils.deserialize_kv(b64, shape=(2, 2, 4, 8))

    assert result.dtype == np.float16, f"Expected float16, got {result.dtype}"
    assert result.shape == (2, 2, 4, 8), f"Shape mismatch: {result.shape}"
    assert np.allclose(original, result), "Values did not survive round-trip"


# ── Test 3: ReplayBuffer weighted sampling ────────────────────────────────

def test_replay_buffer_weighted_sampling(tmp_path):
    """Hot chunks (weight 8) should be sampled far more often than frozen (weight 1)."""
    random.seed(42)
    db_path = tmp_path / "test_replay.db"
    buf = ReplayBuffer(db_path=str(db_path))

    # Add 1 hot chunk and 1 frozen chunk
    hot_chunk   = [{"chunk_id": 1, "text": "hot text",    "tier": "hot"}]
    frozen_chunk = [{"chunk_id": 2, "text": "frozen text", "tier": "frozen"}]
    buf.add_chunks(hot_chunk)
    buf.add_chunks(frozen_chunk)

    assert buf.count() == 2

    hot_count = 0
    frozen_count = 0
    samples = 200
    for _ in range(samples):
        result = buf.sample(1)
        assert len(result) == 1
        if result[0]["tier"] == "hot":
            hot_count += 1
        else:
            frozen_count += 1

    # With weights 8:1, hot should appear ~88% of the time over 200 draws
    assert hot_count > 150, (
        f"Expected hot_count > 150 (got {hot_count} hot, {frozen_count} frozen). "
        f"Weighted sampling may be broken."
    )

    # also exercise weight_by_tier=False
    uniform = buf.sample(2, weight_by_tier=False)
    assert len(uniform) == 2

    buf._con.close()


# ── Test 4: decide_inference_mode logic ──────────────────────────────────

def test_inference_mode_logic():
    """decide_inference_mode returns kv_injection only when all chunks are fresh + cached."""
    current_version = 5

    # All fresh chunks with valid kv_cache → kv_injection
    fresh_chunks = [
        {"chunk_id": 1, "kv_version": 5, "kv_cache": "abc123"},
        {"chunk_id": 2, "kv_version": 5, "kv_cache": "def456"},
    ]
    assert decide_inference_mode(fresh_chunks, current_version) == "kv_injection"

    # One stale chunk (kv_version < current) → text_fallback
    stale_chunks = [
        {"chunk_id": 1, "kv_version": 3, "kv_cache": "abc123"},
        {"chunk_id": 2, "kv_version": 5, "kv_cache": "def456"},
    ]
    assert decide_inference_mode(stale_chunks, current_version) == "text_fallback"

    # Chunk with kv_cache=None → text_fallback
    null_cache_chunks = [
        {"chunk_id": 1, "kv_version": 5, "kv_cache": None},
        {"chunk_id": 2, "kv_version": 5, "kv_cache": "def456"},
    ]
    assert decide_inference_mode(null_cache_chunks, current_version) == "text_fallback"

    # Chunk with kv_version missing entirely → text_fallback
    missing_version_chunks = [
        {"chunk_id": 1, "kv_cache": "abc123"},  # no kv_version key
    ]
    assert decide_inference_mode(missing_version_chunks, current_version) == "text_fallback"

    # Empty list → no stale chunks → kv_injection
    assert decide_inference_mode([], current_version) == "kv_injection"


# ── Test 5: gate pure logic ───────────────────────────────────────────────

def test_gate_pure_logic():
    """compute_hedging_score and decide_gate should behave correctly on known inputs."""

    # Hedging text should produce a score > 0
    hedging_score = compute_hedging_score("I think maybe this is correct")
    assert hedging_score > 0.0, (
        f"Expected hedging score > 0 for hedging text, got {hedging_score}"
    )

    # Confident text with no hedging markers should score exactly 0
    confident_score = compute_hedging_score("The answer is 42.")
    assert confident_score == 0.0, (
        f"Expected hedging score == 0.0 for confident text, got {confident_score}"
    )

    # Low entropy, no hedging, high similarity → should be "direct"
    # p = 0.4*(1-0.1) + 0.3*(1-0.0) + 0.3*0.9 = 0.36+0.30+0.27 = 0.93 >= 0.75
    result_direct = decide_gate(0.1, 0.0, 0.9, threshold=0.75)
    assert result_direct == "direct", f"Expected 'direct', got '{result_direct}'"

    # High entropy, high hedging, low similarity → should be "retrieve"
    # p = 0.4*(1-0.9) + 0.3*(1-0.8) + 0.3*0.1 = 0.04+0.06+0.03 = 0.13 < 0.75
    result_retrieve = decide_gate(0.9, 0.8, 0.1, threshold=0.75)
    assert result_retrieve == "retrieve", f"Expected 'retrieve', got '{result_retrieve}'"
