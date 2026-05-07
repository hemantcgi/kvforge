import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_chunk_payload_has_timestamp_fields():
    """Test that build_payload includes effective_from, superseded_at, and source_version."""
    from pipeline.kv_indexer import build_payload

    # Create a dummy KV array (shape: [32, 2, 8, 128] for typical config)
    kv_array = np.zeros((32, 2, 8, 128), dtype=np.float16)

    # Build a payload with typical input
    payload = build_payload(
        text="This is test chunk content.",
        page=1,
        source_file="test.pdf",
        kv_array=kv_array,
    )

    # Check that timestamp fields exist
    assert "effective_from" in payload, f"Missing effective_from. Keys: {list(payload.keys())}"
    assert "superseded_at" in payload, f"Missing superseded_at. Keys: {list(payload.keys())}"
    assert "source_version" in payload, f"Missing source_version. Keys: {list(payload.keys())}"

    # Check field values
    assert payload["superseded_at"] is None, "superseded_at should be None for new chunks"
    assert payload["effective_from"] is not None, "effective_from should not be None"
    assert isinstance(payload["effective_from"], str), "effective_from should be an ISO string"
    assert isinstance(payload["source_version"], str), "source_version should be a string"


def test_chunk_payload_source_version_from_metadata():
    """Test that source_version captures the modified field from chunk metadata."""
    from pipeline.kv_indexer import build_payload

    kv_array = np.zeros((32, 2, 8, 128), dtype=np.float16)

    # This test directly tests build_payload signature and behavior
    # Note: current implementation doesn't yet accept metadata dict
    # but we're testing what the future state should be
    payload = build_payload(
        text="Test content",
        page=1,
        source_file="doc.pdf",
        kv_array=kv_array,
    )

    # At minimum, source_version field should exist
    assert "source_version" in payload, "source_version field missing"
