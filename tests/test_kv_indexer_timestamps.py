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
    """Test that source_version is populated from chunk metadata when modified is present."""
    from pipeline.kv_indexer import build_payload

    kv_array = np.zeros((32, 2, 8, 128), dtype=np.float16)

    # Test with source_version explicitly passed
    payload = build_payload(
        text="Test content",
        page=1,
        source_file="doc.pdf",
        kv_array=kv_array,
        source_version="2026-01-15T10:30:00+00:00",
    )

    assert "source_version" in payload, "source_version field missing"
    assert payload["source_version"] == "2026-01-15T10:30:00+00:00", (
        f"source_version should be '2026-01-15T10:30:00+00:00', got '{payload['source_version']}'"
    )


def test_effective_from_is_utc_iso8601():
    """Test that effective_from is a UTC ISO 8601 datetime string."""
    from pipeline.kv_indexer import build_payload
    from datetime import datetime

    kv_array = np.zeros((32, 2, 8, 128), dtype=np.float16)

    payload = build_payload(
        text="Test content",
        page=1,
        source_file="doc.pdf",
        kv_array=kv_array,
        source_version="",
    )

    # Should be parseable as ISO 8601 datetime
    dt = datetime.fromisoformat(payload["effective_from"])
    # Check that it's timezone-aware (UTC)
    assert dt.tzinfo is not None, "effective_from must be timezone-aware (UTC)"
    assert dt.tzname() == "UTC", f"effective_from should be UTC, got {dt.tzname()}"
