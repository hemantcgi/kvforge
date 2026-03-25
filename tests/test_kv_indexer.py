# tests/test_kv_indexer.py
"""
Tests for kv_indexer.py.
Uses a mock model so GPU is not required for unit tests.
"""
import json
import numpy as np
import pytest
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_model_outputs(num_layers=28, num_kv_heads=8, head_dim=128, seq_len=10):
    """Return a mock HuggingFace model output with past_key_values."""
    import torch
    past_key_values = tuple(
        (
            torch.randn(1, num_kv_heads, seq_len, head_dim),
            torch.randn(1, num_kv_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )
    mock_out = MagicMock()
    mock_out.past_key_values = past_key_values
    return mock_out


def test_compute_kv_for_chunk_shape():
    import torch
    from pipeline.kv_indexer import compute_kv_for_chunk

    # MagicMock supports context-manager protocol natively (__enter__/__exit__)
    # so torch.no_grad() inside compute_kv_for_chunk works without patching.
    mock_model = MagicMock()
    mock_model.device = "cpu"
    # Setting return_value on a MagicMock makes calling mock_model(...) return this.
    mock_model.return_value = make_mock_model_outputs(
        num_layers=4, num_kv_heads=2, head_dim=8, seq_len=10
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # tiny tokenizer, no GPU needed
    tokenizer.pad_token = tokenizer.eos_token

    # Pass mock directly — compute_kv_for_chunk takes (text, model, tokenizer, ...)
    arr = compute_kv_for_chunk("test text", mock_model, tokenizer,
                                num_layers=4, num_kv_heads=2, head_dim=8)
    assert arr.shape == (4, 2, 2, 8)
    assert arr.dtype == np.float16


def test_kv_indexer_payload_keys():
    """chunk_to_payload must include kv_cache and kv_version=null."""
    from pipeline.kv_indexer import build_payload
    fake_kv = np.zeros((4, 2, 2, 8), dtype=np.float16)
    payload = build_payload(
        text="hello world",
        page=1,
        source_file="test.pdf",
        kv_array=fake_kv,
    )
    assert "kv_cache" in payload
    assert payload["kv_version"] is None
    assert payload["source_file"] == "test.pdf"
    assert payload["access_count"] == 0
    assert payload["tier"] == "frozen"
