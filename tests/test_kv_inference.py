# tests/test_kv_inference.py
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_chunk(kv_version, chunk_id=1):
    import core.kv_utils as kv_utils
    fake_kv = np.zeros((28, 2, 8, 128), dtype=np.float16)
    return {
        "chunk_id": chunk_id,
        "text": "Amazon Bedrock is a managed service.",
        "page": 7,
        "score": 0.9,
        "kv_cache": kv_utils.serialize_kv(fake_kv),
        "kv_version": kv_version,
    }


def test_all_fresh_uses_kv_path():
    """When all chunks have current kv_version, should call generate_with_kv."""
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(kv_version=5, chunk_id=i) for i in range(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "kv_injection"


def test_any_stale_uses_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5), _fake_chunk(None), _fake_chunk(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "text_fallback"


def test_stale_chunks_are_queued():
    from pipeline.kv_inference import decide_inference_mode, get_stale_chunk_ids
    chunks = [_fake_chunk(5), _fake_chunk(None, 2), _fake_chunk(3, 3)]
    stale = get_stale_chunk_ids(chunks, current_lora_version=5)
    assert set(stale) == {2, 3}


def test_kv_stacking_produces_correct_past_key_values_shape():
    """stack_past_key_values must produce HuggingFace-compatible past_key_values."""
    import core.kv_utils as kv_utils
    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, N_CHUNKS = 28, 8, 128, 5
    # Simulate 5 fresh chunks
    chunks = [_fake_chunk(kv_version=3, chunk_id=i) for i in range(N_CHUNKS)]
    chunk_arrs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=(NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM))
        for c in chunks
    ]
    pkv = kv_utils.stack_past_key_values(chunk_arrs, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    layers = list(kv_utils._iter_kv_layers(pkv))
    assert len(layers) == NUM_LAYERS
    k, v = layers[0]
    # [batch=1, num_kv_heads, N_chunks, head_dim]
    assert k.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)
    assert v.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)
