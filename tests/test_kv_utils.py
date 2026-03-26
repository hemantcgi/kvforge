# tests/test_kv_utils.py
import numpy as np
import torch
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.kv_utils import mean_pool_kv, serialize_kv, deserialize_kv, stack_past_key_values

NUM_LAYERS = 4   # use small values for tests
NUM_KV_HEADS = 2
HEAD_DIM = 8
SEQ_LEN = 16


def make_fake_past_key_values(seq_len=SEQ_LEN):
    """Return HuggingFace-style past_key_values: tuple of (K,V) per layer."""
    return tuple(
        (
            torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM),
            torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM),
        )
        for _ in range(NUM_LAYERS)
    )


def test_mean_pool_kv_shape():
    pkv = make_fake_past_key_values()
    result = mean_pool_kv(pkv)
    assert result.shape == (NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM)


def test_mean_pool_kv_dtype():
    pkv = make_fake_past_key_values()
    result = mean_pool_kv(pkv)
    assert result.dtype == np.float16


def test_serialize_deserialize_roundtrip():
    pkv = make_fake_past_key_values()
    arr = mean_pool_kv(pkv)
    b64 = serialize_kv(arr)
    assert isinstance(b64, str)
    restored = deserialize_kv(b64, shape=(NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM))
    np.testing.assert_array_almost_equal(arr.astype(np.float32),
                                          restored.astype(np.float32), decimal=2)


def test_stack_past_key_values_shape():
    from core.kv_utils import _iter_kv_layers
    chunks_kv = [mean_pool_kv(make_fake_past_key_values()) for _ in range(3)]
    pkv = stack_past_key_values(chunks_kv, num_layers=NUM_LAYERS,
                                 num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM)
    layers = list(_iter_kv_layers(pkv))
    assert len(layers) == NUM_LAYERS
    k, v = layers[0]
    # 3 chunks → seq_len=3 (one mean-pooled position per chunk)
    assert k.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
    assert v.shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
