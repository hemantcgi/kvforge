"""Unit tests for kv_fulltoken injection on Gemma4 variable-head-dim architecture.

These tests run on any transformers version >= 5.2.0 with mock configs
matching Gemma4's real shapes (verified on EC2 with transformers 5.14.1).

Fixture: Gemma4-like config with 35 layers, num_kv_shared_layers=20,
head_dim=256, global_head_dim=512, 12 sliding + 3 full non-shared layers.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def gemma4_config():
    """Mock config matching google/gemma-4-E2B-it text_config."""

    class MockGemma4Config:
        num_hidden_layers = 35
        num_key_value_heads = 1
        num_attention_heads = 8
        head_dim = 256
        sliding_window = 512
        num_kv_shared_layers = 20
        _attn_implementation = "eager"

        layer_types = [
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
            "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
            "full_attention",
        ]

        def get_text_config(self, decoder=True):
            return self

    return MockGemma4Config()


@pytest.fixture
def gemma4_past_key_values(gemma4_config):
    """Simulated past_key_values for Gemma4: 15 entries, mixed head_dim."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=gemma4_config)
    for i in range(15):
        lt = gemma4_config.layer_types[i]
        hd = 512 if lt == "full_attention" else 256
        k = torch.randn(1, 1, 16, hd)
        v = torch.randn(1, 1, 16, hd)
        cache.update(k, v, i)
    return cache


# ── Test 1: Config-aware DynamicCache ────────────────────────────────────

def test_dynamic_cache_config_15_layers(gemma4_config):
    """DynamicCache(config=gemma4_config) creates 15 layers: 12 sliding + 3 full."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=gemma4_config)
    layers = cache.layers

    assert len(layers) == 15, f"Expected 15 layers, got {len(layers)}"

    n_sliding = sum(1 for l in layers if hasattr(l, "sliding_window"))
    n_full = len(layers) - n_sliding
    assert n_sliding == 12, f"Expected 12 sliding, got {n_sliding}"
    assert n_full == 3, f"Expected 3 full, got {n_full}"

    # Verify positions: full_attention at indices 4, 9, 14
    for i in [4, 9, 14]:
        assert not hasattr(layers[i], "sliding_window"), f"Layer {i} should be full_attention"
    for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]:
        assert hasattr(layers[i], "sliding_window"), f"Layer {i} should be sliding"


def test_dynamic_cache_sliding_window_value(gemma4_config):
    """Sliding layers have sliding_window=512."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=gemma4_config)
    for i in range(15):
        lt = gemma4_config.layer_types[i]
        sw = getattr(cache.layers[i], "sliding_window", None)
        if lt == "sliding_attention":
            assert sw == 512, f"Layer {i} sliding_window should be 512, got {sw}"
        else:
            assert sw is None, f"Layer {i} should not have sliding_window"


# ── Test 2: Per-layer head_dims in captured KV ───────────────────────────

def test_captured_kv_per_layer_head_dims(gemma4_past_key_values, gemma4_config):
    """The captured past_key_values has correct head_dim per layer."""
    for i, layer in enumerate(gemma4_past_key_values.layers):
        lt = gemma4_config.layer_types[i]
        expected_hd = 512 if lt == "full_attention" else 256
        k = layer.keys
        assert k.shape[-1] == expected_hd, (
            f"Layer {i} ({lt}): expected head_dim={expected_hd}, got {k.shape[-1]}"
        )
        assert k.shape[1] == 1, f"Layer {i}: expected num_kv_heads=1, got {k.shape[1]}"


# ── Test 3: compute_per_token_kv_as_list preserves variable head_dim ─────

def test_compute_per_token_kv_as_list_preserves_all_layers(gemma4_past_key_values, gemma4_config):
    """compute_per_token_kv_as_list returns 15 arrays at native head_dim (no dropping)."""
    from core.kv_utils import compute_per_token_kv_as_list

    kv_list = compute_per_token_kv_as_list(gemma4_past_key_values)
    assert len(kv_list) == 15, f"Expected 15 arrays, got {len(kv_list)}"

    for i, arr in enumerate(kv_list):
        lt = gemma4_config.layer_types[i]
        expected_hd = 512 if lt == "full_attention" else 256
        assert arr.shape == (2, 1, 16, expected_hd), (
            f"Layer {i}: expected shape (2, 1, 16, {expected_hd}), got {arr.shape}"
        )


def test_compute_per_token_kv_drops_variable_head_dim(gemma4_past_key_values, gemma4_config):
    """compute_per_token_kv (current) drops full_attention layers with mismatched head_dim."""
    from core.kv_utils import compute_per_token_kv

    arr = compute_per_token_kv(gemma4_past_key_values)
    # Should only keep the 12 sliding layers (256-dim)
    assert arr.shape[0] == 12, f"Expected 12 layers (256-dim only), got {arr.shape[0]}"
    assert arr.shape[-1] == 256, f"Expected head_dim=256, got {arr.shape[-1]}"


# ── Test 4: Injected cache layer shapes ──────────────────────────────────

def test_injected_cache_layer_shapes(gemma4_past_key_values, gemma4_config):
    """DynamicCache(config, ddp_cache_data=...) creates layers with correct shapes."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(
        config=gemma4_config,
        ddp_cache_data=[(layer.keys, layer.values) for layer in gemma4_past_key_values.layers],
    )
    assert len(cache.layers) == 15
    for i, layer in enumerate(cache.layers):
        lt = gemma4_config.layer_types[i]
        expected_hd = 512 if lt == "full_attention" else 256
        assert layer.keys.shape[-1] == expected_hd, (
            f"Layer {i}: injected head_dim={layer.keys.shape[-1]}, expected {expected_hd}"
        )


# ── Test 5: Multi-chunk concatenation shapes ─────────────────────────────

def test_multi_chunk_concatenation_shapes(gemma4_past_key_values, gemma4_config):
    """Concatenating two chunks along seq_len preserves all dims."""
    from transformers.cache_utils import DynamicCache

    # Two identical chunks concatenated along seq_len
    concat_kvs = []
    for i in range(15):
        k1 = gemma4_past_key_values.layers[i].keys
        v1 = gemma4_past_key_values.layers[i].values
        k = torch.cat([k1, k1], dim=2)
        v = torch.cat([v1, v1], dim=2)
        concat_kvs.append((k, v))

    cache = DynamicCache(config=gemma4_config, ddp_cache_data=concat_kvs)
    assert len(cache.layers) == 15
    for i, layer in enumerate(cache.layers):
        lt = gemma4_config.layer_types[i]
        expected_hd = 512 if lt == "full_attention" else 256
        assert layer.keys.shape == (1, 1, 32, expected_hd), (
            f"Layer {i}: shape={layer.keys.shape}, expected (1, 1, 32, {expected_hd})"
        )


# ── Test 6: Full attention_mask fix ──────────────────────────────────────

def test_full_attention_mask_prevents_empty_input():
    """Simulate how _prefill slices input_ids: full mask prevents empty slicing.

    In Prefill._prefill, if past_length > input_ids.shape[1] and
    input_ids.shape[1] == attention_mask.shape[1], it computes
    next_sequence_length = input_ids.shape[1] - past_length (negative).
    prepare_inputs_for_generation then slices input_ids[:, -next_seq_len:],
    producing empty input. A full attention_mask (past_len + query_len)
    avoids this path because input_ids.shape[1] != attention_mask.shape[1].
    """
    past_len = 44
    query_len = 12

    # Case 1: SHORT attention_mask (current broken behavior)
    short_mask = torch.ones(1, query_len)
    model_inputs_short = {
        "input_ids": torch.ones(1, query_len),
        "attention_mask": short_mask,
    }
    # Simulate _prefill logic
    if model_inputs_short["input_ids"].shape[1] == model_inputs_short["attention_mask"].shape[1]:
        next_sequence_length = query_len - past_len
    else:
        next_sequence_length = None

    assert next_sequence_length < 0, (
        f"Short mask: next_sequence_length should be negative ({next_sequence_length})"
    )
    # This would cause slicing input_ids[:, -(-32):] = input_ids[:, 32:] = empty
    # when prepare_inputs_for_generation does: input_ids[:, -next_sequence_length:]
    if next_sequence_length is not None and next_sequence_length < 0:
        sliced = model_inputs_short["input_ids"][:, -next_sequence_length:]
        assert sliced.shape[1] == 0, "Short mask produce empty input_ids after slicing"

    # Case 2: FULL attention_mask (the fix)
    full_mask = torch.ones(1, past_len + query_len)
    model_inputs_full = {
        "input_ids": torch.ones(1, query_len),
        "attention_mask": full_mask,
    }
    if model_inputs_full["input_ids"].shape[1] == model_inputs_full["attention_mask"].shape[1]:
        next_sequence_length = query_len - past_len  # Would be hit if equal
    else:
        next_sequence_length = None  # Correct: skip slicing when shapes differ

    assert next_sequence_length is None, "Full mask: should skip slicing logic"

    # Verify prepare_inputs_for_generation wouldn't slice
    # (it only slices when next_sequence_length is not None)
    sliced = model_inputs_full["input_ids"]
    if next_sequence_length is not None:
        sliced = sliced[:, -next_sequence_length:]
    assert sliced.shape[1] == query_len, (
        f"Full mask: input_ids should remain {query_len}, got {sliced.shape[1]}"
    )


# ── Test 7: rerotate_keys with matching inv_freq ───────────────────────

def test_rerotate_keys_basic():
    """rerotate_keys applies delta rotation without changing key shape."""
    from core.kv_utils import rerotate_keys

    keys = torch.randn(1, 1, 10, 256)
    inv_freq = torch.randn(128)
    delta = torch.randint(0, 100, (10,))
    result = rerotate_keys(keys, inv_freq, delta)
    assert result.shape == keys.shape, f"Shape changed: {result.shape} vs {keys.shape}"
    assert result.dtype == keys.dtype


def test_rerotate_keys_delta_zero():
    """rerotate_keys with delta=0 should be identity."""
    from core.kv_utils import rerotate_keys

    keys = torch.randn(1, 1, 10, 256)
    inv_freq = torch.randn(128)
    delta = torch.zeros(10, dtype=torch.long)
    result = rerotate_keys(keys, inv_freq, delta)
    torch.testing.assert_close(result, keys, msg="delta=0 rerotation should be identity")


# ── Test 8: No padding to 35 layers ─────────────────────────────────────

def test_no_padding_to_35(gemma4_past_key_values, gemma4_config):
    """Injected cache should have 15 entries, not 35 (no padding)."""
    from transformers.cache_utils import DynamicCache

    # This is the CORRECT behavior (15 entries, no padding)
    correct_cache = DynamicCache(
        config=gemma4_config,
        ddp_cache_data=[(layer.keys, layer.values) for layer in gemma4_past_key_values.layers],
    )
    assert len(correct_cache.layers) == 15

    # Simulate the BROKEN behavior (padding 15 -> 35 with empty tensors)
    all_kvs = [(layer.keys, layer.values) for layer in gemma4_past_key_values.layers]
    n_total = 35
    hd = all_kvs[0][0].shape[-1] if all_kvs else 256
    dt = all_kvs[0][0].dtype if all_kvs else torch.float16
    while len(all_kvs) < n_total:
        z = torch.zeros(1, 1, 0, hd, dtype=dt)
        all_kvs.append((z, z.clone()))

    # Without config: creates 35 layers (one per ddp_cache_data entry)
    broken_cache = DynamicCache(ddp_cache_data=all_kvs)
    assert len(broken_cache.layers) == 35
    empty_layers = sum(1 for l in broken_cache.layers if l.keys is not None and l.keys.shape[2] == 0)
    assert empty_layers == 20, "Should have 20 empty (seq_len=0) padding layers"

    # With config: only 15 layers are created; 35 ddp_cache_data entries cause IndexError
    # (15 layers exist, but update is called for entry 15 which doesn't exist)
    with pytest.raises(IndexError):
        DynamicCache(config=gemma4_config, ddp_cache_data=all_kvs)

    # The correct behavior: pass exactly 15 (non-shared) entries
    correct_entries = all_kvs[:15]
    correct_cache = DynamicCache(config=gemma4_config, ddp_cache_data=correct_entries)
    assert len(correct_cache.layers) == 15
    assert all(l.keys.shape[2] > 0 for l in correct_cache.layers)


# ── Test 9: Store_full_length_kv layer identification ───────────────────

def test_store_full_length_kv_layers(gemma4_config):
    """Layers 13 (sliding) and 14 (full) are the store_full_length_kv layers.

    Among non-shared layers (0-14, layer_types[:15]):
    - Last sliding before sharing starts is layer 13
    - Last full before sharing starts is layer 14
    These are the layers that save to shared_kv_states for shared layers (15-34).
    """
    from transformers.cache_utils import DynamicCache

    # Simulate the model's store_full_length_kv computation
    first_kv_shared = 35 - gemma4_config.num_kv_shared_layers  # = 15
    prev_layers = gemma4_config.layer_types[:first_kv_shared]

    store_full_indices = []
    for layer_type in set(prev_layers):
        idx = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
        store_full_indices.append(idx)

    assert 13 in store_full_indices, "Layer 13 should store full KV (last sliding)"
    assert 14 in store_full_indices, "Layer 14 should store full KV (last full_attention)"

    # Verify layer types at these indices
    assert gemma4_config.layer_types[13] == "sliding_attention"
    assert gemma4_config.layer_types[14] == "full_attention"
