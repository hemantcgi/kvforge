"""
kv_utils.py — KV tensor utilities shared by SP1, SP2, SP3.

Operations:
  mean_pool_kv   — compress past_key_values (per-token) → fixed-size array
  serialize_kv   — numpy float16 array → base64 string (for Qdrant payload)
  deserialize_kv — base64 string → numpy float16 array
  stack_past_key_values — list of chunk KV arrays → HuggingFace past_key_values
"""

import base64
import numpy as np
import torch


def _iter_kv_layers(past_key_values):
    """
    Iterate over (k, v) pairs for each layer, supporting both the legacy tuple
    format (transformers < 4.47) and the DynamicCache format (transformers >= 5.x).

    Yields: (k, v) tensors each shaped [1, num_kv_heads, seq_len, head_dim]
    """
    # transformers >= 5.x: DynamicCache with .layers list of DynamicLayer objects
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
    else:
        # Legacy format: tuple of (k, v) tuples
        for k, v in past_key_values:
            yield k, v


def mean_pool_kv(past_key_values) -> np.ndarray:
    """
    Compress HuggingFace past_key_values to a fixed-size float16 array.

    Input:  past_key_values — either legacy tuple of (K, V) per layer, or
            a DynamicCache object (transformers >= 5.x).
            Each K/V tensor: [1, num_kv_heads, seq_len, head_dim]
    Output: np.ndarray [num_layers, 2, num_kv_heads, head_dim] float16
            (mean-pooled over seq_len dimension — one representative vector per chunk)
    """
    pooled = []
    for k, v in _iter_kv_layers(past_key_values):
        # k, v: [1, num_kv_heads, seq_len, head_dim]
        k = k.squeeze(0)          # [num_kv_heads, seq_len, head_dim]
        v = v.squeeze(0)          # [num_kv_heads, seq_len, head_dim]
        k_pooled = k.mean(dim=1)  # mean over seq_len → [num_kv_heads, head_dim]
        v_pooled = v.mean(dim=1)  # mean over seq_len → [num_kv_heads, head_dim]
        pooled.append(torch.stack([k_pooled, v_pooled]))  # [2, num_kv_heads, head_dim]
    result = torch.stack(pooled)  # [num_layers, 2, num_kv_heads, head_dim]
    return result.cpu().to(torch.float16).numpy()


def serialize_kv(arr: np.ndarray) -> str:
    """Float16 numpy array → base64 string for Qdrant payload storage."""
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def deserialize_kv(b64: str, shape: tuple) -> np.ndarray:
    """Base64 string → float16 numpy array with given shape."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape)


def stack_past_key_values(
    chunks_kv: list,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
):
    """
    Convert list of per-chunk KV arrays into a HuggingFace Cache object.

    Input:  list of N arrays, each [num_layers, 2, num_kv_heads, head_dim]
    Output: DynamicCache (transformers >= 5.x) with N "positions" per layer.
            Falls back to legacy tuple of (K, V) pairs for older transformers.

    Each layer's K/V is shaped [1, num_kv_heads, N, head_dim] where N is the
    number of chunks (they become N virtual token positions).
    """
    if not chunks_kv:
        raise ValueError("chunks_kv must be non-empty")
    for arr in chunks_kv:
        assert arr.shape == (num_layers, 2, num_kv_heads, head_dim), (
            f"Expected chunk shape ({num_layers}, 2, {num_kv_heads}, {head_dim}), got {arr.shape}"
        )

    # Build per-layer (k, v) pairs
    layer_kvs = []
    for layer_idx in range(num_layers):
        ks, vs = [], []
        for chunk_arr in chunks_kv:
            # chunk_arr[layer_idx]: [2, num_kv_heads, head_dim]
            layer = torch.from_numpy(chunk_arr[layer_idx].astype(np.float16))
            ks.append(layer[0])  # [num_kv_heads, head_dim]
            vs.append(layer[1])  # [num_kv_heads, head_dim]
        # stack along new seq dim: [num_kv_heads, N, head_dim] → unsqueeze batch
        k = torch.stack(ks, dim=1).unsqueeze(0)  # [1, num_kv_heads, N, head_dim]
        v = torch.stack(vs, dim=1).unsqueeze(0)  # [1, num_kv_heads, N, head_dim]
        layer_kvs.append((k, v))

    # Wrap in DynamicCache for transformers >= 5.x
    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache(ddp_cache_data=layer_kvs)
    except (ImportError, TypeError):
        # Older transformers: return legacy tuple format
        return tuple(layer_kvs)
