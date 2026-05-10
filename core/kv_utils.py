"""KV-cache tensor utilities shared by the indexing and inference pipelines.

Provides helpers to compress, serialise, deserialise, and reassemble the
HuggingFace ``past_key_values`` KV cache so it can be stored in a vector
store payload and later injected back into the model during inference.

Key functions:

* ``mean_pool_kv``        — compress per-token KV tensors → fixed-size array
* ``serialize_kv``        — numpy float16 array → base64 string (for payload storage)
* ``deserialize_kv``      — base64 string → numpy float16 array
* ``stack_past_key_values`` — list of chunk KV arrays → HuggingFace cache object
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
    """Serialise a float16 numpy KV array to a base64 ASCII string.

    The array is cast to ``float16`` before encoding to halve storage size.
    The result is safe to store in JSON-based payload fields.

    Args:
        arr: KV array of shape ``[num_layers, 2, num_kv_heads, head_dim]``
            (or any shape compatible with float16 storage).

    Returns:
        Base64-encoded ASCII string representing the raw bytes of the array.
    """
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def deserialize_kv(b64: str, shape: tuple) -> np.ndarray:
    """Deserialise a base64 ASCII string back to a float16 numpy array.

    Args:
        b64: Base64-encoded string produced by ``serialize_kv``.
        shape: Expected array shape, e.g.
            ``(num_layers, 2, num_kv_heads, head_dim)``.

    Returns:
        A writeable float16 numpy array of the specified shape.
    """
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float16).copy().reshape(shape)


def compute_per_token_kv(past_key_values) -> np.ndarray:
    """Preserve full token sequence from HuggingFace past_key_values.

    Input:  past_key_values — DynamicCache or legacy tuple.
            Each K/V tensor: [1, num_kv_heads, seq_len, head_dim]
    Output: np.ndarray [num_layers, 2, num_kv_heads, seq_len, head_dim] float16
    """
    layers = []
    for k, v in _iter_kv_layers(past_key_values):
        k = k.squeeze(0)
        v = v.squeeze(0)
        layers.append(torch.stack([k, v]))
    result = torch.stack(layers)
    return result.cpu().to(torch.float16).numpy()


def save_token_kv(arr: np.ndarray, path, tq_config=None) -> None:
    """Save per-token KV array to disk.

    Args:
        arr:       [num_layers, 2, num_kv_heads, seq_len, head_dim] float16
        path:      file path (str or Path)
        tq_config: TurboQuantConfig or None; if provided applies TurboQuant compression
    """
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if tq_config is None:
        np.savez_compressed(str(path), arr=arr, compressed=np.array(False))
        return

    from addons.turboquant.quantizer import TurboQuantKeyCodec, GroupValueCodec
    num_layers, _, num_heads, seq_len, head_dim = arr.shape
    kc = TurboQuantKeyCodec(head_dim, tq_config.key_bits, tq_config.seed)
    vc = GroupValueCodec(tq_config.value_bits, tq_config.group_size)

    payload = {
        "compressed":  np.array(True),
        "num_layers":  np.array(num_layers),
        "num_heads":   np.array(num_heads),
        "seq_len":     np.array(seq_len),
        "head_dim":    np.array(head_dim),
        "key_bits":    np.array(tq_config.key_bits),
        "value_bits":  np.array(tq_config.value_bits),
    }

    t = torch.from_numpy(arr.astype(np.float32))
    for layer_idx in range(num_layers):
        keys   = t[layer_idx, 0]
        values = t[layer_idx, 1]
        ck = kc.compress(keys.unsqueeze(0))
        cv = vc.compress(values.unsqueeze(0))
        for k_name, v_arr in ck.items():
            payload[f"L{layer_idx}_k_{k_name}"] = v_arr.numpy()
        for k_name, v_arr in cv.items():
            payload[f"L{layer_idx}_v_{k_name}"] = v_arr.numpy()

    np.savez_compressed(str(path), **payload)


def load_token_kv(path, tq_config=None) -> np.ndarray:
    """Load per-token KV array from disk. Returns float16 array."""
    data = np.load(str(path), allow_pickle=False)

    if not data["compressed"].item():
        return data["arr"]

    num_layers = int(data["num_layers"])
    num_heads  = int(data["num_heads"])
    seq_len    = int(data["seq_len"])
    head_dim   = int(data["head_dim"])
    key_bits   = int(data["key_bits"])
    value_bits = int(data["value_bits"])

    from addons.turboquant.quantizer import TurboQuantKeyCodec, GroupValueCodec
    seed = tq_config.seed if tq_config else 42
    kc = TurboQuantKeyCodec(head_dim, key_bits, seed)
    vc = GroupValueCodec(value_bits)

    result = np.zeros((num_layers, 2, num_heads, seq_len, head_dim), dtype=np.float32)
    for layer_idx in range(num_layers):
        ck = {k.replace(f"L{layer_idx}_k_", ""): torch.from_numpy(data[k])
              for k in data.files if k.startswith(f"L{layer_idx}_k_")}
        cv = {k.replace(f"L{layer_idx}_v_", ""): torch.from_numpy(data[k])
              for k in data.files if k.startswith(f"L{layer_idx}_v_")}
        keys_rec   = kc.decompress(ck).squeeze(0)
        values_rec = vc.decompress(cv).squeeze(0)
        result[layer_idx, 0] = keys_rec.numpy()
        result[layer_idx, 1] = values_rec.numpy()

    return result.astype(np.float16)


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
