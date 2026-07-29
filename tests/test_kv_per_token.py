import pytest, torch, numpy as np
import tempfile, os
from pathlib import Path


def _make_past_key_values(num_layers, num_heads, seq_len, head_dim):
    pkv = []
    for _ in range(num_layers):
        k = torch.randn(1, num_heads, seq_len, head_dim)
        v = torch.randn(1, num_heads, seq_len, head_dim)
        pkv.append((k, v))
    return tuple(pkv)


def test_compute_per_token_kv_shape():
    from core.kv_utils import compute_per_token_kv
    pkv = _make_past_key_values(32, 8, 64, 128)
    arr = compute_per_token_kv(pkv)
    assert arr.shape == (32, 2, 8, 64, 128)
    assert arr.dtype == np.float16


def test_compute_per_token_kv_values_preserved():
    from core.kv_utils import compute_per_token_kv
    pkv = _make_past_key_values(2, 4, 10, 64)
    arr = compute_per_token_kv(pkv)
    k0 = pkv[0][0].squeeze(0).cpu().numpy().astype(np.float16)
    np.testing.assert_array_equal(arr[0, 0], k0)


def test_mean_pool_unchanged():
    from core.kv_utils import mean_pool_kv
    pkv = _make_past_key_values(32, 8, 64, 128)
    arr = mean_pool_kv(pkv)
    assert arr.shape == (32, 2, 8, 128)


def test_save_load_roundtrip_no_compression(tmp_path):
    from core.kv_utils import save_token_kv, load_token_kv
    arr = np.random.randn(32, 2, 8, 64, 128).astype(np.float16)
    path = tmp_path / "chunk_kv.npz"
    save_token_kv(arr, path, tq_config=None)
    loaded = load_token_kv(path, tq_config=None)
    np.testing.assert_array_equal(arr, loaded)


def test_save_load_roundtrip_with_turboquant(tmp_path):
    from core.kv_utils import save_token_kv, load_token_kv
    from addons.turboquant.config import TurboQuantConfig
    arr = np.random.randn(4, 2, 4, 32, 128).astype(np.float16)
    cfg = TurboQuantConfig(key_bits=3, value_bits=4)
    path = tmp_path / "chunk_kv_tq.npz"
    save_token_kv(arr, path, tq_config=cfg)
    loaded = load_token_kv(path, tq_config=cfg)
    assert loaded.shape == arr.shape
    assert loaded.dtype == np.float16

    # Roundtrip should preserve approximate values, not just shape/dtype.
    def _mean_cos(a, b):
        a = a.reshape(-1, a.shape[-1]).astype(np.float32)
        b = b.reshape(-1, b.shape[-1]).astype(np.float32)
        num = np.sum(a * b, axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
        return float(num.mean() / den.mean())

    assert _mean_cos(arr[:, 0], loaded[:, 0]) > 0.8, "TurboQuant key roundtrip cosine too low"
    assert _mean_cos(arr[:, 1], loaded[:, 1]) > 0.8, "TurboQuant value roundtrip cosine too low"


def test_turboquant_reduces_file_size(tmp_path):
    from core.kv_utils import save_token_kv
    from addons.turboquant.config import TurboQuantConfig
    arr = np.random.randn(32, 2, 8, 128, 128).astype(np.float16)
    raw_path = tmp_path / "raw.npz"
    tq_path  = tmp_path / "tq.npz"
    save_token_kv(arr, raw_path, tq_config=None)
    save_token_kv(arr, tq_path, tq_config=TurboQuantConfig())
    raw_size = raw_path.stat().st_size
    tq_size  = tq_path.stat().st_size
    assert tq_size < raw_size * 0.7, (
        f"TurboQuant did not reduce file size: raw={raw_size}, tq={tq_size}")
