import pytest
import torch
import numpy as np
from addons.turboquant.config import TurboQuantConfig


def test_default_config():
    cfg = TurboQuantConfig()
    assert cfg.key_bits == 3
    assert cfg.value_bits == 4
    assert cfg.group_size == 32
    assert cfg.seed == 42


def test_invalid_bits():
    with pytest.raises(Exception):
        TurboQuantConfig(key_bits=5)  # only 2 or 3 valid


# --- codebooks ---
from addons.turboquant.codebooks import get_codebook


def test_codebook_shape_128_2bit():
    centroids, boundaries = get_codebook(head_dim=128, num_bits=2)
    assert len(centroids) == 4
    assert len(boundaries) == 3
    assert boundaries[0] < boundaries[1] < boundaries[2]


def test_codebook_shape_128_3bit():
    centroids, boundaries = get_codebook(head_dim=128, num_bits=3)
    assert len(centroids) == 8
    assert len(boundaries) == 7


def test_codebook_supported_dims():
    for dim in [64, 96, 128, 256]:
        for bits in [2, 3]:
            c, b = get_codebook(dim, bits)
            assert len(c) == 2**bits


def test_codebook_centroids_ordered():
    centroids, _ = get_codebook(128, 3)
    assert all(centroids[i] < centroids[i + 1] for i in range(len(centroids) - 1))


# --- key codec ---
from addons.turboquant.quantizer import TurboQuantKeyCodec


def test_key_compress_decompress_shape():
    codec = TurboQuantKeyCodec(head_dim=128, num_bits=3, seed=42)
    keys = torch.randn(4, 32, 8, 128)  # [batch, seq, heads, dim]
    compressed = codec.compress(keys)
    assert compressed["norms"].shape == (4, 32, 8)
    assert compressed["indices"].shape == (4, 32, 8, 128)
    assert compressed["qjl_signs"].shape == (4, 32, 8, 128)


def test_key_estimator_is_unbiased():
    torch.manual_seed(0)
    codec = TurboQuantKeyCodec(head_dim=128, num_bits=3, seed=42)
    n_trials = 500
    errors = []
    for _ in range(n_trials):
        k = torch.randn(128)
        k = k / k.norm()
        q = torch.randn(128)
        q = q / q.norm()
        true_dot = (q * k).sum().item()
        compressed = codec.compress(k.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        est_dot = codec.estimate_dot(q, compressed, token_idx=0, head_idx=0, batch_idx=0)
        errors.append(est_dot - true_dot)
    mean_err = np.mean(errors)
    assert abs(mean_err) < 0.05, f"Estimator biased: mean_err={mean_err:.4f}"


def test_key_cosine_similarity_3bit():
    torch.manual_seed(1)
    codec = TurboQuantKeyCodec(head_dim=128, num_bits=3, seed=42)
    k = torch.randn(1, 10, 1, 128)
    k_norm = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    compressed = codec.compress(k)
    k_rec = codec.decompress(compressed)
    k_rec_norm = k_rec / k_rec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (k_norm * k_rec_norm).sum(dim=-1).mean().item()
    assert cos_sim >= 0.95, f"Cosine similarity too low: {cos_sim:.4f}"


# --- value codec ---
from addons.turboquant.quantizer import GroupValueCodec


def test_value_compress_decompress_shape():
    codec = GroupValueCodec(num_bits=4, group_size=32)
    vals = torch.randn(2, 10, 4, 128)
    c = codec.compress(vals)
    assert c["packed"].shape[-1] == 128 // 2   # 4-bit packs 2 per byte
    assert c["scales"].shape[-1] == 4           # 128 // 32 = 4 groups
    rec = codec.decompress(c)
    assert rec.shape == vals.shape


def test_value_cosine_sim_4bit():
    torch.manual_seed(2)
    codec = GroupValueCodec(num_bits=4, group_size=32)
    v = torch.randn(1, 20, 1, 128)
    v_norm = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c = codec.compress(v)
    rec = codec.decompress(c)
    rec_norm = rec / rec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (v_norm * rec_norm).sum(dim=-1).mean().item()
    assert cos_sim >= 0.99, f"4-bit value cos_sim too low: {cos_sim:.4f}"


def test_value_cosine_sim_2bit():
    torch.manual_seed(3)
    codec = GroupValueCodec(num_bits=2, group_size=32)
    v = torch.randn(1, 20, 1, 128)
    v_norm = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c = codec.compress(v)
    rec = codec.decompress(c)
    rec_norm = rec / rec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_sim = (v_norm * rec_norm).sum(dim=-1).mean().item()
    assert cos_sim >= 0.93, f"2-bit value cos_sim too low: {cos_sim:.4f}"


def test_addon_registers():
    from addons.registry import AddonRegistry
    AddonRegistry.load_builtins()
    manifest = AddonRegistry.get("turboquant")
    assert manifest.name == "turboquant"
    assert manifest.config_schema is not None
