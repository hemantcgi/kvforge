"""Tests for the Exp 1 Zipf workload generator."""
import numpy as np

from tools.exp1_workload import zipf_weights, sample_stream, stream_stats


def test_zipf_weights_normalised_and_monotone():
    w = zipf_weights(100, 1.0)
    assert abs(w.sum() - 1.0) < 1e-12
    # strictly decreasing with rank for alpha > 0
    assert np.all(np.diff(w) < 0)


def test_zipf_uniform_when_alpha_zero():
    w = zipf_weights(50, 0.0)
    assert np.allclose(w, 1.0 / 50)


def test_higher_alpha_fewer_distinct():
    # more skew -> fewer distinct queries at the same volume
    s_lo = stream_stats(sample_stream(2000, 0.0, 10000, seed=1))
    s_hi = stream_stats(sample_stream(2000, 1.2, 10000, seed=1))
    assert s_hi["distinct"] < s_lo["distinct"]
    assert s_hi["repeat_rate"] > s_lo["repeat_rate"]


def test_uniform_low_repeat_rate_when_volume_below_pool():
    # alpha=0, volume << pool -> almost all distinct, tiny repeat rate
    s = stream_stats(sample_stream(100000, 0.0, 1000, seed=7))
    assert s["repeat_rate"] < 0.02


def test_stream_stats_shape():
    s = stream_stats(sample_stream(500, 0.8, 5000, seed=3))
    assert s["volume"] == 5000
    assert 0 < s["distinct"] <= 500
    assert 0.0 <= s["repeat_rate"] < 1.0
