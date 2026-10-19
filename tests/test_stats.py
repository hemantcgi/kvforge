import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import stats as eval_stats


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_clear_shift_excludes_zero():
    """A clear shift with per-pair noise should yield a delta CI that excludes 0."""
    rng = np.random.default_rng(0)
    n = 200
    scores_a = np.array(rng.normal(loc=0.5, scale=0.05, size=n))
    noise = rng.normal(loc=0.0, scale=0.03, size=n)
    scores_b = scores_a + 0.2 + noise  # ~+0.2 shift, paired, with real per-pair variance

    result = eval_stats.paired_bootstrap(
        list(scores_a), list(scores_b), n_resamples=2000, seed=42
    )

    assert result["delta"] == pytest.approx(0.2, abs=0.02)
    assert result["ci_low"] > 0
    assert result["ci_high"] > result["ci_low"]
    assert result["p_value"] < 0.05


def test_paired_bootstrap_identical_distributions_includes_zero():
    """Identical paired samples should give delta=0 and a CI that includes 0."""
    rng = np.random.default_rng(1)
    scores = list(rng.normal(loc=0.5, scale=0.1, size=100))

    result = eval_stats.paired_bootstrap(scores, list(scores), n_resamples=2000, seed=42)

    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] <= 0 <= result["ci_high"]
    assert result["p_value"] == pytest.approx(1.0, abs=1e-9)


def test_paired_bootstrap_unequal_lengths_raises():
    with pytest.raises(ValueError):
        eval_stats.paired_bootstrap([0.1, 0.2, 0.3], [0.1, 0.2])


def test_paired_bootstrap_is_deterministic_given_seed():
    scores_a = [0.1, 0.4, 0.6, 0.9, 0.2]
    scores_b = [0.3, 0.5, 0.7, 0.95, 0.4]

    r1 = eval_stats.paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=7)
    r2 = eval_stats.paired_bootstrap(scores_a, scores_b, n_resamples=500, seed=7)

    assert r1 == r2


def test_paired_bootstrap_result_keys():
    result = eval_stats.paired_bootstrap([0.1, 0.2], [0.2, 0.3], n_resamples=100, seed=1)
    assert set(result.keys()) == {"delta", "ci_low", "ci_high", "p_value"}


# ---------------------------------------------------------------------------
# mcnemar_exact
# ---------------------------------------------------------------------------


def test_mcnemar_exact_hand_computed_b01_8_b10_2():
    """Hand-computed exact two-sided binomial McNemar for b01=8, b10=2, n=10, p=0.5.

    p = 2 * P(X <= 2) where X ~ Binomial(10, 0.5), capped at 1.0.
    P(X<=2) = (C(10,0)+C(10,1)+C(10,2)) / 2**10 = (1+10+45)/1024 = 56/1024
    p = 2 * 56/1024 = 112/1024 = 0.109375
    """
    p = eval_stats.mcnemar_exact(8, 2)
    assert p == pytest.approx(0.109375, abs=1e-9)


def test_mcnemar_exact_symmetric_in_b01_b10():
    assert eval_stats.mcnemar_exact(8, 2) == pytest.approx(eval_stats.mcnemar_exact(2, 8))


def test_mcnemar_exact_zero_discordant_pairs_is_one():
    assert eval_stats.mcnemar_exact(0, 0) == pytest.approx(1.0)


def test_mcnemar_exact_equal_counts_is_one():
    """When b01 == b10 the two-sided exact p-value should be 1.0 (perfectly balanced)."""
    assert eval_stats.mcnemar_exact(5, 5) == pytest.approx(1.0)


def test_mcnemar_exact_extreme_discordance_is_small():
    """b01=0, b10=20 should give an extremely small p-value."""
    p = eval_stats.mcnemar_exact(0, 20)
    assert p < 1e-4


# ---------------------------------------------------------------------------
# summarize_ci
# ---------------------------------------------------------------------------


def test_summarize_ci_format():
    deltas = {"delta": 0.083, "ci_low": 0.041, "ci_high": 0.126, "p_value": 0.002}
    summary = eval_stats.summarize_ci(deltas)
    assert summary == "Δ=+0.083 [95% CI +0.041, +0.126], p=0.002"


def test_summarize_ci_negative_delta_has_sign():
    deltas = {"delta": -0.05, "ci_low": -0.09, "ci_high": -0.01, "p_value": 0.01}
    summary = eval_stats.summarize_ci(deltas)
    assert summary.startswith("Δ=-0.050")
    assert "-0.090" in summary
    assert "-0.010" in summary
