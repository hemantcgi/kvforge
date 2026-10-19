"""Statistical tools for comparing evaluation conditions.

Paired bootstrap confidence intervals for a metric delta between two
conditions (e.g. RAG vs LoRA-adapted), exact McNemar test for paired binary
outcomes, and a human-readable one-line summary formatter.

This module is intentionally numpy-only: no scipy dependency. The exact
binomial McNemar test is implemented directly with ``math.comb``.
"""

from __future__ import annotations

import math

import numpy as np


def paired_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Paired bootstrap CI and two-sided p-value for the delta mean(b) - mean(a).

    ``scores_a`` and ``scores_b`` must be paired by question (same length,
    same index refers to the same question in both lists).

    Args:
        scores_a: Per-question scores for condition A.
        scores_b: Per-question scores for condition B.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        ci: Confidence level (e.g. 0.95).

    Returns:
        Dict with ``delta`` (mean(b) - mean(a)), ``ci_low``, ``ci_high``, and
        ``p_value`` (two-sided bootstrap p-value for delta != 0).

    Raises:
        ValueError: If ``scores_a`` and ``scores_b`` have unequal lengths.
    """
    arr_a = np.asarray(scores_a, dtype=float)
    arr_b = np.asarray(scores_b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"scores_a and scores_b must be paired (equal length): "
            f"got {len(arr_a)} vs {len(arr_b)}"
        )

    n = len(arr_a)
    diffs = arr_b - arr_a
    delta = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_deltas[i] = np.mean(diffs[idx])

    alpha = 1 - ci
    ci_low, ci_high = np.percentile(boot_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value for H0: delta == 0.
    # Center the bootstrap distribution on 0 (the null) and measure how
    # extreme the observed delta is relative to that null distribution.
    centered = boot_deltas - delta
    p_value = float(np.mean(np.abs(centered) >= np.abs(delta)))
    p_value = min(p_value, 1.0)

    return {
        "delta": delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def mcnemar_exact(b01: int, b10: int) -> float:
    """Exact two-sided binomial McNemar p-value for paired binary outcomes.

    Args:
        b01: Count of pairs where A was wrong and B was right (discordant).
        b10: Count of pairs where A was right and B was wrong (discordant).

    Returns:
        Two-sided exact p-value under the null that b01 and b10 are drawn
        from Binomial(n, 0.5), where n = b01 + b10.
    """
    n = b01 + b10
    if n == 0:
        return 1.0

    k = min(b01, b10)
    # P(X <= k) under Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_value = min(1.0, 2 * tail)
    return p_value


def summarize_ci(deltas: dict) -> str:
    """Format a bootstrap result dict as a one-line human-readable summary.

    Example: "Δ=+0.083 [95% CI +0.041, +0.126], p=0.002"
    """
    delta = deltas["delta"]
    ci_low = deltas["ci_low"]
    ci_high = deltas["ci_high"]
    p_value = deltas["p_value"]
    return (
        f"Δ={delta:+.3f} [95% CI {ci_low:+.3f}, {ci_high:+.3f}], p={p_value:.3f}"
    )
