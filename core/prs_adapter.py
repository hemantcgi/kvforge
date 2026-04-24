"""Per-cluster PRS computation, logistic regression weight learning, and
phase-advancement logic for KVForge dynamic PRS.

All numerics use pure NumPy — no sklearn.

Public API
----------
* ``compute_cluster_prs(faq, vdb, realtime, weights)`` → weighted PRS float.
* ``compute_slope(prs_history)`` → linear slope (positive = improving).
* ``should_advance(cluster_state, cfg_threshold, stability_window)`` → bool.
* ``adapt_weights(cluster_state, min_samples)`` → new weight dict or None.
* ``initial_threshold(base, difficulty_score, global_mean)`` → adjusted float.
* ``update_cluster_after_round(...)`` — full per-cluster update + persistence.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import core.version as ver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _logistic_regression_weights(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 300
) -> np.ndarray:
    """Fit a one-layer logistic regression via gradient descent; return normalised weights.

    Weights are clipped to [0.10, 0.70] then L1-normalised so they sum to 1.

    Args:
        X: ``(n, 3)`` feature matrix — columns are [faq, vdb, realtime].
        y: ``(n,)`` binary labels (1 = correct, 0 = incorrect).
        lr: Learning rate.
        epochs: Number of gradient steps.

    Returns:
        ``(3,)`` normalised weight vector summing to 1.0.
    """
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        logits = np.clip(X @ w, -20, 20)
        preds = 1 / (1 + np.exp(-logits))
        w -= lr * (X.T @ (preds - y)) / len(y)
    w = np.clip(np.abs(w), 0.10, 0.70)
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1 / len(w))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_cluster_prs(
    faq_coverage: float,
    vdb_coverage: float,
    realtime_coverage: float,
    weights: dict,
) -> float:
    """Compute a weighted PRS score from three signals, clipped to [0, 1].

    Args:
        faq_coverage: FAQ accuracy signal in [0, 1].
        vdb_coverage: VDB sampling coverage signal in [0, 1].
        realtime_coverage: Real-time query coverage signal in [0, 1].
        weights: Dict with keys ``'faq'``, ``'vdb'``, ``'realtime'``.

    Returns:
        Weighted PRS in [0, 1].
    """
    return float(
        np.clip(
            weights.get("faq", 0.4) * faq_coverage
            + weights.get("vdb", 0.4) * vdb_coverage
            + weights.get("realtime", 0.2) * realtime_coverage,
            0.0,
            1.0,
        )
    )


def compute_slope(prs_history: list[float]) -> float:
    """Ordinary least-squares slope of a PRS history sequence.

    Args:
        prs_history: List of PRS floats ordered by LoRA round (oldest first).

    Returns:
        Slope (positive = improving, negative = regressing, 0 = flat/single point).
    """
    n = len(prs_history)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(prs_history, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def should_advance(
    cluster_state: dict,
    cfg_threshold: float,
    stability_window: int = 3,
) -> bool:
    """Return True if this cluster should advance to the next phase.

    Advancement requires:
    1. ``prs >= threshold`` (uses cluster-specific threshold if available).
    2. Non-negative slope over the last ``stability_window`` rounds.

    Args:
        cluster_state: Per-cluster state dict from ``ver.get_cluster_state()``.
        cfg_threshold: Global config advancement threshold fallback.
        stability_window: Number of trailing PRS rounds to measure slope over.

    Returns:
        ``True`` if both conditions are met.
    """
    prs = cluster_state.get("prs", 0.0)
    threshold = cluster_state.get("threshold", cfg_threshold)
    if prs < threshold:
        return False
    history = cluster_state.get("prs_history", [])
    window = history[-stability_window:] if len(history) >= stability_window else history
    return compute_slope(window) >= 0.0


def adapt_weights(
    cluster_state: dict, min_samples: int = 10
) -> Optional[dict]:
    """Fit logistic regression on labeled query history; return updated weights or None.

    Returns ``None`` when there are too few labeled samples or when the labels
    are all the same (no learning signal).

    Args:
        cluster_state: Per-cluster state dict; must contain ``'labeled_history'``
            — a list of dicts with keys ``'faq'``, ``'vdb'``, ``'realtime'``,
            ``'correct'``.
        min_samples: Minimum labeled samples required before adapting.

    Returns:
        Dict ``{'faq': float, 'vdb': float, 'realtime': float}`` summing to 1.0,
        or ``None`` if adaptation is skipped.
    """
    history = cluster_state.get("labeled_history", [])
    if len(history) < min_samples:
        return None
    X = np.array([[h["faq"], h["vdb"], h["realtime"]] for h in history])
    y = np.array([float(h["correct"]) for h in history])
    if y.std() == 0:
        return None
    w = _logistic_regression_weights(X, y)
    return {"faq": float(w[0]), "vdb": float(w[1]), "realtime": float(w[2])}


def initial_threshold(
    base_threshold: float,
    difficulty_score: float,
    global_mean_difficulty: float,
) -> float:
    """Scale *base_threshold* by relative cluster difficulty, clamped to [0.5, 0.95].

    Harder clusters (difficulty > mean) get a higher threshold; easier ones get
    a lower threshold.  The scaling multiplier is clamped to [0.85, 1.15].

    Args:
        base_threshold: Global PRS advancement threshold (e.g. 0.72).
        difficulty_score: This cluster's difficulty score from the estimator.
        global_mean_difficulty: Mean difficulty across all clusters.

    Returns:
        Adjusted threshold float in [0.5, 0.95].
    """
    if global_mean_difficulty == 0:
        return base_threshold
    ratio = difficulty_score / global_mean_difficulty
    multiplier = float(np.clip(ratio, 0.85, 1.15))
    return float(np.clip(base_threshold * multiplier, 0.5, 0.95))


def update_cluster_after_round(
    cluster_id: str,
    faq_coverage: float,
    vdb_coverage: float,
    realtime_stats: dict,
    cfg: dict,
) -> dict:
    """Compute per-cluster PRS, optionally adapt weights, and persist state.

    This is the main entry point called by the PRS evaluator after each LoRA
    training round.

    Args:
        cluster_id: Cluster identifier string.
        faq_coverage: FAQ signal value in [0, 1].
        vdb_coverage: VDB sampling coverage in [0, 1].
        realtime_stats: Dict from ``query_logger.get_cluster_stats()`` with keys
            ``'realtime_coverage'`` and ``'query_count'``.
        cfg: Config dict (or ``DatasourceConfig.model_dump()``).

    Returns:
        Updated cluster state dict (also persisted via :func:`ver.save_cluster_state`).
    """
    state = ver.get_cluster_state(cluster_id)
    realtime_coverage = realtime_stats.get("realtime_coverage", 0.0)
    weights = state.get("learned_weights") or cfg.get(
        "prs_signal_weights", {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    )
    prs = compute_cluster_prs(faq_coverage, vdb_coverage, realtime_coverage, weights)

    history = state.get("prs_history", [])
    history.append(round(prs, 4))
    state.update(
        {
            "faq_coverage": round(faq_coverage, 4),
            "vdb_coverage": round(vdb_coverage, 4),
            "realtime_coverage": round(realtime_coverage, 4),
            "prs": round(prs, 4),
            "prs_history": history,
            "query_count": realtime_stats.get(
                "query_count", state.get("query_count", 0)
            ),
        }
    )

    if cfg.get("prs_auto_weight", True):
        new_weights = adapt_weights(
            state, cfg.get("min_cluster_samples_for_adaptation", 10)
        )
        if new_weights:
            state["learned_weights"] = new_weights
            prs = compute_cluster_prs(
                faq_coverage, vdb_coverage, realtime_coverage, new_weights
            )
            state["prs"] = round(prs, 4)

    threshold = state.get("threshold", cfg.get("prs_advancement_threshold", 0.72))
    window = cfg.get("prs_stability_window", 3)
    current_phase = state.get("phase", 1)
    if current_phase < 3 and should_advance(state, threshold, window):
        state["phase"] = current_phase + 1
        print(f"Cluster {cluster_id} advanced to Phase {state['phase']}")

    ver.save_cluster_state(cluster_id, state)
    return state
