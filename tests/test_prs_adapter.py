"""Tests for core/prs_adapter.py — per-cluster PRS and logistic regression weights."""

import numpy as np
import pytest
from core.prs_adapter import (
    compute_cluster_prs,
    compute_slope,
    should_advance,
    adapt_weights,
    initial_threshold,
)


def test_compute_cluster_prs_weighted():
    weights = {"faq": 0.4, "vdb": 0.4, "realtime": 0.2}
    prs = compute_cluster_prs(0.8, 0.6, 0.5, weights)
    assert abs(prs - (0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.5)) < 1e-6


def test_compute_cluster_prs_clipped():
    weights = {"faq": 0.5, "vdb": 0.5, "realtime": 0.5}
    prs = compute_cluster_prs(1.0, 1.0, 1.0, weights)
    assert prs == 1.0


def test_compute_slope_increasing():
    slope = compute_slope([0.5, 0.6, 0.7, 0.8])
    assert slope > 0


def test_compute_slope_decreasing():
    slope = compute_slope([0.8, 0.7, 0.6])
    assert slope < 0


def test_compute_slope_flat():
    slope = compute_slope([0.7, 0.7, 0.7])
    assert abs(slope) < 1e-6


def test_compute_slope_single():
    assert compute_slope([0.7]) == 0.0


def test_should_advance_true_when_prs_above_threshold_and_slope_positive():
    state = {"prs": 0.75, "threshold": 0.72, "prs_history": [0.65, 0.70, 0.75]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is True


def test_should_advance_false_when_prs_below_threshold():
    state = {"prs": 0.60, "threshold": 0.72, "prs_history": [0.60, 0.60, 0.60]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is False


def test_should_advance_false_when_slope_negative():
    state = {"prs": 0.75, "threshold": 0.72, "prs_history": [0.80, 0.77, 0.75]}
    assert should_advance(state, cfg_threshold=0.72, stability_window=3) is False


def test_adapt_weights_returns_none_when_too_few_samples():
    state = {"labeled_history": [{"faq": 0.8, "vdb": 0.7, "realtime": 0.6, "correct": 1}]}
    result = adapt_weights(state, min_samples=10)
    assert result is None


def test_adapt_weights_returns_normalized_weights():
    history = [
        {"faq": 0.9, "vdb": 0.2, "realtime": 0.5, "correct": 1},
        {"faq": 0.8, "vdb": 0.3, "realtime": 0.4, "correct": 1},
        {"faq": 0.2, "vdb": 0.8, "realtime": 0.1, "correct": 0},
        {"faq": 0.1, "vdb": 0.9, "realtime": 0.2, "correct": 0},
        {"faq": 0.85, "vdb": 0.25, "realtime": 0.6, "correct": 1},
        {"faq": 0.15, "vdb": 0.75, "realtime": 0.3, "correct": 0},
        {"faq": 0.9, "vdb": 0.1, "realtime": 0.7, "correct": 1},
        {"faq": 0.1, "vdb": 0.85, "realtime": 0.2, "correct": 0},
        {"faq": 0.88, "vdb": 0.2, "realtime": 0.5, "correct": 1},
        {"faq": 0.12, "vdb": 0.82, "realtime": 0.1, "correct": 0},
    ]
    state = {"labeled_history": history}
    weights = adapt_weights(state, min_samples=10)
    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-4
    for v in weights.values():
        assert 0.10 <= v <= 0.70


def test_initial_threshold_scales_with_difficulty():
    t_easy = initial_threshold(0.72, difficulty_score=0.3, global_mean_difficulty=0.5)
    t_hard = initial_threshold(0.72, difficulty_score=0.7, global_mean_difficulty=0.5)
    assert t_hard > t_easy


def test_initial_threshold_clamped():
    t = initial_threshold(0.72, difficulty_score=100.0, global_mean_difficulty=0.5)
    assert t <= 0.95
