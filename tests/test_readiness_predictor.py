import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fake_data_points(n=30):
    rng = np.random.RandomState(42)
    points = []
    for i in range(n):
        n_chunks = rng.randint(500, 6000)
        factual_density = rng.uniform(0.3, 0.9)
        uniqueness = rng.uniform(0.2, 0.8)
        delta = 0.1 * (n_chunks / 6000) + 0.3 * factual_density - 0.1 * uniqueness + rng.normal(0, 0.02)
        points.append({
            "dataset_id": f"ds_{i % 5}",
            "N": n_chunks,
            "factual_density": factual_density,
            "mean_cis_uniqueness": uniqueness,
            "topic_entropy": rng.uniform(1, 5),
            "model_params_b": 2.0,
            "delta": delta,
        })
    return points


def test_build_feature_matrix():
    from tools.readiness_predictor import build_feature_matrix
    points = _fake_data_points(20)
    X, y, ds_ids = build_feature_matrix(points)
    assert X.shape[0] == 20
    assert len(y) == 20
    assert len(ds_ids) == 20
    assert X.shape[1] >= 4


def test_train_predictor_ridge():
    from tools.readiness_predictor import train_predictor, build_feature_matrix
    points = _fake_data_points(30)
    X, y, _ = build_feature_matrix(points)
    model = train_predictor(X, y, model_type="ridge")
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == 30


def test_evaluate_lodo():
    from tools.readiness_predictor import evaluate_lodo, build_feature_matrix
    points = _fake_data_points(25)
    X, y, ds_ids = build_feature_matrix(points)
    result = evaluate_lodo(X, y, ds_ids)
    assert "r2" in result
    assert "per_dataset" in result
    assert len(result["per_dataset"]) == 5


def test_derive_decision_rule():
    from tools.readiness_predictor import derive_decision_rule, build_feature_matrix
    from sklearn.linear_model import Ridge
    points = _fake_data_points(30)
    X, y, _ = build_feature_matrix(points)
    model = Ridge().fit(X, y)
    feature_names = ["N", "factual_density", "mean_cis_uniqueness", "topic_entropy", "model_params_b"]
    rule = derive_decision_rule(model, feature_names)
    assert "thresholds" in rule
    assert "description" in rule
