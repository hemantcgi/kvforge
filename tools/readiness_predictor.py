"""Pre-training readiness predictor for the knowledge absorption curve.

Trains a regression model on corpus properties to predict whether Path B
(parametric answering) will match Path A (text RAG) before expensive training.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_NAMES = [
    "N",
    "total_tokens",
    "mean_cis_uniqueness",
    "factual_density",
    "topic_entropy",
    "contradiction_rate",
    "model_params_b",
]


def build_feature_matrix(data_points: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features and outcome from a list of data point dicts.

    Each data point must have: ``dataset_id``, ``delta`` (Path B - Path A F1),
    and feature fields matching ``FEATURE_NAMES`` (missing fields default to 0).

    Returns:
        ``(X, y, dataset_ids)`` where X is [n_points, n_features], y is [n_points].
    """
    X = []
    y = []
    ds_ids = []
    for dp in data_points:
        row = [float(dp.get(f, 0.0)) for f in FEATURE_NAMES]
        X.append(row)
        y.append(float(dp.get("delta", 0.0)))
        ds_ids.append(dp.get("dataset_id", "unknown"))
    return np.array(X), np.array(y), ds_ids


def train_predictor(X: np.ndarray, y: np.ndarray, model_type: str = "ridge") -> object:
    """Train a regression model to predict delta from corpus features."""
    if model_type == "ridge":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])
    elif model_type == "logistic":
        y_binary = (y >= -0.02).astype(int)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        model.fit(X, y_binary)
        return model
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    model.fit(X, y)
    return model


def evaluate_lodo(X: np.ndarray, y: np.ndarray, dataset_ids: list[str]) -> dict:
    """Leave-one-dataset-out cross-validation.

    Returns:
        ``{r2: float, per_dataset: {ds_id: {r2, mse, n}}, feature_importance: dict}``.
    """
    unique_ds = list(set(dataset_ids))
    per_dataset: dict[str, dict] = {}
    all_preds = []
    all_true = []

    for holdout_ds in unique_ds:
        train_mask = np.array([d != holdout_ds for d in dataset_ids])
        test_mask = ~train_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ])
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        all_preds.extend(preds.tolist())
        all_true.extend(y[test_mask].tolist())
        ss_res = np.sum((y[test_mask] - preds) ** 2)
        ss_tot = np.sum((y[test_mask] - y[test_mask].mean()) ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot
        per_dataset[holdout_ds] = {
            "r2": round(float(r2), 4),
            "mse": round(float(np.mean((y[test_mask] - preds) ** 2)), 6),
            "n": int(test_mask.sum()),
        }

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    ss_res = np.sum((all_true - all_preds) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2) + 1e-10
    overall_r2 = 1 - ss_res / ss_tot

    full_model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])
    full_model.fit(X, y)
    regressor = full_model.named_steps["regressor"]
    scaler = full_model.named_steps["scaler"]
    coefs = regressor.coef_ * scaler.scale_
    importance = {FEATURE_NAMES[i]: round(float(abs(coefs[i])), 4) for i in range(len(FEATURE_NAMES))}
    sorted_importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return {
        "r2": round(float(overall_r2), 4),
        "per_dataset": per_dataset,
        "feature_importance": sorted_importance,
    }


def derive_decision_rule(model: object, feature_names: list[str]) -> dict:
    """Derive a human-readable decision rule from the fitted model.

    Extracts feature importance (absolute standardized coefficient magnitude)
    and identifies which features most strongly predict Path B viability.

    Note: Raw coefficient sign and magnitude are shown but should not be
    interpreted as precise effect sizes without un-standardization against
    training data statistics. This function gives qualitative guidance.
    """
    if hasattr(model, "named_steps") and "regressor" in model.named_steps:
        regressor = model.named_steps["regressor"]
        coefs = regressor.coef_
        intercept = float(regressor.intercept_)
    else:
        coefs = getattr(model, "coef_", np.zeros(len(feature_names)))
        intercept = float(getattr(model, "intercept_", 0.0))

    if len(coefs) != len(feature_names):
        coefs = np.zeros(len(feature_names))

    feature_contrib = {feature_names[i]: float(coefs[i]) for i in range(len(feature_names))}
    ranked = sorted(feature_contrib.items(), key=lambda x: -abs(x[1]))

    thresholds = {}
    for name, coef in ranked[:3]:
        thresholds[name] = "higher is better" if coef > 0 else "lower is better"

    return {
        "thresholds": thresholds,
        "top_features": [name for name, _ in ranked[:3]],
        "all_coefficients": {name: round(coef, 4) for name, coef in ranked},
        "intercept": round(intercept, 4),
        "description": (
            f"Model intercept: {intercept:.4f}. "
            f"Top feature: {ranked[0][0]} (coef={ranked[0][1]:.4f}). "
            f"Path B is predicted viable when the top features favor high delta."
        ),
    }
