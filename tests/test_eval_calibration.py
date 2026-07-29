"""Tests for pipeline/eval_calibration.py — Sprint 2.5 calibration tooling."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.eval_calibration import (
    load_confidence_token_rows,
    extract_confidence_and_correctness,
    evaluate_calibration,
)


def test_load_confidence_token_json(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps([
        {"question": "Q1", "confidence": 0.9, "correctness": 1},
        {"question": "Q2", "confidence": 0.3, "correctness": 0},
    ]))
    rows = load_confidence_token_rows(str(p))
    assert len(rows) == 2
    assert rows[0]["confidence"] == 0.9


def test_load_confidence_token_csv(tmp_path):
    p = tmp_path / "rows.csv"
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "confidence", "correctness"])
        writer.writeheader()
        writer.writerow({"question": "Q1", "confidence": "0.9", "correctness": "1"})
        writer.writerow({"question": "Q2", "confidence": "0.3", "correctness": "0"})
    rows = load_confidence_token_rows(str(p))
    assert len(rows) == 2
    assert rows[0]["confidence"] == "0.9"


def test_extract_confidence_and_correctness_normalizes():
    rows = [
        {"confidence": 0.9, "correctness": True},
        {"confidence": 0.3, "correctness": False},
        {"confidence": "0.7", "judge_correct": 0},
    ]
    confidences, correctness = extract_confidence_and_correctness(rows)
    assert confidences == pytest.approx([0.9, 0.3, 0.7])
    assert correctness == [1, 0, 0]


def test_evaluate_calibration_perfectly_calibrated():
    # 10 samples in each bin with confidence == accuracy.
    confidences = []
    correctness = []
    for i in range(10):
        confidences.extend([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness.extend([0, 0, 0, 1, 1])
    # This is not perfectly calibrated, but ECE should be small.
    result = evaluate_calibration(confidences, correctness, bins=5)
    assert "ece" in result
    assert 0.0 <= result["ece"] <= 1.0
    assert result["n"] == 50


def test_evaluate_calibration_empty():
    result = evaluate_calibration([], [], bins=10)
    assert result["ece"] == 0.0
    assert result["n"] == 0


def test_evaluate_calibration_with_plot(tmp_path):
    from unittest.mock import MagicMock, patch
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
    correctness = [0, 0, 1, 1, 1]
    plot_path = tmp_path / "rel.png"

    mock_fig, mock_ax = MagicMock(), MagicMock()
    mock_plt = MagicMock()
    mock_plt.subplots = MagicMock(return_value=(mock_fig, mock_ax))
    with patch.dict("sys.modules", {"matplotlib.pyplot": mock_plt}):
        result = evaluate_calibration(
            confidences, correctness, bins=5, plot_path=str(plot_path),
        )

    assert result["ece"] >= 0.0
    mock_plt.subplots.assert_called_once()
    mock_fig.savefig.assert_called_once_with(str(plot_path), dpi=150)


def test_main_legacy_phase_quality(tmp_path):
    from pipeline.eval_calibration import main
    input_path = tmp_path / "in.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(json.dumps({
        "modes": {
            "text_rag": {
                "per_question": [
                    {"judge_correct": 1, "confidence": 0.9, "token_f1": 0.9},
                    {"judge_correct": 0, "confidence": 0.3, "token_f1": 0.2},
                ]
            }
        }
    }))

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "eval_calibration.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--format", "phase_quality",
        ]
        main()
    finally:
        sys.argv = old_argv

    output = json.loads(output_path.read_text())
    assert output["format"] == "phase_quality"
    assert output["mode"] == "text_rag"
    assert output["n"] == 2


def test_main_confidence_token_csv(tmp_path):
    from pipeline.eval_calibration import main
    input_path = tmp_path / "in.csv"
    output_path = tmp_path / "out.json"
    with open(input_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "confidence", "correctness"])
        writer.writeheader()
        writer.writerow({"question": "Q1", "confidence": "0.9", "correctness": "1"})
        writer.writerow({"question": "Q2", "confidence": "0.2", "correctness": "0"})

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "eval_calibration.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--format", "confidence_token",
        ]
        main()
    finally:
        sys.argv = old_argv

    output = json.loads(output_path.read_text())
    assert output["format"] == "confidence_token"
    assert output["n"] == 2
