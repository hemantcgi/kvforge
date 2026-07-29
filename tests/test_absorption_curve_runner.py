import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_check_heldout_quarantine_passes():
    from tools.absorption_curve_runner import check_heldout_quarantine
    check_heldout_quarantine({1, 2, 3}, {4, 5, 6})


def test_check_heldout_quarantine_raises():
    from tools.absorption_curve_runner import check_heldout_quarantine
    with pytest.raises(AssertionError, match="quarantine"):
        check_heldout_quarantine({1, 2, 3}, {3, 4, 5})


def test_aggregate_results(tmp_path):
    from tools.absorption_curve_runner import aggregate_results
    for seed, delta in [(42, 0.05), (43, 0.03), (44, 0.07)]:
        p = tmp_path / f"condition_a_seed{seed}.json"
        p.write_text(json.dumps({
            "condition": "condition_a",
            "seed": seed,
            "path_a_f1": 0.20,
            "path_b_f1": 0.20 + delta,
            "delta": delta,
            "per_question": [{"q": "Q1", "path_a_f1": 0.2, "path_b_f1": 0.2 + delta}],
        }))
    summary = aggregate_results([tmp_path / f"condition_a_seed{s}.json" for s in [42, 43, 44]])
    assert "condition_a" in summary
    assert "mean_delta" in summary["condition_a"]
    assert abs(summary["condition_a"]["mean_delta"] - 0.05) < 0.01
    assert "ci_low" in summary["condition_a"]
    assert "ci_high" in summary["condition_a"]


def test_build_condition_name():
    from tools.absorption_curve_runner import _build_condition_name
    name = _build_condition_name({"N": 2000, "quality": "natural", "diversity": "entity_graph"})
    assert "N2000" in name
    assert "natural" in name
    assert "entity_graph" in name
