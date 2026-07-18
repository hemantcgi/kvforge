import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.measure_baseline_fkds import (
    estimate_judge_cost,
    summarize_fkds,
    summarize_latency,
)


def test_estimate_judge_cost_non_negative():
    cost = estimate_judge_cost("What is Bedrock?", "A service.", "A service.", "gpt-4o-mini")
    assert cost > 0


def test_estimate_judge_cost_scales_with_model():
    cheap = estimate_judge_cost("What is Bedrock?", "A service.", "A service.", "gpt-4o-mini")
    expensive = estimate_judge_cost("What is Bedrock?", "A service.", "A service.", "gpt-4o")
    assert expensive > cheap


def test_summarize_latency():
    lat = [0.1, 0.2, 0.3, 0.4, 0.5]
    s = summarize_latency(lat)
    assert s["n"] == 5
    assert s["mean"] == 0.3
    assert s["p50"] == 0.3


def test_summarize_fkds():
    fkds = [0.5, 0.6, 0.7, 0.8]
    s = summarize_fkds(fkds)
    assert s["n"] == 4
    assert abs(s["mean"] - 0.65) < 1e-6
    assert s["sem"] >= 0

