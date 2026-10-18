"""Pin the pre-registered length-controlled metrics for Exp 5 Tier 1.

These tests lock the budget-F1 definition (truncate BOTH sides to k tokens)
so the metric cannot silently drift into a third incompatible FA formula
(§0.1 / cross-experiment threat #12).
"""
from eval.metrics import (
    token_prf, budget_f1, first_k_precision, non_inferiority, _truncate_to_budget,
)


def test_first_k_precision_recovers_plan_definition():
    # first k predicted tokens vs FULL gold, divided by k-pred length.
    # pred opens with 3 gold-matching tokens then 1 junk; gold is long.
    gold = "paris is the capital city of france in western europe"
    pred = "paris capital france zzz and more filler words here"
    # first 4 pred tokens = [paris, capital, france, zzz]; 3 in gold -> 3/4
    assert abs(first_k_precision(pred, gold, 4) - 0.75) < 1e-9
    # gold is NOT truncated: a short dense opening scores high despite long gold
    assert first_k_precision("paris", gold, 20) == 1.0
    assert first_k_precision("", gold, 20) == 0.0


def test_token_prf_separates_precision_and_recall():
    # pred = 2 tokens, gold = 4 tokens, 2 overlap -> P=1.0, R=0.5
    r = token_prf("alpha beta", "alpha beta gamma delta")
    assert abs(r["precision"] - 1.0) < 1e-9
    assert abs(r["recall"] - 0.5) < 1e-9
    assert abs(r["f1"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-9


def test_token_prf_empty_cases():
    assert token_prf("", "") == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert token_prf("x", "")["f1"] == 0.0
    assert token_prf("", "y")["f1"] == 0.0


def test_truncate_to_budget():
    assert _truncate_to_budget("a b c d e", 3) == "a b c"
    assert _truncate_to_budget("a b", 5) == "a b"      # shorter than budget: unchanged
    assert _truncate_to_budget("", 3) == ""


def test_budget_truncates_both_sides_not_just_prediction():
    # The core reason the metric exists: a short dense answer vs a long gold.
    gold = " ".join(f"w{i}" for i in range(100))
    pred = " ".join(f"w{i}" for i in range(20))        # first 20 gold tokens, correct
    full = token_prf(pred, gold)
    b20 = budget_f1(pred, gold, 20)
    assert full["recall"] < 0.25        # 20/100 full-gold recall: penalised for brevity
    assert b20["recall"] == 1.0         # gold also truncated to 20 -> full credit
    assert b20["f1"] == 1.0
    assert b20["k"] == 20


def test_budget_does_not_touch_already_short_answers():
    # Short cells (~4-word golds) must be unchanged by a 20-token budget.
    pred, gold = "paris", "paris france"
    assert budget_f1(pred, gold, 20)["f1"] == token_prf(pred, gold)["f1"]


def test_non_inferiority_verdicts():
    assert non_inferiority(0.01, 0.05)["verdict"] == "SUPERIOR"
    assert non_inferiority(-0.01, 0.03)["verdict"] == "NON_INFERIOR"
    assert non_inferiority(-0.10, -0.05)["verdict"] == "INFERIOR"
    assert non_inferiority(-0.05, 0.05)["verdict"] == "INCONCLUSIVE"
    assert non_inferiority(-0.02, 0.01, margin=-0.02)["verdict"] == "INCONCLUSIVE"  # ci_low == margin, not >
