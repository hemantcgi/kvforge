"""Tests for the Exp 1 break-even cost math."""
from tools.exp1_cost_model import break_even


def test_break_even_basic():
    # one_time=3068, c_par=0.082, c_comp=0.243, rag_one_time=60
    # V = (3068-60)/(0.243-0.082) = 3008/0.161 ~= 18683 -> +1
    n = break_even(3068.0, 0.082, 0.243, 60.0)
    assert 18600 <= n <= 18700


def test_break_even_none_when_competitor_not_more_expensive():
    # if parametric per-query >= competitor per-query, never breaks even
    assert break_even(3000.0, 0.20, 0.20, 60.0) is None
    assert break_even(3000.0, 0.25, 0.20, 60.0) is None


def test_break_even_cheaper_competitor_per_query_pushes_n_up():
    # a cheaper competitor per-query (e.g. cached RAG) raises N*
    n_plain = break_even(3068.0, 0.082, 0.243, 60.0)
    n_cached = break_even(3068.0, 0.082, 0.162, 60.0)   # cached repeat cost measured
    assert n_cached > n_plain
