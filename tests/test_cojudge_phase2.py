"""TDD for tools/cojudge_phase2.py — blind grouped-by-fact co-judge harness.

Covers: within-fact dedup of identical prediction strings (so judge noise can't
manufacture discordant pairs), blind-key round-trip (grades map back to the right
arms), grade validation, McNemar discordant counts, and the stratified
diff-in-diff bootstrap (resamples hits/misses separately, keeping strata sizes).
"""
import pytest

from tools import cojudge_phase2 as CJ


def _preds(**arm_rows):
    """arm_rows: arm -> {cid: (question, gold, prediction)} -> nested dict shape."""
    out = {}
    for arm, rows in arm_rows.items():
        out[arm] = {
            cid: {"question": q, "gold": g, "prediction": p}
            for cid, (q, g, p) in rows.items()
        }
    return out


def test_build_dedups_identical_strings_within_fact():
    # R0 and R2 give the SAME string; R3 differs -> 2 unique candidates, not 3.
    preds = _preds(
        R0={"7": ("Q?", "gold", "same answer")},
        R2={"7": ("Q?", "gold", "same answer")},
        R3={"7": ("Q?", "gold", "different answer")},
    )
    records, key = CJ.build_blind_facts(preds, ["R0", "R2", "R3"], seed=1)
    assert len(records) == 1
    cands = records[0]["candidates"]
    assert len(cands) == 2  # deduped: "same answer" once, "different answer" once
    # the shared string's label maps to BOTH R0 and R2
    labels = key[records[0]["fact_id"]]["labels"]
    shared = [lab for lab, info in labels.items() if info["prediction"] == "same answer"]
    assert len(shared) == 1
    assert sorted(labels[shared[0]]["arms"]) == ["R0", "R2"]


def test_unblind_copies_grade_to_all_arms_sharing_a_string():
    preds = _preds(
        R0={"7": ("Q?", "gold", "same")},
        R2={"7": ("Q?", "gold", "same")},
        R3={"7": ("Q?", "gold", "other")},
    )
    records, key = CJ.build_blind_facts(preds, ["R0", "R2", "R3"], seed=1)
    fid = records[0]["fact_id"]
    labels = key[fid]["labels"]
    same_lab = next(l for l, i in labels.items() if i["prediction"] == "same")
    other_lab = next(l for l, i in labels.items() if i["prediction"] == "other")
    grades = {fid: {same_lab: "CORRECT", other_lab: "INCORRECT"}}
    cov, strict = CJ.unblind(key, grades)
    assert cov["7"] == {"R0": 1.0, "R2": 1.0, "R3": 0.0}
    assert strict["7"] == {"R0": 1.0, "R2": 1.0, "R3": 0.0}


def test_partial_counts_as_covered_not_strict():
    preds = _preds(R0={"7": ("Q?", "g", "p")})
    records, key = CJ.build_blind_facts(preds, ["R0"], seed=1)
    fid = records[0]["fact_id"]
    lab = records[0]["candidates"][0]["label"]
    cov, strict = CJ.unblind(key, {fid: {lab: "PARTIAL"}})
    assert cov["7"]["R0"] == 1.0
    assert strict["7"]["R0"] == 0.0


def test_build_requires_chunk_in_all_arms():
    preds = _preds(
        R0={"7": ("Q?", "g", "a"), "8": ("Q2?", "g2", "b")},
        R2={"7": ("Q?", "g", "a")},  # missing "8"
    )
    records, key = CJ.build_blind_facts(preds, ["R0", "R2"], seed=1)
    # only the intersection (chunk "7") is built
    assert [k["chunk_id"] for k in key.values()] == ["7"]


def test_validate_flags_missing_and_bad_verdicts():
    preds = _preds(
        R0={"7": ("Q?", "g", "a")},
        R2={"7": ("Q?", "g", "b")},
    )
    records, key = CJ.build_blind_facts(preds, ["R0", "R2"], seed=1)
    fid = records[0]["fact_id"]
    labs = [c["label"] for c in records[0]["candidates"]]
    # one good, one garbage verdict
    grades = {fid: {labs[0]: "CORRECT", labs[1]: "MAYBE"}}
    problems = CJ.validate_grades(key, grades)
    assert (fid, labs[1]) in problems
    assert (fid, labs[0]) not in problems
    # missing fact entirely
    assert CJ.validate_grades(key, {}) == [(fid, labs[0]), (fid, labs[1])] or \
           set(CJ.validate_grades(key, {})) == {(fid, labs[0]), (fid, labs[1])}


def test_mcnemar_counts_discordant_pairs():
    cov = {
        "1": {"T": 1.0, "C": 0.0},  # treatment-only
        "2": {"T": 1.0, "C": 0.0},  # treatment-only
        "3": {"T": 0.0, "C": 1.0},  # control-only
        "4": {"T": 1.0, "C": 1.0},  # concordant
        "5": {"T": 0.0, "C": 0.0},  # concordant
    }
    b, c = CJ.mcnemar_counts(cov, list(cov), "T", "C")
    assert (b, c) == (1, 2)  # b=control-only=1, c=treatment-only=2


def test_rate_is_mean_coverage():
    cov = {"1": {"T": 1.0}, "2": {"T": 0.0}, "3": {"T": 1.0}}
    assert CJ.rate(cov, list(cov), "T") == pytest.approx(2 / 3)


def test_stratified_did_keeps_strata_and_signs():
    # hits: treatment == control -> Δ_hit = 0
    # misses: treatment all covered, control none -> Δ_miss = +1
    # => DiD = +1, CI must be a degenerate point at +1 (no variance since all equal)
    cov = {}
    hit = ["h1", "h2", "h3"]
    miss = ["m1", "m2", "m3"]
    for c in hit:
        cov[c] = {"T": 1.0, "C": 1.0}
    for c in miss:
        cov[c] = {"T": 1.0, "C": 0.0}
    res = CJ.stratified_did_bootstrap(cov, hit, miss, "T", "C", n=2000, seed=7)
    assert res["delta_hit"] == pytest.approx(0.0)
    assert res["delta_miss"] == pytest.approx(1.0)
    assert res["did"] == pytest.approx(1.0)
    assert res["ci_low"] == pytest.approx(1.0) and res["ci_high"] == pytest.approx(1.0)


def test_stratified_did_detects_no_interaction():
    # Both strata have identical +0.5 effect -> DiD ~ 0, CI brackets 0.
    cov = {}
    hit = [f"h{i}" for i in range(20)]
    miss = [f"m{i}" for i in range(20)]
    for i, c in enumerate(hit):
        cov[c] = {"T": 1.0 if i % 2 else 0.0, "C": 0.0}  # Δ_hit = 0.5
    for i, c in enumerate(miss):
        cov[c] = {"T": 1.0 if i % 2 else 0.0, "C": 0.0}  # Δ_miss = 0.5
    res = CJ.stratified_did_bootstrap(cov, hit, miss, "T", "C", n=3000, seed=7)
    assert abs(res["did"]) < 1e-9
    assert res["ci_low"] <= 0.0 <= res["ci_high"]
