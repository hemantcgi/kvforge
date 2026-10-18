"""Blind, grouped-by-fact co-judge harness for the RAFT Phase-2 complementarity study.

Pure logic only (no GPU, no judge calls here): builds blind judge records, maps
graded verdicts back to arms, and runs the pre-registered paired statistics.

Design (plan docs/raft_phase2_experiment_plan.md §4/§5.6/§8):
  * unit = fact (chunk_id); metric coverage@0.5 = (CORRECT+PARTIAL)/N.
  * One judge record per fact, holding that fact's UNIQUE prediction strings only
    (R0/R1/R2 share a prompt and frequently emit byte-identical answers; judging
    the same string twice would turn judge noise into fake discordant pairs in
    McNemar). Identical strings are collapsed to one label and the grade is copied
    back to every arm that produced it.
  * Candidates are shuffled and labelled A/B/C… per fact, so the judge is blind to
    arm identity. The un-blinding key is kept separate from the shards.
  * Interaction (Δ_miss − Δ_hit) significance via a stratified paired bootstrap
    that resamples the hit and miss strata independently, holding their sizes fixed.
"""
from __future__ import annotations

import random
import string

import numpy as np

COVERED = ("CORRECT", "PARTIAL")
VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT"}


def covered(verdict: str) -> float:
    """coverage@0.5: CORRECT or PARTIAL -> 1.0, else 0.0."""
    return 1.0 if verdict.strip().upper() in COVERED else 0.0


def strict_correct(verdict: str) -> float:
    return 1.0 if verdict.strip().upper() == "CORRECT" else 0.0


def build_blind_facts(preds_by_arm: dict, arms: list, seed: int = 42):
    """Build blind judge records + un-blinding key.

    preds_by_arm: {arm: {chunk_id(str): {question, gold, prediction}}}.
    Only chunk_ids present in EVERY listed arm are built (intersection).

    Returns (records, key):
      records: [{fact_id, question, gold, candidates: [{label, prediction}, …]}]
      key:     {fact_id: {chunk_id, question, gold,
                          labels: {label: {prediction, arms: [...]}}}}
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    common = set(preds_by_arm[arms[0]])
    for a in arms[1:]:
        common &= set(preds_by_arm[a])
    chunk_ids = sorted(common, key=lambda x: int(x))

    rng = random.Random(seed)
    records, key = [], {}
    for idx, cid in enumerate(chunk_ids):
        ref = preds_by_arm[arms[0]][cid]
        question, gold = ref["question"], ref["gold"]
        # unique prediction string -> arms that produced it (preserve arm order)
        uniq: dict[str, list] = {}
        for a in arms:
            p = preds_by_arm[a][cid]["prediction"]
            uniq.setdefault(p, []).append(a)
        preds = list(uniq.keys())
        rng.shuffle(preds)
        if len(preds) > len(string.ascii_uppercase):
            raise ValueError(f"fact {cid}: too many unique candidates ({len(preds)})")
        labels, cands = {}, []
        for li, p in enumerate(preds):
            lab = string.ascii_uppercase[li]
            labels[lab] = {"prediction": p, "arms": uniq[p]}
            cands.append({"label": lab, "prediction": p})
        fact_id = f"f{idx:04d}"
        records.append({"fact_id": fact_id, "question": question,
                        "gold": gold, "candidates": cands})
        key[fact_id] = {"chunk_id": cid, "question": question,
                        "gold": gold, "labels": labels}
    return records, key


def shard_records(records: list, n_shards: int) -> list:
    """Round-robin partition (balances shard sizes). Grouping stays by fact."""
    shards = [[] for _ in range(n_shards)]
    for i, r in enumerate(records):
        shards[i % n_shards].append(r)
    return shards


def validate_grades(key: dict, grades: dict) -> list:
    """Return [(fact_id, label)] for every pair missing or with a bad verdict."""
    problems = []
    for fid, meta in key.items():
        fg = grades.get(fid)
        for lab in meta["labels"]:
            v = None if fg is None else fg.get(lab)
            if v is None or v.strip().upper() not in VERDICTS:
                problems.append((fid, lab))
    return problems


def unblind(key: dict, grades: dict):
    """Map graded verdicts back to arms.

    Returns (cov, strict): {chunk_id: {arm: 1.0/0.0}} for coverage@0.5 and strict.
    """
    cov, strict = {}, {}
    for fid, meta in key.items():
        cid = meta["chunk_id"]
        fg = grades[fid]
        for lab, info in meta["labels"].items():
            v = fg[lab]
            for a in info["arms"]:
                cov.setdefault(cid, {})[a] = covered(v)
                strict.setdefault(cid, {})[a] = strict_correct(v)
    return cov, strict


def rate(cov: dict, chunks: list, arm: str):
    """Mean coverage of `arm` over `chunks` (None if empty)."""
    return sum(cov[c][arm] for c in chunks) / len(chunks) if chunks else None


def mcnemar_counts(cov: dict, chunks: list, arm_t: str, arm_c: str):
    """Discordant pair counts: (b = control-only covered, c = treatment-only)."""
    b = sum(1 for x in chunks if cov[x][arm_c] == 1 and cov[x][arm_t] == 0)
    c = sum(1 for x in chunks if cov[x][arm_t] == 1 and cov[x][arm_c] == 0)
    return b, c


def stratified_did_bootstrap(cov: dict, hit_chunks: list, miss_chunks: list,
                             arm_t: str, arm_c: str, n: int = 10000, seed: int = 42):
    """Stratified paired bootstrap for the interaction Δ_miss − Δ_hit.

    Resamples the hit and miss strata independently (sizes held fixed), so the CI
    reflects variability in both subset deltas. Two-sided bootstrap p for DiD != 0.
    """
    def arr(chunks, arm):
        return np.array([cov[c][arm] for c in chunks], dtype=float)

    d_hit = arr(hit_chunks, arm_t) - arr(hit_chunks, arm_c)
    d_miss = arr(miss_chunks, arm_t) - arr(miss_chunks, arm_c)
    did = float(d_miss.mean() - d_hit.mean())

    rng = np.random.default_rng(seed)
    nh, nm = len(hit_chunks), len(miss_chunks)
    boots = np.empty(n)
    for i in range(n):
        ih = rng.integers(0, nh, nh)
        im = rng.integers(0, nm, nm)
        boots[i] = d_miss[im].mean() - d_hit[ih].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    centered = boots - did
    p = min(1.0, float(np.mean(np.abs(centered) >= abs(did))))
    return {
        "delta_hit": float(d_hit.mean()),
        "delta_miss": float(d_miss.mean()),
        "did": did,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p": p,
    }
