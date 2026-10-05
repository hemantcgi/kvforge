#!/usr/bin/env python3
"""Tier-1 re-scoring of the matched-format study (Exp 5, offline, judge-free).

Recomputes length-controlled, judge-free metrics on the already-collected
matched-format records and emits a factorial table with paired bootstrap CIs,
non-inferiority verdicts, and the pooled interaction estimand D. No API calls,
no GPU, no retraining.

PRE-REGISTRATION (2026-10-05):
  * Primary length-control metric = first_k_precision at k in {20, 40}: overlap of
    the first k predicted tokens with the FULL gold, over k. This is the RECOVERED
    definition behind evaluation_plan.md's "common-token-budget recall" figures
    (expertqa 0.615/0.539, techqa 0.694/0.517 at k=20) — verified exact against
    these records. (The plan's "recall" label is a misnomer; it is a precision of
    the opening k tokens.)
  * Also reported: full token precision / recall / F1, and a SECONDARY symmetric
    budget-F1 (both arms truncated) matching the literal §0.1 prose.
  * Primary decision: non-inferiority of parametric vs RAG on first_k_precision at
    BOTH budgets, margin -0.02 (§0.2). Full-F1 NI reported as secondary/confounded.
  * Pooled interaction D over the external panel at each k (Exp 5's actual question:
    does the gap shrink under length control?).
  * SINGLE-SEED and EXPLORATORY (§0.2): gates whether Exp 4/6 training is worth
    starting; cannot support a headline claim.
  * Judge fields (judge_correct / factual_accuracy) are NOT used (single judge,
    length-biased per Revision-note-2). Judge-free by design.

Usage:
    python tools/rescore_matched_format.py \
        --root results/matched_format \
        --out  results/matched_format/tier1_rescore.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from eval.metrics import token_prf, budget_f1, first_k_precision, non_inferiority
from eval.stats import paired_bootstrap
from tools.mf_common import interaction_delta

CORPORA = ["uc1", "expertqa", "techqa", "emanual"]
EXTERNAL = ["expertqa", "techqa", "emanual"]
LENGTHS = ["short", "long"]
ARMS = {"parametric": "parametric_records.json", "rag": "text_rag_records.json"}
BUDGETS = [20, 40]
CMP_METRICS = ["f1", "recall", "precision", "fkp_20", "fkp_40", "sbf1_20", "sbf1_40"]


def _norm_q(q: str) -> str:
    return " ".join((q or "").lower().split())


def load_cell(root: str, corpus: str, length: str, arm: str) -> list[dict]:
    cell_dir = os.path.join(root, corpus, f"{length}_{arm}")
    fn = os.path.join(cell_dir, ARMS[arm])
    if not os.path.exists(fn):
        cand = glob.glob(os.path.join(cell_dir, "*records*.json"))
        if not cand:
            return []
        fn = cand[0]
    with open(fn) as fh:
        return json.load(fh)


def score_records(recs: list[dict]) -> dict[str, dict]:
    """Attach judge-free metrics to each record, keyed by normalised question."""
    by_q: dict[str, dict] = {}
    for r in recs:
        q = _norm_q(r.get("question", ""))
        if not q:
            continue
        pred = r.get("answer", "") or ""
        gold = r.get("ground_truth", "") or ""
        prf = token_prf(pred, gold)
        n_words = len(pred.split())
        rec = {
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
            "ans_words": n_words,
            "empty": 1 if not pred.strip() else 0,
        }
        for k in BUDGETS:
            rec[f"fkp_{k}"] = first_k_precision(pred, gold, k)      # PRIMARY (recovered)
            rec[f"sbf1_{k}"] = budget_f1(pred, gold, k)["f1"]       # secondary (symmetric)
            rec[f"short_lt_{k}"] = 1 if n_words < k else 0          # underfilled-window diag
        by_q[q] = rec
    return by_q


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median_int(xs: list[int]) -> int:
    return sorted(xs)[len(xs) // 2] if xs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/matched_format")
    ap.add_argument("--out", default="results/matched_format/tier1_rescore.json")
    args = ap.parse_args()

    report = {
        "meta": {
            "experiment": "Exp5 Tier1 (offline, judge-free, length-controlled)",
            "budgets": BUDGETS, "margin": -0.02, "bootstrap_resamples": 10000,
            "single_seed": True, "exploratory": True,
            "primary_metric": "first_k_precision (recovered plan definition)",
            "primary_metric_def": "overlap(tokens(pred)[:k], tokens(gold)) / len(tokens(pred)[:k])",
            "secondary_metric": "sbf1_k = symmetric budget-F1 (both arms truncated to k)",
            "judge_fields_used": False,
        },
        "cells": {}, "comparisons": {}, "interaction_D_external": {},
    }

    scored: dict[tuple, dict] = {}
    for corpus in CORPORA:
        for length in LENGTHS:
            for arm in ARMS:
                by_q = score_records(load_cell(args.root, corpus, length, arm))
                scored[(corpus, length, arm)] = by_q
                cell = {
                    "n": len(by_q),
                    "precision": _mean([r["precision"] for r in by_q.values()]),
                    "recall": _mean([r["recall"] for r in by_q.values()]),
                    "f1": _mean([r["f1"] for r in by_q.values()]),
                    "ans_words_median": _median_int([r["ans_words"] for r in by_q.values()]),
                    "empty_rate": _mean([r["empty"] for r in by_q.values()]),
                }
                for k in BUDGETS:
                    cell[f"fkp_{k}"] = _mean([r[f"fkp_{k}"] for r in by_q.values()])
                    cell[f"sbf1_{k}"] = _mean([r[f"sbf1_{k}"] for r in by_q.values()])
                    cell[f"short_lt_{k}"] = _mean([r[f"short_lt_{k}"] for r in by_q.values()])
                report["cells"][f"{corpus}/{length}/{arm}"] = cell

    # paired parametric-vs-RAG within each (corpus, length)
    for corpus in CORPORA:
        for length in LENGTHS:
            par, rag = scored[(corpus, length, "parametric")], scored[(corpus, length, "rag")]
            common = sorted(set(par) & set(rag))
            if not common:
                continue
            cmp = {"n_paired": len(common)}
            for metric in CMP_METRICS:
                control = [rag[q][metric] for q in common]      # A = RAG
                treatment = [par[q][metric] for q in common]    # B = parametric
                cmp[metric] = paired_bootstrap(control, treatment)   # delta = par - rag
            cmp["NI_fkp_20"] = non_inferiority(cmp["fkp_20"]["ci_low"], cmp["fkp_20"]["ci_high"])
            cmp["NI_fkp_40"] = non_inferiority(cmp["fkp_40"]["ci_low"], cmp["fkp_40"]["ci_high"])
            cmp["NI_f1_secondary"] = non_inferiority(cmp["f1"]["ci_low"], cmp["f1"]["ci_high"])
            report["comparisons"][f"{corpus}/{length}"] = cmp

    # pooled interaction D over the external panel, per metric per budget
    for metric in ["f1", "fkp_20", "fkp_40"]:
        pooled_perq: list[float] = []
        per_corpus = {}
        for corpus in EXTERNAL:
            pl, rl = scored[(corpus, "long", "parametric")], scored[(corpus, "long", "rag")]
            ps, rs = scored[(corpus, "short", "parametric")], scored[(corpus, "short", "rag")]
            common = sorted(set(pl) & set(rl) & set(ps) & set(rs))
            if not common:
                continue
            d = interaction_delta(
                [pl[q][metric] for q in common], [rl[q][metric] for q in common],
                [ps[q][metric] for q in common], [rs[q][metric] for q in common],
            )
            per_corpus[corpus] = {"D": d["D"], "n": d["n"]}
            pooled_perq.extend(d["per_question"])
        if pooled_perq:
            bs = paired_bootstrap([0.0] * len(pooled_perq), pooled_perq)  # CI on mean(D)
            report["interaction_D_external"][metric] = {
                "pooled_D": bs["delta"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                "p_value": bs["p_value"], "n": len(pooled_perq), "per_corpus": per_corpus,
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)

    # ---- human-readable summary ----
    print("\n=== Exp5 Tier1 re-score (judge-free, length-controlled) ===")
    print("PRIMARY = first_k_precision (recovered plan metric); margin=-0.02; "
          "SINGLE-SEED / EXPLORATORY\n")
    hdr = (f"{'cell':26} {'n':>4} {'prec':>6} {'rec':>6} {'f1':>6} "
           f"{'fkp20':>6} {'fkp40':>6} {'ans_w':>6} {'<40%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for corpus in CORPORA:
        for length in LENGTHS:
            for arm in ARMS:
                c = report["cells"][f"{corpus}/{length}/{arm}"]
                print(f"{corpus + '/' + length + '/' + arm:26} {c['n']:>4} "
                      f"{c['precision']:>6.3f} {c['recall']:>6.3f} {c['f1']:>6.3f} "
                      f"{c['fkp_20']:>6.3f} {c['fkp_40']:>6.3f} "
                      f"{c['ans_words_median']:>6} {c['short_lt_40']:>6.2f}")

    print("\n--- parametric vs RAG (paired; Δ = par - rag; 95% CI) ---")
    for corpus in CORPORA:
        for length in LENGTHS:
            key = f"{corpus}/{length}"
            if key not in report["comparisons"]:
                continue
            cmp = report["comparisons"][key]
            a, b = cmp["fkp_20"], cmp["fkp_40"]
            print(f"{key:16} n={cmp['n_paired']:>4}  "
                  f"Δfkp20={a['delta']:+.3f}[{a['ci_low']:+.3f},{a['ci_high']:+.3f}] {cmp['NI_fkp_20']['verdict']:12}  "
                  f"Δfkp40={b['delta']:+.3f}[{b['ci_low']:+.3f},{b['ci_high']:+.3f}] {cmp['NI_fkp_40']['verdict']}")

    print("\n--- pooled interaction D over external panel (Δlong - Δshort) ---")
    for metric, v in report["interaction_D_external"].items():
        print(f"  {metric:7} D={v['pooled_D']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}] "
              f"p={v['p_value']:.3f} n={v['n']}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
