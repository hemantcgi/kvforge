#!/usr/bin/env python3
"""Score the RAFT Phase-2 blind co-judge and emit the pre-registered verdict.

Consumes (under --cojudge-dir):
  cojudge_p2_key.json, cojudge_p2_facts_meta.json, judge_out/shard_*.json
Produces: raft_phase2_verdict.json (+ console summary). Uses the tested pure
functions in tools/cojudge_phase2.py and eval/stats.mcnemar_exact.

This regenerates the COMPUTED statistical core of the verdict. Three fields in
the checked-in verdict are curated/added-after and NOT written here:
HEADLINE (interpretation prose), wilcoxon_3level_paired, cross_session_drift.
See results/knowledge/uc4/cojudge_phase2/README.md.

Bands (plan §8):
  Primary R2 vs R0 (coverage@0.5, unit=fact):
    R2>=R0+0.040 AND chi2>=3.84 -> beats plain RAG (necessary)
    |R2-R0|<=0.012 -> RAG-DOMINATED (DEAD); R2<=R0-0.012 -> INTERFERENCE; else AMBIGUOUS.
  Interaction DiD = Δ_miss - Δ_hit (stratified bootstrap CI):
    overall gain AND DiD>0 CI excl 0 -> COMPLEMENTARITY CONFIRMED
    overall gain AND DiD<=0 -> GROUNDED-READING GAIN
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import cojudge_phase2 as CJ  # noqa: E402
from eval.stats import mcnemar_exact  # noqa: E402

ARMS = ["R0", "R1", "R2", "R3", "R1cb"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cojudge-dir", default=os.path.join(ROOT, "results/knowledge/uc4/cojudge_phase2"),
                    help="dir holding cojudge_p2_key.json, cojudge_p2_facts_meta.json, judge_out/")
    ap.add_argument("--out", default=os.path.join(ROOT, "results/knowledge/uc4/raft_phase2_verdict.json"),
                    help="verdict output path")
    args = ap.parse_args()

    key = json.load(open(os.path.join(args.cojudge_dir, "cojudge_p2_key.json")))
    meta = json.load(open(os.path.join(args.cojudge_dir, "cojudge_p2_facts_meta.json")))

    # ---- merge judge shards ----
    grades = {}
    shard_files = sorted(glob.glob(os.path.join(args.cojudge_dir, "judge_out", "shard_*.json")))
    for f in shard_files:
        g = json.load(open(f))
        for fid, labmap in g.items():
            grades[fid] = {k: v.strip().upper() for k, v in labmap.items()}
    print(f"loaded {len(shard_files)} shard outputs, {len(grades)} facts graded")

    # ---- validate completeness BEFORE scoring ----
    problems = CJ.validate_grades(key, grades)
    if problems:
        missing_facts = sorted({fid for fid, _ in problems})
        print(f"!!! {len(problems)} ungraded/invalid label-pairs across {len(missing_facts)} facts")
        print("    offending facts:", missing_facts[:20])
        bad_shards = sorted({int(fid[1:]) % 10 for fid in missing_facts})
        print("    re-dispatch shard(s):", bad_shards)
        sys.exit(2)
    print("validation OK: every fact/label graded with a valid verdict")

    cov, strict = CJ.unblind(key, grades)
    chunks = sorted(cov, key=lambda x: int(x))
    assert len(chunks) == 481

    # ---- subsets ----  (meta is keyed by fact_id; index by chunk_id)
    CHUNK_META = {m["chunk_id"]: m for m in meta.values()}
    hit = [c for c in chunks if CHUNK_META[c]["hit"]]
    miss = [c for c in chunks if not CHUNK_META[c]["hit"]]
    clean = [c for c in chunks if not CHUNK_META[c]["leaked"]]
    leak = [c for c in chunks if CHUNK_META[c]["leaked"]]
    assert len(hit) == 290 and len(miss) == 191 and len(clean) == 406 and len(leak) == 75

    def rates(arm):
        return {
            "all": round(CJ.rate(cov, chunks, arm), 4),
            "hit": round(CJ.rate(cov, hit, arm), 4),
            "miss": round(CJ.rate(cov, miss, arm), 4),
            "clean": round(CJ.rate(cov, clean, arm), 4),
            "leaked": round(CJ.rate(cov, leak, arm), 4),
            "strict_all": round(CJ.rate(strict, chunks, arm), 4),
        }
    coverage = {a: rates(a) for a in ARMS}

    def mcnemar(arm_t, arm_c, subset=None):
        cs = subset if subset is not None else chunks
        b, c = CJ.mcnemar_counts(cov, cs, arm_t, arm_c)   # b=control-only, c=treat-only
        m = b + c
        chi2 = (abs(b - c) - 1) ** 2 / m if m else 0.0
        return {"b_control_only": b, "c_treat_only": c, "chi2": round(chi2, 3),
                "exact_p": round(mcnemar_exact(b, c), 5), "sig_chi2": chi2 >= 3.84}

    def contrast(arm_t, arm_c, subset=None, label=""):
        cs = subset if subset is not None else chunks
        d = CJ.rate(cov, cs, arm_t) - CJ.rate(cov, cs, arm_c)
        return {"label": label, "n": len(cs),
                f"{arm_t}": round(CJ.rate(cov, cs, arm_t), 4),
                f"{arm_c}": round(CJ.rate(cov, cs, arm_c), 4),
                "delta": round(d, 4), "mcnemar": mcnemar(arm_t, arm_c, cs)}

    # ---- PRIMARY: R2 vs R0 ----
    prim = contrast("R2", "R0", label="PRIMARY R2 vs R0 (coverage@0.5, all 481)")
    d = prim["delta"]
    sig = prim["mcnemar"]["sig_chi2"]
    if d >= 0.040 and sig:
        prim_band = "R2 BEATS PLAIN RAG (Δ>=+0.040 AND χ²>=3.84) — necessary; see interaction"
    elif abs(d) <= 0.012:
        prim_band = "RAG-DOMINATED (DEAD): |Δ|<=0.012 — FT adds nothing once passage in context"
    elif d <= -0.012:
        prim_band = "INTERFERENCE (publishable negative): priors override the passage"
    else:
        prim_band = "AMBIGUOUS -> needs 2nd paired seed (noise SE≈0.016 > 0.012)"

    # ---- INTERACTION: DiD = Δ_miss - Δ_hit, R2 vs R0 ----
    did = CJ.stratified_did_bootstrap(cov, hit, miss, "R2", "R0", n=20000, seed=42)
    ci_excl_0 = not (did["ci_low"] <= 0.0 <= did["ci_high"])
    overall_gain = (d >= 0.040 and sig)
    if overall_gain and did["did"] > 0 and ci_excl_0:
        inter_band = "COMPLEMENTARITY CONFIRMED (WORKS): overall gain AND Δ_miss-Δ_hit>0, CI excludes 0"
    elif overall_gain and did["did"] <= 0:
        inter_band = "GROUNDED-READING GAIN: overall gain but lands on hits (Δ_miss-Δ_hit<=0)"
    elif overall_gain:
        inter_band = "overall gain; interaction positive but CI includes 0 (miss-concentration not significant)"
    else:
        inter_band = "no overall gain -> interaction not applicable as 'WORKS' path"

    verdict = {
        "experiment": "RAFT Phase-2 FT+RAG complementarity — blind same-session Opus co-judge (5 arms)",
        "judge": "blind grouped-by-fact, deduped within fact, 10 shards x fresh Opus; coverage@0.5=(CORRECT+PARTIAL)/481",
        "n_facts": 481, "hit": 290, "miss": 191, "clean": 406, "leaked": 75,
        "arms": {"R0": "base+top5", "R1": "promo adapter+top5", "R2": "RAFT adapter+top5",
                 "R3": "RAFT closed-book", "R1cb": "promo adapter closed-book (prior deterministic dump)"},
        "coverage@0.5": coverage,
        "PRIMARY_R2_vs_R0": prim,
        "PRIMARY_band": prim_band,
        "INTERACTION_R2_vs_R0_diff_in_diff": {
            "delta_hit": round(did["delta_hit"], 4), "delta_miss": round(did["delta_miss"], 4),
            "structure_effect_miss_minus_hit": round(did["did"], 4),
            "ci95": [round(did["ci_low"], 4), round(did["ci_high"], 4)],
            "bootstrap_p": round(did["p"], 5), "ci_excludes_0": ci_excl_0},
        "INTERACTION_band": inter_band,
        "CLEAN_vs_LEAKED_R2_vs_R0": {
            "clean": contrast("R2", "R0", clean, "R2 vs R0 | CLEAN (n=406)"),
            "leaked": contrast("R2", "R0", leak, "R2 vs R0 | LEAKED (n=75)")},
        "SECONDARY_R1_vs_R0_rag": contrast("R1", "R0", label="does Phase-1 adapter add on top of retrieval?"),
        "SECONDARY_R3_vs_R1cb_closedbook": {
            **contrast("R3", "R1cb", label="RAFT objective vs plain-QA, closed-book (data-matched)"),
            "caveat": "R1cb generated in a prior session (deterministic greedy, identical gen cfg: cap48/chat/bare-Q); "
                      "re-judged fresh in THIS blind session. R3 closed-book is OOD for R2 (always trained WITH context) — directional."},
        "confounds_sec9": {
            "answer_in_context_on_misses": sum(1 for c in miss if CHUNK_META[c]["answer_in_context"]),
            "answer_in_context_on_misses_note": "gold chunk not in top-5 but answer present in another retrieved chunk; "
                "these misses can be solved by reading, not parametrics",
            "recovery_source_asymmetry": "CORPUS-SCAN-only oracle 21.7% miss vs 5.4% hit; at 77%/95% precision the "
                "effectively-closed-book-via-wrong-oracle excess on misses ≈3.8pp — could inflate Δ_miss; small vs ±0.040 bands",
            "length_ood_R2": "13/481 eval prompts exceed R2's 1207-tok training max_length (mild length-OOD; R0/R1 base/plain-QA unaffected)",
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(verdict, open(args.out, "w"), indent=2)

    print(json.dumps({"coverage@0.5": {a: coverage[a]["all"] for a in ARMS},
                      "PRIMARY": prim_band, "delta_R2_R0": d,
                      "INTERACTION": inter_band,
                      "DiD": verdict["INTERACTION_R2_vs_R0_diff_in_diff"]}, indent=2))
    print(f"\nwrote verdict -> {args.out}")


if __name__ == "__main__":
    main()
