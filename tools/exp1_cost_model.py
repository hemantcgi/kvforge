#!/usr/bin/env python3
"""Exp 1 break-even cost model: $/1M queries and N* from measured serving costs.

Consumes the per-query serving costs measured by exp1_serve_bench.py and the
workload repeat-structure from exp1_workload.py, and computes:
  * $/1M queries for each arm vs volume V,
  * the break-even volume N* where parametric becomes cheaper than each RAG arm,
  * how N* moves with query skew alpha.

Cost accounting (all in GPU-seconds, converted to $ at the instance rate):

  parametric one-time = index + KV_precompute + LoRA_train + post-train KV_recompute
    (KV terms included per the plan's §Exp1 setup; toggle with --no-kv to model a
     pure phase-3 deployment that never injects KV)
  parametric per-query = c_par           (a repeat is answered from weights, same cost)

  plain RAG one-time   = index           (no training, no KV)
  plain RAG per-query  = c_rag           (every query re-prefills; flat in alpha)

  prefix-cached RAG    = index one-time; a DISTINCT query costs c_rag (cold prefill),
                         a REPEAT costs c_warm (prefix cached). Effective per-query
                         cost falls as the repeat rate (hence alpha) rises.

KEY STRUCTURAL RESULT (not a bug): against *plain* RAG, N* is finite and
alpha-independent. Against *prefix-cached* RAG, higher alpha LOWERS RAG's
effective cost, so N* GROWS with alpha and can diverge — the opposite of the
plan's stated "N* decreases with alpha". Caching amortises repeats for RAG too,
and parametric repeats were already cheap. This directly tests the plan's
falsification criterion ("a prefix-cached baseline matches KVForge at all alpha").

CAVEAT: c_warm (true cached-repeat cost) is proxied by c_par (prefill-free, short
generation) unless --c-warm is given. It should be measured directly with an
all-repeats workload; flagged in the output.

Usage:
    python tools/exp1_cost_model.py \
        --par results/exp1/uc4_parametric.json --rag results/exp1/uc4_rag.json \
        --chunks 2520 --train-s 2000 --out results/exp1/uc4_costmodel.json
"""
from __future__ import annotations

import argparse
import json
import os

from tools.exp1_workload import sample_stream, stream_stats

A10G_USD_PER_HR = 1.006          # g5.xlarge on-demand
ALPHAS = [0.0, 0.5, 0.8, 1.0, 1.2]
VOLUMES = [10**3, 10**4, 10**5, 10**6]


def gpu_s(path: str) -> float:
    d = json.load(open(path))
    return d.get("realized_gpu_s_per_query", d.get("gpu_s_per_query"))


def break_even(one_time_s: float, c_par: float, c_comp: float, rag_one_time_s: float):
    """Smallest integer V with parametric cumulative cost <= competitor's.

    par(V)  = one_time_s      + V*c_par
    comp(V) = rag_one_time_s  + V*c_comp
    Returns None (never) if c_par >= c_comp.
    """
    denom = c_comp - c_par
    if denom <= 0:
        return None
    v = (one_time_s - rag_one_time_s) / denom
    return max(0, int(v) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", required=True, help="parametric bench/replay JSON")
    ap.add_argument("--rag", required=True, help="plain RAG bench/replay JSON")
    ap.add_argument("--rag-cache", help="prefix-cached RAG bench JSON (optional)")
    ap.add_argument("--c-warm", type=float, help="measured cached-repeat gpu_s/query; "
                    "default = c_par (prefill-free proxy)")
    ap.add_argument("--cache-replays", help="comma-sep per-alpha rag_cache REPLAY JSONs; "
                    "when given, N* vs cached RAG uses MEASURED realized costs (not the proxy)")
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--kv-rate", type=float, default=0.20, help="s/chunk KV precompute (§5.3)")
    ap.add_argument("--train-s", type=float, default=2000.0)
    ap.add_argument("--index-s", type=float, default=60.0)
    ap.add_argument("--no-kv", action="store_true", help="exclude KV precompute/recompute "
                    "(pure phase-3 deployment)")
    ap.add_argument("--usd-per-hr", type=float, default=A10G_USD_PER_HR)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rate = args.usd_per_hr / 3600.0           # $/gpu-second
    c_par, c_rag = gpu_s(args.par), gpu_s(args.rag)
    c_warm = args.c_warm if args.c_warm is not None else c_par

    kv_s = 0.0 if args.no_kv else args.kv_rate * args.chunks
    par_one_time_s = args.index_s + kv_s + args.train_s + kv_s   # precompute + post-train recompute
    rag_one_time_s = args.index_s

    report = {
        "meta": {
            "c_par": c_par, "c_rag": c_rag, "c_warm": c_warm,
            "c_warm_is_proxy": args.c_warm is None,
            "rate_usd_per_gpu_s": round(rate, 8), "usd_per_hr": args.usd_per_hr,
            "chunks": args.chunks, "kv_included": not args.no_kv,
            "par_one_time_s": par_one_time_s, "rag_one_time_s": rag_one_time_s,
            "par_one_time_usd": round(par_one_time_s * rate, 4),
            "measured_cache": bool(args.cache_replays),
            "note": ("cached-RAG N* from MEASURED replay costs" if args.cache_replays
                     else "cached-RAG N* from PROXY c_warm=c_par (measure directly)"),
        },
        "per_query_usd": {
            "parametric": c_par * rate, "rag": c_rag * rate, "rag_cache_warm": c_warm * rate,
        },
        "break_even_N": {}, "cost_curve_usd_per_1M": {},
    }

    # N* vs plain RAG (alpha-independent)
    n_plain = break_even(par_one_time_s, c_par, c_rag, rag_one_time_s)
    report["break_even_N"]["vs_plain_rag"] = n_plain

    # N* vs prefix-cached RAG. Cached cost at volume V is
    #   index + distinct(a,V)*c_rag + (V-distinct)*c_warm,
    # so par<=cache  <=>  (par_one_time-index) <= distinct(a,V)*(c_rag-c_warm).
    # The break-even is therefore governed by a DISTINCT-query threshold, not V:
    #   distinct_threshold = (par_one_time-index)/(c_rag-c_warm).
    # Since distinct <= pool_size, if the pool has fewer distinct queries than the
    # threshold, parametric NEVER breaks even against cached RAG (at any volume/alpha).
    # PROXY path only when no measured replays are supplied (withdrawn otherwise).
    if not args.cache_replays:
        import numpy as np
        report["break_even_N"]["vs_prefix_cached_rag_PROXY"] = {}
        if c_rag - c_warm > 0:
            distinct_threshold = (par_one_time_s - rag_one_time_s) / (c_rag - c_warm)
        else:
            distinct_threshold = float("inf")
        report["break_even_N"]["distinct_threshold_vs_cached_PROXY"] = (
            None if distinct_threshold == float("inf") else int(distinct_threshold) + 1)
        Vgrid = np.unique(np.logspace(3, 7, 60).astype(int))
        for alpha in ALPHAS:
            nstar = None
            for V in Vgrid:
                d = stream_stats(sample_stream(args.pool_size, alpha, int(V), seed=42))["distinct"]
                if par_one_time_s + V * c_par <= rag_one_time_s + d * c_rag + (V - d) * c_warm:
                    nstar = int(V)
                    break
            report["break_even_N"]["vs_prefix_cached_rag_PROXY"][f"alpha_{alpha}"] = nstar

    # $/1M cost curves
    for V in VOLUMES:
        report["cost_curve_usd_per_1M"][str(V)] = {
            "parametric": round((par_one_time_s + V * c_par) * rate / V * 1e6, 2),
            "plain_rag": round((rag_one_time_s + V * c_rag) * rate / V * 1e6, 2),
        }

    # MEASURED N* vs prefix-cached RAG, from real replay realized costs (preferred).
    if args.cache_replays:
        report["break_even_N"]["vs_prefix_cached_rag_measured"] = {}
        for path in args.cache_replays.split(","):
            r = json.load(open(path.strip()))
            a = r["alpha"]
            c_eff = r["realized_gpu_s_per_query"]        # measured blended cost at this alpha
            n = break_even(par_one_time_s, c_par, c_eff, rag_one_time_s)
            report["break_even_N"]["vs_prefix_cached_rag_measured"][f"alpha_{a}"] = {
                "c_rag_eff_measured": c_eff,
                "prefix_cache_hit_rate": r.get("prefix_cache_hit_rate"),
                "repeat_rate": r.get("repeat_rate"),
                "N_star": n,
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)

    m = report["meta"]
    print("\n=== Exp1 break-even cost model (UC4, measured) ===")
    print(f"per-query $: parametric={c_par*rate:.6f}  rag={c_rag*rate:.6f}  "
          f"rag_warm={c_warm*rate:.6f}  (rate ${rate:.6f}/gpu-s)")
    print(f"parametric one-time: {par_one_time_s:.0f} gpu-s = ${m['par_one_time_usd']} "
          f"(KV {'in' if m['kv_included'] else 'ex'}cluded)")
    print(f"\nN* vs PLAIN rag: {n_plain} queries (alpha-independent)")
    be = report["break_even_N"]
    if "vs_prefix_cached_rag_measured" in be:
        print("\nN* vs PREFIX-CACHED rag by alpha, MEASURED (realized replay cost):")
        for a, v in be["vs_prefix_cached_rag_measured"].items():
            print(f"    {a}: c_eff={v['c_rag_eff_measured']:.5f} "
                  f"hit_rate={v['prefix_cache_hit_rate']} repeat_rate={v['repeat_rate']} "
                  f"N*={v['N_star'] if v['N_star'] is not None else '∞ (parametric not cheaper)'}")
    else:
        dt = be.get("distinct_threshold_vs_cached_PROXY")
        print(f"\nvs PREFIX-CACHED rag (PROXY c_warm=c_par): needs >= {dt} DISTINCT queries")
        for a, v in be.get("vs_prefix_cached_rag_PROXY", {}).items():
            print(f"    {a}: N*={v if v is not None else '∞'}")
    print("\n$/1M queries:")
    for V, c in report["cost_curve_usd_per_1M"].items():
        print(f"  V={int(V):>8}: parametric=${c['parametric']:>8}  plain_rag=${c['plain_rag']:>8}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
