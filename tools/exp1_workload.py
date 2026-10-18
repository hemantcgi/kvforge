#!/usr/bin/env python3
"""Exp 1 workload generator: Zipf(alpha) query streams over a query pool.

The break-even analysis needs a *stream* of queries with controlled repetition,
because repeats are what amortise one-time cost (parametric answers a repeat from
weights at the same low cost; prefix-cached RAG serves a repeat's shared prefix
cheaply). This module turns a flat query pool into a Zipf-skewed stream and
reports the repeat structure that the cost model consumes.

It is pure (stdlib + numpy), version-independent, and testable without a GPU.

Zipf convention: items ranked 1..n, P(rank r) ∝ 1 / r**alpha.
  alpha = 0   -> uniform (no popularity skew; distinct count maximal)
  alpha > 0   -> skew; larger alpha -> fewer distinct queries at a given volume.

Usage:
    python tools/exp1_workload.py --pool query_pool.json \
        --alphas 0,0.5,0.8,1.0,1.2 --volumes 1000,10000,100000 \
        --out results/exp1/workload_stats.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ALPHAS = [0.0, 0.5, 0.8, 1.0, 1.2]
VOLUMES = [1000, 10000, 100000, 1000000]


def zipf_weights(n: int, alpha: float) -> np.ndarray:
    """Normalised Zipf probabilities over n ranked items. alpha=0 -> uniform."""
    if n <= 0:
        raise ValueError("n must be > 0")
    ranks = np.arange(1, n + 1, dtype=float)
    w = ranks ** (-alpha)
    return w / w.sum()


def sample_stream(n_pool: int, alpha: float, volume: int, seed: int = 42) -> np.ndarray:
    """Sample `volume` query indices from a pool of `n_pool` under Zipf(alpha).

    A random permutation assigns pool items to ranks per seed, so the popular
    set is not an artefact of pool ordering.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pool)          # which pool item holds each rank
    probs = zipf_weights(n_pool, alpha)
    ranks = rng.choice(n_pool, size=volume, p=probs)
    return perm[ranks]


def stream_stats(stream: np.ndarray) -> dict:
    """Repeat structure the cost model needs.

    distinct        : number of unique queries served (one-time work per distinct).
    repeat_rate     : fraction of requests that are repeats of an earlier request
                      = 1 - distinct/volume. This is the achievable cache-hit /
                      amortisation ceiling for that stream.
    empirical_alpha : slope of log(freq) vs log(rank), a sanity check on skew.
    """
    volume = int(stream.size)
    if volume == 0:
        return {"volume": 0, "distinct": 0, "repeat_rate": 0.0, "empirical_alpha": 0.0}
    _, counts = np.unique(stream, return_counts=True)
    distinct = int(counts.size)
    freqs = np.sort(counts)[::-1].astype(float)
    ranks = np.arange(1, distinct + 1, dtype=float)
    if distinct >= 2:
        slope = np.polyfit(np.log(ranks), np.log(freqs), 1)[0]
        empirical_alpha = float(-slope)
    else:
        empirical_alpha = 0.0
    return {
        "volume": volume,
        "distinct": distinct,
        "repeat_rate": 1.0 - distinct / volume,
        "empirical_alpha": empirical_alpha,
        "max_freq": int(freqs[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="JSON list of queries, or {'queries': [...]}")
    ap.add_argument("--pool-size", type=int, default=2000,
                    help="Used when --pool is omitted (synthetic pool).")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--volumes", default=",".join(str(v) for v in VOLUMES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/exp1/workload_stats.json")
    args = ap.parse_args()

    if args.pool:
        with open(args.pool) as fh:
            data = json.load(fh)
        pool = data["queries"] if isinstance(data, dict) else data
        n_pool = len(pool)
    else:
        n_pool = args.pool_size

    alphas = [float(x) for x in args.alphas.split(",")]
    volumes = [int(x) for x in args.volumes.split(",")]

    report = {"meta": {"n_pool": n_pool, "seed": args.seed}, "grid": {}}
    print(f"pool size = {n_pool}\n")
    hdr = f"{'alpha':>6} {'volume':>9} {'distinct':>9} {'repeat_rate':>12} {'emp_alpha':>10}"
    print(hdr)
    print("-" * len(hdr))
    for alpha in alphas:
        for v in volumes:
            st = stream_stats(sample_stream(n_pool, alpha, v, seed=args.seed))
            report["grid"][f"a{alpha}_v{v}"] = {"alpha": alpha, **st}
            print(f"{alpha:>6} {v:>9} {st['distinct']:>9} "
                  f"{st['repeat_rate']:>12.4f} {st['empirical_alpha']:>10.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
