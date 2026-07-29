"""Sweep recompute_ratio on a held-out eval set and plot quality vs latency.

Runs the baseline measurement tool for each ratio in {0.0, 0.05, 0.10, 0.15,
0.20, 0.30, 0.50, 1.0} and emits a CSV + PNG plot of fKDS vs p50 latency.

Usage::

    python3 -m tools.sweep_recompute_ratio \
        --config examples/usecase4_bedrock_userguide/config.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --output-dir examples/usecase4_bedrock_userguide/recompute_sweep

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]


def run_ratio(
    ratio: float,
    config: str,
    eval_set: str,
    output_dir: Path,
    judge_provider: str,
    judge_model: str,
) -> dict:
    """Run baseline measurement for a single recompute_ratio and return summary."""
    ratio_out = output_dir / f"ratio_{ratio:.2f}"
    ratio_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tools.measure_baseline_fkds",
        "--config", config,
        "--eval-set", eval_set,
        "--output-dir", str(ratio_out),
        "--modes", "kv_meanpool",
        "--judge-provider", judge_provider,
        "--judge-model", judge_model,
        "--recompute-ratio", f"{ratio:.3f}",
    ]
    print(f"\n🔬 Running recompute_ratio={ratio:.2f}")
    subprocess.run(cmd, check=True)

    with open(ratio_out / "summary.json") as f:
        summary = json.load(f)
    mode = summary["modes"]["kv_meanpool"]
    return {
        "ratio": ratio,
        "fkds_mean": mode["fkds"]["mean"],
        "fkds_sem": mode["fkds"]["sem"],
        "p50_latency": mode["latency"]["p50"],
        "p95_latency": mode["latency"]["p95"],
        "cost": mode["cost_usd_total"],
    }


def plot(results: list[dict], output_dir: Path) -> None:
    """Plot fKDS vs p50 latency with error bars."""
    ratios = [r["ratio"] for r in results]
    fkds = [r["fkds_mean"] for r in results]
    sems = [r["fkds_sem"] for r in results]
    latencies = [r["p50_latency"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(latencies, fkds, yerr=sems, fmt="o-", capsize=4)
    for x, y, ratio in zip(latencies, fkds, ratios):
        ax.annotate(f"{ratio:.2f}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("p50 latency (s)")
    ax.set_ylabel("mean fKDS")
    ax.set_title("fKDS vs latency for recompute_ratio sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fkds_vs_latency.png", dpi=150)
    print(f"💾 Plot saved to {output_dir / 'fkds_vs_latency.png'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--eval-set", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-provider", default="anthropic")
    p.add_argument("--judge-model", default="claude-fable-5")
    p.add_argument("--ratios", nargs="+", type=float, default=RATIOS,
                   help="List of recompute_ratio values to sweep")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ratio in args.ratios:
        results.append(run_ratio(
            ratio, args.config, args.eval_set, out_dir,
            args.judge_provider, args.judge_model,
        ))

    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w") as f:
        f.write("ratio,fkds_mean,fkds_sem,p50_latency,p95_latency,cost_usd\n")
        for r in results:
            f.write(f"{r['ratio']},{r['fkds_mean']},{r['fkds_sem']},"
                    f"{r['p50_latency']},{r['p95_latency']},{r['cost']}\n")
    print(f"💾 CSV saved to {csv_path}")

    plot(results, out_dir)

    # Find the Pareto-optimal point with the best fKDS / latency trade-off.
    best = max(results, key=lambda r: r["fkds_mean"] / max(r["p50_latency"], 0.1))
    print(f"\n🏆 Best trade-off (fKDS / latency): ratio={best['ratio']:.2f}, "
          f"fKDS={best['fkds_mean']:.3f}, p50={best['p50_latency']:.2f}s")


if __name__ == "__main__":
    main()
