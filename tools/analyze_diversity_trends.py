"""Trend analysis and visualization for Path B diversity validation experiments.

Reads `summary.json` files produced by `tools.measure_baseline_fkds` (and optional
`experiment_metadata.json` files) and produces:

- Per-dataset / per-model diversity trendline plots
- Coverage (Path B / Path A) plots with 95% confidence intervals
- Gap-to-90% plots
- CSV summary tables
- Trend-shape classification (rising, plateau, noisy flat, divergent)

Usage::

    python -m tools.analyze_diversity_trends \
        --results-dir results/pathb_diversity \
        --output-dir results/pathb_diversity/figures

Expected directory layout::

    results/pathb_diversity/
      uc4_bedrock_llama32_3b_1x_seed42/
        summary.json
        experiment_metadata.json  (optional)
      uc4_bedrock_llama32_3b_10x_seed42/
        summary.json
        ...

`experiment_metadata.json` schema::

    {
      "experiment_id": "uc4_bedrock_llama32_3b_10x_seed42",
      "dataset": "uc4_bedrock",
      "model_family": "llama3",
      "model_id": "meta-llama/Llama-3.2-3B-Instruct",
      "diversity_level": 10,
      "seed": 42
    }

If metadata is missing, the script attempts to parse the directory name with the
heuristic pattern::

    <dataset>_<model_family>_<diversity>x_seed<seed>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _mean_sem_ci(values: list[float], alpha: float = 0.05) -> dict[str, float]:
    """Return mean, SEM, and bootstrap percentile CI for a list of scalar values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "sem": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if n > 1:
        # bootstrap percentile CI
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(10_000):
            sample = rng.choice(arr, size=n, replace=True)
            boot_means.append(float(np.mean(sample)))
        boot = np.sort(boot_means)
        lower = float(np.percentile(boot, alpha / 2 * 100))
        upper = float(np.percentile(boot, (1 - alpha / 2) * 100))
    else:
        lower = upper = mean
    return {"mean": round(mean, 4), "sem": round(sem, 4),
            "ci_lower": round(lower, 4), "ci_upper": round(upper, 4), "n": n}


def _parse_dir_name(name: str) -> dict[str, Any] | None:
    """Try to parse experiment metadata from a directory name.

    Expected pattern: <dataset>_<model_family>_<diversity>x_seed<seed>
    Dataset and model_family may contain underscores, so we greedily match the
    trailing _<digits>x_seed<digits> suffix and treat the rest as
    dataset_model_family.
    """
    m = re.match(r"(.+)_(\d+)x_seed(\d+)$", name)
    if not m:
        return None
    prefix = m.group(1)
    diversity_level = int(m.group(2))
    seed = int(m.group(3))
    # Split prefix into dataset and model_family by last underscore.
    # This is a heuristic and may need manual correction via metadata.
    if "_" in prefix:
        dataset, model_family = prefix.rsplit("_", 1)
    else:
        dataset = prefix
        model_family = "unknown"
    return {
        "experiment_id": name,
        "dataset": dataset,
        "model_family": model_family,
        "diversity_level": diversity_level,
        "seed": seed,
    }


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Load a single experimental run from a result directory.

    Accepts runs with text_rag OR parametric mode (not both).
    Coverage ratios are computed in _load_all_runs by cross-referencing
    Path A baselines per dataset.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    meta_path = run_dir / "experiment_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = {}
    else:
        meta = _parse_dir_name(run_dir.name) or {}
    if not meta:
        return None

    modes = summary.get("modes", {})
    mode_key = "text_rag" if "text_rag" in modes else ("parametric" if "parametric" in modes else None)
    if mode_key is None:
        return None

    mode_data = modes.get(mode_key, {})
    # measure_baseline_fkds.py now reports raw factual_accuracy under that key.
    # Accept the legacy "fkds" key as a fallback for older summaries.
    fa_key = "factual_accuracy" if "factual_accuracy" in mode_data else "fkds"
    fa_summary = mode_data.get(fa_key, {})
    if not fa_summary:
        return None

    latency = mode_data.get("latency", {})
    cost = mode_data.get("cost_usd_total", 0.0)

    # Load per-question records
    records_path = run_dir / f"{mode_key}_records.json"
    fa_list = []
    if records_path.exists():
        try:
            with open(records_path) as f:
                records = json.load(f)
            fa_list = [
                r.get("factual_accuracy", r.get("fkds"))
                for r in records if isinstance(r, dict)
                and ("factual_accuracy" in r or "fkds" in r)
            ]
        except (json.JSONDecodeError, OSError):
            fa_list = []

    return {
        "experiment_id": meta.get("experiment_id", run_dir.name),
        "dataset": meta.get("dataset", "unknown"),
        "model_family": meta.get("model_family", "unknown"),
        "model_id": meta.get("model_id", "unknown"),
        "diversity_level": int(meta.get("diversity_level", 0)),
        "seed": int(meta.get("seed", 0)),
        "mode": mode_key,
        "fkds_mean": float(fa_summary.get("mean", 0.0)),
        "fkds_sem": float(fa_summary.get("sem", 0.0)),
        "fkds_n": int(fa_summary.get("n", 0)),
        "fkds_list": fa_list,
        "latency_p50": float(latency.get("p50", 0.0)),
        "cost_usd": float(cost),
    }


def _load_all_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load all runs, cross-reference Path A baselines, and compute coverage ratios."""
    raw_runs = []
    for run_dir in sorted(results_dir.rglob("summary.json")):
        run = _load_run(run_dir.parent)
        if run:
            raw_runs.append(run)

    # Separate into path_a baselines and path_b runs
    path_a = {r["experiment_id"]: r for r in raw_runs if r["mode"] == "text_rag"}
    path_b_runs = [r for r in raw_runs if r["mode"] == "parametric"]

    # For each path B run, find the matching path A baseline by dataset+model_family
    combined = []
    for b in path_b_runs:
        dataset = b["dataset"]
        model_family = b["model_family"]
        # Find the path A baseline for this dataset
        a = None
        for a_id, a_run in path_a.items():
            if a_run["dataset"] == dataset and a_run["model_family"] == model_family:
                a = a_run
                break
        if a is None:
            continue

        mean_text = a["fkds_mean"]
        mean_param = b["fkds_mean"]
        relative = mean_param / mean_text if mean_text > 0 else 0.0

        # Bootstrap CI on coverage ratio if paired records exist
        text_fkds_list = a.get("fkds_list", [])
        param_fkds_list = b.get("fkds_list", [])
        if len(text_fkds_list) == len(param_fkds_list) and len(text_fkds_list) > 1:
            rng = np.random.default_rng(42)
            ratios = []
            n = len(text_fkds_list)
            for _ in range(10_000):
                idx = rng.integers(0, n, size=n)
                t = np.mean([text_fkds_list[i] for i in idx])
                p = np.mean([param_fkds_list[i] for i in idx])
                if t > 0:
                    ratios.append(p / t)
            ratios = np.sort(ratios)
            rel_ci = {"ci_lower": float(np.percentile(ratios, 2.5)),
                       "ci_upper": float(np.percentile(ratios, 97.5))}
        else:
            rel_ci = {"ci_lower": 0.0, "ci_upper": 0.0}

        combined.append({
            "experiment_id": b["experiment_id"],
            "dataset": dataset,
            "model_family": model_family,
            "model_id": b.get("model_id", "unknown"),
            "diversity_level": b["diversity_level"],
            "seed": b.get("seed", 0),
            "path_a_fkds_mean": mean_text,
            "path_a_fkds_sem": a.get("fkds_sem", 0.0),
            "path_a_fkds_n": a.get("fkds_n", 0),
            "path_b_fkds_mean": mean_param,
            "path_b_fkds_sem": b.get("fkds_sem", 0.0),
            "path_b_fkds_n": b.get("fkds_n", 0),
            "relative_quality": relative,
            "relative_quality_ci_lower": rel_ci["ci_lower"],
            "relative_quality_ci_upper": rel_ci["ci_upper"],
            "gap_to_90": 0.90 - relative,
            "latency_p50": b.get("latency_p50", 0.0),
            "cost_usd": b.get("cost_usd", 0.0),
        })

    return combined


def _classify_trend(rows: list[dict[str, Any]]) -> str:
    """Classify a single dataset/model trendline shape.

    Expects rows sorted by diversity_level. Returns one of:
    - "rising": each step increases by >0.02
    - "plateau": last two levels within 0.02
    - "noisy": no consistent ordering
    - "insufficient_data": fewer than 2 levels
    """
    if len(rows) < 2:
        return "insufficient_data"
    rows = sorted(rows, key=lambda r: r["diversity_level"])
    fkdss = [r["path_b_fkds_mean"] for r in rows]

    diffs = [fkdss[i + 1] - fkdss[i] for i in range(len(fkdss) - 1)]
    if all(d > 0.02 for d in diffs):
        return "rising"
    if len(fkdss) >= 2 and abs(fkdss[-1] - fkdss[-2]) <= 0.02:
        return "plateau"
    return "noisy"


def _decision_recommendation(runs: list[dict[str, Any]]) -> list[str]:
    """Return a human-readable decision recommendation based on loaded runs."""
    if not runs:
        return ["No valid runs found."]

    # Group by dataset/model
    from collections import defaultdict
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["model_family"])].append(r)

    recommendations = []
    for (dataset, model), rows in groups.items():
        rows = sorted(rows, key=lambda r: r["diversity_level"])
        trend = _classify_trend(rows)
        max_rel = max(r["relative_quality"] for r in rows)
        best = max(rows, key=lambda r: r["relative_quality"])
        line = (
            f"{dataset} / {model}: {trend} trend, "
            f"max coverage = {max_rel:.1%} at {best['diversity_level']}x"
        )
        recommendations.append(line)

    # Cross-dataset consistency
    trends = {_classify_trend(rows) for rows in groups.values()}
    if len(trends) == 1:
        recommendations.append(f"All dataset/model trends agree: {trends.pop()}.")
    else:
        recommendations.append(
            f"Trends diverge across datasets/models: {sorted(trends)}. "
            "Do not generalize; expand to more corpora before model decision."
        )

    # Overall recommendation
    max_coverage = max(r["relative_quality"] for r in runs)
    if max_coverage >= 0.90:
        recommendations.append(
            "At least one condition reached 90% of Path A. Stop and productionize."
        )
    elif max_coverage >= 0.85 and any(
        _classify_trend(rows) == "rising" for rows in groups.values()
    ):
        recommendations.append(
            "Best coverage is 85-89% and still rising. Run cross-architecture A/B (Sprint 2)."
        )
    else:
        recommendations.append(
            "Best coverage <85% or plateaued. Skip cross-architecture and scale to 7-8B (Sprint 3)."
        )

    return recommendations


def _write_csv(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Write a CSV summary table of all runs."""
    import csv

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "experiment_id", "dataset", "model_family", "model_id",
        "diversity_level", "seed", "path_a_fkds_mean", "path_a_fkds_sem",
        "path_b_fkds_mean", "path_b_fkds_sem", "relative_quality",
        "relative_quality_ci_lower", "relative_quality_ci_upper",
        "gap_to_90", "latency_p50", "cost_usd",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({k: run.get(k, "") for k in fieldnames})
    print(f"Wrote CSV summary to {csv_path}")


def _plot_fkds_trend(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot fKDS vs diversity level, one line per dataset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in runs:
        by_dataset[r["dataset"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda r: r["diversity_level"])
        xs = [r["diversity_level"] for r in rows]
        ys = [r["path_b_fkds_mean"] for r in rows]
        ax.plot(xs, ys, marker="o", label=dataset)
    ax.set_xlabel("Diversity level (variations per chunk)")
    ax.set_ylabel("Path B fKDS")
    ax.set_title("Path B fKDS by Diversity Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fkds_by_diversity.png", dpi=150)
    fig.savefig(output_dir / "fkds_by_diversity.svg")
    plt.close(fig)


def _plot_coverage_trend(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot Path B / Path A coverage vs diversity level, one line per dataset."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in runs:
        by_dataset[r["dataset"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda r: r["diversity_level"])
        xs = [r["diversity_level"] for r in rows]
        ys = [r["relative_quality"] for r in rows]
        ax.plot(xs, ys, marker="o", label=dataset)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("Diversity level (variations per chunk)")
    ax.set_ylabel("Path B / Path A coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage vs Diversity Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_by_diversity.png", dpi=150)
    fig.savefig(output_dir / "coverage_by_diversity.svg")
    plt.close(fig)


def _plot_gap_to_90(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot gap to 90% target vs diversity level."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in runs:
        by_dataset[r["dataset"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda r: r["diversity_level"])
        xs = [r["diversity_level"] for r in rows]
        ys = [r["gap_to_90"] for r in rows]
        ax.plot(xs, ys, marker="o", label=dataset)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("Diversity level (variations per chunk)")
    ax.set_ylabel("Gap to 90% target (0.90 - coverage)")
    ax.set_title("Gap to 90% Target vs Diversity Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "gap_to_90.png", dpi=150)
    fig.savefig(output_dir / "gap_to_90.svg")
    plt.close(fig)


def _plot_coverage_by_model(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot coverage faceted by dataset, one line per model family."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_dataset_model = defaultdict(list)
    for r in runs:
        by_dataset_model[(r["dataset"], r["model_family"])].append(r)

    datasets = sorted({r["dataset"] for r in runs})
    n = len(datasets)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        models = sorted({k[1] for k in by_dataset_model if k[0] == dataset})
        for model in models:
            rows = sorted(by_dataset_model[(dataset, model)], key=lambda r: r["diversity_level"])
            xs = [r["diversity_level"] for r in rows]
            ys = [r["relative_quality"] for r in rows]
            ax.plot(xs, ys, marker="o", label=model)
        ax.axhline(0.90, color="red", linestyle="--", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Coverage")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_by_model.png", dpi=150)
    fig.savefig(output_dir / "coverage_by_model.svg")
    plt.close(fig)


def _plot_coverage_with_ci(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot coverage with 95% bootstrap confidence intervals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_dataset = defaultdict(list)
    for r in runs:
        by_dataset[r["dataset"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, rows in sorted(by_dataset.items()):
        rows = sorted(rows, key=lambda r: r["diversity_level"])
        xs = [r["diversity_level"] for r in rows]
        ys = [r["relative_quality"] for r in rows]
        lower = [r["relative_quality_ci_lower"] for r in rows]
        upper = [r["relative_quality_ci_upper"] for r in rows]
        ax.plot(xs, ys, marker="o", label=dataset)
        ax.fill_between(xs, lower, upper, alpha=0.2)
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("Diversity level (variations per chunk)")
    ax.set_ylabel("Path B / Path A coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage with 95% Bootstrap CI")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_with_ci.png", dpi=150)
    fig.savefig(output_dir / "coverage_with_ci.svg")
    plt.close(fig)


def _plot_coverage_heatmap(runs: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot a heatmap of coverage by (dataset, model, diversity level)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    from collections import defaultdict
    by_key = defaultdict(float)
    for r in runs:
        key = (r["dataset"], r["model_family"], r["diversity_level"])
        by_key[key] = max(by_key[key], r["relative_quality"])

    datasets = sorted({k[0] for k in by_key})
    models = sorted({k[1] for k in by_key})
    levels = sorted({k[2] for k in by_key})
    if not datasets or not models or not levels:
        return

    matrix = np.zeros((len(datasets) * len(models), len(levels)))
    ylabels = []
    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ylabels.append(f"{dataset}\n{model}")
            row_idx = i * len(models) + j
            for col_idx, level in enumerate(levels):
                matrix[row_idx, col_idx] = by_key.get((dataset, model, level), 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"{l}x" for l in levels])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Diversity level")
    ax.set_title("Coverage Heatmap (Path B / Path A)")
    fig.colorbar(im, ax=ax, label="Coverage")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_heatmap.png", dpi=150)
    fig.savefig(output_dir / "coverage_heatmap.svg")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Path B diversity validation trends.")
    p.add_argument("--results-dir", required=True, help="Directory containing run subdirectories")
    p.add_argument("--output-dir", required=True, help="Directory for plots and CSV")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib plot generation")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_all_runs(results_dir)
    if not runs:
        print(f"No valid runs found under {results_dir}")
        return

    print(f"Loaded {len(runs)} experimental runs")
    _write_csv(runs, output_dir)

    if not args.no_plots:
        _plot_fkds_trend(runs, output_dir)
        _plot_coverage_trend(runs, output_dir)
        _plot_gap_to_90(runs, output_dir)
        _plot_coverage_by_model(runs, output_dir)
        _plot_coverage_with_ci(runs, output_dir)
        _plot_coverage_heatmap(runs, output_dir)
        print(f"Wrote plots to {output_dir}")

    print("\nTrend classification and recommendations:")
    for line in _decision_recommendation(runs):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
