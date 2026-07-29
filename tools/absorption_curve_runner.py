"""Orchestrates absorption-curve train-eval grid across conditions and seeds.

Each condition specifies a corpus variant (size, quality, diversity) and
training recipe. The runner trains a LoRA adapter, evaluates Path A and Path B,
and writes a structured JSON result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from eval.metrics import token_f1


def _build_condition_name(condition: dict) -> str:
    """Build a human-readable condition name from its parameters."""
    parts = [f"N{condition.get('N', 'all')}"]
    if "quality" in condition:
        parts.append(condition["quality"])
    if "diversity" in condition:
        parts.append(condition["diversity"])
    if "model" in condition:
        parts.append(condition["model"].split("/")[-1])
    return "_".join(parts)


def check_heldout_quarantine(train_qids: set, heldout_qids: set) -> None:
    """Hard assertion: no held-out question ID may appear in training data."""
    overlap = train_qids & heldout_qids
    assert not overlap, (
        f"Held-out quarantine violated: {len(overlap)} question IDs "
        f"appear in both training and held-out sets: {list(overlap)[:5]}"
    )


def run_condition(
    condition: dict,
    cfg: dict,
    seed: int,
    output_dir: Path,
) -> Path:
    """Train and evaluate one condition at one seed. Writes result JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_name = _build_condition_name(condition)
    result_path = output_dir / f"{condition_name}_seed{seed}.json"

    cfg_with_seed = {**cfg, "lora_seed": seed, **condition}

    train_cmd = [
        sys.executable, "-m", "pipeline.lora_trainer",
        "--config", cfg.get("config_path", ""),
        "--faqs", condition.get("faq_path", ""),
        "--seed", str(seed),
    ]
    if condition.get("rewrite_path"):
        train_cmd.extend(["--rewrites", condition["rewrite_path"]])

    eval_cmd = [
        sys.executable, "-m", "tools.measure_baseline_fkds",
        "--config", cfg.get("config_path", ""),
        "--eval-set", condition.get("eval_path", ""),
        "--output-dir", str(output_dir / condition_name),
        "--modes", "text_rag", "parametric",
        "--judge-model", cfg.get("judge_model", "gemini-2.5-flash"),
    ]

    if cfg.get("dry_run", False):
        result = {
            "condition": condition_name,
            "seed": seed,
            "condition_params": condition,
            "path_a_f1": 0.0,
            "path_b_f1": 0.0,
            "delta": 0.0,
            "per_question": [],
        }
        result_path.write_text(json.dumps(result, indent=2))
        return result_path

    subprocess.run(train_cmd, check=True)
    subprocess.run(eval_cmd, check=True)

    eval_summary = output_dir / condition_name / "summary.json"
    if eval_summary.exists():
        eval_data = json.loads(eval_summary.read_text())
    else:
        eval_data = {"modes": {}}

    path_a = eval_data.get("modes", {}).get("text_rag", {}).get("factual_accuracy", {})
    path_b = eval_data.get("modes", {}).get("parametric", {}).get("factual_accuracy", {})

    result = {
        "condition": condition_name,
        "seed": seed,
        "condition_params": condition,
        "path_a_f1": path_a.get("mean", 0.0),
        "path_b_f1": path_b.get("mean", 0.0),
        "delta": path_b.get("mean", 0.0) - path_a.get("mean", 0.0),
        "path_a_sem": path_a.get("sem", 0.0),
        "path_b_sem": path_b.get("sem", 0.0),
        "per_question": eval_data.get("modes", {}).get("text_rag", {}).get("records", []),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result_path


def run_grid(grid_config: dict, cfg: dict, output_dir: Path) -> list[Path]:
    """Run all conditions × seeds in a grid configuration.

    Args:
        grid_config: ``{conditions: [dict, ...], seeds: [42, 43, 44]}``.
        cfg: Base config dict.
        output_dir: Where to write result JSONs.

    Returns:
        List of result file paths.
    """
    output_dir = Path(output_dir)
    results: list[Path] = []
    for condition in grid_config.get("conditions", []):
        for seed in grid_config.get("seeds", [42]):
            result = run_condition(condition, cfg, seed, output_dir)
            results.append(result)
    return results


def aggregate_results(result_paths: list[Path]) -> dict:
    """Collect result JSONs into a per-condition summary with bootstrap CIs."""
    by_condition: dict[str, list[dict]] = {}
    for p in result_paths:
        data = json.loads(Path(p).read_text())
        by_condition.setdefault(data["condition"], []).append(data)

    summary: dict[str, dict] = {}
    for cond, runs in by_condition.items():
        deltas = np.array([r["delta"] for r in runs])
        rng = np.random.RandomState(42)
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(deltas, size=len(deltas), replace=True)
            boot_means.append(np.mean(sample))
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        summary[cond] = {
            "mean_delta": round(float(np.mean(deltas)), 4),
            "std_delta": round(float(np.std(deltas)), 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "n_seeds": len(runs),
            "crossover": ci_low > 0,
        }
    return summary
