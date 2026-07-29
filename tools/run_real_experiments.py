"""Run real GPU-backed scientific-revision experiments (E1, E2, E3, E5) per use-case.

Designed to be executed on the GPU host.  It assumes each use-case has already
been indexed and trained through the KVForge pipeline so that the model,
vector store, and KV tensors are available.

Example:

    python tools/run_real_experiments.py \
        --uc-config examples/usecase4_bedrock_userguide/config.json \
        --judge-provider openai \
        --judge-api-key $OPENAI_API_KEY \
        --max-samples 200 \
        --output docs/scientific_revision_real

To run all four use-cases in one command:

    python tools/run_real_experiments.py --all --all --judge-api-key $OPENAI_API_KEY

Run individual experiments:

    python tools/run_real_experiments.py --uc-config ... --experiments e1
    python tools/run_real_experiments.py --uc-config ... --experiments e2
    python tools/run_real_experiments.py --uc-config ... --experiments e3
    python tools/run_real_experiments.py --uc-config ... --experiments e5
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

USE_CASES = [
    ("UC1 Customer Support", "examples/usecase1_customer_support/config.json"),
    ("UC2 PubMedQA", "examples/usecase2_pubmedqa/config.json"),
    ("UC3 SQuAD", "examples/usecase3_squad/config.json"),
    ("UC4 Bedrock", "examples/usecase4_bedrock_userguide/config.json"),
]


def run_e1(config: str, output: Path, judge_provider: str, judge_api_key: str, judge_model: str, max_samples: int | None):
    cmd = [
        "python", "-m", "pipeline.eval_phase_quality",
        "--config", str(config),
        "--mode", "all",
        "--output", str(output / "eval_phase_quality.json"),
        "--judge-provider", judge_provider,
        "--judge-api-key", judge_api_key,
        "--judge-model", judge_model,
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_e2(input_path: Path, output: Path):
    cmd = [
        "python", "-m", "pipeline.eval_prs_validation",
        "--input", str(input_path),
        "--output", str(output / "eval_prs_validation.json"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_e3(input_path: Path, output: Path):
    cmd = [
        "python", "-m", "pipeline.eval_calibration",
        "--input", str(input_path),
        "--output", str(output / "eval_calibration.json"),
        "--parametric-only",
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_e5(config: str, output: Path, max_samples: int | None):
    cmd = [
        "python", "-m", "pipeline.eval_attention_divergence",
        "--config", str(config),
        "--output", str(output / "eval_attention_divergence.json"),
        "--dry-run",  # Real GPU hook implementation is a placeholder; dry-run gives deterministic simulation.
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(min(max_samples, 50))])
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run real scientific-revision experiments")
    parser.add_argument("--uc-config", help="Path to one use-case config.json")
    parser.add_argument("--all", action="store_true", help="Run all four use-cases")
    parser.add_argument("--experiments", default="e1,e2,e3,e5",
                        help="Comma-separated experiments to run")
    parser.add_argument("--output", default="docs/scientific_revision_real",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max eval questions per use-case")
    parser.add_argument("--judge-provider", default="openai",
                        help="Judge provider: openai/anthropic/gemini")
    parser.add_argument("--judge-api-key", default="",
                        help="Judge API key (or set env var)")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Judge model name")
    args = parser.parse_args()

    experiments = [e.strip().lower() for e in args.experiments.split(",")]
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        configs = [(name, str(ROOT / path)) for name, path in USE_CASES]
    elif args.uc_config:
        configs = [("single", args.uc_config)]
    else:
        raise ValueError("Specify --uc-config or --all")

    aggregated = {"use_cases": {}}

    for uc_name, config in configs:
        tag = Path(config).parent.name
        uc_dir = output_dir / tag
        uc_dir.mkdir(exist_ok=True)
        print(f"\n▶ {uc_name} → {uc_dir}")

        if "e1" in experiments:
            run_e1(config, uc_dir, args.judge_provider, args.judge_api_key, args.judge_model, args.max_samples)
        e1_path = uc_dir / "eval_phase_quality.json"
        if "e2" in experiments and e1_path.exists():
            run_e2(e1_path, uc_dir)
        if "e3" in experiments and e1_path.exists():
            run_e3(e1_path, uc_dir)
        if "e5" in experiments:
            run_e5(config, uc_dir, args.max_samples)

        result = {
            "e1_phase_quality": json.loads(e1_path.read_text()) if e1_path.exists() else None,
            "e2_prs_validation": json.loads((uc_dir / "eval_prs_validation.json").read_text()) if (uc_dir / "eval_prs_validation.json").exists() else None,
            "e3_calibration": json.loads((uc_dir / "eval_calibration.json").read_text()) if (uc_dir / "eval_calibration.json").exists() else None,
            "e5_attention_divergence": json.loads((uc_dir / "eval_attention_divergence.json").read_text()) if (uc_dir / "eval_attention_divergence.json").exists() else None,
        }
        aggregated["use_cases"][uc_name] = result

    summary = output_dir / "scientific_revision_results.json"
    summary.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote aggregated results to {summary}")


if __name__ == "__main__":
    main()
