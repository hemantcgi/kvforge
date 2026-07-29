"""Run E4 ablations and E5 real attention divergence for the scientific revision.

Designed to run on the GPU host.  Assumes each use-case has already been
indexed and trained to LoRA v1 so that the model, vector store, and KV
tensors are available.

Usage:

    export ANTHROPIC_API_KEY=...
    python tools/run_e4_e5_experiments.py \
        --output docs/scientific_revision_real \
        --e4-ucs usecase2_pubmedqa,usecase3_squad \
        --max-samples 50

Arguments:
    --output          Output directory for all results (default: docs/scientific_revision_real)
    --e4-ucs          Comma-separated use-case dir names for the ablation grid.
    --max-samples     Cap evaluation questions per use-case.
    --judge-provider  Judge provider (anthropic/openai/gemini).
    --judge-model     Judge model name.
    --skip-e5         Run only E4 ablations.
    --skip-e4         Run only E5 real attention divergence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

USE_CASES = [
    ("UC1 Customer Support", "examples/usecase1_customer_support"),
    ("UC2 PubMedQA", "examples/usecase2_pubmedqa"),
    ("UC3 SQuAD", "examples/usecase3_squad"),
    ("UC4 Bedrock", "examples/usecase4_bedrock_userguide"),
]


def run(cmd: list[str], cwd: str | Path = ROOT, check: bool = True) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=check)


def run_e5(config: str, output_dir: Path, max_samples: int, checkpoint: str | None = None):
    out = output_dir / "eval_attention_divergence.json"
    cmd = [
        "python", "-m", "pipeline.eval_attention_divergence",
        "--config", config,
        "--output", str(out),
        "--max-samples", str(max_samples),
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    run(cmd)
    return json.loads(out.read_text())


def train_variant(config: str, faqs: str, checkpoint_dir: str, uniform: bool, seed: int) -> None:
    cmd = [
        "python", "-m", "pipeline.lora_trainer",
        "--config", config,
        "--faqs", faqs,
        "--checkpoint-dir", checkpoint_dir,
        "--seed", str(seed),
        "--from-base",
    ]
    if uniform:
        cmd.append("--uniform-sampling")
    run(cmd)


def generate_heuristic_faqs(config: str, output: str, count: int = 50) -> None:
    if Path(output).exists():
        print(f"Using existing heuristic FAQs: {output}")
        return
    cmd = [
        "python", "tools/generate_faqs.py",
        "--config", config,
        "--count", str(count),
        "--output", output,
    ]
    run(cmd)


def set_version_checkpoint(config: str, checkpoint_dir: str, lora_version: int) -> None:
    """Temporarily point version.json at a specific checkpoint for compute-kv/eval."""
    sys.path.insert(0, str(ROOT))
    import core.version as ver
    cfg = json.loads(Path(config).read_text())
    ver.init(cfg)
    data = ver.load()
    data["checkpoint_path"] = checkpoint_dir
    data["current_lora_version"] = lora_version
    ver.save(data)
    print(f"Set version.json → checkpoint={checkpoint_dir} version={lora_version}")


def backup_version(config: str, backup_path: str) -> None:
    cfg = json.loads(Path(config).read_text())
    src = Path(cfg.get("version_file", "version.json"))
    dst = Path(backup_path)
    dst.write_text(src.read_text())
    print(f"Backed up version.json → {backup_path}")


def restore_version(backup_path: str, config: str) -> None:
    cfg = json.loads(Path(config).read_text())
    dst = Path(cfg.get("version_file", "version.json"))
    dst.write_text(Path(backup_path).read_text())
    print(f"Restored version.json from {backup_path}")


def recompute_kv(config: str) -> None:
    run(["python", "-m", "pipeline.kv_indexer", "--config", config, "compute-kv"])


def evaluate_prs(config: str, faqs: str, checkpoint: str, output: Path,
                 sample: int = 10) -> float:
    # PRS evaluator writes only to stdout; we capture and parse it.
    cmd = [
        "python", "-m", "pipeline.prs_evaluator",
        "--config", config,
        "--faqs", faqs,
        "--checkpoint", checkpoint,
        "--skip-version-update",
        "--sample", str(sample),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    prs = None
    for line in result.stdout.splitlines():
        if "PRS after round" in line:
            try:
                prs = float(line.split(":")[-1].strip())
            except Exception:
                pass
    output.write_text(json.dumps({"prs": prs}, indent=2))
    return prs


def evaluate_e1(config: str, checkpoint: str, output: Path, max_samples: int,
                judge_provider: str, judge_model: str) -> dict:
    cmd = [
        "python", "-m", "pipeline.eval_phase_quality",
        "--config", config,
        "--mode", "all",
        "--output", str(output),
        "--checkpoint", checkpoint,
        "--judge-provider", judge_provider,
        "--judge-model", judge_model,
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    env = os.environ.copy()
    # Ensure the key is visible to the child process.
    run(cmd)
    return json.loads(output.read_text())


def train_faqs_path(uc_dir: str) -> str:
    """Return the cloud-LLM FAQ file used for the main LoRA training."""
    base = Path(uc_dir)
    if (base / "faqs_train.json").exists():
        return str(base / "faqs_train.json")
    return str(base / "faqs.json")


def run_ablation(uc_name: str, uc_dir: str, output_dir: Path, args) -> dict:
    print(f"\n{'='*70}")
    print(f"▶ E4 ablation: {uc_name}")
    print(f"{'='*70}")
    config = str(Path(uc_dir) / "config.json")
    base = Path(uc_dir)
    lora_dir = base / "lora_checkpoints"
    backup = output_dir / "version.json.bak"

    faqs_cloud = train_faqs_path(uc_dir)
    faqs_heuristic = str(base / "faqs_heuristic.json")

    # Save original v1 state.
    backup_version(config, str(backup))

    # Generate heuristic FAQs once.
    generate_heuristic_faqs(config, faqs_heuristic, count=args.faq_count)

    variants = {
        "v1_seed": {
            "faqs": faqs_cloud,
            "checkpoint": str(lora_dir / "v1_seed"),
            "uniform": False,
            "label": "tier_weighted_cloud",
        },
        "v2_uniform": {
            "faqs": faqs_cloud,
            "checkpoint": str(lora_dir / "v2_uniform"),
            "uniform": True,
            "label": "uniform_cloud",
        },
        "v3_heuristic": {
            "faqs": faqs_heuristic,
            "checkpoint": str(lora_dir / "v3_heuristic"),
            "uniform": True,
            "label": "uniform_heuristic",
        },
    }

    results = {}
    for vname, v in variants.items():
        print(f"\n▶ Training {vname} ({v['label']}) ...")
        train_variant(config, v["faqs"], v["checkpoint"], v["uniform"], seed=args.seed)

        # Recompute KV under this variant's checkpoint.
        version_num = int(Path(v["checkpoint"]).name.lstrip("v").split("_")[0])
        set_version_checkpoint(config, v["checkpoint"], version_num)
        recompute_kv(config)

        # PRS evaluation.
        prs_out = output_dir / f"prs_{v['label']}.json"
        prs = evaluate_prs(config, v["faqs"], v["checkpoint"], prs_out,
                           sample=args.prs_sample)

        # E1 factual evaluation.
        e1_out = output_dir / f"eval_phase_quality_{v['label']}.json"
        e1 = evaluate_e1(config, v["checkpoint"], e1_out, args.max_samples,
                         args.judge_provider, args.judge_model)

        results[v["label"]] = {
            "checkpoint": v["checkpoint"],
            "prs": prs,
            "e1_summary": {
                mode: {
                    "em": e1["modes"][mode]["summary"]["em"]["mean"],
                    "token_f1": e1["modes"][mode]["summary"]["token_f1"]["mean"],
                    "judge": e1["modes"][mode]["summary"]["judge"]["mean"],
                    "n": e1["modes"][mode]["summary"]["n"],
                }
                for mode in e1["modes"]
            },
        }

    # Restore original v1 version.json.
    restore_version(str(backup), config)

    ablation_summary = output_dir / "eval_ablations.json"
    ablation_summary.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote ablation summary to {ablation_summary}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run E4 ablations and E5 real attention divergence")
    parser.add_argument("--output", default="docs/scientific_revision_real",
                        help="Output directory for results")
    parser.add_argument("--e4-ucs", default="usecase2_pubmedqa,usecase3_squad",
                        help="Comma-separated use-case dir names for E4 ablations")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max eval questions per use-case")
    parser.add_argument("--e5-samples", type=int, default=30,
                        help="Max questions per use-case for E5")
    parser.add_argument("--faq-count", type=int, default=50,
                        help="Number of heuristic FAQs to generate")
    parser.add_argument("--prs-sample", type=int, default=10,
                        help="Number of FAQs to sample for PRS evaluation per variant")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible LoRA training")
    parser.add_argument("--judge-provider", default="anthropic",
                        help="Judge provider: anthropic/openai/gemini")
    parser.add_argument("--judge-model", default="claude-3-haiku-20240307",
                        help="Judge model name")
    parser.add_argument("--skip-e5", action="store_true",
                        help="Skip E5 attention divergence")
    parser.add_argument("--skip-e4", action="store_true",
                        help="Skip E4 ablations")
    args = parser.parse_args()

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    e4_ucs = [u.strip() for u in args.e4_ucs.split(",") if u.strip()]

    aggregated = {"e5": {}, "e4": {}}

    # ── E5 real attention divergence for all four use-cases ─────────────────
    if not args.skip_e5:
        for uc_name, uc_dir in USE_CASES:
            tag = Path(uc_dir).name
            uc_out = output_dir / tag
            uc_out.mkdir(exist_ok=True)
            config = str(ROOT / uc_dir / "config.json")
            print(f"\n▶ E5 real attention divergence: {uc_name}")
            e5_result = run_e5(config, uc_out, args.e5_samples)
            aggregated["e5"][uc_name] = e5_result

    # ── E4 ablations for selected use-cases ─────────────────────────────────
    if not args.skip_e4:
        for uc_name, uc_dir in USE_CASES:
            tag = Path(uc_dir).name
            if tag not in e4_ucs:
                continue
            uc_out = output_dir / tag
            uc_out.mkdir(exist_ok=True)
            e4_result = run_ablation(uc_name, uc_dir, uc_out, args)
            aggregated["e4"][uc_name] = e4_result

    summary = output_dir / "e4_e5_results.json"
    # Preserve existing E5 results if we are running only E4 (so repeated runs
    # do not overwrite the divergence measurements).
    if args.skip_e5 and summary.exists():
        existing = json.loads(summary.read_text())
        aggregated["e5"].update(existing.get("e5", {}))
        aggregated["e4"].update(existing.get("e4", {}))
    summary.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote aggregated E4/E5 results to {summary}")


if __name__ == "__main__":
    main()
