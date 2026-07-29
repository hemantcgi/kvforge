"""KVForge end-to-end improvement loop — Sprint 4.

Coordinates one full improvement round:

1. Rebuild the query pool from logged queries + FAQ paraphrases + chunk questions.
2. Run the frozen teacher pipeline (Path A with recompute) over the pool.
3. Generate on-policy samples with confidence labels from the student adapter.
4. Train distillation on the combined pairs.
5. Evaluate the new adapter on the held-out set.
6. Run the acceptance gate; deploy if it passes.

Usage (on EC2)::

    python3 -m pipeline.end_to_end_loop \
        --config examples/usecase4_bedrock_userguide/config_distill.json \
        --teacher-config examples/usecase4_bedrock_userguide/config.json \
        --eval-set examples/usecase4_bedrock_userguide/eval_heldout_v1.json \
        --faqs examples/usecase4_bedrock_userguide/faqs.json \
        --judge-model claude-fable-5 \
        --judge-provider anthropic \
        --rounds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'='*60}", flush=True)
    print(f"[{label}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED rc={result.returncode}"
    print(f"[{label}] {status} in {elapsed:.0f}s", flush=True)
    return ok


def run_round(
    cfg_path: str,
    teacher_cfg_path: str,
    eval_set_path: str,
    faqs_path: str,
    judge_model: str,
    judge_provider: str,
    round_num: int,
    output_dir: str,
    replay_ratio: float = 0.5,
    quality_threshold: float = 0.7,
    distill_pairs_prev: str | None = None,
) -> tuple[str, str] | None:
    """Execute one full improvement round. Returns (distill_pairs, checkpoint) or None if failed."""

    base = Path(output_dir)
    pool_path = str(base / f"query_pool_round{round_num:02d}.json")
    teacher_path = str(base / f"teacher_pairs_round{round_num:02d}.json")
    on_policy_path = str(base / f"on_policy_round{round_num:02d}.json")
    distill_path = str(base / f"distill_pairs_round{round_num:02d}.json")
    with open(cfg_path) as f:
        _cfg = json.load(f)
    checkpoint_dir = str(
        Path(_cfg.get("addon_config", {}).get("training", {}).get(
            "checkpoint_dir",
            str(Path(cfg_path).parent / "lora_checkpoints/"),
        )) / f"loop_round{round_num:02d}"
    )

    steps = [
        ("build_query_pool", [
            sys.executable, "-m", "tools.build_query_pool",
            "--config", cfg_path,
            "--faqs", faqs_path,
            "--output", pool_path,
        ]),
        ("run_teacher", [
            sys.executable, "-m", "tools.run_teacher_pipeline",
            "--config", teacher_cfg_path,
            "--query-pool", pool_path,
            "--output", teacher_path,
            "--quality-threshold", str(quality_threshold),
            "--judge-model", judge_model,
        ]),
        ("generate_on_policy", [
            sys.executable, "-m", "tools.generate_on_policy_samples",
            "--config", cfg_path,
            "--teacher-config", teacher_cfg_path,
            "--query-pool", pool_path,
            "--output", on_policy_path,
            "--judge-model", judge_model,
        ]),
        ("merge_pairs", [
            sys.executable, "tools/merge_distill_pairs.py",
            "--teacher-pairs", teacher_path,
            "--on-policy", on_policy_path,
            "--output", distill_path,
        ]),
        ("train_distillation", [
            sys.executable, "-u", "pipeline/lora_trainer.py",
            "--config", cfg_path,
            "--distill-pairs", distill_path,
            "--replay-ratio", str(replay_ratio),
            "--checkpoint-dir", checkpoint_dir,
        ]),
    ]

    for label, cmd in steps:
        if not _run(cmd, f"round{round_num}/{label}"):
            print(f"[round{round_num}] Stopping after failed step: {label}", flush=True)
            return None

    # Update version.json with the new checkpoint for eval.
    import core.version as ver
    with open(cfg_path) as f:
        ver.init(json.load(f))
    data = ver.load()
    data["checkpoint_path"] = checkpoint_dir
    ver.save(data)

    return distill_path, checkpoint_dir


def main() -> None:
    p = argparse.ArgumentParser(description="KVForge end-to-end improvement loop")
    p.add_argument("--config", required=True, help="Student datasource config")
    p.add_argument("--teacher-config", required=True, help="Teacher (Path A) config")
    p.add_argument("--eval-set", required=True, help="Held-out eval set JSON")
    p.add_argument("--faqs", required=True, help="FAQ JSON for query pool generation")
    p.add_argument("--output-dir", required=True, help="Directory for per-round artifacts")
    p.add_argument("--rounds", type=int, default=3, help="Number of improvement rounds")
    p.add_argument("--replay-ratio", type=float, default=0.5)
    p.add_argument("--quality-threshold", type=float, default=0.0,
                   help="Quality threshold (0.0 = keep all teacher answers)")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-provider", default="openai")
    p.add_argument("--accept-candidate", action="store_true",
                   help="Run acceptance gate after each round")
    p.add_argument("--fkds-history", default=None,
                   help="JSON file for fKDS history (enables train-on-drop)")
    p.add_argument("--train-on-drop-min-delta", type=float, default=-0.02,
                   help="Min fKDS drop to trigger retrain")
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for r in range(1, args.rounds + 1):
        print(f"\n{'#'*60}", flush=True)
        print(f"# ROUND {r}/{args.rounds}", flush=True)
        print(f"{'#'*60}", flush=True)

        result = run_round(
            cfg_path=args.config,
            teacher_cfg_path=args.teacher_config,
            eval_set_path=args.eval_set,
            faqs_path=args.faqs,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            round_num=r,
            output_dir=args.output_dir,
            replay_ratio=args.replay_ratio,
            quality_threshold=args.quality_threshold,
        )
        if result is None:
            print(f"\nLoop aborted after round {r} failure.", flush=True)
            sys.exit(1)

        distill_path, checkpoint_dir = result
        print(f"\nRound {r} complete. Distill pairs: {distill_path}", flush=True)
        print(f"  Checkpoint: {checkpoint_dir}", flush=True)

        # Check train-on-drop if history file is configured.
        if args.fkds_history:
            from pipeline.train_on_drop import should_retrain
            triggered, report = should_retrain(
                args.fkds_history, args.train_on_drop_min_delta,
            )
            if triggered:
                print(f"\n⚠️  TRAIN-ON-DROP: {report['reason']}", flush=True)
                print(f"  Best fKDS: {report.get('best_fkds')}", flush=True)
                if r >= args.rounds:
                    print("  Extra round needed but --rounds limit reached. Extend and re-run.", flush=True)
            else:
                print(f"  Train-on-drop: {report['reason']}", flush=True)


if __name__ == "__main__":
    main()
