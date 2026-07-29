"""E4 — Ablations: tier-weighted vs uniform replay, and FAQ source quality.

Usage:

    python -m pipeline.eval_ablations \
        --config examples/usecase4_bedrock_userguide/config.json \
        --faqs-train examples/usecase4_bedrock_userguide/faqs_train.json \
        --mode dry-run \
        --output examples/usecase4_bedrock_userguide/eval_ablations.json

This script is a skeleton for the two ablations requested in the scientific
revision plan:

(a) Tier-weighted vs uniform replay: train a LoRA with the standard tier-weighted
replay buffer, then retrain with uniform sampling (``--uniform-sampling``) and
compare the resulting PRS and E1 factual metrics.

(b) FAQ source: compare sleep-time (cloud LLM) FAQs against heuristic FAQs by
running the same E1 factual evaluation on the model trained with each FAQ source.

In ``--dry-run`` mode the script returns deterministic synthetic results that
encode the expected ordering: tier-weighted replay outperforms uniform replay;
cloud-LLM FAQs outperform heuristic FAQs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)


_DRY_RUN_ABLATIONS = {
    "tier_weighted_cloud": {"em": 0.45, "f1": 0.78, "judge": 0.80, "prs": 0.85},
    "uniform_cloud": {"em": 0.35, "f1": 0.68, "judge": 0.70, "prs": 0.78},
    "tier_weighted_heuristic": {"em": 0.30, "f1": 0.60, "judge": 0.62, "prs": 0.72},
}


def _dry_run(config: dict, seed: int) -> dict:
    rng = random.Random(seed)
    results = {}
    for name, params in _DRY_RUN_ABLATIONS.items():
        results[name] = {
            "em": round(params["em"] + rng.gauss(0, 0.02), 3),
            "token_f1": round(params["f1"] + rng.gauss(0, 0.02), 3),
            "judge": round(params["judge"] + rng.gauss(0, 0.02), 3),
            "prs": round(params["prs"] + rng.gauss(0, 0.01), 3),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E4: Ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--faqs-train", required=True)
    parser.add_argument("--uniform-sampling", action="store_true",
                        help="Train with uniform replay instead of tier-weighted")
    parser.add_argument("--heuristic-faqs", action="store_true",
                        help="Use heuristic FAQ source instead of cloud LLM")
    parser.add_argument("--mode", choices=["real", "dry-run"], default="dry-run")
    parser.add_argument("--dry-run-seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    if args.mode == "dry-run":
        results = _dry_run(cfg, seed=args.dry_run_seed)
    else:
        raise NotImplementedError(
            "Real ablation training requires GPU. Use --mode dry-run for the simulation."
        )

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    for name, scores in results.items():
        print(f"  {name}: EM={scores['em']} F1={scores['token_f1']} Judge={scores['judge']} PRS={scores['prs']}")


if __name__ == "__main__":
    main()
