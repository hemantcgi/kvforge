"""E2 — Validate PRS cosine-based accuracy against factual metrics.

Usage:

    python -m pipeline.eval_prs_validation \
        --input examples/usecase4_bedrock_userguide/eval_phase_quality.json \
        --output examples/usecase4_bedrock_userguide/eval_prs_validation.json

This script takes the per-question outputs from ``eval_phase_quality.py`` and
computes the correlation between (a) the legacy cosine-based PRS accuracy
component and (b) the factual metrics (exact-match, token-F1, LLM-judge).  It
also simulates a factual PRS variant that replaces cosine with token-F1, and
reports how often phase-gating decisions would flip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys_path = str(Path(__file__).resolve().parent.parent)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from eval import metrics


def _compute_cosine_prs(
    parametric_answers: list[str],
    rag_answers: list[str],
    ground_truths: list[str],
    embed_model: str = "BAAI/bge-small-en-v1.5",
) -> list[float]:
    """Legacy cosine-based accuracy_ratio used by PRS evaluator."""
    from fastembed import TextEmbedding
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    ratios = []
    for p, r, g in zip(parametric_answers, rag_answers, ground_truths):
        vecs = np.array(list(embedder.embed([p, r, g])))
        a = vecs[0] / (np.linalg.norm(vecs[0]) + 1e-9)
        b = vecs[2] / (np.linalg.norm(vecs[2]) + 1e-9)
        c = vecs[1] / (np.linalg.norm(vecs[1]) + 1e-9)
        param_sim = float(np.dot(a, b))
        rag_sim = float(np.dot(c, b))
        ratios.append(min(param_sim / (rag_sim + 1e-9), 1.0))
    return ratios


def _simulate_factual_prs(
    cosine_ratios: list[float],
    token_f1s: list[float],
    judges: list[int],
) -> dict:
    """Build a factual PRS variant and count flipped phase-gating decisions."""
    # Factual accuracy component: 50% token-F1 + 50% judge correctness.
    factual_acc = [0.5 * f1 + 0.5 * jc for f1, jc in zip(token_f1s, judges)]

    # PRS thresholds
    THRESHOLD_1 = 0.75
    THRESHOLD_2 = 0.80

    cos_mean = float(np.mean(cosine_ratios))
    fac_mean = float(np.mean(factual_acc))

    # Simulate two-round stability: require two consecutive rounds > threshold.
    # We use the same per-question distribution both rounds (best-case for cosine).
    cos_reaches_phase3 = cos_mean >= THRESHOLD_2
    fac_reaches_phase3 = fac_mean >= THRESHOLD_2

    return {
        "cosine_mean": round(cos_mean, 4),
        "factual_mean": round(fac_mean, 4),
        "cosine_reaches_phase3": cos_reaches_phase3,
        "factual_reaches_phase3": fac_reaches_phase3,
        "decision_flips": int(cos_reaches_phase3 != fac_reaches_phase3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E2: PRS validation against factual metrics")
    parser.add_argument("--input", required=True, help="eval_phase_quality.json output")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                        help="Embedding model for cosine PRS")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    modes = data.get("modes", {})

    if "parametric" not in modes or "text_rag" not in modes:
        raise ValueError("Input must contain parametric and text_rag modes")

    param_rows = modes["parametric"]["per_question"]
    rag_rows = modes["text_rag"]["per_question"]
    n = len(param_rows)

    ground_truths = [r["ground_truth"] for r in param_rows]
    param_answers = [r["prediction"] for r in param_rows]
    rag_answers = [r["prediction"] for r in rag_rows]
    ems = [r["em"] for r in param_rows]
    f1s = [r["token_f1"] for r in param_rows]
    judges = [r["judge_correct"] for r in param_rows]

    cosine_ratios = _compute_cosine_prs(
        param_answers, rag_answers, ground_truths, embed_model=args.embed_model
    )

    correlations = {}
    factual = [max(0.5 * f1 + 0.5 * jc, 0.0) for f1, jc in zip(f1s, judges)]
    for name, y in [("em", ems), ("token_f1", f1s), ("judge", judges), ("factual", factual)]:
        if len(set(cosine_ratios)) <= 1 or len(set(y)) <= 1:
            correlations[name] = {"pearson": None, "spearman": None,
                                   "note": "zero variance, correlation undefined"}
            continue
        correlations[name] = {
            "pearson": round(float(pearsonr(cosine_ratios, y)[0]), 4),
            "spearman": round(float(spearmanr(cosine_ratios, y)[0]), 4),
        }

    # Find disagreement cases: cosine says high but factual says low.
    disagreements = []
    for i, (cos, f1, em, jc) in enumerate(zip(cosine_ratios, f1s, ems, judges)):
        if cos >= 0.85 and f1 < 0.5:
            disagreements.append({
                "index": i,
                "question": param_rows[i]["question"],
                "cosine_ratio": round(cos, 4),
                "token_f1": round(f1, 4),
                "exact_match": em,
                "judge": jc,
            })

    prs_simulation = _simulate_factual_prs(cosine_ratios, f1s, judges)

    output = {
        "n": n,
        "cosine_ratios": [round(x, 4) for x in cosine_ratios],
        "correlations": correlations,
        "disagreements": disagreements,
        "prs_simulation": prs_simulation,
    }

    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    print(f"  Cosine vs factual Pearson: {correlations.get('factual', {}).get('pearson')}")
    print(f"  Disagreement cases: {len(disagreements)}")
    print(f"  Phase-3 decision flips: {prs_simulation['decision_flips']}")


if __name__ == "__main__":
    main()
