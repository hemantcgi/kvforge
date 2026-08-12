"""Capacity gate integration for the KVForge pipeline.

After the first LoRA training round, this module compares Path A (text RAG)
against Path B (parametric / answer-from-weights) on a held-out set and
recommends whether the corpus+model has enough capacity to proceed to
Phase 2/3.

The implementation reuses the production inference and PRS generation paths
instead of duplicating context-building and generation logic.
"""
from __future__ import annotations

import numpy as np


def _make_judge_client():
    """Return an Anthropic judge client if credentials are available, else None."""
    try:
        from anthropic import Anthropic
        from dotenv import load_dotenv
        import os
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("anthropic_api_key")
        return Anthropic(api_key=api_key) if api_key else None
    except ImportError:
        return None


def _score_answer(question: str, answer: str, ground_truth: str,
                  judge_client, judge_model: str) -> tuple[float, dict, float]:
    """Return (f1, judge_dict, factual_accuracy) for a single answer."""
    from eval import metrics as eval_metrics
    f1 = eval_metrics.token_f1(answer, ground_truth)
    judge = eval_metrics.llm_judge(question, answer, ground_truth,
                                   client=judge_client, model=judge_model)
    factual_accuracy = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
    return f1, judge, factual_accuracy


def _summarize_records(records: list[dict]) -> dict:
    """Return mean, SEM and n for a list of per-question records."""
    vals = [r["factual_accuracy"] for r in records]
    arr = np.array(vals)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "factual_accuracy_mean": round(float(np.mean(arr)), 4),
        "factual_accuracy_sem": round(sem, 4),
        "n": len(arr),
        "records": records,
    }


def evaluate_text_rag(
    questions: list[dict],
    cfg: dict,
    judge_client,
    judge_model: str,
) -> dict:
    """Evaluate Path A using the production text-in-context fallback path."""
    from pipeline.kv_inference import answer_with_mode

    records = []
    for idx, item in enumerate(questions):
        q = item["question"]
        gt = item["answer"]
        ans, _ = answer_with_mode(q, cfg, force_mode="text_rag")
        f1, judge, fa = _score_answer(q, ans, gt, judge_client, judge_model)
        records.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "factual_accuracy": round(fa, 4),
            "f1": round(f1, 4),
            "judge_correct": int(judge["factually_correct"]),
        })
        print(f"  [{idx+1}/{len(questions)}] Path A factual_accuracy={fa:.3f}", flush=True)
    return _summarize_records(records)


def evaluate_parametric(
    questions: list[dict],
    pipe,
    tokenizer,
    judge_client,
    judge_model: str,
) -> dict:
    """Evaluate Path B using the production parametric generation helper."""
    from pipeline.prs_evaluator import _generate_parametric

    records = []
    for idx, item in enumerate(questions):
        q = item["question"]
        gt = item["answer"]
        ans = _generate_parametric(q, pipe, tokenizer, sft_format="chat")
        f1, judge, fa = _score_answer(q, ans, gt, judge_client, judge_model)
        records.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "factual_accuracy": round(fa, 4),
            "f1": round(f1, 4),
            "judge_correct": int(judge["factually_correct"]),
        })
        print(f"  [{idx+1}/{len(questions)}] Path B factual_accuracy={fa:.3f}", flush=True)
    return _summarize_records(records)


def run_capacity_gate(
    questions: list[dict],
    cfg: dict,
    pipe,
    tokenizer,
    judge_client=None,
    judge_model: str = "claude-sonnet-4-5-20250929",
) -> dict:
    """Compare Path A vs Path B and return a capacity-gate decision dict.

    Args:
        questions: Held-out eval items with ``question`` and ``answer`` keys.
        cfg: Flat datasource config dict.
        pipe: HuggingFace text-generation pipeline for parametric answers.
        tokenizer: Model tokenizer.
        judge_client: Optional LLM judge client (e.g., Anthropic).
        judge_model: Judge model identifier passed to ``llm_judge``.

    Returns:
        Dict with ``path_a``, ``path_b``, ``beats_path_a``, ``margin`` and
        ``recommendation`` keys.
    """
    print("\n--- Evaluating Path A (text RAG) ---")
    path_a = evaluate_text_rag(questions, cfg, judge_client, judge_model)
    print(f"  Path A factual_accuracy: {path_a['factual_accuracy_mean']:.4f} +/- {path_a['factual_accuracy_sem']:.4f}")

    print("\n--- Evaluating Path B (parametric) ---")
    path_b = evaluate_parametric(questions, pipe, tokenizer, judge_client, judge_model)
    print(f"  Path B factual_accuracy: {path_b['factual_accuracy_mean']:.4f} +/- {path_b['factual_accuracy_sem']:.4f}")

    beats_path_a = path_b["factual_accuracy_mean"] > path_a["factual_accuracy_mean"]
    margin = path_b["factual_accuracy_mean"] - path_a["factual_accuracy_mean"]

    return {
        "path_a": path_a,
        "path_b": path_b,
        "beats_path_a": beats_path_a,
        "margin": round(margin, 4),
        "recommendation": (
            "Proceed to Phase 2"
            if beats_path_a
            else "Stay on Phase 1 — model capacity may be insufficient. Try a larger model."
        ),
    }
