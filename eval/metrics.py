"""Factual evaluation metrics for the KVForge scientific revision.

Provides SQuAD-style exact-match and token-F1, an LLM-as-judge correctness
rubric, expected calibration error, and bootstrap confidence intervals.

This module is intentionally dependency-light: it uses only the standard
library, numpy, and an optional external judge client.  It is designed to be
imported by both the existing ``pipeline/ab_evaluator.py`` and the new
scientific-revision scripts.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Callable

import numpy as np


def normalize_text(text: str) -> str:
    """SQuAD-style normalization: lower-case, strip punctuation and articles."""
    text = text.lower()
    text = text.replace("\u2019", "'")  # smart apostrophe
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> int:
    """Return 1 if normalized prediction matches normalized ground truth, else 0."""
    return int(normalize_text(prediction) == normalize_text(ground_truth))


def _tokens(text: str) -> list[str]:
    """Tokenize on whitespace after normalization."""
    return normalize_text(text).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-overlap F1 between prediction and ground truth."""
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    context: str | None = None,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> dict[str, Any]:
    """LLM-as-judge: binary factual correctness + short rationale.

    Args:
        question: The original question.
        prediction: The model's predicted answer.
        ground_truth: The reference answer.
        context: Optional retrieved context (for RAG-mode answers).
        client: Optional external judge client.  Must support
            ``client.chat.completions.create(...)``.  If ``None``, a
            deterministic heuristic fallback is used (useful for testing and
            dry-runs without API access).
        model: Judge model name when a client is provided.
        temperature: Sampling temperature for the judge (default 0 for
            determinism).

    Returns:
        Dict with keys ``factually_correct`` (bool), ``rationale`` (str), and
        ``raw_response`` (str).
    """
    if client is None:
        return _heuristic_judge(question, prediction, ground_truth, context)

    system_prompt = (
        "You are a strict factual correctness judge for question answering.\n\n"
        "A prediction is CORRECT only if it contains all the key facts in the "
        "ground-truth answer and does not contradict it. Minor wording differences, "
        "omissions of extra detail, and differences in article or phrasing are acceptable. "
        "A prediction that is partially correct but missing a critical fact, or that adds a "
        "contradictory fact, is INCORRECT.\n\n"
        "Reply with EXACTLY one of the following two lines, followed by a one-sentence rationale:\n"
        "CORRECT: <rationale>\n"
        "INCORRECT: <rationale>"
    )
    user_text = (
        f"Question: {question}\n\n"
        f"Ground-truth answer: {ground_truth}\n\n"
        f"Predicted answer: {prediction}\n"
    )
    if context:
        user_text += f"\nRetrieved context: {context[:2000]}"

    try:
        if hasattr(client, "messages"):
            # Anthropic client path: translate messages to Anthropic format.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            system = ""
            anthropic_messages = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    anthropic_messages.append({"role": m["role"], "content": m["content"]})
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system,
                messages=anthropic_messages,
            )
            text_blocks = [c for c in response.content if c.type == "text"]
            raw = text_blocks[0].text.strip() if text_blocks else ""
        else:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            raw = response.choices[0].message.content.strip()
        correct = raw.upper().startswith("CORRECT")
        rationale = raw.split(":", 1)[1].strip() if ":" in raw else raw
    except Exception as exc:
        raw = f"judge-error: {exc}"
        correct = False
        rationale = raw

    return {
        "factually_correct": correct,
        "rationale": rationale,
        "raw_response": raw,
    }


def _heuristic_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    context: str | None = None,
) -> dict[str, Any]:
    """Deterministic heuristic judge for dry-runs / CI.

    A prediction is marked correct if the token-F1 to the ground truth is at
    least 0.5.  This is intentionally a conservative lower bound and is NOT a
    substitute for a real LLM judge.
    """
    f1 = token_f1(prediction, ground_truth)
    correct = f1 >= 0.5 or exact_match(prediction, ground_truth) == 1
    rationale = (
        f"token-F1={f1:.2f} against ground truth; "
        f"threshold=0.5" + (" (heuristic fallback)" if context is not None else "")
    )
    return {
        "factually_correct": correct,
        "rationale": rationale,
        "raw_response": f"{'CORRECT' if correct else 'INCORRECT'}: {rationale}",
    }


def expected_calibration_error(
    confidences: list[float] | np.ndarray,
    correctness: list[int] | np.ndarray,
    bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) and per-bin statistics.

    Args:
        confidences: Model self-reported confidence in [0, 1].
        correctness: Binary correctness labels (0/1).
        bins: Number of equal-width confidence bins.

    Returns:
        Dict with ``ece`` (float), ``per_bin_accuracy`` (list), and
        ``per_bin_confidence`` (list).
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correctness, dtype=int)
    if len(conf) == 0:
        return {"ece": 0.0, "per_bin_accuracy": [], "per_bin_confidence": []}

    ece = 0.0
    per_bin_accuracy = []
    per_bin_confidence = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        if i == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            per_bin_accuracy.append(0.0)
            per_bin_confidence.append((lo + hi) / 2)
            continue
        bin_acc = float(np.mean(corr[mask]))
        bin_conf = float(np.mean(conf[mask]))
        per_bin_accuracy.append(bin_acc)
        per_bin_confidence.append(bin_conf)
        ece += np.sum(mask) / len(conf) * abs(bin_acc - bin_conf)

    return {
        "ece": float(ece),
        "per_bin_accuracy": per_bin_accuracy,
        "per_bin_confidence": per_bin_confidence,
    }


def bootstrap_ci(
    values: list[float] | np.ndarray,
    statistic: Callable | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a sample statistic.

    Args:
        values: Numeric observations.
        statistic: Function mapping an array to a scalar.  Defaults to np.mean.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (statistic, lower_bound, upper_bound).
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0)
    stat = statistic or np.mean
    point = float(stat(arr))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot[i] = stat(sample)
    alpha = 1 - ci
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def summarize_binary_metric(scores: list[int] | np.ndarray, **kwargs) -> dict[str, Any]:
    """Return mean, SEM, and bootstrap CI for a binary metric."""
    arr = np.asarray(scores, dtype=float)
    mean, lo, hi = bootstrap_ci(arr, **kwargs)
    sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "sem": sem,
        "ci_lower": lo,
        "ci_upper": hi,
        "n": len(arr),
    }
