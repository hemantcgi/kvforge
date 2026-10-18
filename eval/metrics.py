"""Factual evaluation metrics for the KVForge scientific revision.

Provides exact-match and token-F1, an LLM-as-judge correctness
rubric, expected calibration error, and bootstrap confidence intervals.

This module is intentionally dependency-light: it uses only the standard
library, numpy, and an optional external judge client.  It is designed to be
imported by both the existing ``pipeline/ab_evaluator.py`` and the new
scientific-revision scripts.
"""

from __future__ import annotations

import math
import re
import time
import string
from collections import Counter
from typing import Any, Callable

import numpy as np


def normalize_text(text: str) -> str:
    """standard span-QA normalization: lower-case, strip punctuation and articles."""
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
    """span-overlap token F1 between prediction and ground truth."""
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


# Partial-credit judge constants.
PARTIAL_SCORE = 0.5  # score assigned when judge says PARTIAL
PARTIAL_SYSTEM_PROMPT = (
    "You are a strict factual correctness judge for question answering.\n\n"
    "A prediction is CORRECT only if it contains all the key facts. "
    "It is PARTIAL if it contains some correct information but is missing a critical fact. "
    "It is INCORRECT if it is completely wrong, contradictory, or empty.\n\n"
    "Minor wording differences and omissions of extra detail are acceptable.\n\n"
    "Reply with EXACTLY one of the following three lines:\n"
    "CORRECT: <rationale>\n"
    "PARTIAL: <rationale>\n"
    "INCORRECT: <rationale>"
)

BINARY_SYSTEM_PROMPT = (
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

_JUDGE_PROMPTS = {"binary": BINARY_SYSTEM_PROMPT, "partial": PARTIAL_SYSTEM_PROMPT}


def _parse_judge_response(raw: str, judge_mode: str) -> tuple[float, str]:
    """Parse judge raw response into a numeric score and rationale.

    Returns:
        Tuple of (score in [0.0, 0.5, 1.0], rationale string).
    """
    upper = raw.upper()

    if judge_mode == "partial":
        # Option C: CORRECT / PARTIAL / INCORRECT
        for line in upper.split("\n"):
            line = line.strip()
            if line.startswith("CORRECT:"):
                score = 1.0
                break
            if line.startswith("PARTIAL:"):
                score = PARTIAL_SCORE
                break
            if line.startswith("INCORRECT:"):
                score = 0.0
                break
        else:
            # Fallback: search anywhere in response
            if "PARTIAL" in upper:
                score = PARTIAL_SCORE
            elif "CORRECT" in upper and "PARTIAL" not in upper and "INCORRECT" not in upper:
                score = 1.0
            elif "INCORRECT" in upper:
                score = 0.0
            else:
                score = 0.0
    else:
        # Binary: CORRECT / INCORRECT
        score = 1.0 if upper.startswith("CORRECT") else 0.0

    rationale = raw.split(":", 1)[1].strip() if ":" in raw else raw
    return score, rationale


def llm_judge(
    question: str,
    prediction: str,
    ground_truth: str,
    context: str | None = None,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    judge_mode: str = "binary",
    max_tokens: int = 1024,
    max_attempts: int = 4,
) -> dict[str, Any]:
    """LLM-as-judge: factual correctness scoring with optional partial credit.

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
        judge_mode: One of "binary" (CORRECT/INCORRECT, default) or
            "partial" (CORRECT/PARTIAL/INCORRECT with PARTIAL=0.5).

    Returns:
        Dict with keys ``factually_correct`` (bool, binary only),
        ``judge_score`` (float, 0.0/0.5/1.0), ``rationale`` (str), and
        ``raw_response`` (str).  In binary mode ``judge_score`` equals
        ``float(factually_correct)``.
    """
    system_prompt = _JUDGE_PROMPTS.get(judge_mode, BINARY_SYSTEM_PROMPT)

    if client is None:
        result = _heuristic_judge(question, prediction, ground_truth, context)
        result["judge_mode"] = judge_mode
        result["judge_score"] = float(result["factually_correct"])
        return result

    user_text = (
        f"Question: {question}\n\n"
        f"Ground-truth answer: {ground_truth}\n\n"
        f"Predicted answer: {prediction}\n"
    )
    if context:
        user_text += f"\nRetrieved context: {context[:2000]}"

    last_exc: Exception | None = None
    for _attempt in range(max_attempts):
      try:
        if hasattr(client, "messages"):
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
                max_tokens=256,
                system=system,
                messages=anthropic_messages,
            )
            text_blocks = [c for c in response.content if c.type == "text"]
            raw = text_blocks[0].text.strip() if text_blocks else ""
        else:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            choice = response.choices[0]
            msg = choice.message
            raw = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()
            if not raw:
                raise ValueError(
                    "judge returned empty content "
                    f"(finish_reason={getattr(choice, 'finish_reason', '?')}, "
                    f"max_tokens={max_tokens}) - the budget was likely consumed by "
                    "reasoning tokens")
        judge_score, rationale = _parse_judge_response(raw, judge_mode)
        return {
            "factually_correct": bool(judge_score >= 1.0),
            "judge_score": judge_score,
            "judge_mode": judge_mode,
            "rationale": rationale,
            "raw_response": raw,
            "judge_failed": False,
        }
      except Exception as exc:
        # A transport/parse failure is NOT evidence that the answer was wrong.
        # Scoring it 0.0 silently injects false negatives, and because longer
        # answers exhaust an undersized token budget more often, that bias lands
        # hardest on exactly the long-format cells under test.
        last_exc = exc
        if _attempt + 1 < max_attempts:
            time.sleep(0.7 * (_attempt + 1))
    raw = f"judge-error: {last_exc}"
    return {
        "factually_correct": None,
        "judge_score": None,
        "judge_mode": judge_mode,
        "rationale": raw,
        "raw_response": raw,
        "judge_failed": True,
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


# --------------------------------------------------------------------------- #
# Length-controlled scoring (§0.1) — pre-registered for Exp 5 Tier 1
# --------------------------------------------------------------------------- #
#
# PRE-REGISTRATION (2026-10-05, Exp 5 length-control factor):
#   budget_f1(pred, gold, k):
#     1. Truncate BOTH pred and gold to their first k whitespace tokens (the
#        model's first k emitted words).
#     2. Normalise (lower-case, strip punctuation/articles) and compute standard
#        token precision / recall / F1 on the truncated pair.
#   Reported at k in {20, 40}. This DEFINES the "common-token-budget F1" named in
#   evaluation_plan.md §0.1. The plan's prose figures (0.615 vs 0.539, etc.) are
#   NOT reproducible: a whole-repo + GPU-host sweep on 2026-10-05 found no code
#   that produced them, so this pre-registered definition supersedes them.
#   Rationale: a symmetric budget measures "at a fixed answer length, how good is
#   each arm", the density-vs-coverage question Exp 5 actually asks. Truncating
#   only the prediction caps recall at k/len(gold) and re-introduces the length
#   confound the metric exists to remove.


def token_prf(prediction: str, ground_truth: str) -> dict[str, float]:
    """Token precision, recall, and F1 after normalisation, reported separately.

    ``token_f1`` collapses precision and recall into one number; Exp 5's
    density-vs-coverage finding (parametric = higher precision, lower recall)
    needs them apart.
    """
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(ground_truth)
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _truncate_to_budget(text: str, k: int) -> str:
    """First ``k`` whitespace tokens of the raw text (the model's first k words)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return " ".join((text or "").split()[:k])


def budget_f1(prediction: str, ground_truth: str, k: int) -> dict[str, float]:
    """Common-token-budget precision/recall/F1 (§0.1 length control).

    Both prediction and gold are truncated to their first ``k`` whitespace tokens,
    then scored with :func:`token_prf`. See the pre-registration note above.
    """
    pred_k = _truncate_to_budget(prediction, k)
    gold_k = _truncate_to_budget(ground_truth, k)
    out = token_prf(pred_k, gold_k)
    out["k"] = k
    return out


def first_k_precision(prediction: str, ground_truth: str, k: int) -> float:
    """Overlap of the first k *predicted* tokens with the FULL gold, over k.

    = |tokens(pred)[:k] ∩ tokens(gold)| / |tokens(pred)[:k]|

    This REPRODUCES the "common-token-budget recall" figures reported in
    evaluation_plan.md (expertqa 0.615 / 0.539, techqa 0.694 / 0.517 at k=20),
    verified exactly against the stored records on 2026-10-05. Note the plan's
    label is a misnomer: the quantity is the *precision* of the opening k tokens
    against the full reference, not a recall. The gold is NOT truncated, so it
    avoids rewarding whatever a long reference happens to say first.

    This is the recovered original metric; it is preferred over a freshly
    invented one so the plan's numbers remain comparable (§0.1, threat #12).
    """
    pred_k = _tokens(prediction)[:k]
    gold = _tokens(ground_truth)
    if not pred_k or not gold:
        return 0.0
    common = sum((Counter(pred_k) & Counter(gold)).values())
    return common / len(pred_k)


def non_inferiority(ci_low: float, ci_high: float, margin: float = -0.02) -> dict[str, Any]:
    """Non-inferiority verdict for a paired delta (treatment - control), per §0.2.

    Treatment is NON_INFERIOR if the lower bound of the 95% CI on the delta
    exceeds ``margin`` (default -0.02 FA). Also classifies SUPERIOR (CI strictly
    above 0), INFERIOR (CI strictly below the margin), and INCONCLUSIVE (the CI
    straddles the margin, i.e. underpowered for the decision).
    """
    if ci_low > 0:
        verdict = "SUPERIOR"
    elif ci_low > margin:
        verdict = "NON_INFERIOR"
    elif ci_high < margin:
        verdict = "INFERIOR"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "margin": margin, "ci_low": ci_low, "ci_high": ci_high}
