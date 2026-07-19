"""Confidence pseudo-token helpers for KVForge.

Implements the Sprint 2.5 "yes" / "no" confidence-supervision scheme:

* Training: append ``\\nConfidence: <yes/no>`` to the assistant answer.
* Inference: force-append ``\\nConfidence:`` and read the restricted two-token
  softmax over the single tokens ``" yes"`` and ``" no"``.
* Metrics: strip the confidence suffix before token-F1, judge, and fKDS scoring.

No tokenizer modification or reserved tokens are required.
"""

from __future__ import annotations

import re
from typing import Any


CONFIDENCE_PREFIX = "\nConfidence:"
YES_TOKEN = " yes"
NO_TOKEN = " no"

# Default threshold for deriving a yes/no label from a factual-accuracy score.
DEFAULT_CONFIDENCE_LABEL_THRESHOLD = 0.5


class ConfidenceTokenError(Exception):
    """Raised when confidence-token assumptions are violated (e.g. not single-token)."""


# ---------------------------------------------------------------------------
# Suffix formatting / stripping
# ---------------------------------------------------------------------------


def append_confidence_suffix(answer: str, is_confident: bool) -> str:
    """Return *answer* with the confidence pseudo-token suffix appended.

    Args:
        answer: The assistant answer text.
        is_confident: ``True`` → suffix uses ``yes``, ``False`` → ``no``.

    Returns:
        ``answer\\nConfidence: <yes/no>``
    """
    label = "yes" if is_confident else "no"
    return f"{answer}{CONFIDENCE_PREFIX} {label}"


def strip_confidence_suffix(text: str) -> str:
    """Remove the confidence suffix if present.

    Handles both fully-formed suffixes and truncated suffixes where the model
    began emitting ``\\nConfidence:`` but stopped. This ensures downstream
    token-F1, judge, and fKDS scoring never see the pseudo-token text.

    Args:
        text: Generated text that may end with a confidence suffix.

    Returns:
        *text* with any trailing confidence suffix removed.
    """
    if not text:
        return text
    # Strip a fully-formed suffix (yes/no optional because generation may stop).
    pattern = re.escape(CONFIDENCE_PREFIX) + r"(?:\s*(?:yes|no)?)?\s*$"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).rstrip()


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


def factual_accuracy_to_label(
    factual_accuracy: float,
    threshold: float = DEFAULT_CONFIDENCE_LABEL_THRESHOLD,
) -> bool:
    """Convert a factual-accuracy score into a confidence label.

    Args:
        factual_accuracy: Score in [0, 1] (e.g. 0.5*token_F1 + 0.5*judge).
        threshold: Minimum score for a ``yes`` label.

    Returns:
        ``True`` if factual_accuracy >= threshold, else ``False``.
    """
    return factual_accuracy >= threshold


def generate_confidence_label(
    question: str,
    student_answer: str,
    teacher_answer: str,
    client: Any | None = None,
    judge_model: str = "gpt-4o-mini",
    threshold: float = DEFAULT_CONFIDENCE_LABEL_THRESHOLD,
) -> bool:
    """Score a student answer against a teacher answer and return yes/no label.

    This is the on-policy distillation labeler: the student is sampled, then
    its answer is compared to the teacher answer using token-F1 + LLM judge.
    A combined factual accuracy >= *threshold* yields ``True`` (yes).

    Args:
        question: Original question.
        student_answer: Student-generated answer.
        teacher_answer: Teacher (Path A) reference answer.
        client: Optional external judge client. ``None`` uses the heuristic.
        judge_model: Judge model name when a client is provided.
        threshold: Minimum factual accuracy for a ``yes`` label.

    Returns:
        ``True`` for confident (yes), ``False`` for not confident (no).
    """
    from eval.metrics import llm_judge, token_f1

    f1 = token_f1(student_answer, teacher_answer)
    judge = llm_judge(question, student_answer, teacher_answer, client=client, model=judge_model)
    factual_acc = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
    return factual_accuracy_to_label(factual_acc, threshold=threshold)


# ---------------------------------------------------------------------------
# Tokenizer verification
# ---------------------------------------------------------------------------


def verify_confidence_tokens(tokenizer) -> tuple[int, int]:
    """Verify that ``" yes"`` and ``" no"`` are single tokens.

    Args:
        tokenizer: HuggingFace tokenizer with ``encode`` method.

    Returns:
        ``(yes_token_id, no_token_id)``.

    Raises:
        ConfidenceTokenError: if either token is not a single token.
    """
    yes_ids = tokenizer.encode(YES_TOKEN, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_TOKEN, add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ConfidenceTokenError(
            f"Confidence tokens must be single-token: {YES_TOKEN!r} -> {yes_ids}, "
            f"{NO_TOKEN!r} -> {no_ids}. Re-verify if the base model changes."
        )
    return int(yes_ids[0]), int(no_ids[0])


# ---------------------------------------------------------------------------
# Inference-time probability extraction
# ---------------------------------------------------------------------------


def extract_confidence_probability(
    answer_text: str,
    model,
    tokenizer,
    verify_single_token: bool = True,
) -> float:
    """Return P(" yes") restricted to the two-token softmax ``{" yes", " no"}``.

    The procedure is:

    1. Strip any existing confidence suffix from *answer_text*.
    2. Force-append ``\\nConfidence:``.
    3. Run one forward pass.
    4. Read the next-position logits, restrict to the token IDs for ``" yes"``
       and ``" no"``, normalize, and return the probability of ``" yes"``.

    Args:
        answer_text: Generated answer text (with or without confidence suffix).
        model: HuggingFace causal-LM in eval mode.
        tokenizer: Corresponding tokenizer.
        verify_single_token: If ``True`` (default), assert that ``" yes"`` and
            ``" no"`` are single tokens.

    Returns:
        P(" yes") in [0, 1]. Returns 0.5 if both token probabilities are zero.
    """
    import torch

    text = strip_confidence_suffix(answer_text)
    prompt = text + CONFIDENCE_PREFIX

    yes_id, no_id = verify_confidence_tokens(tokenizer) if verify_single_token else _token_ids_fallback(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = torch.as_tensor(outputs.logits[0, -1, :])
    probs = torch.softmax(logits, dim=-1)
    yes_prob = probs[yes_id].item()
    no_prob = probs[no_id].item()
    total = yes_prob + no_prob
    if total <= 0:
        return 0.5
    return yes_prob / total


def _token_ids_fallback(tokenizer) -> tuple[int, int]:
    """Return the first token id of the encoded confidence tokens.

    Used only when ``verify_single_token`` is disabled; callers that care about
    the single-token guarantee should use ``verify_confidence_tokens``.
    """
    yes_ids = tokenizer.encode(YES_TOKEN, add_special_tokens=False)
    no_ids = tokenizer.encode(NO_TOKEN, add_special_tokens=False)
    return int(yes_ids[0]), int(no_ids[0])


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_with_confidence_suffix(
    model,
    tokenizer,
    inputs: Any,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    **gen_kwargs,
) -> tuple[str, float]:
    """Generate an answer, then extract its confidence-token probability.

    This is a convenience wrapper for the inference path: it generates up to
    *max_new_tokens*, strips any confidence suffix that the model may have
    emitted, and then runs a forced forward pass over ``\\nConfidence:`` to read
    the restricted two-token softmax.

    Args:
        model: HuggingFace causal-LM.
        tokenizer: Corresponding tokenizer.
        inputs: Tokenized inputs (dict with ``input_ids`` / ``attention_mask``).
        max_new_tokens: Generation budget.
        do_sample: Sampling flag.
        **gen_kwargs: Additional kwargs passed to ``model.generate``.

    Returns:
        ``(answer_text, p_yes)`` where *answer_text* has the confidence suffix
        stripped.
    """
    import torch

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **gen_kwargs,
        )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    answer = strip_confidence_suffix(generated)
    p_yes = extract_confidence_probability(answer, model, tokenizer)
    return answer, p_yes
