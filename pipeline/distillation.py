"""Distillation helpers for KVForge Sprint 2.

Provides non-GPU data-pipeline functions for:

* Building a distillation query pool from real queries, FAQ paraphrases, and
  chunk-conditioned generated questions.
* Quality-filtering teacher answers.
* Generating on-policy confidence labels for Sprint 2.5.

GPU-dependent generation functions accept ``model``/``tokenizer`` and are
intended to be called with the loaded HuggingFace model on the EC2 worker.
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_JUDGE_THRESHOLD = 0.7


def normalize_question(text: str) -> str:
    """Lower-case and strip punctuation for deduplication."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def deduplicate_pool(entries: list[dict]) -> list[dict]:
    """Remove duplicate questions from the pool, keeping the first occurrence.

    Args:
        entries: List of pool dicts, each with a ``question`` key.

    Returns:
        Deduplicated list. Dropped duplicates are omitted entirely.
    """
    seen: set[str] = set()
    result = []
    for e in entries:
        key = normalize_question(e["question"])
        if key in seen:
            continue
        seen.add(key)
        result.append(e)
    return result


def quality_filter(
    pairs: list[dict],
    threshold: float = DEFAULT_JUDGE_THRESHOLD,
    judge_client: Any | None = None,
    judge_model: str = "gpt-4o-mini",
) -> tuple[list[dict], dict]:
    """Filter teacher answers by factual accuracy against the expected answer.

    Each pair must contain ``question``, ``teacher_answer``, and ``expected_answer``.
    Factual accuracy is ``0.5 * token_f1 + 0.5 * judge_correct``. Pairs scoring
    below *threshold* are dropped.

    Args:
        pairs: List of teacher pairs.
        threshold: Minimum factual accuracy to keep the pair.
        judge_client: Optional external judge client; heuristic fallback if None.
        judge_model: Judge model name when a client is provided.

    Returns:
        ``(filtered_pairs, stats)`` where stats contains ``kept``, ``dropped``,
        ``drop_rate``, and ``mean_factual_accuracy``.
    """
    from eval.metrics import llm_judge, token_f1

    kept = []
    scores = []
    for p in pairs:
        q = p["question"]
        pred = p["teacher_answer"]
        gold = p["expected_answer"]
        f1 = token_f1(pred, gold)
        judge = llm_judge(q, pred, gold, client=judge_client, model=judge_model)
        factual_acc = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
        scores.append(factual_acc)
        if factual_acc >= threshold:
            kept.append({**p, "factual_accuracy": factual_acc})

    n = len(pairs)
    stats = {
        "kept": len(kept),
        "dropped": n - len(kept),
        "drop_rate": (n - len(kept)) / n if n else 0.0,
        "mean_factual_accuracy": sum(scores) / len(scores) if scores else 0.0,
    }
    return kept, stats


def load_real_queries(db_path: str, limit: int = 1000) -> list[dict]:
    """Load real retrieval-routed queries from the query log as pool entries.

    Args:
        db_path: Path to the query_log SQLite database.
        limit: Maximum number of records to return.

    Returns:
        List of pool entries with keys ``question``, ``expected_answer``,
        ``source``, ``cluster_id``.
    """
    from pipeline.query_logger import get_training_pairs

    rows = get_training_pairs(db_path, limit=limit)
    return [
        {
            "question": r["question"],
            "expected_answer": r["answer"],
            "source": "real_query",
            "cluster_id": r.get("cluster_id"),
        }
        for r in rows
    ]


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> str:
    """Generate text from a prompt using a HuggingFace model in eval mode."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def expand_faq_questions(
    faqs: list[dict],
    model,
    tokenizer,
    n: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate paraphrases of FAQ questions and return them as pool entries.

    Args:
        faqs: FAQ dicts with ``question`` and ``answer`` keys.
        model, tokenizer: Loaded HuggingFace model.
        n: Number of paraphrases to generate per FAQ.
        max_new_tokens: Generation budget for the paraphrase batch.
        temperature: Sampling temperature.

    Returns:
        Pool entries with ``question``, ``expected_answer``, ``source``,
        and ``original_question``.
    """
    entries = []
    for f in faqs:
        original = f["question"].strip()
        answer = f.get("answer", "")
        prompt = (
            f"Generate {n} different paraphrases of the following question, "
            f"one per line. Do not answer the question.\n\n"
            f"Question: {original}\n\n"
            f"Paraphrases:"
        )
        generated = _generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        lines = [line.strip("-• ") for line in generated.splitlines() if line.strip()]
        for line in lines[:n]:
            entries.append({
                "question": line,
                "expected_answer": answer,
                "source": "faq_paraphrase",
                "original_question": original,
            })
    return entries


def generate_chunk_questions(
    chunks: list[dict],
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> list[dict]:
    """Generate a factual question for each chunk and return pool entries.

    Args:
        chunks: Chunk dicts with ``text`` and ``chunk_id`` keys.
        model, tokenizer: Loaded HuggingFace model.
        max_new_tokens: Generation budget per question.
        temperature: Sampling temperature.

    Returns:
        Pool entries with ``question``, ``source``, ``chunk_id``, and ``chunk_text``.
    """
    entries = []
    for c in chunks:
        text = c["text"].strip()
        if not text:
            continue
        prompt = (
            "Write a single factual question that can be answered from the "
            "following passage. Answer only with the question.\n\n"
            f"Passage:\n{text[:500]}\n\n"
            "Question:"
        )
        question = _generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        question = question.split("\n")[0].strip()
        if question:
            entries.append({
                "question": question,
                "expected_answer": "",
                "source": "chunk_question",
                "chunk_id": c.get("chunk_id"),
                "chunk_text": text,
            })
    return entries


def build_query_pool(
    faqs: list[dict] | None,
    model,
    tokenizer,
    query_log_db: str | None = None,
    chunks: list[dict] | None = None,
    faq_paraphrases: int = 3,
    real_query_limit: int = 1000,
) -> list[dict]:
    """Assemble a deduplicated distillation query pool.

    Combines (in order of priority):

    1. Real logged queries from ``query_log_db``.
    2. FAQ paraphrases.
    3. Chunk-conditioned generated questions.

    Args:
        faqs: FAQ dicts with ``question`` and ``answer`` keys.
        model, tokenizer: Loaded HuggingFace model.
        query_log_db: Path to query log SQLite DB; if None, real queries are skipped.
        chunks: Chunk dicts for chunk-conditioned questions; if None, skipped.
        faq_paraphrases: Number of paraphrases per FAQ.
        real_query_limit: Maximum real queries to include.

    Returns:
        Deduplicated list of pool entries.
    """
    pool: list[dict] = []
    if query_log_db:
        pool.extend(load_real_queries(query_log_db, limit=real_query_limit))
    if faqs:
        pool.extend(expand_faq_questions(faqs, model, tokenizer, n=faq_paraphrases))
    if chunks:
        pool.extend(generate_chunk_questions(chunks, model, tokenizer))
    return deduplicate_pool(pool)


def generate_on_policy_samples(
    query_pool: list[dict],
    student_model,
    student_tokenizer,
    teacher_model,
    teacher_tokenizer,
    cfg: dict,
    sft_format: str = "chat",
    judge_client: Any | None = None,
    judge_model: str = "gpt-4o-mini",
) -> list[dict]:
    """Generate student answers for a query pool and label them for distillation.

    For each pool entry:

    1. Generate a student answer (parametric or with the same format used at eval).
    2. Generate a teacher answer (Path A reference).
    3. Score the student answer against the teacher answer and assign a
       confidence label (yes/no) using the Sprint 2.5 logic.

    Args:
        query_pool: Deduplicated pool entries.
        student_model, student_tokenizer: Current student model.
        teacher_model, teacher_tokenizer: Frozen teacher (Path A) model.
        cfg: Datasource config dict.
        sft_format: ``"chat"`` or ``"bare"``.
        judge_client: Optional judge client.
        judge_model: Judge model name.

    Returns:
        List of dicts with ``question``, ``student_answer``, ``teacher_answer``,
        ``confidence_label``, ``factual_accuracy``.
    """
    from eval.metrics import llm_judge, token_f1
    from pipeline.confidence_token import generate_confidence_label
    from pipeline.prs_evaluator import _generate_parametric
    from transformers import pipeline as hf_pipeline

    student_pipe = hf_pipeline(
        "text-generation", model=student_model, tokenizer=student_tokenizer,
        max_new_tokens=256, do_sample=False,
    )
    teacher_pipe = hf_pipeline(
        "text-generation", model=teacher_model, tokenizer=teacher_tokenizer,
        max_new_tokens=256, do_sample=False,
    )

    samples = []
    for entry in query_pool:
        q = entry["question"]
        student_ans = _generate_parametric(q, student_pipe, student_tokenizer, sft_format)
        teacher_ans = _generate_parametric(q, teacher_pipe, teacher_tokenizer, sft_format)
        label = generate_confidence_label(
            q, student_ans, teacher_ans, client=judge_client, judge_model=judge_model
        )
        f1 = token_f1(student_ans, teacher_ans)
        judge = llm_judge(q, student_ans, teacher_ans, client=judge_client, model=judge_model)
        factual_acc = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
        samples.append({
            "question": q,
            "student_answer": student_ans,
            "teacher_answer": teacher_ans,
            "confidence_label": label,
            "factual_accuracy": factual_acc,
            "source": entry.get("source", "on_policy"),
        })
    return samples
