"""Compute the Parametric Readiness Score (PRS) after each LoRA training round.

PRS measures how close the fine-tuned model is to being able to answer queries
reliably from its weights alone (without retrieval).  It is a weighted
combination of three sub-scores:

.. math::

    PRS = 0.5 \\cdot accuracy\\_ratio
        + 0.3 \\cdot calibration\\_score
        + 0.2 \\cdot self\\_consistency

* **accuracy_ratio** — semantic similarity of the parametric answer vs. the
  RAG answer, normalised by the RAG-vs-ground-truth similarity.
* **calibration_score** — how well the model's self-reported confidence
  matches its actual answer quality.
* **self_consistency** — mean pairwise cosine similarity across *n* stochastic
  answers to the same question.

Run automatically by ``index_and_train.py`` after ``lora_trainer.py`` completes.
"""

import json
import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.version as ver
import core.model_loader as model_loader
from vectorstore.registry import get_store
# kv_inference is imported lazily inside evaluate() — SP3 may not exist yet


CONFIDENCE_PROMPT_SUFFIX = (
    "\n\nOn a scale of 0 to 100, how confident are you in your answer above? "
    "Reply with a single integer only."
)


def _extract_qa(faq: dict, q_key: str = "question", a_key: str = "answer") -> tuple[str, str]:
    """Extract question and answer using configurable key names."""
    if q_key not in faq:
        raise KeyError(f"FAQ missing key '{q_key}'. Available keys: {list(faq.keys())}")
    if a_key not in faq:
        raise KeyError(f"FAQ missing key '{a_key}'. Available keys: {list(faq.keys())}")
    return faq[q_key], faq[a_key]


def _embed(texts: list[str], model_name: str) -> np.ndarray:
    embedder = TextEmbedding(model_name=model_name, show_download_progress=False)
    return np.array(list(embedder.embed(texts)))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _format_query(query: str, tokenizer, sft_format: str) -> str:
    """Format a query for parametric generation.

    With ``sft_format == "chat"`` the query is wrapped in the model's chat template with an
    assistant generation prompt, so eval matches how the model was trained (chat SFT). The
    chat branch also strips the ``(variant N)`` augmentation suffix, matching
    ``lora_trainer.build_sft_example`` so eval content matches chat-SFT training. With
    ``"bare"`` the raw query is returned unchanged (legacy behavior) — bare mode does not
    strip at train time either, so eval stays consistent with it unstripped.
    """
    if sft_format == "chat":
        from pipeline.lora_trainer import _strip_variant_suffix
        q = _strip_variant_suffix(query)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return query


def _factual_accuracy(f1: float, judge_correct: bool) -> float:
    """Combine token-F1 and LLM-judge correctness into a single accuracy score.

    Replaces the legacy cosine-similarity accuracy_ratio, which the paper's
    own validation (pipeline/eval_prs_validation.py) found correlates only
    weakly with real factual correctness. This is the same formula that
    validation script used to simulate a factual PRS variant.
    """
    return 0.5 * f1 + 0.5 * float(judge_correct)


def _fixed_calibration(self_conf: float, factual_acc: float) -> float:
    """How well self-reported confidence matches factual accuracy.

    Replaces comparing confidence against cosine similarity (param_sim),
    which measured confidence against the wrong scale once accuracy_ratio
    itself moved to factual_acc.
    """
    return 1.0 - abs(self_conf - factual_acc)


def _coverage_ratio(scores: list[float], threshold: float) -> float:
    """Fraction of scores at or above threshold. Returns 0.0 for an empty list."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def _generate_parametric(query: str, pipe, tokenizer=None, sft_format: str = "bare") -> str:
    """Generate an answer to *query* from model weights with no retrieved context.

    Args:
        query: The question string.
        pipe: A HuggingFace ``text-generation`` pipeline.
        tokenizer: The model's tokenizer, used to apply the chat template when
            ``sft_format == "chat"``.
        sft_format: ``"chat"`` or ``"bare"`` — see ``_format_query``.

    Returns:
        The generated answer text (prompt prefix stripped).
    """
    prompt = _format_query(query, tokenizer, sft_format)
    if hasattr(pipe, "generate"):
        # pipe is actually a model — use direct generate (avoids pipeline
        # text-stripping bugs with Gemma4 chat-template tokens)
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(pipe.device)
        with torch.no_grad():
            outputs = pipe.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    else:
        out = pipe(prompt, add_special_tokens=(sft_format != "chat"))
        generated = out[0]["generated_text"][len(prompt):].strip()
    from pipeline.confidence_token import strip_confidence_suffix
    return strip_confidence_suffix(generated)


def _extract_confidence(
    answer: str,
    pipe_short,
    model,
    tokenizer,
    use_confidence_token: bool = False,
) -> float:
    """Return a self-reported confidence value in [0, 1].

    Two modes:

    * ``use_confidence_token=False`` (legacy): append a 0-100 integer prompt and
      parse the model's sampled integer.
    * ``use_confidence_token=True`` (Sprint 2.5): force-append ``\\nConfidence:``,
      run one forward pass, and read the restricted two-token softmax over
      ``" yes"`` and ``" no"``.
    """
    if use_confidence_token:
        from pipeline.confidence_token import extract_confidence_probability
        return extract_confidence_probability(answer, model, tokenizer)

    # Legacy integer prompt path.
    prompt = answer + CONFIDENCE_PROMPT_SUFFIX
    out = pipe_short(prompt)
    tail = out[0]["generated_text"][len(prompt):].strip()
    try:
        val = int("".join(c for c in tail if c.isdigit())[:3])
        return min(val, 100) / 100.0
    except ValueError:
        return 0.5


def _self_consistency(query: str, pipe_sample, embedder, tokenizer=None,
                        sft_format: str = "bare", n: int = 3,
                        return_embeddings: bool = False):
    """Generate n answers at temperature 0.7; return mean pairwise cosine sim.

    Args:
        return_embeddings: If True, return ``(score, embs)`` where ``embs`` is the
            array of sampled answer embeddings. Default False preserves the
            original scalar-only return value.
    """
    prompt = _format_query(query, tokenizer, sft_format)
    # Chat-template strings already contain a literal <|begin_of_text|> BOS;
    # suppress the pipeline's own BOS insertion so eval tokenization matches
    # the single-BOS tokenization used by chat-SFT training. Bare mode has no
    # BOS in the raw string, so the pipeline's default add-BOS is kept.
    answers = [
        _strip_confidence_suffix(
            pipe_sample(prompt, add_special_tokens=(sft_format != "chat"))[0]
            ["generated_text"][len(prompt):].strip()
        )
        for _ in range(n)
    ]
    embs = np.array(list(embedder.embed(answers)))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim(embs[i], embs[j]))
    score = float(np.mean(sims)) if sims else 1.0
    if return_embeddings:
        return score, embs
    return score


def _strip_confidence_suffix(text: str) -> str:
    """Convenience alias for the Sprint 2.5 confidence suffix stripper."""
    from pipeline.confidence_token import strip_confidence_suffix
    return strip_confidence_suffix(text)


# Provisional weights (data-derived, 4-corpus backtest — see
# docs/superpowers/specs/2026-07-12-prs-gate-rework-design.md): accuracy-dominant so PRS
# ranks corpora by real quality rather than by confident-but-wrong self-consistency.
_DEFAULT_PRS_WEIGHTS = {"accuracy": 0.7, "calibration": 0.15, "consistency": 0.15}


def _compute_prs(accuracy_ratios: list, calibrations: list, consistencies: list,
                 weights: dict | None) -> float:
    """Compute the weighted PRS from per-question component score lists.

    Args:
        accuracy_ratios: Per-question accuracy ratio values (list of floats in
            [0, 1]).
        calibrations: Per-question calibration scores (list of floats in
            [0, 1]).
        consistencies: Per-question self-consistency scores (list of floats in
            [0, 1]).
        weights: Dict with keys ``'accuracy'``, ``'calibration'``,
            ``'consistency'`` mapping to floats that sum to 1.0.
            Defaults to ``{accuracy: 0.7, calibration: 0.15, consistency: 0.15}``.

    Returns:
        PRS score clipped to ``[0.0, 1.0]``.
    """
    import numpy as np
    w = weights or _DEFAULT_PRS_WEIGHTS
    return float(np.clip(
        w.get("accuracy", 0.5) * np.mean(accuracy_ratios)
        + w.get("calibration", 0.3) * np.mean(calibrations)
        + w.get("consistency", 0.2) * np.mean(consistencies),
        0.0, 1.0
    ))


def compute_kds(
    faqs: list[dict],
    cfg: dict,
    lora_checkpoint: str | None = None,
    sample_cap: int = 300,
    n: int = 3,
) -> tuple[float, dict]:
    """Compute per-chunk Knowledge Differentiation Score (KDS).

    For each chunk with at least one tagged FAQ (via ``source_chunk_ids``),
    generate *n* parametric answers per FAQ question at temperature, embed them,
    and compute a variance-ratio score that measures how distinct the model's
    answers are for that chunk relative to the corpus mean.

    Chunks without tagged FAQs are excluded (fail-closed).  Measured chunks get
    ``kds`` and ``last_kds_round`` persisted in their vector-store payload, and
    a corpus-level ``mean_kds`` record is appended to ``version.json``.

    Args:
        faqs: List of FAQ dicts.  Each must contain the question key configured
            by ``cfg['faq_question_key']`` and ``source_chunk_ids``.
        cfg: Datasource configuration dict.
        lora_checkpoint: Path to a LoRA adapter directory to load before
            evaluation.  ``None`` uses the base model.
        sample_cap: Maximum number of chunks to sample in this round.
        n: Number of parametric answers to sample per FAQ question.

    Returns:
        ``(mean_kds, kds_by_chunk)`` where ``mean_kds`` is the average KDS over
        chunks measured this round and ``kds_by_chunk`` maps chunk id to its KDS
        value in ``[0, 1]``.
    """
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **indexing_cfg, **training_cfg}
    sft_format = effective_cfg.get("sft_format", "chat")
    q_key = effective_cfg.get("faq_question_key", "question")

    ver.init(cfg)
    model_loader.init(cfg)

    round_num = ver.get_lora_version()
    kds_by_chunk: dict[str, float] = {}
    mean_kds = 0.0
    measured_chunks = 0

    # Group FAQs by source chunk id.
    faq_by_chunk: dict[str, list[dict]] = {}
    for faq in faqs:
        for cid in faq.get("source_chunk_ids", []):
            faq_by_chunk.setdefault(str(cid), []).append(faq)

    if faq_by_chunk:
        store = get_store(cfg)
        collection = effective_cfg.get("collection")
        if collection:
            # Scroll all chunks to read existing last_kds_round values.
            all_points: list = []
            offset = None
            while True:
                page, offset = store.scroll(
                    collection,
                    limit=1000,
                    with_payload=True,
                    offset=offset,
                )
                all_points.extend(page)
                if offset is None:
                    break

            chunk_meta: dict[str, tuple[Any, dict]] = {}
            for p in all_points:
                cid = str(p.id)
                chunk_meta[cid] = (p.id, p.payload or {})

            # Rotating coverage: never-measured first, then oldest last_kds_round.
            def _sort_key(cid: str):
                payload = chunk_meta[cid][1]
                last = payload.get("last_kds_round")
                if last is None:
                    return (0, 0)
                return (1, last)

            eligible = [cid for cid in chunk_meta if cid in faq_by_chunk]
            eligible.sort(key=_sort_key)
            selected = eligible[:sample_cap]

            if selected:
                # Load model and shared sampling resources.
                model, tokenizer = model_loader.load(lora_checkpoint)
                embed_model = effective_cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
                from transformers import pipeline as hf_pipeline
                pipe_sample = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                )
                embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)

                # Prefer stripped questions when chat/paraphrase augmentation.
                _strip = None
                if sft_format == "chat":
                    from pipeline.lora_trainer import _strip_variant_suffix
                    _strip = _strip_variant_suffix

                chunk_means: dict[str, np.ndarray] = {}
                per_chunk_embeddings: dict[str, np.ndarray] = {}
                all_embeddings: list[np.ndarray] = []

                for cid in selected:
                    questions = []
                    for faq in faq_by_chunk[cid]:
                        q = faq.get(q_key, "")
                        if _strip:
                            q = _strip(q)
                        if q:
                            questions.append(q)

                    if not questions:
                        continue

                    pooled: list[np.ndarray] = []
                    for q in questions:
                        _, embs = _self_consistency(
                            q,
                            pipe_sample,
                            embedder,
                            tokenizer,
                            sft_format,
                            n=n,
                            return_embeddings=True,
                        )
                        pooled.append(embs)

                    if not pooled:
                        continue

                    chunk_embs = np.vstack(pooled)
                    chunk_means[cid] = chunk_embs.mean(axis=0)
                    per_chunk_embeddings[cid] = chunk_embs
                    all_embeddings.append(chunk_embs)

                if chunk_means:
                    all_embeddings_arr = np.vstack(all_embeddings)
                    grand_mean = all_embeddings_arr.mean(axis=0)

                    for cid, mu_i in chunk_means.items():
                        embs = per_chunk_embeddings[cid]
                        W_i = float(np.mean(np.sum((embs - mu_i) ** 2, axis=1)))
                        B_i = float(np.sum((mu_i - grand_mean) ** 2))
                        denom = B_i + W_i
                        kds = (B_i / denom) if denom > 0 else 0.0
                        kds_by_chunk[cid] = float(np.clip(kds, 0.0, 1.0))

                        original_id, _ = chunk_meta[cid]
                        store.set_payload(
                            collection,
                            original_id,
                            {
                                "kds": kds_by_chunk[cid],
                                "last_kds_round": round_num,
                            },
                        )

                    mean_kds = float(np.mean(list(kds_by_chunk.values())))
                    measured_chunks = len(kds_by_chunk)

    ver.append_kds(round_num, mean_kds, measured_chunks)
    return mean_kds, kds_by_chunk


def compute_fkds(
    faqs: list[dict],
    cfg: dict,
    lora_checkpoint: str | None = None,
    sample_cap: int = 300,
    n: int = 3,
    factual_weight: float = 0.1,
    judge_model: str = "claude-sonnet-4-6",
) -> tuple[float, float, dict]:
    """Compute factual KDS (fKDS) and persist it alongside the existing KDS.

    fKDS blends the existing consistency-based KDS with a factual-accuracy
    component scored against FAQ ground truth. This addresses the validation
    finding that plain consistency KDS does not correlate with KV-injection
    quality.

    For each chunk, N parametric answers are generated per FAQ question, then:
      - consistency_KDS uses the answer-embedding variance-ratio (same as KDS).
      - factual_accuracy uses 0.5 * token_F1 + 0.5 * LLM-judge correctness.
      - fKDS = factual_weight * consistency_KDS + (1 - factual_weight) * factual_accuracy.

    Args:
        faqs: List of FAQ dicts, each with question/answer keys and source_chunk_ids.
        cfg: Datasource configuration dict.
        lora_checkpoint: Path to a LoRA adapter directory; None uses the base model.
        sample_cap: Maximum number of chunks to sample this round.
        n: Number of parametric answers to sample per FAQ question.
        factual_weight: Weight on the consistency component; 0.1 means 90% factual.
        judge_model: Judge model name for the factual-accuracy LLM judge.

    Returns:
        ``(mean_kds, mean_fkds, fkds_by_chunk)`` where ``fkds_by_chunk`` maps
        chunk id to a dict with keys ``kds``, ``factual_accuracy``, and ``fkds``.
    """
    from eval import metrics as eval_metrics

    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **indexing_cfg, **training_cfg}
    sft_format = effective_cfg.get("sft_format", "chat")
    q_key = effective_cfg.get("faq_question_key", "question")
    a_key = effective_cfg.get("faq_answer_key", "answer")

    ver.init(cfg)
    model_loader.init(cfg)

    round_num = ver.get_lora_version()
    fkds_by_chunk: dict[str, dict] = {}
    mean_kds = 0.0
    mean_fkds = 0.0
    measured_chunks = 0

    # Group FAQs by source chunk id.
    faq_by_chunk: dict[str, list[dict]] = {}
    for faq in faqs:
        for cid in faq.get("source_chunk_ids", []):
            faq_by_chunk.setdefault(str(cid), []).append(faq)

    if faq_by_chunk:
        store = get_store(cfg)
        collection = effective_cfg.get("collection")
        if collection:
            all_points: list = []
            offset = None
            while True:
                page, offset = store.scroll(
                    collection,
                    limit=1000,
                    with_payload=True,
                    offset=offset,
                )
                all_points.extend(page)
                if offset is None:
                    break

            chunk_meta: dict[str, tuple[Any, dict]] = {}
            for p in all_points:
                cid = str(p.id)
                chunk_meta[cid] = (p.id, p.payload or {})

            # Rotating coverage: never-measured first, then oldest last_kds_round.
            def _sort_key(cid: str):
                payload = chunk_meta[cid][1]
                last = payload.get("last_kds_round")
                if last is None:
                    return (0, 0)
                return (1, last)

            eligible = [cid for cid in chunk_meta if cid in faq_by_chunk]
            eligible.sort(key=_sort_key)
            selected = eligible[:sample_cap]

            if selected:
                model, tokenizer = model_loader.load(lora_checkpoint)
                embed_model = effective_cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
                from transformers import pipeline as hf_pipeline

                pipe_sample = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                )
                embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)

                # Judge client
                judge_client = None
                if judge_model.startswith("claude"):
                    try:
                        import anthropic
                        judge_client = anthropic.Anthropic()
                    except Exception as e:
                        print(f"⚠️ Could not create Anthropic judge client: {e}", flush=True)

                # Strip variant suffix for chat mode
                _strip = None
                if sft_format == "chat":
                    from pipeline.lora_trainer import _strip_variant_suffix
                    _strip = _strip_variant_suffix

                chunk_means: dict[str, np.ndarray] = {}
                per_chunk_embeddings: dict[str, np.ndarray] = {}
                all_embeddings: list[np.ndarray] = []
                per_chunk_factual: dict[str, list[float]] = {}

                for cid in selected:
                    questions = []
                    for faq in faq_by_chunk[cid]:
                        q = faq.get(q_key, "")
                        if _strip:
                            q = _strip(q)
                        if q:
                            questions.append((q, faq))

                    if not questions:
                        continue

                    pooled_embs: list[np.ndarray] = []
                    factual_scores: list[float] = []

                    for q, faq in questions:
                        answers = []
                        for _ in range(n):
                            answers.append(_generate_parametric(q, pipe_sample, tokenizer, sft_format))
                        embs = np.array(list(embedder.embed(answers)))
                        pooled_embs.append(embs)

                        gt = faq.get(a_key, "")
                        for ans in answers:
                            f1 = eval_metrics.token_f1(ans, gt)
                            judge = eval_metrics.llm_judge(
                                q, ans, gt, client=judge_client, model=judge_model
                            )
                            factual_scores.append(0.5 * f1 + 0.5 * float(judge["factually_correct"]))

                    if not pooled_embs:
                        continue

                    chunk_embs = np.vstack(pooled_embs)
                    chunk_means[cid] = chunk_embs.mean(axis=0)
                    per_chunk_embeddings[cid] = chunk_embs
                    all_embeddings.append(chunk_embs)
                    per_chunk_factual[cid] = factual_scores

                if chunk_means:
                    all_embeddings_arr = np.vstack(all_embeddings)
                    grand_mean = all_embeddings_arr.mean(axis=0)

                    kds_sum = 0.0
                    fkds_sum = 0.0
                    for cid, mu_i in chunk_means.items():
                        embs = per_chunk_embeddings[cid]
                        W_i = float(np.mean(np.sum((embs - mu_i) ** 2, axis=1)))
                        B_i = float(np.sum((mu_i - grand_mean) ** 2))
                        denom = B_i + W_i
                        kds = (B_i / denom) if denom > 0 else 0.0
                        kds = float(np.clip(kds, 0.0, 1.0))

                        factual_scores = per_chunk_factual[cid]
                        mean_factual = sum(factual_scores) / len(factual_scores) if factual_scores else 0.0

                        fkds = factual_weight * kds + (1.0 - factual_weight) * mean_factual
                        fkds = float(np.clip(fkds, 0.0, 1.0))

                        fkds_by_chunk[cid] = {
                            "kds": kds,
                            "factual_accuracy": mean_factual,
                            "fkds": fkds,
                        }

                        original_id, _ = chunk_meta[cid]
                        store.set_payload(
                            collection,
                            original_id,
                            {
                                "kds": kds,
                                "fkds": fkds,
                                "factual_accuracy": mean_factual,
                                "last_kds_round": round_num,
                            },
                        )
                        kds_sum += kds
                        fkds_sum += fkds

                    n_chunks = len(chunk_means)
                    mean_kds = kds_sum / n_chunks
                    mean_fkds = fkds_sum / n_chunks
                    measured_chunks = n_chunks

    ver.append_kds(round_num, mean_kds, measured_chunks)
    ver.append_fkds(round_num, mean_fkds, measured_chunks)
    return mean_kds, mean_fkds, fkds_by_chunk


def evaluate(faqs: list[dict], cfg: dict, lora_checkpoint: str | None = None) -> float:
    """Compute the Parametric Readiness Score on a sample of FAQs.

    For each FAQ:

    1. Generate a parametric answer (no context) and a RAG answer (with
       retrieval, if ``kv_inference`` is available).
    2. Embed both answers and the ground-truth answer.
    3. Compute ``accuracy_ratio``, ``calibration``, and ``self_consistency``.

    After evaluation, queries whose factual accuracy clears
    ``known_good_accuracy_threshold`` are recorded as "known-good" in
    ``version.json`` for use by the Phase 2/3 confidence gate.

    Args:
        faqs: List of FAQ dicts.  Each must have the keys specified by
            ``cfg['faq_question_key']`` and ``cfg['faq_answer_key']``.
        cfg: Datasource configuration dict.
        lora_checkpoint: Path to a LoRA adapter directory to load before
            evaluation.  ``None`` uses the base model.

    Returns:
        PRS score in ``[0.0, 1.0]``.
    """
    # Support both flat configs and nested addon_config.
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **indexing_cfg, **training_cfg}
    sft_format = effective_cfg.get("sft_format", "chat")
    use_confidence_token = effective_cfg.get("use_confidence_token", False)

    model, tokenizer = model_loader.load(lora_checkpoint)
    embed_model = effective_cfg.get("embed_model", "BAAI/bge-small-en-v1.5")

    # Lazy import — SP3 may not be built yet; graceful degradation
    try:
        from pipeline.kv_inference import answer_with_retrieval
        has_sp3 = True
    except ImportError:
        has_sp3 = False

    # Create shared resources once — avoid reconstructing per FAQ
    from transformers import pipeline as hf_pipeline
    pipe_gen = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                            max_new_tokens=256, do_sample=False)
    pipe_conf = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                             max_new_tokens=5, do_sample=False)
    pipe_sample = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                               max_new_tokens=128, do_sample=True, temperature=0.7)
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)

    accuracy_ratios, calibrations, consistencies = [], [], []
    factual_accs = []  # consumed by Task 3's known_good_queries selection

    q_key = effective_cfg.get("faq_question_key", "question")
    a_key = effective_cfg.get("faq_answer_key", "answer")

    total = len(faqs)
    for idx, faq in enumerate(faqs, 1):
        q, gt = _extract_qa(faq, q_key=q_key, a_key=a_key)
        print(f"⏳ Evaluating FAQ {idx}/{total}: {q[:60]}…", flush=True)
        param_ans = _generate_parametric(q, pipe_gen, tokenizer, sft_format)
        if has_sp3:
            rag_ans = _strip_confidence_suffix(answer_with_retrieval(q, cfg))
        else:
            rag_ans = gt
        embs = np.array(list(embedder.embed([param_ans, rag_ans, gt])))
        param_sim = _cosine_sim(embs[0], embs[2])
        rag_sim   = _cosine_sim(embs[1], embs[2])
        cosine_accuracy_ratio = min(param_sim / (rag_sim + 1e-9), 1.0)  # diagnostic only, not scored
        self_conf = _extract_confidence(
            param_ans, pipe_conf, model, tokenizer,
            use_confidence_token=use_confidence_token,
        )
        consistencies.append(_self_consistency(q, pipe_sample, embedder, tokenizer, sft_format))

        # Factual metrics — now the actual scoring signal, not diagnostic-only.
        from eval import metrics as _eval_metrics
        em = _eval_metrics.exact_match(param_ans, gt)
        f1 = _eval_metrics.token_f1(param_ans, gt)
        judge = _eval_metrics.llm_judge(q, param_ans, gt)
        factual_acc = _factual_accuracy(f1, judge["factually_correct"])
        factual_accs.append(factual_acc)
        accuracy_ratios.append(factual_acc)
        calibrations.append(_fixed_calibration(self_conf, factual_acc))

        print(f"   acc={factual_acc:.3f} (cosine={cosine_accuracy_ratio:.3f}) "
              f"conf={calibrations[-1]:.3f} cons={consistencies[-1]:.3f} "
              f"EM={em} F1={f1:.3f} judge={int(judge['factually_correct'])}", flush=True)

    weights = effective_cfg.get("prs_weights", None)
    prs = _compute_prs(accuracy_ratios, calibrations, consistencies, weights)

    # Populate known_good_queries: queries where factual accuracy clears the
    # known-good threshold. Stored as pre-computed embeddings for use by
    # confidence_gate._query_similarity.
    known_good_threshold = effective_cfg.get("known_good_accuracy_threshold", 0.5)
    good_queries = [faqs[i].get(q_key, faqs[i].get("question", ""))
                    for i, r in enumerate(factual_accs) if r >= known_good_threshold]
    if good_queries:
        good_embs = [e.astype(float).tolist() for e in embedder.embed(good_queries)]
        data = ver.load()
        data["known_good_queries"] = good_embs
        ver.save(data)

    # Hook 1: Dynamic PRS — per-cluster three-signal update
    lora_version = ver.get_lora_version()
    cluster_states: dict = {}
    try:
        from pathlib import Path as _Path
        cluster_file = _Path(effective_cfg.get("checkpoint_dir", ".")) / "clusters.json"
        if cluster_file.exists():
            from core.prs_adapter import update_cluster_after_round
            from pipeline.query_logger import get_cluster_stats
            from core.cluster_manager import load_clusters
            cluster_data = load_clusters(str(cluster_file))
            k = cluster_data["k"]
            faq_coverage = _coverage_ratio(factual_accs, known_good_threshold)
            vdb_coverage = min(len(faqs) / max(effective_cfg.get("scout_initial_faq_count", 20), 1), 1.0)
            for cid_int in range(k):
                cid = str(cid_int)
                realtime_stats = get_cluster_stats(
                    effective_cfg.get("query_log_db", "query_log.db"), cid
                )
                state = update_cluster_after_round(cid, faq_coverage, vdb_coverage, realtime_stats, cfg)
                cluster_states[cid] = state
    except Exception:
        pass

    # Hook 2: Flywheel — record training round snapshot
    try:
        from core.analytics import record_round, init_db
        init_db(cfg)
        if not cluster_states:
            cluster_states = {"global": {
                "prs": prs, "phase": ver.load().get("phase", 1),
                "query_count": len(faqs), "faq_coverage": prs,
            }}
        record_round(cfg, lora_version, cluster_states, tier_distribution={})
    except Exception:
        pass

    # Component 2: Knowledge Differentiation Score (KDS) for KV-injection eligibility.
    if effective_cfg.get("compute_kds", True):
        try:
            compute_kds(faqs, cfg, lora_checkpoint)
        except Exception as e:
            print(f"⚠️ KDS computation failed: {e}", flush=True)

    return prs


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--faqs", default="examples/bedrock_50_faqs.json")
    p.add_argument("--sample", type=int, default=50)
    p.add_argument("--checkpoint", default=None,
                   help="Override LoRA checkpoint path (default: use version.json)")
    p.add_argument("--skip-version-update", action="store_true",
                   help="Do not write PRS result to version.json (useful for ablations)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.faqs) as f:
        all_faqs = json.load(f)

    ver.init(cfg)
    model_loader.init(cfg)

    import random
    faqs = random.sample(all_faqs, min(args.sample, len(all_faqs)))

    v = ver.load()
    prs = evaluate(faqs, cfg, args.checkpoint or v.get("checkpoint_path"))
    round_num = v["current_lora_version"]
    if not args.skip_version_update:
        # PRS thresholds commonly live under addon_config.training in real
        # config files, not at cfg's top level — flatten before reading them
        # so configured values are actually honored instead of silently
        # falling back to the Python-level defaults every time.
        training_cfg = cfg.get("addon_config", {}).get("training", {})
        threshold_cfg = {**cfg, **training_cfg}
        ver.append_prs(
            round_num,
            prs,
            regression_threshold=threshold_cfg.get("prs_regression_threshold", 0.25),
            stability_window=threshold_cfg.get("prs_stability_window", 3),
            phase2_advance_threshold=threshold_cfg.get("phase2_advance_threshold", 0.30),
            phase3_advance_threshold=threshold_cfg.get("phase3_advance_threshold", 0.55),
        )
        mean_kds, kds_by_chunk = compute_kds(
            all_faqs, cfg, args.checkpoint or v.get("checkpoint_path")
        )
        print(f"📊 KDS after round {round_num}: {mean_kds:.4f} (measured {len(kds_by_chunk)} chunks)")
    print(f"📊 PRS after round {round_num}: {prs:.4f}")
    print(f"   Phase: {ver.get_phase()}")


if __name__ == "__main__":
    main()
