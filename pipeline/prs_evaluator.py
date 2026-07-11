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


def _generate_parametric(query: str, pipe) -> str:
    """Generate an answer to *query* from model weights with no retrieved context.

    Args:
        query: The question string.
        pipe: A HuggingFace ``text-generation`` pipeline.

    Returns:
        The generated answer text (prompt prefix stripped).
    """
    out = pipe(query)
    return out[0]["generated_text"][len(query):].strip()


def _extract_confidence(answer: str, pipe_short) -> float:
    """Ask model to self-rate confidence; return value in [0, 1]."""
    prompt = answer + CONFIDENCE_PROMPT_SUFFIX
    out = pipe_short(prompt)
    tail = out[0]["generated_text"][len(prompt):].strip()
    try:
        val = int("".join(c for c in tail if c.isdigit())[:3])
        return min(val, 100) / 100.0
    except ValueError:
        return 0.5


def _self_consistency(query: str, pipe_sample, embedder, n: int = 3) -> float:
    """Generate n answers at temperature 0.7; return mean pairwise cosine sim."""
    answers = [pipe_sample(query)[0]["generated_text"][len(query):].strip()
               for _ in range(n)]
    embs = np.array(list(embedder.embed(answers)))
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(_cosine_sim(embs[i], embs[j]))
    return float(np.mean(sims)) if sims else 1.0


_DEFAULT_PRS_WEIGHTS = {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2}


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
            Defaults to ``{accuracy: 0.5, calibration: 0.3, consistency: 0.2}``.

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


def evaluate(faqs: list[dict], cfg: dict, lora_checkpoint: str | None = None) -> float:
    """Compute the Parametric Readiness Score on a sample of FAQs.

    For each FAQ:

    1. Generate a parametric answer (no context) and a RAG answer (with
       retrieval, if ``kv_inference`` is available).
    2. Embed both answers and the ground-truth answer.
    3. Compute ``accuracy_ratio``, ``calibration``, and ``self_consistency``.

    After evaluation, queries where ``accuracy_ratio >= 0.85`` are recorded as
    "known-good" in ``version.json`` for use by the confidence gate.

    Args:
        faqs: List of FAQ dicts.  Each must have the keys specified by
            ``cfg['faq_question_key']`` and ``cfg['faq_answer_key']``.
        cfg: Datasource configuration dict.
        lora_checkpoint: Path to a LoRA adapter directory to load before
            evaluation.  ``None`` uses the base model.

    Returns:
        PRS score in ``[0.0, 1.0]``.
    """
    model, tokenizer = model_loader.load(lora_checkpoint)
    embed_model = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")

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

    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")

    total = len(faqs)
    for idx, faq in enumerate(faqs, 1):
        q, gt = _extract_qa(faq, q_key=q_key, a_key=a_key)
        print(f"⏳ Evaluating FAQ {idx}/{total}: {q[:60]}…", flush=True)
        param_ans = _generate_parametric(q, pipe_gen)
        if has_sp3:
            rag_ans = answer_with_retrieval(q, cfg)
        else:
            rag_ans = gt
        embs = np.array(list(embedder.embed([param_ans, rag_ans, gt])))
        param_sim = _cosine_sim(embs[0], embs[2])
        rag_sim   = _cosine_sim(embs[1], embs[2])
        accuracy_ratio = min(param_sim / (rag_sim + 1e-9), 1.0)
        accuracy_ratios.append(accuracy_ratio)
        self_conf = _extract_confidence(param_ans, pipe_conf)
        calibrations.append(1.0 - abs(self_conf - param_sim))
        consistencies.append(_self_consistency(q, pipe_sample, embedder))
        print(f"   acc={accuracy_ratio:.3f} conf={calibrations[-1]:.3f} cons={consistencies[-1]:.3f}", flush=True)

    weights = cfg.get("prs_weights", None)
    prs = _compute_prs(accuracy_ratios, calibrations, consistencies, weights)

    # Populate known_good_queries: queries where accuracy_ratio >= 0.85
    # Stored as pre-computed embeddings for use by confidence_gate._query_similarity
    good_queries = [faqs[i].get(q_key, faqs[i].get("question", ""))
                    for i, r in enumerate(accuracy_ratios) if r >= 0.85]
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
        cluster_file = _Path(cfg.get("checkpoint_dir", ".")) / "clusters.json"
        if cluster_file.exists():
            from core.prs_adapter import update_cluster_after_round
            from pipeline.query_logger import get_cluster_stats
            from core.cluster_manager import load_clusters
            cluster_data = load_clusters(str(cluster_file))
            k = cluster_data["k"]
            faq_coverage = sum(1 for r in accuracy_ratios if r >= 0.85) / max(len(accuracy_ratios), 1)
            vdb_coverage = min(len(faqs) / max(cfg.get("scout_initial_faq_count", 20), 1), 1.0)
            for cid_int in range(k):
                cid = str(cid_int)
                realtime_stats = get_cluster_stats(
                    cfg.get("query_log_db", "query_log.db"), cid
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

    return prs


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    p.add_argument("--faqs", default="examples/bedrock_50_faqs.json")
    p.add_argument("--sample", type=int, default=50)
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
    prs = evaluate(faqs, cfg, v.get("checkpoint_path"))
    round_num = v["current_lora_version"]
    ver.append_prs(
        round_num,
        prs,
        regression_threshold=cfg.get("prs_regression_threshold", 0.60),
        stability_window=cfg.get("prs_stability_window", 3),
        advance_threshold=cfg.get("prs_threshold", 0.50),
    )
    print(f"📊 PRS after round {round_num}: {prs:.4f}")
    print(f"   Phase: {ver.get_phase()}")


if __name__ == "__main__":
    main()
