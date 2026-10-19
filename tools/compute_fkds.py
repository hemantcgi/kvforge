"""Compute factual KDS (fKDS) for validation.

fKDS blends the existing consistency-based KDS with a factual-accuracy
component scored against FAQ ground truth. This script is standalone and does
not modify the core KDS pipeline; it is meant to validate whether fKDS
correlates with KV-injection factual accuracy better than the current KDS.

Output: JSON with per-chunk consistency, factual accuracy, and fKDS.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import metrics
from pipeline.prs_evaluator import _generate_parametric, _self_consistency
from core import model_loader, version as ver
from vectorstore.registry import get_store
from fastembed import TextEmbedding


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _make_judge_client(model: str):
    """Create a judge client for the requested model."""
    if model.startswith("claude"):
        try:
            import anthropic
            return anthropic.Anthropic()
        except Exception as e:
            print(f"⚠️ Could not create Anthropic client: {e}", flush=True)
            return None
    # Add OpenAI path if needed
    return None


def compute_fkds(
    faqs: list[dict],
    cfg: dict,
    lora_checkpoint: str | None = None,
    sample_cap: int = 100,
    n: int = 2,
    factual_weight: float = 0.1,
    judge_model: str = "claude-sonnet-4-6",
    ignore_last_kds: bool = False,
) -> dict:
    """Compute factual KDS (fKDS) for a corpus.

    Returns a dict with:
      - mean_kds: average consistency KDS
      - mean_factual: average factual accuracy
      - mean_fkds: average fKDS
      - by_chunk: dict mapping chunk_id to scores
    """
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    training_cfg = cfg.get("addon_config", {}).get("training", {})
    effective_cfg = {**cfg, **indexing_cfg, **training_cfg}
    sft_format = effective_cfg.get("sft_format", "chat")
    q_key = effective_cfg.get("faq_question_key", "question")
    a_key = effective_cfg.get("faq_answer_key", "answer")

    ver.init(cfg)
    model_loader.init(cfg)

    # Group FAQs by chunk
    faq_by_chunk: dict[str, list[dict]] = {}
    for faq in faqs:
        for cid in faq.get("source_chunk_ids", []):
            faq_by_chunk.setdefault(str(cid), []).append(faq)

    # Load chunk metadata
    store = get_store(cfg)
    collection = effective_cfg.get("collection")
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

    # Rotating coverage (default) or deterministic numeric order
    def _sort_key(cid: str):
        if ignore_last_kds:
            try:
                return int(cid)
            except ValueError:
                return cid
        payload = chunk_meta[cid][1]
        last = payload.get("last_kds_round")
        if last is None:
            return (0, 0)
        return (1, last)

    eligible = [cid for cid in chunk_meta if cid in faq_by_chunk]
    eligible.sort(key=_sort_key)
    selected = eligible[:sample_cap]

    result = {
        "mean_kds": 0.0,
        "mean_factual": 0.0,
        "mean_fkds": 0.0,
        "by_chunk": {},
        "factual_weight": factual_weight,
        "judge_model": judge_model,
        "n": n,
        "sample_cap": sample_cap,
        "measured_chunks": 0,
    }

    if not selected:
        return result

    # Load model and resources
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
    judge_client = _make_judge_client(judge_model)

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
            # Generate N parametric answers
            answers = []
            for _ in range(n):
                ans = _generate_parametric(q, pipe_sample, tokenizer, sft_format)
                answers.append(ans)

            # Embed answers
            embs = np.array(list(embedder.embed(answers)))
            pooled_embs.append(embs)

            # Score factual accuracy
            gt = faq.get(a_key, "")
            for ans in answers:
                f1 = metrics.token_f1(ans, gt)
                judge = metrics.llm_judge(q, ans, gt, client=judge_client, model=judge_model)
                factual_acc = 0.5 * f1 + 0.5 * float(judge["factually_correct"])
                factual_scores.append(factual_acc)

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
        factual_sum = 0.0
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

            result["by_chunk"][cid] = {
                "kds": kds,
                "factual_accuracy": mean_factual,
                "fkds": fkds,
            }
            kds_sum += kds
            factual_sum += mean_factual
            fkds_sum += fkds

        n_chunks = len(chunk_means)
        result["mean_kds"] = kds_sum / n_chunks
        result["mean_factual"] = factual_sum / n_chunks
        result["mean_fkds"] = fkds_sum / n_chunks
        result["measured_chunks"] = n_chunks

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--faqs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sample-cap", type=int, default=100)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--factual-weight", type=float, default=0.1,
                   help="Weight on the consistency KDS component; 0.1 matches the production pipeline/prs_evaluator.py default.")
    p.add_argument("--judge-model", default="claude-sonnet-4-6")
    p.add_argument("--ignore-last-kds", action="store_true",
                   help="Select chunks in numeric order ignoring last_kds_round")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    addon = cfg.get("addon_config", {})
    for section in ("indexing", "inference", "training", "background", "sync", "monitoring"):
        cfg.update(addon.get(section, {}))

    faqs = json.loads(Path(args.faqs).read_text())
    v = ver.load()
    result = compute_fkds(
        faqs,
        cfg,
        lora_checkpoint=v.get("checkpoint_path"),
        sample_cap=args.sample_cap,
        n=args.n,
        factual_weight=args.factual_weight,
        judge_model=args.judge_model,
        ignore_last_kds=args.ignore_last_kds,
    )

    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")
    print(f"mean KDS: {result['mean_kds']:.4f}")
    print(f"mean factual: {result['mean_factual']:.4f}")
    print(f"mean fKDS: {result['mean_fkds']:.4f}")


if __name__ == "__main__":
    main()
