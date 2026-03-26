"""Phase 3 confidence gate: answer parametrically or fall back to retrieval.

When the system reaches Phase 3 (``version.json["phase"] >= 3``), every
incoming query is first evaluated against three confidence signals:

1. **Token entropy** — low entropy during a short draft generation indicates
   the model is confident in its answer.
2. **Hedging score** — presence of hedging phrases (``"I think"``,
   ``"maybe"``, etc.) in the draft indicates uncertainty.
3. **Query similarity** — cosine similarity to previously seen queries that
   the model answered accurately.

If the weighted combination exceeds ``gate_threshold`` the query is answered
directly from the fine-tuned model weights (parametric mode); otherwise the
full KV-retrieval pipeline is invoked.

If the phase is below 3, all queries are routed to the retrieval pipeline
unconditionally.
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline.kv_background as kv_background
import core.model_loader as model_loader
import core.version as ver

HEDGING_MARKERS = [
    "i think", "i'm not sure", "i am not sure", "approximately",
    "maybe", "or maybe", "i don't know", "i do not know",
    "possibly", "it might", "it may",
]

DRAFT_TOKENS = 20
DEFAULT_THRESHOLD = 0.75
_embedder = None  # cached TextEmbedding instance — avoid reload per query


# ── Pure functions (testable without model) ───────────────────────────────

def compute_hedging_score(text: str) -> float:
    """Fraction of distinct hedging marker concepts present in text (0.0–1.0).
    Checks longest markers first to avoid double-counting substrings (e.g. 'or maybe' vs 'maybe').
    """
    lower = text.lower()
    # Sort longest first so 'or maybe' is matched before 'maybe'
    hits = 0
    remaining = lower
    for m in sorted(HEDGING_MARKERS, key=len, reverse=True):
        if m in remaining:
            hits += 1
            remaining = remaining.replace(m, "")  # remove matched text to prevent substring overlap
    return round(hits / len(HEDGING_MARKERS), 4)


def decide_gate(
    token_entropy: float,
    hedging_score: float,
    query_similarity: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """
    Compute P(no_retrieval) from three signals and apply threshold.
    Returns 'direct' or 'retrieve'.

    Weights: entropy 0.4, hedging 0.3, similarity 0.3
    Low entropy → high confidence, high hedging → low confidence.
    """
    entropy_score = max(0.0, 1.0 - token_entropy)       # invert: low entropy = good
    hedging_contribution = max(0.0, 1.0 - hedging_score) # invert: low hedging = good
    p_no_retrieval = (0.4 * entropy_score
                      + 0.3 * hedging_contribution
                      + 0.3 * query_similarity)
    return "direct" if p_no_retrieval >= threshold else "retrieve"


def _token_entropy(logits: torch.Tensor) -> float:
    """Compute mean per-token entropy from a stack of draft generation logits.

    Args:
        logits: Tensor of shape ``[num_tokens, vocab_size]`` containing the
            raw (pre-softmax) logits for each generated token.

    Returns:
        Mean Shannon entropy in nats, averaged across all generated tokens.
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-9)
    entropy_per_token = -(probs * log_probs).sum(dim=-1)
    return float(entropy_per_token.mean().item())


def _generate_draft(query: str, model, tokenizer,
                     max_tokens: int = DRAFT_TOKENS) -> tuple[str, float]:
    """Generate a short draft answer to *query* and compute its mean token entropy.

    Args:
        query: The user query string.
        model: Loaded HuggingFace causal language model.
        tokenizer: Corresponding tokenizer.
        max_tokens: Maximum number of new tokens to generate for the draft.

    Returns:
        A ``(draft_text, mean_entropy)`` tuple where *draft_text* is the
        decoded draft answer and *mean_entropy* is the mean per-token entropy.
    """
    inputs = tokenizer(query, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    scores = torch.stack(output.scores, dim=1)  # [1, num_tokens, vocab]
    entropy = _token_entropy(scores.squeeze(0))
    draft = tokenizer.decode(
        output.sequences[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return draft, entropy


def _query_similarity_to_known_good(query: str, cfg: dict) -> float:
    """Compute the maximum cosine similarity between *query* and known-good query embeddings.

    Known-good embeddings are pre-computed by ``prs_evaluator.evaluate`` and
    stored in ``version.json["known_good_queries"]``.

    Args:
        query: The incoming user query.
        cfg: Datasource configuration dict; uses ``cfg['embed_model']``.

    Returns:
        Maximum cosine similarity in [0, 1].  Returns ``0.5`` (neutral) when
        no known-good queries have been recorded yet.
    """
    global _embedder
    v = ver.load()
    known_good = v.get("known_good_queries", [])
    if not known_good:
        return 0.5  # neutral when no history yet

    from fastembed import TextEmbedding
    if _embedder is None:
        _embedder = TextEmbedding(
            model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
            show_download_progress=False,
        )
    embedder = _embedder
    q_emb = np.array(list(embedder.embed([query]))[0])
    known_embs = np.array(known_good)  # [N, dim]

    sims = known_embs @ q_emb / (
        np.linalg.norm(known_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
    )
    return float(sims.max())


def answer(query: str, cfg: dict) -> str:
    """Phase 3 entry point: route *query* to parametric or retrieval inference.

    If the current phase is below 3, delegates unconditionally to
    ``kv_inference.answer_with_retrieval``.  In Phase 3, generates a short
    draft, evaluates the three confidence signals, and either returns a full
    parametric answer or falls back to retrieval.

    Args:
        query: The user query string.
        cfg: Datasource configuration dict.

    Returns:
        Final answer string.
    """
    if ver.get_phase() < 3:
        from pipeline.kv_inference import answer_with_retrieval
        return answer_with_retrieval(query, cfg)

    threshold = cfg.get("gate_threshold", DEFAULT_THRESHOLD)
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    draft, entropy = _generate_draft(query, model, tokenizer)
    hedging = compute_hedging_score(draft)
    similarity = _query_similarity_to_known_good(query, cfg)

    decision = decide_gate(entropy, hedging, similarity, threshold)
    print(f"  Gate: entropy={entropy:.2f} hedging={hedging:.2f} "
          f"sim={similarity:.2f} -> {decision}", flush=True)

    if decision == "direct":
        # Full generation from weights
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        result = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
        # Log parametric hit for access tracker
        _log_would_have_retrieved(query, cfg)
        return result
    else:
        from pipeline.kv_inference import answer_with_retrieval
        return answer_with_retrieval(query, cfg)


def _log_would_have_retrieved(query: str, cfg: dict) -> None:
    """Record parametric hits for the chunks that would have been retrieved for *query*.

    Called after a successful parametric answer to update the
    ``parametric_hit_count`` metric on the relevant vector store chunks.
    Errors are silently swallowed as this is a non-critical bookkeeping step.

    Args:
        query: The user query that was answered parametrically.
        cfg: Datasource configuration dict.
    """
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient
        from pipeline.bedrock_rag import Config, _run_search
        embedder = TextEmbedding(model_name=cfg["embed_model"],
                                  show_download_progress=False)
        client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})
        hits = _run_search(query, embedder, client, rag_cfg)
        chunk_ids = [h.id for h in hits]
        kv_background.record_parametric_hit(chunk_ids)
    except Exception:
        pass  # non-critical


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--config", default="my_config.json")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    kv_background.start(cfg)
    print(answer(args.query, cfg))
