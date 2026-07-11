"""Sleep-time corpus curation pass.

Runs after FAQ generation. Computes CIS scores and determines tier actions:
  - promote_to_enhanced: chunks to schedule for per-token KV computation
  - archive_candidates:  chunks to surface to the dashboard for user review
"""
import json
import numpy as np
from pathlib import Path


def run_coverage_sweep(
    faqs: list[str],
    chunks: list[dict],
    embedder,
    top_k: int = 5,
) -> dict[int, list[str]]:
    """Run FAQs against corpus. Returns {faq_idx: [chunk_id_rank1, ...]}."""
    chunk_ids  = [c["id"] for c in chunks]
    chunk_vecs = np.stack([np.array(c["vector"], dtype=np.float32) for c in chunks])
    norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True).clip(min=1e-8)
    chunk_vecs_n = chunk_vecs / norms

    results = {}
    for faq_idx, faq_text in enumerate(faqs):
        faq_emb = embedder.embed([faq_text])[0].astype(np.float32)
        faq_emb = faq_emb / np.linalg.norm(faq_emb).clip(min=1e-8)
        sims    = chunk_vecs_n @ faq_emb
        ranked  = np.argsort(-sims)[:top_k].tolist()
        results[faq_idx] = [chunk_ids[i] for i in ranked]
    return results


def identify_tier_actions(
    cis_scores:    dict[str, float],
    unique_scores: dict[str, float],
    cfg,
    already_enhanced: set[str] | None = None,
) -> dict[str, list[str]]:
    """Classify chunks into tier actions based on CIS and uniqueness.

    Returns dict with keys:
      promote_to_enhanced: list of chunk IDs to schedule for background promotion
      archive_candidates:  list of chunk IDs to surface as archival recommendations
    """
    already_enhanced = already_enhanced or set()
    promote, archive = [], []

    for cid, score in cis_scores.items():
        unique = unique_scores.get(cid, 0.5)

        if score >= cfg.enhanced_tier_threshold and cid not in already_enhanced:
            promote.append(cid)
        elif (score < cfg.archive_candidate_threshold
              and unique < cfg.uniqueness_floor):
            archive.append(cid)

    return {"promote_to_enhanced": promote, "archive_candidates": archive}


def run_curation_pass(
    faqs: list[str],
    chunks: list[dict],
    embedder,
    cfg,
    version_file: str,
) -> dict:
    """Full curation pass: coverage sweep → CIS → tier actions → persist.

    Returns the tier actions dict.
    """
    from addons.corpus_intelligence.cis import (
        compute_access_score, compute_uniqueness_score,
        compute_coverage_score, compute_cis,
    )

    hit_counts  = {c["id"]: c["payload"].get("hit_count", 0) for c in chunks}
    embeddings  = {c["id"]: np.array(c["vector"], dtype=np.float32) for c in chunks}
    already_enh = {c["id"] for c in chunks if c["payload"].get("kv_token_path")}

    coverage_map = run_coverage_sweep(faqs, chunks, embedder, top_k=5)

    access_scores   = compute_access_score(hit_counts)
    unique_scores   = compute_uniqueness_score(embeddings)
    coverage_scores = compute_coverage_score(coverage_map)
    cis_scores      = compute_cis(
        access_scores, unique_scores, coverage_scores,
        alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
    )

    cis_path = Path(version_file).with_suffix(".cis.json")
    cis_path.write_text(json.dumps(cis_scores, indent=2), encoding="utf-8")

    return identify_tier_actions(cis_scores, unique_scores, cfg,
                                 already_enhanced=already_enh)
