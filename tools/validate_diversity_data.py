"""Quality gates for EntiGraph and rewrite training data.

Validates that synthetic training data meets minimum quality thresholds
before it is used in absorption-curve experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import token_f1


def _token_overlap(a: str, b: str) -> float:
    """Return token-F1 as a lexical overlap proxy."""
    return token_f1(a, b)


def check_rewrite_coverage(rewrites: list[dict], source_chunks: list[dict]) -> dict:
    """Check that rewrites preserve factual content of their source chunks.

    Pass criterion: >= 90% of rewrites have token-F1 >= 0.3 with their source.
    """
    source_by_id = {str(c["chunk_id"]): c["text"] for c in source_chunks}
    n_pass = 0
    for rw in rewrites:
        src_text = source_by_id.get(str(rw["chunk_id"]), "")
        if src_text and _token_overlap(rw["text"], src_text) >= 0.3:
            n_pass += 1
    rate = n_pass / len(rewrites) if rewrites else 0.0
    return {
        "pass": rate >= 0.9,
        "coverage_rate": round(rate, 4),
        "details": f"{n_pass}/{len(rewrites)} rewrites preserve source content",
    }


def check_qa_grounding(qa_pairs: list[dict], source_chunks: list[dict]) -> dict:
    """Check that QA answers are grounded in their source chunk text.

    Pass criterion: >= 90% of answers have token-F1 >= 0.2 with their source.
    """
    source_by_id = {str(c["chunk_id"]): c["text"] for c in source_chunks}
    n_pass = 0
    for qa in qa_pairs:
        src_text = source_by_id.get(str(qa.get("chunk_id", "")), "")
        if src_text and _token_overlap(qa.get("answer", ""), src_text) >= 0.2:
            n_pass += 1
    rate = n_pass / len(qa_pairs) if qa_pairs else 0.0
    return {
        "pass": rate >= 0.9,
        "grounding_rate": round(rate, 4),
        "details": f"{n_pass}/{len(qa_pairs)} QA answers grounded in source",
    }


def check_entity_graph_quality(relations: list[dict], entities: dict) -> dict:
    """Check EntiGraph output quality.

    Pass criteria:
    - entity_coverage: >= 60% of entities appear in multiple chunks (>= 2)
    - cross_chunk_rate: >= 50% of relations involve cross-chunk entity pairs
    """
    multi_chunk = sum(1 for e in entities.values() if len(e.get("chunk_ids", [])) >= 2)
    entity_coverage = multi_chunk / len(entities) if entities else 0.0

    n_cross = 0
    for rel in relations:
        pair = rel.get("entity_pair", ())
        if len(pair) == 2:
            ent_a = entities.get(pair[0], {})
            ent_b = entities.get(pair[1], {})
            chunks_a = set(ent_a.get("chunk_ids", []))
            chunks_b = set(ent_b.get("chunk_ids", []))
            if chunks_a and chunks_b and not chunks_a.issubset(chunks_b):
                n_cross += 1
    cross_chunk_rate = n_cross / len(relations) if relations else 0.0

    return {
        "pass": entity_coverage >= 0.6 and cross_chunk_rate >= 0.5,
        "cross_chunk_rate": round(cross_chunk_rate, 4),
        "entity_coverage": round(entity_coverage, 4),
        "details": f"entity_coverage={entity_coverage:.2f}, cross_chunk_rate={cross_chunk_rate:.2f}",
    }
