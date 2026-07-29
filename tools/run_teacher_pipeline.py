"""Run the distillation query pool through the frozen Path A teacher.

Usage:

    python3 tools/run_teacher_pipeline.py \
        --config teacher_datasource.json \
        --query-pool query_pool.json \
        --output teacher_pairs.json \
        --quality-threshold 0.7

For each query in the pool, the script:

1. Retrieves chunks using the teacher config.
2. Generates a teacher answer via Path A (text-RAG with partial recompute).
3. Records the answer and retrieved chunk ids.
4. Quality-filters the answers against the expected answer (when available).

The output JSON contains ``teacher_pairs`` (list of filtered pairs) and a
``stats`` dict. GPU usage: one forward pass per query.

Run this on EC2 as a chained job after the query pool is built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.model_loader as model_loader
import core.version as ver
from pipeline.distillation import quality_filter
from pipeline.kv_inference import answer_with_mode
from pipeline.bedrock_rag import _run_search, Config
from vectorstore.registry import get_store
from fastembed import TextEmbedding


def _retrieve_chunk_ids(query: str, cfg: dict, embedder, store) -> list[str]:
    """Return the IDs of chunks retrieved for *query* using the teacher config."""
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    inference_cfg = cfg.get("addon_config", {}).get("inference", {})
    flat_cfg = {**cfg, **indexing_cfg, **inference_cfg}
    rag_cfg = Config(**{k: flat_cfg[k] for k in Config.__dataclass_fields__ if k in flat_cfg})
    hits = _run_search(query, embedder, store, rag_cfg)
    return [str(h.id) for h in hits]


def run_teacher_pipeline(
    cfg: dict,
    query_pool: list[dict],
    quality_threshold: float = 0.7,
    judge_model: str = "gpt-4o-mini",
    max_samples: int | None = None,
    max_new_tokens: int | None = None,
) -> dict:
    """Run each query through Path A and return quality-filtered teacher pairs.

    Args:
        cfg: Teacher datasource config.
        query_pool: Deduplicated pool entries from ``build_query_pool.py``.
        quality_threshold: Minimum factual accuracy to keep a teacher answer.
        judge_model: Judge model name for quality filtering.
        max_samples: Optional cap for quick dry-runs.
        max_new_tokens: Override max_new_tokens in config for shorter answers.

    Returns:
        Dict with ``teacher_pairs`` and ``stats``.
    """
    model_loader.init(cfg)
    ver.init(cfg)
    # Teacher always uses the base model, not any fine-tuned adapter.
    model, tokenizer = model_loader.load(None)

    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    flat_cfg = {**cfg, **indexing_cfg}
    embed_model = flat_cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    store = get_store(flat_cfg)

    if max_samples:
        query_pool = query_pool[:max_samples]

    if max_new_tokens is not None:
        inference_cfg = cfg.get("addon_config", {}).get("inference", {})
        inference_cfg["max_new_tokens"] = max_new_tokens
        cfg.setdefault("addon_config", {})["inference"] = inference_cfg
        cfg["max_new_tokens"] = max_new_tokens

    raw_pairs = []
    for idx, entry in enumerate(query_pool, 1):
        q = entry["question"]
        print(f"⏳ Teacher {idx}/{len(query_pool)}: {q[:60]}…", flush=True)
        answer, mode = answer_with_mode(q, cfg, force_mode=None)
        chunk_ids = _retrieve_chunk_ids(q, cfg, embedder, store)
        raw_pairs.append({
            "question": q,
            "teacher_answer": answer,
            "expected_answer": entry.get("expected_answer", ""),
            "source": entry.get("source", "unknown"),
            "chunk_ids": chunk_ids,
            "mode": mode,
        })

    # Only filter pairs that have an expected answer.
    evaluable = [p for p in raw_pairs if p["expected_answer"]]
    non_evaluable = [p for p in raw_pairs if not p["expected_answer"]]

    if quality_threshold <= 0.0:
        # Skip quality filter — keep all teacher answers regardless of score.
        for p in evaluable:
            p["factual_accuracy"] = 0.0
        filtered = evaluable + non_evaluable
        stats = {"kept": len(evaluable), "dropped": 0, "drop_rate": 0.0,
                 "mean_factual_accuracy": 0.0}
    else:
        filtered, stats = quality_filter(
            evaluable, threshold=quality_threshold, judge_model=judge_model
        )
        # Keep non-evaluable (chunk-generated) questions with a neutral accuracy.
        for p in non_evaluable:
            p["factual_accuracy"] = 0.0
        filtered.extend(non_evaluable)

    stats["total"] = len(raw_pairs)
    stats["evaluable"] = len(evaluable)
    stats["non_evaluable"] = len(non_evaluable)
    return {"teacher_pairs": filtered, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query pool through Path A teacher")
    parser.add_argument("--config", required=True, help="Teacher datasource config JSON")
    parser.add_argument("--query-pool", required=True, help="Query pool JSON")
    parser.add_argument("--output", required=True, help="Output teacher pairs JSON")
    parser.add_argument("--quality-threshold", type=float, default=0.7)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override max_new_tokens for shorter teacher answers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.query_pool) as f:
        query_pool = json.load(f)

    result = run_teacher_pipeline(
        cfg, query_pool,
        quality_threshold=args.quality_threshold,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✓ Wrote {args.output}")
    print(f"  Teacher pairs: {len(result['teacher_pairs'])}/{len(query_pool)} "
          f"(kept evaluable={result['stats']['kept']}, "
          f"dropped evaluable={result['stats']['dropped']})")


if __name__ == "__main__":
    main()
