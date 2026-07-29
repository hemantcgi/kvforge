"""Build a distillation query pool for Sprint 2.

Usage:

    python3 tools/build_query_pool.py \
        --config datasource.json \
        --faqs faqs.json \
        --output query_pool.json \
        --faq-paraphrases 3 \
        --real-query-limit 500

The tool loads:

1. Real retrieval-routed queries from the query log database.
2. Paraphrases of the FAQ questions.
3. Chunk-conditioned generated questions (requires a chunk sample from the vector store).

It deduplicates the pool by normalized question text and writes a JSON file.

GPU usage: this script loads the LLM to generate paraphrases and chunk questions.
Run on EC2 as a chained job after training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.model_loader as model_loader
import core.version as ver
from pipeline.distillation import build_query_pool
from vectorstore.registry import get_store


def _sample_chunks(store, collection: str, n: int) -> list[dict]:
    """Sample *n* chunks evenly across the collection."""
    all_points = []
    offset = None
    while True:
        page, offset = store.scroll(collection, limit=200, with_payload=True, offset=offset)
        all_points.extend(page)
        if offset is None or len(all_points) >= n:
            break
    if len(all_points) > n:
        step = max(1, len(all_points) // n)
        sampled = [all_points[i] for i in range(0, len(all_points), step)][:n]
    else:
        sampled = all_points
    return [
        {
            "chunk_id": p.id,
            "text": p.payload.get("text", ""),
        }
        for p in sampled
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a distillation query pool")
    parser.add_argument("--config", required=True, help="Datasource config JSON")
    parser.add_argument("--faqs", required=True, help="FAQ JSON file")
    parser.add_argument("--output", required=True, help="Output query pool JSON")
    parser.add_argument("--faq-paraphrases", type=int, default=3)
    parser.add_argument("--real-query-limit", type=int, default=1000)
    parser.add_argument("--chunk-samples", type=int, default=0,
                        help="Number of chunks to sample for chunk-conditioned questions")
    parser.add_argument("--checkpoint", default=None,
                        help="Override LoRA checkpoint for generation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    ver.init(cfg)
    model_loader.init(cfg)

    with open(args.faqs) as f:
        faqs = json.load(f)

    lora_ckpt = args.checkpoint or ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    query_log_db = cfg.get("query_log_db")
    chunks = None
    if args.chunk_samples > 0:
        indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
        effective_cfg = {**cfg, **indexing_cfg}
        store = get_store(effective_cfg)
        chunks = _sample_chunks(store, effective_cfg["collection"], args.chunk_samples)

    pool = build_query_pool(
        faqs=faqs,
        model=model,
        tokenizer=tokenizer,
        query_log_db=query_log_db,
        chunks=chunks,
        faq_paraphrases=args.faq_paraphrases,
        real_query_limit=args.real_query_limit,
    )

    Path(args.output).write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"✓ Wrote query pool: {len(pool)} entries -> {args.output}")
    sources = {}
    for e in pool:
        sources[e.get("source", "unknown")] = sources.get(e.get("source", "unknown"), 0) + 1
    print(f"  Sources: {sources}")


if __name__ == "__main__":
    main()
