#!/usr/bin/env python3
"""
ask.py — Query KVForge from the command line.

Usage:
    python3 ask.py "What is Bedrock?"
    python3 ask.py --config examples/usecase4_bedrock_userguide/config.json "What is Amazon Bedrock?"
    python3 ask.py --config examples/usecase4_bedrock_userguide/config.json --top-k 3 "What is Amazon Bedrock?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="Query KVForge")
    parser.add_argument("query", help="Question to ask")
    parser.add_argument("--config", default="my_config.json",
                        help="Config file (default: my_config.json)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--chunk-chars", type=int, default=500,
                        help="Max chars per chunk in prompt (default: 500)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Use base model without LoRA adapter")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cfg = dict(cfg)
    cfg["top_k"] = args.top_k

    print(f"Searching Qdrant for: {args.query!r}", flush=True)

    import torch
    import core.version as ver
    import core.model_loader as model_loader
    import pipeline.kv_background as kv_background
    from fastembed import TextEmbedding
    from vectorstore.registry import get_store
    from pipeline.bedrock_rag import _run_search, Config

    ver.init(cfg)
    model_loader.init(cfg)
    kv_background.start(cfg)

    # Support both flat configs and nested addon_config.
    embed_model = cfg.get("embed_model", cfg.get("addon_config", {}).get("indexing", {}).get("embed_model", "BAAI/bge-small-en-v1.5"))
    embedder = TextEmbedding(model_name=embed_model, show_download_progress=False)
    store = get_store(cfg)
    indexing_cfg = cfg.get("addon_config", {}).get("indexing", {})
    inference_cfg = cfg.get("addon_config", {}).get("inference", {})
    flat_cfg = {**cfg, **indexing_cfg, **inference_cfg}
    rag_cfg = Config(**{k: flat_cfg[k] for k in Config.__dataclass_fields__ if k in flat_cfg})

    hits = _run_search(args.query, embedder, store, rag_cfg)
    if not hits:
        print("No relevant chunks found.")
        sys.exit(0)

    print(f"Found {len(hits)} chunks. Top score: {hits[0].score:.4f}")

    chunks = [
        {
            "page": h.payload.get("page", 0),
            "score": round(h.score, 4),
            "text": h.payload["text"][:args.chunk_chars],
        }
        for h in hits
    ]

    print("Loading model…", flush=True)
    lora_ckpt = None if args.no_lora else ver.load().get("checkpoint_path")
    if lora_ckpt:
        print(f"  LoRA: {lora_ckpt}")
    else:
        print("  Base model (no LoRA)")
    model, tokenizer = model_loader.load(lora_ckpt)
    model = model.half()  # ensure consistent float16 after LoRA merge

    # Simple instruction prompt — avoids chat template tokens that confuse the
    # LoRA-finetuned model when used with greedy decoding.
    context_parts = []
    for c in chunks:
        context_parts.append(f"[page {c['page']}, score {c['score']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        f"Using only the context below, answer the question in 2-4 sentences. "
        f"Cite page numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {args.query}\n\n"
        f"Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600,
    ).to(model.device)

    token_count = inputs["input_ids"].shape[1]
    print(f"Prompt: {token_count} tokens. Generating…", flush=True)
    print()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
