"""
kv_inference.py — KV-injected inference with text-in-context fallback.

Decision logic per query:
  ALL chunks have kv_version == current_lora_version → KV injection (fast)
  ANY chunk stale or null                            → text-in-context fallback
  Either path                                        → enqueue stale chunks for bg heal
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
import kv_utils
import kv_background
import model_loader
import version as ver
from bedrock_rag import _run_search, Config
from fastembed import TextEmbedding
from vectorstore.registry import get_store


SYSTEM_PROMPT = (
    "You are a precise assistant. Answer ONLY using the provided context. "
    "Cite sources inline as [page P]. "
    "End with: Confidence: <0-100>%  — <one sentence explanation>"
)


# ── Pure decision functions (testable without GPU) ────────────────────────

def decide_inference_mode(chunks: list[dict], current_lora_version: int) -> str:
    """Return 'kv_injection' if all chunks are fresh and have kv_cache, else 'text_fallback'."""
    for chunk in chunks:
        v = chunk.get("kv_version")
        if v is None or v < current_lora_version:
            return "text_fallback"
        if chunk.get("kv_cache") is None:
            return "text_fallback"  # kv_cache missing — can't inject
    return "kv_injection"


def get_stale_chunk_ids(chunks: list[dict], current_lora_version: int) -> list[int]:
    return [
        c["chunk_id"] for c in chunks
        if c.get("kv_version") is None or c["kv_version"] < current_lora_version
    ]


# ── Inference paths ───────────────────────────────────────────────────────

def generate_with_kv(query: str, chunks: list[dict],
                      model, tokenizer, cfg: dict) -> str:
    """Fast path: inject pre-computed KV tensors as past_key_values."""
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
    kv_shape = (num_layers, 2, num_kv_heads, head_dim)

    chunk_kvs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape)
        for c in chunks
    ]
    past_kv = kv_utils.stack_past_key_values(
        chunk_kvs, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    # Move past_kv tensors to model device.
    # DynamicCache (transformers >= 5.x): move each layer's keys/values in place.
    # Legacy tuple format: rebuild tuple with device-moved tensors.
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            for layer in past_kv.layers:
                layer.keys = layer.keys.to(model.device)
                layer.values = layer.values.to(model.device)
        else:
            past_kv = tuple(
                (k.to(model.device), v.to(model.device)) for k, v in past_kv
            )
    except ImportError:
        past_kv = tuple(
            (k.to(model.device), v.to(model.device)) for k, v in past_kv
        )

    prompt = f"Based on the context provided, answer: {query}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            past_key_values=past_kv,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)


def generate_text_in_context(query: str, chunks: list[dict],
                               model, tokenizer,
                               max_new_tokens: int = 256,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               repetition_penalty: float = 1.2) -> str:
    """Fallback path: include chunk text in prompt."""
    context = "\n\n---\n\n".join(
        f"[page {c['page']}, score {c['score']}]\n{c['text']}"
        for c in chunks
    )
    # Direct instruction prompt — avoids chat-template tokens that confuse the
    # model when used without the exact fine-tuning format.
    prompt = (
        f"Using only the context below, answer the question in 2-4 sentences. "
        f"Cite page numbers.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1600,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()


def answer_with_retrieval(query: str, cfg: dict) -> str:
    """
    Full SP3 pipeline: search → version check → KV inject or text fallback.
    Called by prs_evaluator.py for RAG-mode answers.
    """
    embedder = TextEmbedding(model_name=cfg["embed_model"],
                              show_download_progress=False)
    store = get_store(cfg)

    # Build Config from cfg dict — use only keys that Config expects
    rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})

    hits = _run_search(query, embedder, store, rag_cfg)
    if not hits:
        return ""

    chunks = [
        {
            "chunk_id": h.id,
            "text": h.payload["text"],
            "page": h.payload["page"],
            "score": round(h.score, 4),
            "kv_cache": h.payload.get("kv_cache"),
            "kv_version": h.payload.get("kv_version"),
        }
        for h in hits
    ]

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    # Record access
    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    # Enqueue stale for background healing
    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    mode = decide_inference_mode(chunks, current_ver)
    if mode == "kv_injection":
        return generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        return generate_text_in_context(query, chunks, model, tokenizer)


def main() -> None:
    """Pipe-compatible: read JSON from stdin (from bedrock_rag.py search)."""
    if sys.stdin.isatty():
        print('Usage: python3 bedrock_rag.py search "query" | python3 kv_inference.py')
        sys.exit(1)

    data = json.load(sys.stdin)
    query = data["query"]
    chunks = data["chunks"]

    with open("my_config.json") as f:
        cfg = json.load(f)

    kv_background.start(cfg)

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    for rank, chunk in enumerate(chunks, start=1):
        kv_background.record_access(chunk["chunk_id"], rank)

    stale = get_stale_chunk_ids(chunks, current_ver)
    if stale:
        kv_background.enqueue_kv_recompute(stale)

    mode = decide_inference_mode(chunks, current_ver)
    print(f"Mode: {mode}  |  lora_version={current_ver}  |  "
          f"stale_chunks={len(stale)}/{len(chunks)}")
    print("-" * 62)

    if mode == "kv_injection":
        answer = generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        answer = generate_text_in_context(query, chunks, model, tokenizer)

    print(answer)


if __name__ == "__main__":
    main()
