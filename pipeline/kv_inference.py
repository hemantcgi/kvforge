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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.kv_utils as kv_utils
import pipeline.kv_background as kv_background
import core.model_loader as model_loader
import core.version as ver
from core.kv_utils import deserialize_kv, load_token_kv
from pipeline.bedrock_rag import _run_search, Config
from fastembed import TextEmbedding
from vectorstore.registry import get_store

# Track which query_log_db paths have already been initialised in this process
# so init_db() is not called on every inference request.
_initialized_query_log_dbs: set[str] = set()


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


def route_chunk_injection(chunk: dict, cfg: dict, tq_config=None, vector_store=None) -> dict:
    """Route a single chunk to the correct injection path.

    Returns a dict with keys: path ("enhanced"/"active"/"archive"), kv_arr, text.
    """
    payload = chunk.get("payload", chunk)
    status = payload.get("status", "active")
    kv_token_path = payload.get("kv_token_path")

    if kv_token_path and status != "archived":
        arr = load_token_kv(kv_token_path, tq_config=tq_config)
        return {"path": "enhanced", "kv_arr": arr, "text": None}

    if status == "archived":
        text = _fetch_archive_text(payload.get("archive_path", ""), payload.get("text", ""))
        count = payload.get("archive_retrieval_count", 0) + 1
        if vector_store is not None:
            vector_store.update_payload(
                collection=cfg.get("collection", ""),
                point_id=chunk.get("id"),
                payload={"archive_retrieval_count": count},
            )
        return {"path": "archive", "kv_arr": None, "text": text}

    shape = (cfg["kv_num_layers"], 2, cfg["kv_num_heads"], cfg["kv_head_dim"])
    arr = deserialize_kv(payload["kv_cache"], shape)
    return {"path": "active", "kv_arr": arr, "text": None}


def _fetch_archive_text(archive_path: str, fallback_text: str = "") -> str:
    """Return text from archive_path if it exists, else fallback_text."""
    if not archive_path:
        return fallback_text
    try:
        p = Path(archive_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return fallback_text


# ── Inference paths ───────────────────────────────────────────────────────

def generate_with_kv(query: str, chunks: list[dict],
                      model, tokenizer, cfg: dict,
                      extra_context: str = "") -> str:
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

    context_prefix = f"Additional context:\n{extra_context}\n\n" if extra_context else ""
    prompt = f"{context_prefix}Based on the context provided, answer: {query}"
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
                               repetition_penalty: float = 1.2,
                               extra_context: str = "") -> str:
    """Fallback path: include chunk text in prompt."""
    context = "\n\n---\n\n".join(
        f"[page {c['page']}, score {c['score']}]\n{c['text']}"
        for c in chunks
    )
    if extra_context:
        context += f"\n\n---\n\n{extra_context}"
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
            "page": h.payload.get("page"),
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
        answer = generate_with_kv(query, chunks, model, tokenizer, cfg)
    else:
        answer = generate_text_in_context(query, chunks, model, tokenizer)

    try:
        from pipeline import query_logger as _ql
        _db = cfg.get("query_log_db", "query_log.db")
        if _db not in _initialized_query_log_dbs:
            _ql.init_db(_db)
            _initialized_query_log_dbs.add(_db)
        _ql.log_query(
            db_path=_db,
            query_text=query,
            answer_text=answer,
            routed_to="retrieval",
            cluster_id=None,
            chunk_id=str(chunks[0]["chunk_id"]) if chunks else None,
        )
    except Exception:
        pass

    return answer


def route_query(query: str, cfg: dict) -> list[dict]:
    """Dynamic PRS cluster-aware retrieval.

    Embeds *query*, finds its nearest cluster, then calls
    ``answer_with_retrieval`` restricted to chunks from that cluster.
    Falls back to full-collection retrieval when no cluster data exists.

    Args:
        query: User query string.
        cfg: Datasource config dict.

    Returns:
        List of chunk dicts from the nearest cluster (or full collection).
    """
    from pathlib import Path as _Path
    cluster_file = _Path(cfg.get("checkpoint_dir", ".")) / "clusters.json"
    if not cluster_file.exists():
        return []

    try:
        from core.cluster_manager import load_clusters, nearest_cluster
        cluster_data = load_clusters(str(cluster_file))
        embedder = TextEmbedding(model_name=cfg["embed_model"],
                                  show_download_progress=False)
        q_vec = list(embedder.embed([query]))[0]
        import numpy as np
        cluster_id = nearest_cluster(
            np.array(q_vec), np.array(cluster_data["centroids"])
        )
        store = get_store(cfg)
        from pipeline.bedrock_rag import Config
        rag_cfg = Config(**{k: cfg[k] for k in Config.__dataclass_fields__ if k in cfg})
        hits = store.query(
            cfg["collection"], q_vec.tolist(), top_k=cfg.get("top_k", 5),
            scroll_filter={"cluster_id": str(cluster_id)},
        )
        return [
            {
                "chunk_id": h.id,
                "text": h.payload.get("text", ""),
                "page": h.payload.get("page"),
                "score": round(h.score, 4),
                "kv_cache": h.payload.get("kv_cache"),
                "kv_version": h.payload.get("kv_version"),
                "cluster_id": str(cluster_id),
            }
            for h in hits
        ]
    except Exception:
        return []


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
