"""Long-lived background workers for KV healing and access-counter flushing.

Two daemon threads are started via ``start(cfg)``:

1. **KV recompute worker** — pulls chunk IDs from ``_kv_queue`` and
   recomputes their KV tensor using the current (possibly updated) LoRA model.
   Automatically reloads the model when a new LoRA version is detected.

2. **Access flush worker** — periodically drains the in-memory
   ``_access_buffer`` and writes the accumulated access counts, retrieval
   ranks, and parametric-hit counts back to the vector store.  Flush is
   triggered when either the query count threshold or the time interval is
   reached.

Public API (called from inference threads):

* ``enqueue_kv_recompute(chunk_ids)`` — schedule KV recomputation.
* ``record_access(chunk_id, rank)`` — record a retrieval event (zero latency).
* ``record_parametric_hit(chunk_ids)`` — record parametric-mode hits.
* ``start(cfg)`` — start both background threads (idempotent).

Run standalone alongside ``kv_inference.py``::

    python3 kv_background.py &
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.kv_utils as kv_utils
import core.model_loader as model_loader
import core.version as ver
from core.kv_utils import compute_per_token_kv, save_token_kv
from vectorstore.registry import get_store

_kv_queue: queue.Queue = queue.Queue()
_image_kv_queue: queue.Queue = queue.Queue()
_access_buffer: dict[int, dict] = {}
_access_lock = threading.Lock()
_query_count = 0
_query_lock = threading.Lock()


# ── Public API (called by kv_inference.py) ────────────────────────────────

def enqueue_kv_recompute(chunk_ids: list[int]) -> None:
    """Schedule KV tensor recomputation for the given chunk IDs.

    Puts each ID onto the internal queue consumed by ``_kv_worker``.
    This call returns immediately and does not block the inference thread.

    Args:
        chunk_ids: List of chunk identifiers whose KV cache needs refreshing.
    """
    for cid in chunk_ids:
        _kv_queue.put(cid)


def enqueue_image_kv_recompute(chunk_ids: list[int]) -> None:
    """Schedule image KV tensor recomputation for the given chunk IDs.

    Puts each ID onto the internal image queue consumed by ``_image_kv_worker``.
    Returns immediately; does not block the inference thread.

    Args:
        chunk_ids: List of image chunk identifiers whose KV cache needs refreshing.
    """
    for cid in chunk_ids:
        _image_kv_queue.put(cid)


def record_access(chunk_id: int, rank: int) -> None:
    """Record a retrieval event for *chunk_id* at result position *rank*.

    Updates the in-memory ``_access_buffer`` under a lock and increments the
    global query counter.  No I/O is performed; the buffer is flushed
    asynchronously by ``_access_worker``.

    Args:
        chunk_id: Identifier of the retrieved chunk.
        rank: 1-based position in the retrieval result list.
    """
    global _query_count
    with _access_lock:
        if chunk_id not in _access_buffer:
            _access_buffer[chunk_id] = {"count": 0, "rank_sum": 0.0, "last_ts": 0}
        _access_buffer[chunk_id]["count"] += 1
        _access_buffer[chunk_id]["rank_sum"] += rank
        _access_buffer[chunk_id]["last_ts"] = int(time.time())
    with _query_lock:
        _query_count += 1


def record_parametric_hit(chunk_ids: list[int]) -> None:
    """Record parametric-mode hits for chunks that would have been retrieved.

    Called by the confidence gate after answering a query directly from model
    weights.  Increments the in-memory ``parametric_hits`` counter for each
    chunk ID so the access flush worker can persist the metric.

    Args:
        chunk_ids: List of chunk IDs that would have been retrieved for this
            query.
    """
    with _access_lock:
        for cid in chunk_ids:
            if cid not in _access_buffer:
                _access_buffer[cid] = {"count": 0, "rank_sum": 0.0, "last_ts": 0,
                                        "parametric_hits": 0}
            _access_buffer[cid].setdefault("parametric_hits", 0)
            _access_buffer[cid]["parametric_hits"] += 1


# ── KV recompute worker ───────────────────────────────────────────────────

def _kv_worker(cfg: dict) -> None:
    """Background thread: drain the KV recompute queue and update stale chunks.

    Runs indefinitely.  When a new LoRA version is detected (``version.json``
    changes), the model is reloaded via ``model_loader.reload`` so that
    recomputed KV tensors reflect the latest fine-tuned weights.

    Args:
        cfg: Datasource configuration dict.
    """
    client = get_store(cfg)
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    # Load model once at startup (model_loader singleton reuses it across calls)
    v = ver.load()
    _cached_lora_version = v.get("current_lora_version", 0)
    model, tokenizer = model_loader.load(v.get("checkpoint_path"))

    while True:
        chunk_id = _kv_queue.get()
        try:
            current_ver = ver.get_lora_version()
            # Reload if a new LoRA version has been written since worker started
            if current_ver != _cached_lora_version:
                lora_ckpt = ver.load().get("checkpoint_path")
                model, tokenizer = model_loader.reload(lora_ckpt)
                _cached_lora_version = current_ver

            results, _ = client.scroll(
                cfg["collection"],
                limit=1,
                with_payload=True,
            )
            # Filter for the specific chunk_id
            results = [r for r in results if r.id == chunk_id]
            if not results:
                continue
            text = results[0].payload.get("text", "")
            from pipeline.kv_indexer import compute_kv_for_chunk
            kv_arr = compute_kv_for_chunk(
                text, model, tokenizer, num_layers, num_kv_heads, head_dim
            )
            client.set_payload(
                cfg["collection"],
                chunk_id,
                {"kv_cache": kv_utils.serialize_kv(kv_arr), "kv_version": current_ver},
            )
        except Exception as e:
            print(f"[kv_background] KV recompute error for chunk {chunk_id}: {e}",
                  flush=True)
        finally:
            _kv_queue.task_done()


# ── Access flush worker ───────────────────────────────────────────────────

def _flush_access(cfg: dict, store) -> None:
    """Write accumulated access metrics from ``_access_buffer`` to the vector store.

    Atomically snapshots and clears the in-memory buffer, then for each chunk
    retrieves its current payload, merges in the delta counts, and calls
    ``store.set_payload``.  Errors for individual chunks are logged but do not
    abort the flush.

    Args:
        cfg: Datasource configuration dict.
        store: VectorStore instance to write updated payloads to.
    """
    global _query_count
    with _access_lock:
        if not _access_buffer:
            return
        snapshot = dict(_access_buffer)
        _access_buffer.clear()
    with _query_lock:
        _query_count = 0

    current_ts = int(time.time())
    for chunk_id, delta in snapshot.items():
        try:
            # Retrieve existing payload; use scroll with ID filter (Qdrant) or native_client
            if hasattr(store, "native_client"):
                existing = store.native_client.retrieve(
                    collection_name=cfg["collection"],
                    ids=[chunk_id],
                    with_payload=True,
                )
                payload = existing[0].payload if existing else {}
            else:
                results, _ = store.scroll(cfg["collection"], limit=100, with_payload=True)
                match = [r for r in results if r.id == chunk_id]
                if not match:
                    continue
                payload = match[0].payload

            old_count = payload.get("access_count", 0) or 0
            old_rank_sum = old_count * (payload.get("avg_retrieval_rank") or 0.0)
            new_count = old_count + delta["count"]
            new_rank_avg = (old_rank_sum + delta["rank_sum"]) / new_count

            updates = {
                "access_count": new_count,
                "last_accessed_ts": delta["last_ts"],
                "avg_retrieval_rank": round(new_rank_avg, 3),
            }
            if "parametric_hits" in delta:
                updates["parametric_hit_count"] = (
                    payload.get("parametric_hit_count", 0) + delta["parametric_hits"]
                )
            store.set_payload(cfg["collection"], chunk_id, updates)
        except Exception as e:
            print(f"[kv_background] Access flush error for {chunk_id}: {e}", flush=True)


def _access_worker(cfg: dict) -> None:
    """Flush on every 3 queries or every 30 sec — whichever comes first."""
    flush_interval = cfg.get("access_flush_seconds", 30)
    flush_queries = cfg.get("access_flush_queries", 3)
    store = get_store(cfg)
    last_flush = time.time()

    while True:
        time.sleep(5)
        with _query_lock:
            qc = _query_count
        elapsed = time.time() - last_flush
        if qc >= flush_queries or elapsed >= flush_interval:
            _flush_access(cfg, store)
            last_flush = time.time()


_started = False
_start_lock = threading.Lock()


def start(cfg: dict) -> None:
    """Start both background threads. Idempotent — safe to call multiple times."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    t1 = threading.Thread(target=_kv_worker, args=(cfg,), daemon=True)
    t2 = threading.Thread(target=_access_worker, args=(cfg,), daemon=True)
    t1.start()
    t2.start()
    print("✅ kv_background workers started", flush=True)


def promote_chunk_to_enhanced_tier(
    chunk_id: str,
    chunk_text: str,
    cfg: dict,
    model,
    tokenizer,
    vector_store,
    tq_config=None,
    existing_kv_token_path: str | None = None,
) -> str | None:
    """Run LLM forward pass, save per-token KVs, update kv_token_path in VectorStore.

    Returns the path written, or None if skipped (already enhanced).
    """
    import torch

    if existing_kv_token_path:
        return None

    inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True,
                       max_length=cfg.get("chunk_size", 512))
    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    arr = compute_per_token_kv(out.past_key_values)

    kv_dir = Path(cfg["per_token_kv_dir"])
    kv_dir.mkdir(parents=True, exist_ok=True)
    path = kv_dir / f"{chunk_id}.npz"
    save_token_kv(arr, path, tq_config=tq_config)

    vector_store.update_payload(
        collection=cfg["collection"],
        point_id=chunk_id,
        payload={"kv_token_path": str(path)},
    )
    return str(path)


if __name__ == "__main__":
    with open("my_config.json") as f:
        cfg = json.load(f)
    start(cfg)
    print("Background workers running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
