"""
kv_background.py — Background worker with two jobs:
  1. KV recompute queue: heal stale chunks after they are first retrieved
  2. Access tracker flush: batch-write access counters to Qdrant every 50 queries or 5 min

Run as a long-lived process alongside kv_inference.py:
  python3 kv_background.py &
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kv_utils
import model_loader
import version as ver
from qdrant_client import QdrantClient

_kv_queue: queue.Queue = queue.Queue()
_access_buffer: dict[int, dict] = {}
_access_lock = threading.Lock()
_query_count = 0
_query_lock = threading.Lock()


# ── Public API (called by kv_inference.py) ────────────────────────────────

def enqueue_kv_recompute(chunk_ids: list[int]) -> None:
    """Called from inference thread when stale chunks are detected."""
    for cid in chunk_ids:
        _kv_queue.put(cid)


def record_access(chunk_id: int, rank: int) -> None:
    """Called from inference thread — zero latency, in-memory only."""
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
    """Called when confidence gate answers without retrieval."""
    with _access_lock:
        for cid in chunk_ids:
            if cid not in _access_buffer:
                _access_buffer[cid] = {"count": 0, "rank_sum": 0.0, "last_ts": 0,
                                        "parametric_hits": 0}
            _access_buffer[cid].setdefault("parametric_hits", 0)
            _access_buffer[cid]["parametric_hits"] += 1


# ── KV recompute worker ───────────────────────────────────────────────────

def _kv_worker(cfg: dict) -> None:
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    num_layers = cfg["kv_num_layers"]
    num_kv_heads = cfg["kv_num_heads"]
    head_dim = cfg["kv_head_dim"]

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
                collection_name=cfg["collection"],
                ids=[chunk_id],
                with_payload=True,
                limit=1,
            )
            if not results:
                continue
            text = results[0].payload.get("text", "")
            from kv_indexer import compute_kv_for_chunk
            kv_arr = compute_kv_for_chunk(
                text, model, tokenizer, num_layers, num_kv_heads, head_dim
            )
            client.set_payload(
                collection_name=cfg["collection"],
                payload={"kv_cache": kv_utils.serialize_kv(kv_arr),
                         "kv_version": current_ver},
                points=[chunk_id],
            )
        except Exception as e:
            print(f"[kv_background] KV recompute error for chunk {chunk_id}: {e}",
                  flush=True)
        finally:
            _kv_queue.task_done()


# ── Access flush worker ───────────────────────────────────────────────────

def _flush_access(cfg: dict, client: QdrantClient) -> None:
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
            existing = client.retrieve(
                collection_name=cfg["collection"],
                ids=[chunk_id],
                with_payload=True,
            )
            if not existing:
                continue
            payload = existing[0].payload
            old_count = payload.get("access_count", 0)
            old_rank_sum = old_count * payload.get("avg_retrieval_rank", 0.0)
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
            client.set_payload(
                collection_name=cfg["collection"],
                payload=updates,
                points=[chunk_id],
            )
        except Exception as e:
            print(f"[kv_background] Access flush error for {chunk_id}: {e}", flush=True)


def _access_worker(cfg: dict) -> None:
    """Flush on every 50 queries or every 5 min — whichever comes first."""
    flush_interval = cfg.get("access_flush_seconds", 300)
    flush_queries = cfg.get("access_flush_queries", 50)
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    last_flush = time.time()

    while True:
        time.sleep(5)
        with _query_lock:
            qc = _query_count
        elapsed = time.time() - last_flush
        if qc >= flush_queries or elapsed >= flush_interval:
            _flush_access(cfg, client)
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
