"""Extended indexing pipeline: chunk, embed, and compute KV tensors per document.

Extends the basic indexing pipeline with LLM KV-cache computation so that each
stored chunk carries a pre-computed ``kv_cache`` payload field.

Commands
--------
``index <pdf>``
    Full pipeline: load PDF → chunk → embed → compute KV tensors → upsert.
``compute-kv``
    Recompute KV tensors for filtered chunks (skips re-embedding).
    Useful after a LoRA training round to bring stale chunks up to date.

    Options:

    * ``--filter kv_version=null`` — process only un-cached chunks.
    * ``--stale-version N`` — heal all chunks with ``kv_version < N``.
    * ``--source-file FILE`` — restrict to a specific source document.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import core.model_loader as model_loader
import core.kv_utils as kv_utils
import core.version as ver
from pipeline.bedrock_rag import chunk_pages, read_pdf, embed_chunks
from fastembed import TextEmbedding
from vectorstore.base import Point
from vectorstore.registry import get_store


def compute_kv_for_chunk(
    text: str,
    model,
    tokenizer,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """Run a single text chunk through the LLM and return its mean-pooled KV array.

    Tokenises *text* (truncated to 512 tokens), runs a forward pass with
    ``use_cache=True``, then calls ``kv_utils.mean_pool_kv`` to compress the
    per-token KV tensors into a fixed-size float16 array.

    Args:
        text: Plain-text content of the chunk.
        model: Loaded HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        num_layers: Expected number of transformer layers (used for shape assertion).
        num_kv_heads: Expected number of KV attention heads.
        head_dim: Expected head dimensionality.

    Returns:
        Float16 numpy array of shape ``[num_layers, 2, num_kv_heads, head_dim]``.

    Raises:
        AssertionError: If the produced KV shape does not match the expected
            dimensions.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    arr = kv_utils.mean_pool_kv(outputs.past_key_values)
    expected = (num_layers, 2, num_kv_heads, head_dim)
    assert arr.shape == expected, (
        f"KV shape mismatch: expected {expected}, got {arr.shape}. "
        "Check kv_num_layers/kv_num_heads/kv_head_dim in config."
    )
    return arr


def build_payload(
    text: str,
    page: int,
    source_file: str,
    kv_array: np.ndarray,
    indexed_at: int | None = None,
    source_version: str = "",
) -> dict:
    """Construct the full vector store payload dict for a newly indexed chunk.

    Initialises all tracking fields (access counts, tier, etc.) to their
    default zero-state values.

    Args:
        text: Chunk text content.
        page: Source page number (1-indexed).
        source_file: Filename (not full path) of the source document.
        kv_array: Pre-computed KV array from ``compute_kv_for_chunk``.
        indexed_at: Unix timestamp to record as the indexing time.  Defaults
            to the current time if ``None``.
        source_version: ISO 8601 timestamp string from chunk metadata's ``modified``
            field, indicating when the source document was last modified.  Defaults
            to empty string if not provided.

    Returns:
        Dict suitable for use as a ``Point.payload`` argument.
    """
    return {
        "text": text,
        "page": page,
        "source_file": source_file,
        "indexed_at": indexed_at or int(time.time()),
        "kv_cache": kv_utils.serialize_kv(kv_array),
        "kv_version": None,
        "access_count": 0,
        "last_accessed_ts": None,
        "avg_retrieval_rank": None,
        "parametric_hit_count": 0,
        "tier": "frozen",
        "effective_from": datetime.now(timezone.utc).isoformat(),
        "superseded_at": None,
        "source_version": source_version,
    }


def cmd_index(cfg: dict) -> None:
    """Run the full index pipeline for a source document or corpus.

    Dispatches through the ingestion registry based on ``cfg["loader"]``.
    For pdf loaders, ``cfg["_source_path"]`` must be set.  For jsonl/hf
    loaders, the source path is optional and defaults to
    ``<version_file_dir>/data/corpus.jsonl``.

    Steps:

    1. Load and chunk the source using the configured loader.
    2. Embed all chunks with the configured embedding model.
    3. Load the LLM and compute mean-pooled KV tensors for every chunk.
    4. Upsert the resulting ``Point`` objects to the vector store in batches.

    Args:
        cfg: Datasource configuration dict (already flattened).
    """
    from ingestion.registry import get_loader

    store = get_store(cfg)
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    # Resolve source path
    source_path = cfg.get("_source_path", "")
    if not source_path and cfg.get("loader", "pdf") != "pdf":
        ver_dir = Path(cfg.get("version_file", "version.json")).parent
        candidate = ver_dir / "data" / "corpus.jsonl"
        source_path = str(candidate) if candidate.exists() else ""

    # 1. Chunk + embed via ingestion registry
    loader = get_loader(cfg)
    chunks = loader.load(source_path)
    source_label = Path(source_path).name if source_path else cfg.get("dataset_id", "unknown")
    print(f"  {len(chunks)} chunks from {source_label}")

    embedder = TextEmbedding(model_name=cfg["embed_model"],
                              show_download_progress=False)
    vectors = embed_chunks(chunks, embedder, cfg["embed_batch"])

    # 2. Load LLM for KV computation (optional — skipped gracefully on CPU/no-GPU)
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = None, None
    import traceback as _tb
    if model_loader.DEVICE == "cpu":
        print(
            "⚠️  No CUDA GPU detected — skipping KV tensor computation.\n"
            "   Phase 1 (text RAG) will work normally.\n"
            "   Run 'Recompute KV' on a GPU machine later to enable Phase 2."
        )
    else:
        try:
            model, tokenizer = model_loader.load(lora_ckpt)
        except Exception as exc:
            print(
                f"⚠️  LLM load failed ({exc.__class__.__name__}: {exc})\n"
                + _tb.format_exc() +
                "   KV tensors will be skipped — Phase 1 (text RAG) will still work.\n"
                "   Run 'recompute' on a GPU machine to enable Phase 2 (KV injection)."
            )

    # 3. Compute KV (if model available) + upsert
    import uuid as _uuid
    kv_label = "with KV" if model is not None else "without KV (Phase 1 only)"
    print(f"Upserting {len(chunks)} chunks {kv_label} ...")
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        kv_arr = None
        if model is not None:
            kv_arr = compute_kv_for_chunk(
                chunk["text"], model, tokenizer, num_layers, num_kv_heads, head_dim
            )
        meta = chunk.get("metadata", {})
        payload = build_payload(
            text=chunk["text"],
            page=meta.get("page", chunk.get("page", 0)),
            source_file=source_label,
            kv_array=kv_arr,
            source_version=meta.get("modified", ""),
        )
        point_id = chunk.get("chunk_id") or str(_uuid.uuid4())
        points.append(Point(id=point_id, vector=vec, payload=payload))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)}", end="\r", flush=True)

    # Ensure collection exists before upserting
    if not store.collection_exists(cfg["collection"]):
        store.create_collection(cfg["collection"], cfg["vector_dim"])
        print(f"Created collection '{cfg['collection']}' (dim={cfg['vector_dim']})")

    # batch upsert
    for start in range(0, len(points), cfg["upsert_batch"]):
        store.upsert(cfg["collection"], points[start:start + cfg["upsert_batch"]])
    print(f"\nIndexed {len(points)} chunks {kv_label}")

    # Cluster embeddings and tag each chunk with its cluster_id
    try:
        from core.cluster_manager import cluster_embeddings, save_clusters
        from pathlib import Path as _Path
        vec_array = np.array(vectors)
        k_range = tuple(cfg.get("cluster_k_range", [3, 20]))
        centroids, labels = cluster_embeddings(vec_array, k_range=k_range)
        cluster_file = str(_Path(cfg["checkpoint_dir"]) / "clusters.json")
        save_clusters(cluster_file, centroids, labels, lora_version=ver.get_lora_version())
        for point, label in zip(points, labels):
            store.set_payload(cfg["collection"], point.id, {"cluster_id": str(int(label))})
        print(f"Clustered {len(points)} chunks into {len(centroids)} clusters")
    except Exception as exc:
        print(f"  (clustering skipped: {exc})")

    # Write version.json to signal Phase 1 complete (has_index=True in Studio)
    state = ver.load()
    ver.save(state)


def cmd_compute_kv(cfg: dict, filter_type: str, filter_value) -> None:
    """Recompute KV tensors for chunks that match the specified filter.

    Scrolls through the collection with the appropriate Qdrant filter and
    calls ``compute_kv_for_chunk`` for each matching point, then updates its
    ``kv_cache`` and ``kv_version`` payload fields in-place.

    Args:
        cfg: Datasource configuration dict.
        filter_type: One of:

            * ``'null'`` — select chunks where ``kv_version`` is null.
            * ``'stale'`` — select chunks where ``kv_version < filter_value``.
            * ``'source'`` — select chunks with ``source_file == filter_value``.
        filter_value: Numeric version threshold (for ``'stale'``) or source
            filename string (for ``'source'``); ignored for ``'null'``.
    """
    store = get_store(cfg)
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    lora_ckpt = ver.load().get("checkpoint_path")
    current_ver = ver.get_lora_version()
    model, tokenizer = model_loader.load(lora_ckpt)

    # Build scroll_filter for Qdrant (passed through store.scroll as scroll_filter kwarg)
    scroll_filter = None
    if cfg.get("vector_store", "qdrant") == "qdrant":
        from qdrant_client.models import Filter, FieldCondition, IsNullCondition, Range
        if filter_type == "null":
            scroll_filter = Filter(must=[IsNullCondition(is_null={"key": "kv_version"})])
        elif filter_type == "stale":
            # Qdrant drops null payload fields so IsNullCondition misses absent
            # kv_version; do a full scan and rely on the client-side filter.
            scroll_filter = None
        else:
            scroll_filter = Filter(must=[
                FieldCondition(key="source_file", match={"value": filter_value})
            ])

    offset = None
    updated = 0
    while True:
        results, offset = store.scroll(
            cfg["collection"],
            limit=50,
            with_payload=True,
            offset=offset,
            scroll_filter=scroll_filter,
        )
        if not results:
            break
        for point in results:
            # For non-Qdrant stores scroll_filter is None, so all chunks are
            # returned. Apply the equivalent filter client-side.
            if filter_type == "null" and point.payload.get("kv_version") is not None:
                continue
            if filter_type == "stale" and point.payload.get("kv_version") is not None:
                try:
                    if int(point.payload["kv_version"]) >= int(filter_value):
                        continue
                except (TypeError, ValueError):
                    pass
            kv_arr = compute_kv_for_chunk(
                point.payload["text"], model, tokenizer,
                num_layers, num_kv_heads, head_dim
            )
            store.set_payload(
                cfg["collection"],
                point.id,
                {"kv_cache": kv_utils.serialize_kv(kv_arr), "kv_version": current_ver},
            )
            updated += 1
        if offset is None:
            break

    # Flush any remaining buffered writes (FAISSStore batches saves).
    if hasattr(store, "flush"):
        store.flush()
    print(f"Recomputed KV for {updated} chunks -> kv_version={current_ver}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index")
    idx.add_argument("pdf_file", nargs="?", default=None,
                     help="Path to PDF (pdf loader only). Omit for jsonl/hf loaders.")

    kv = sub.add_parser("compute-kv")
    kv.add_argument("--filter", choices=["kv_version=null"], default=None)
    kv.add_argument("--stale-version", type=int, default=None)
    kv.add_argument("--source-file", default=None)

    args = p.parse_args()
    with open(args.config) as f:
        raw = json.load(f)
    # Flatten nested addon_config format used by Studio
    if "addon_config" in raw:
        from core.config import KVForgeConfig
        dc = KVForgeConfig(**raw)
        cfg = dc.get_merged_config("indexing", "inference", "training")
        cfg.setdefault("version_file", raw.get("version_file", "version.json"))
        cfg.setdefault("collection", raw.get("collection", ""))
    else:
        cfg = raw
    ver.init(cfg)
    model_loader.init(cfg)

    if args.cmd == "index":
        if args.pdf_file:
            cfg["_source_path"] = args.pdf_file
        cmd_index(cfg)
    elif args.cmd == "compute-kv":
        if args.stale_version is not None:
            cmd_compute_kv(cfg, "stale", args.stale_version)
        elif args.source_file:
            cmd_compute_kv(cfg, "source", args.source_file)
        else:
            cmd_compute_kv(cfg, "null", None)


if __name__ == "__main__":
    main()
