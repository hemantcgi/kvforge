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
        "source_version": "",
    }


def cmd_index(pdf_path: Path, cfg: dict) -> None:
    """Run the full index pipeline for a single PDF file.

    Steps:

    1. Read and chunk the PDF using the ``bedrock_rag`` pipeline.
    2. Embed all chunks with the configured embedding model.
    3. Load the LLM and compute mean-pooled KV tensors for every chunk.
    4. Upsert the resulting ``Point`` objects to the vector store in batches.

    Args:
        pdf_path: Path to the PDF file to index.
        cfg: Datasource configuration dict.
    """
    store = get_store(cfg)
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    # 1. Chunk + embed (reuse bedrock_rag pipeline)
    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages, cfg["chunk_size"], cfg["chunk_overlap"])
    print(f"  {len(chunks)} chunks from {pdf_path.name}")

    embedder = TextEmbedding(model_name=cfg["embed_model"],
                              show_download_progress=False)
    vectors = embed_chunks(chunks, embedder, cfg["embed_batch"])

    # 2. Load LLM for KV computation
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    # 3. Compute KV + upsert
    print(f"Computing KV tensors for {len(chunks)} chunks ...")
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        kv_arr = compute_kv_for_chunk(
            chunk["text"], model, tokenizer, num_layers, num_kv_heads, head_dim
        )
        payload = build_payload(
            text=chunk["text"],
            page=chunk["page"],
            source_file=pdf_path.name,
            kv_array=kv_arr,
        )
        points.append(Point(id=chunk["chunk_id"], vector=vec, payload=payload))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)}", end="\r", flush=True)

    # batch upsert
    for start in range(0, len(points), cfg["upsert_batch"]):
        store.upsert(cfg["collection"], points[start:start + cfg["upsert_batch"]])
    print(f"\nIndexed {len(points)} chunks with KV (kv_version=null)")

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
    idx.add_argument("pdf_file")

    kv = sub.add_parser("compute-kv")
    kv.add_argument("--filter", choices=["kv_version=null"], default=None)
    kv.add_argument("--stale-version", type=int, default=None)
    kv.add_argument("--source-file", default=None)

    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    ver.init(cfg)
    model_loader.init(cfg)

    if args.cmd == "index":
        cmd_index(Path(args.pdf_file), cfg)
    elif args.cmd == "compute-kv":
        if args.stale_version is not None:
            cmd_compute_kv(cfg, "stale", args.stale_version)
        elif args.source_file:
            cmd_compute_kv(cfg, "source", args.source_file)
        else:
            cmd_compute_kv(cfg, "null", None)


if __name__ == "__main__":
    main()
