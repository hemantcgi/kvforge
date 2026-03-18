"""
kv_indexer.py — Extended indexer: chunk + embed + compute KV tensors.

Commands:
  index      <pdf>       — full index (embed + KV compute + upsert)
  compute-kv             — recompute KV for filtered chunks (no re-embed)
    --filter kv_version=null
    --stale-version N    — heal all chunks with kv_version < N
    --source-file FILE
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

sys.path.insert(0, str(Path(__file__).parent))
import model_loader
import kv_utils
import version as ver
from bedrock_rag import chunk_pages, read_pdf, embed_chunks
from fastembed import TextEmbedding


def compute_kv_for_chunk(
    text: str,
    model,
    tokenizer,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    """
    Run a single chunk through the LLM forward pass and return mean-pooled KV.
    Output shape: [num_layers, 2, num_kv_heads, head_dim] float16
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
    """Construct the full Qdrant payload for a new chunk."""
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
    }


def cmd_index(pdf_path: Path, cfg: dict) -> None:
    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
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
        points.append(PointStruct(id=chunk["chunk_id"], vector=vec, payload=payload))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(chunks)}", end="\r", flush=True)

    # batch upsert
    for start in range(0, len(points), cfg["upsert_batch"]):
        client.upsert(
            collection_name=cfg["collection"],
            points=points[start:start + cfg["upsert_batch"]],
        )
    print(f"\nIndexed {len(points)} chunks with KV (kv_version=null)")


def cmd_compute_kv(cfg: dict, filter_type: str, filter_value) -> None:
    """Recompute KV for chunks matching the given filter."""
    from qdrant_client.models import Filter, FieldCondition, IsNullCondition, Range

    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)

    lora_ckpt = ver.load().get("checkpoint_path")
    current_ver = ver.get_lora_version()
    model, tokenizer = model_loader.load(lora_ckpt)

    # Scroll through matching chunks
    if filter_type == "null":
        scroll_filter = Filter(must=[IsNullCondition(is_null={"key": "kv_version"})])
    elif filter_type == "stale":
        scroll_filter = Filter(must=[
            FieldCondition(key="kv_version",
                           range=Range(lt=int(filter_value)))
        ])
    else:
        scroll_filter = Filter(must=[
            FieldCondition(key="source_file", match={"value": filter_value})
        ])

    offset = None
    updated = 0
    while True:
        results, offset = client.scroll(
            collection_name=cfg["collection"],
            scroll_filter=scroll_filter,
            limit=50,
            with_payload=True,
            offset=offset,
        )
        if not results:
            break
        for point in results:
            kv_arr = compute_kv_for_chunk(
                point.payload["text"], model, tokenizer,
                num_layers, num_kv_heads, head_dim
            )
            client.set_payload(
                collection_name=cfg["collection"],
                payload={"kv_cache": kv_utils.serialize_kv(kv_arr),
                         "kv_version": current_ver},
                points=[point.id],
            )
            updated += 1
        if offset is None:
            break

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
