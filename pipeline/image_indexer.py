"""Image indexing pipeline: extract → CLIP embed → LLaVA KV + caption → upsert.

Commands
--------
``index-images <pdf>``
    Full pipeline: extract images from PDF → embed with CLIP → compute KV
    tensors and caption via LLaVA → upsert to image collection.
``compute-kv-images``
    Recompute KV tensors for stale image chunks (skips re-embedding).
    Options:
    * ``--filter kv_version=null`` — process only un-cached chunks.
    * ``--stale-version N``        — recompute all with kv_version < N.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.kv_utils as kv_utils
import core.version as ver
from core.multimodal_loader import LLaVALoader
from embeddings.clip_embedder import CLIPEmbedder
from ingestion.image_extractor import PDFImageExtractor
from vectorstore.base import Point
from vectorstore.registry import get_store


def image_chunk_id(source_file: str, page: int, idx: int) -> int:
    """Return a deterministic non-negative integer ID for an image chunk."""
    key = f"{source_file}:{page}:{idx}"
    digest = hashlib.sha256(key.encode()).digest()[:8]
    return int.from_bytes(digest, "big") % (2**62)


def build_image_payload(
    image_path: str,
    page: int,
    source_file: str,
    kv_array,
    caption: str,
    indexed_at: int | None = None,
) -> dict:
    return {
        "image_path": image_path,
        "caption": caption,
        "page": page,
        "source_file": source_file,
        "indexed_at": indexed_at or int(time.time()),
        "kv_cache": kv_utils.serialize_kv(kv_array),
        "kv_version": None,
        "access_count": 0,
        "last_accessed_ts": None,
        "tier": "frozen",
    }


def cmd_index_images(pdf_path: Path, cfg: dict) -> None:
    store = get_store(cfg)
    image_collection = cfg["collection"] + cfg.get("image_collection_suffix", "_images")

    extractor = PDFImageExtractor(cfg)
    clip = CLIPEmbedder(cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    mm_llm = LLaVALoader(cfg)

    store.create_collection(image_collection, dim=clip.dim)

    images = extractor.load(str(pdf_path))
    print(f"  {len(images)} images extracted from {pdf_path.name}")

    points = []
    for idx, img in enumerate(images):
        vector = clip.encode_image(img["image_path"])
        kv_arr = mm_llm.encode_image_kv(img["image_path"])
        caption = mm_llm.caption(img["image_path"])
        chunk_id = image_chunk_id(img["source_file"], img["page"], idx)
        payload = build_image_payload(
            image_path=img["image_path"],
            page=img["page"],
            source_file=img["source_file"],
            kv_array=kv_arr,
            caption=caption,
        )
        points.append(Point(id=chunk_id, vector=vector, payload=payload))
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(images)}", end="\r", flush=True)

    for start in range(0, len(points), cfg.get("upsert_batch", 128)):
        store.upsert(image_collection, points[start:start + cfg.get("upsert_batch", 128)])
    print(f"\nIndexed {len(points)} image chunks (kv_version=null)")


def cmd_compute_kv_images(cfg: dict, filter_type: str, filter_value) -> None:
    store = get_store(cfg)
    image_collection = cfg["collection"] + cfg.get("image_collection_suffix", "_images")
    mm_llm = LLaVALoader(cfg)
    current_ver = ver.get_lora_version()

    offset = None
    updated = 0
    while True:
        results, offset = store.scroll(
            image_collection, limit=50, with_payload=True, offset=offset,
        )
        if not results:
            break
        for point in results:
            if filter_type == "null" and point.payload.get("kv_version") is not None:
                continue
            if filter_type == "stale" and point.payload.get("kv_version") is not None:
                try:
                    if int(point.payload["kv_version"]) >= int(filter_value):
                        continue
                except (TypeError, ValueError):
                    pass
            kv_arr = mm_llm.encode_image_kv(point.payload["image_path"])
            store.set_payload(
                image_collection,
                point.id,
                {"kv_cache": kv_utils.serialize_kv(kv_arr), "kv_version": current_ver},
            )
            updated += 1
        if offset is None:
            break

    print(f"Recomputed KV for {updated} image chunks -> kv_version={current_ver}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="my_config.json")
    sub = p.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index-images")
    idx.add_argument("pdf_file")

    kv = sub.add_parser("compute-kv-images")
    kv.add_argument("--filter", choices=["kv_version=null"], default=None)
    kv.add_argument("--stale-version", type=int, default=None)

    args = p.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    ver.init(cfg)

    if args.cmd == "index-images":
        cmd_index_images(Path(args.pdf_file), cfg)
    elif args.cmd == "compute-kv-images":
        if args.stale_version is not None:
            cmd_compute_kv_images(cfg, "stale", args.stale_version)
        else:
            cmd_compute_kv_images(cfg, "null", None)


if __name__ == "__main__":
    main()
