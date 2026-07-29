"""Parallel text + image retrieval merged by score, and multimodal answer generation.

multimodal_search   — parallel search across text and image collections, merged.
multimodal_answer   — full RAG: search → decide inference mode → generate answer.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.kv_utils as kv_utils
import core.version as ver
import core.model_loader as model_loader
from embeddings.clip_embedder import CLIPEmbedder
from embeddings.registry import get_embedder
from pipeline.image_inference import (
    decide_image_inference_mode,
    get_image_context,
    get_stale_image_chunk_ids,
)
from pipeline.kv_inference import (
    decide_inference_mode,
    get_stale_chunk_ids,
    generate_with_kv,
    generate_text_in_context,
)
from vectorstore.registry import get_store


def _text_hit_to_dict(hit) -> dict:
    return {
        "chunk_id": hit.id,
        "text": hit.payload.get("text", ""),
        "page": hit.payload.get("page"),
        "score": round(hit.score, 4),
        "kv_cache": hit.payload.get("kv_cache"),
        "kv_version": hit.payload.get("kv_version"),
        "kds": hit.payload.get("kds"),
    }


def _image_hit_to_dict(hit) -> dict:
    return {
        "chunk_id": hit.id,
        "caption": hit.payload.get("caption", ""),
        "image_path": hit.payload.get("image_path", ""),
        "page": hit.payload.get("page"),
        "score": round(hit.score, 4),
        "kv_cache": hit.payload.get("kv_cache"),
        "kv_version": hit.payload.get("kv_version"),
    }


def multimodal_search(query: str, cfg: dict) -> list[dict]:
    """Parallel search across text and image collections, merged by score.

    Returns at most ``cfg['top_k']`` chunks sorted by score descending.
    Each chunk dict has a ``'modality'`` key: ``'text'`` or ``'image'``.
    Stale image chunks are enqueued for background KV recomputation.
    """
    store = get_store(cfg)
    text_embedder = get_embedder(cfg)
    clip = CLIPEmbedder(cfg.get("clip_model", "openai/clip-vit-base-patch32"))
    image_collection = cfg["collection"] + cfg.get("image_collection_suffix", "_images")
    top_k = cfg.get("top_k", 5)

    text_hits = store.query(
        cfg["collection"], text_embedder.encode([query])[0], top_k=top_k
    )
    image_hits = store.query(
        image_collection, clip.encode_text(query), top_k=top_k
    )

    text_chunks  = [{"modality": "text",  **_text_hit_to_dict(h)}  for h in text_hits]
    image_chunks = [{"modality": "image", **_image_hit_to_dict(h)} for h in image_hits]

    merged = sorted(text_chunks + image_chunks, key=lambda c: c["score"], reverse=True)

    # Enqueue stale image chunks for background KV recomputation
    import pipeline.kv_background as kv_background
    current_ver = ver.get_lora_version()
    stale_image_ids = get_stale_image_chunk_ids(
        [c for c in merged if c["modality"] == "image"], current_ver
    )
    if stale_image_ids:
        kv_background.enqueue_image_kv_recompute(stale_image_ids)

    return merged[:top_k]


def multimodal_answer(query: str, cfg: dict) -> str:
    """Combined text + image retrieval and generation.

    Default path (image_kv_inference=False):
      - Text chunks: KV injection if all fresh, else text-in-context.
      - Image chunks: captions appended as extra_context to whichever text path runs.

    Path A (image_kv_inference=True, all image chunks fresh):
      - Image KV tensors injected into multimodal LLM.
      - Text chunks passed as text-in-context to the multimodal LLM.
    """
    chunks = multimodal_search(query, cfg)
    text_ch  = [c for c in chunks if c["modality"] == "text"]
    image_ch = [c for c in chunks if c["modality"] == "image"]

    current_ver = ver.get_lora_version()
    lora_ckpt = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    image_context = get_image_context(image_ch) if image_ch else ""
    image_mode = decide_image_inference_mode(
        image_ch, current_ver, cfg.get("image_kv_inference", False)
    )

    if image_mode == "image_kv_injection":
        # Path A: image KV injected via multimodal LLM; text is in-context.
        from core.multimodal_loader import LLaVALoader
        mm_llm = LLaVALoader(cfg)
        num_layers, num_kv_heads, head_dim = mm_llm.kv_shape
        full_shape = (num_layers, 2, num_kv_heads, head_dim)
        image_kv_arrays = [
            kv_utils.deserialize_kv(c["kv_cache"], shape=full_shape)
            for c in image_ch if c.get("kv_cache")
        ]
        past_kv = kv_utils.stack_past_key_values(
            image_kv_arrays, num_layers=num_layers,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
        )
        text_context = "\n\n".join(
            f"[page {c['page']}, score {c['score']:.3f}]\n{c['text']}"
            for c in text_ch
        )
        prompt = (
            f"Using the context and images below, answer: {query}\n\n"
            f"Context:\n{text_context}"
        )
        inputs = mm_llm._processor(prompt, return_tensors="pt").to(mm_llm._model.device)
        with torch.no_grad():
            output = mm_llm._model.generate(
                **inputs, past_key_values=past_kv,
                max_new_tokens=cfg.get("max_new_tokens", 256), do_sample=False,
            )
        return mm_llm._processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # Default path: text KV or text-in-context; image captions as extra_context.
    text_mode = decide_inference_mode(text_ch, current_ver, kds_threshold=cfg.get("kds_threshold"))

    stale = get_stale_chunk_ids(text_ch, current_ver)
    if stale:
        import pipeline.kv_background as kv_background
        kv_background.enqueue_kv_recompute(stale)

    if text_mode == "kv_injection":
        return generate_with_kv(query, text_ch, model, tokenizer, cfg,
                                  extra_context=image_context)
    return generate_text_in_context(query, text_ch, model, tokenizer,
                                     extra_context=image_context)
