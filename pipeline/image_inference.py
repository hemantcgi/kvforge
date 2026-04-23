"""Decision logic and context formatting for image chunks at query time.

Two paths:
  caption_fallback     — image captions used as text context; default.
  image_kv_injection   — image KV tensors injected into multimodal LLM;
                         only when image_kv_inference=True in config and
                         all image chunks have fresh KV tensors.
"""


def decide_image_inference_mode(
    image_chunks: list[dict],
    current_lora_version: int,
    image_kv_inference: bool,
) -> str:
    """Return 'image_kv_injection' or 'caption_fallback'."""
    if not image_kv_inference:
        return "caption_fallback"
    for chunk in image_chunks:
        kv_ver = chunk.get("kv_version")
        if kv_ver is None or kv_ver < current_lora_version:
            return "caption_fallback"
        if chunk.get("kv_cache") is None:
            return "caption_fallback"
    return "image_kv_injection"


def get_image_context(image_chunks: list[dict]) -> str:
    """Format image captions as text context for the caption fallback path."""
    return "\n\n".join(
        f"[Image, page {c['page']}, score {c['score']:.3f}]\n{c['caption']}"
        for c in image_chunks
    )


def get_stale_image_chunk_ids(
    image_chunks: list[dict], current_lora_version: int
) -> list[int]:
    """Return chunk IDs whose KV tensors are missing or behind the current LoRA version."""
    return [
        c["chunk_id"]
        for c in image_chunks
        if c.get("kv_version") is None or c["kv_version"] < current_lora_version
    ]
