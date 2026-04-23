# KVForge Multimodal: Image Vectors in KV Cache

**Date:** 2026-04-23
**Status:** Approved for implementation

---

## Problem

KVForge currently indexes and retrieves only text chunks. Documents that contain images — charts, diagrams, figures, schematics — are indexed as text only; the visual content is either lost (if the PDF has no alt-text) or captured only as surrounding prose. KV tensors are computed from text token sequences only. There is no mechanism to embed images into the CLIP vector space, compute KV tensors from visual tokens, or retrieve image chunks alongside text chunks at query time.

---

## Solution

Two parallel pipelines at index time, merged at query time:

1. **Text pipeline** — unchanged. `kv_indexer.py`, `kv_inference.py`, all vector stores, all existing tests untouched.
2. **Image pipeline** — new files only. Extract images from source documents → embed with CLIP → compute KV tensors via a multimodal LLM (LLaVA default) → store in a separate image collection.

At query time, both collections are searched in parallel. Results are merged by cosine score. Image chunks use caption-in-context fallback by default (multimodal LLM not needed at query time); setting `image_kv_inference: true` in config enables full KV injection via the multimodal LLM (Path to A).

---

## Architecture

### Two Collections Per Datasource

| Collection | Embedder | Dim | Stores |
|---|---|---|---|
| `<collection>` | text embedder (FastEmbed / ST / OpenAI) | `vector_dim` | text chunks + text KV tensors |
| `<collection>_images` | CLIP image encoder | 512 | image chunks + image KV tensors + captions |

The suffix is configurable (`image_collection_suffix`, default `"_images"`).

---

### `MultimodalLLM` Protocol (`core/multimodal_loader.py`)

Protocol-agnostic, following the existing `Embedder` / `VectorStore` pattern:

```python
@runtime_checkable
class MultimodalLLM(Protocol):
    def encode_image_kv(self, image_path: str) -> np.ndarray: ...
    # Returns float16 array [num_layers, 2, num_kv_heads, head_dim]
    # — same shape as text KV tensors produced by kv_utils.mean_pool_kv

    def caption(self, image_path: str) -> str: ...
    # Returns a text description of the image for caption-fallback path

    @property
    def kv_shape(self) -> tuple[int, int, int]: ...
    # (num_layers, num_kv_heads, head_dim)
```

`LLaVALoader` is the default implementation. It:
- Loads `llava-hf/llava-1.5-7b-hf` (or `cfg["multimodal_model"]`) via `AutoModelForCausalLM` + `AutoProcessor`
- `encode_image_kv`: loads image with PIL → prefill forward pass with `use_cache=True` → mean-pools KV tensors from visual token positions via `kv_utils.mean_pool_kv` → returns `[num_layers, 2, num_kv_heads, head_dim]` float16 array
- `caption`: generates text with prompt `"Describe this image concisely."` → returns decoded string
- Module-level `_instance` singleton (same pattern as `core/model_loader.py`) — loads once per process

---

### `CLIPEmbedder` (`embeddings/clip_embedder.py`)

```python
class CLIPEmbedder:
    def encode_image(self, image_path: str) -> list[float]: ...
    # CLIP image encoder → 512-dim vector

    def encode_text(self, text: str) -> list[float]: ...
    # CLIP text encoder → 512-dim vector (for querying the image collection)

    @property
    def dim(self) -> int: ...
    # 512 for openai/clip-vit-base-patch32
```

Uses `transformers.CLIPModel` + `CLIPProcessor`. The embeddings registry gains a `"clip"` entry.

---

### `ImageLoader` Protocol + `PDFImageExtractor` (`ingestion/image_extractor.py`)

```python
@runtime_checkable
class ImageLoader(Protocol):
    def load(self, source: str) -> list[dict]: ...
    # Returns: [{"image_path": str, "page": int, "source_file": str}, ...]
```

`PDFImageExtractor.load(source)`:
1. Opens PDF with `pdfplumber`
2. Iterates pages; calls `page.images` for each raster image object
3. Crops with `PIL` using the image's bounding box
4. Saves each image as PNG to `<image_store_dir>/<collection>/<stem>_p<page>_<idx>.png`
5. Returns one dict per image: `{image_path, page, source_file}`

Minimum image size filter: images smaller than 32×32 pixels are skipped (typically decorative rules or icons).

---

### `pipeline/image_indexer.py`

Mirrors `kv_indexer.py` structure. Two commands: `index-images` and `compute-kv-images`.

**`cmd_index_images(source, cfg)`:**
1. `PDFImageExtractor(cfg).load(source)` → list of image dicts
2. `CLIPEmbedder(cfg["clip_model"]).encode_image(path)` → 512-dim vector per image
3. `LLaVALoader(cfg).encode_image_kv(path)` → `[layers, 2, heads, head_dim]` KV array
4. `LLaVALoader(cfg).caption(path)` → text caption
5. Build payload: `{image_path, caption, page, source_file, kv_cache, kv_version=None, access_count=0, tier="frozen"}`
6. Deterministic integer ID: `abs(hash(f"{source_file}:{page}:{idx}")) % (2**62)`
7. Upsert to `<collection>_images`

**`cmd_compute_kv_images(cfg, filter_type, filter_value)`:**
Scroll image collection → re-run `encode_image_kv` for stale/null entries → `store.set_payload(...)` with new `kv_cache` and `kv_version`. Identical scroll-filter pattern to `kv_indexer.cmd_compute_kv`.

---

### `pipeline/image_inference.py`

```python
def decide_image_inference_mode(
    image_chunks: list[dict],
    current_lora_version: int,
    image_kv_inference: bool,
) -> str:
    """Return 'image_kv_injection' or 'caption_fallback'."""
    if not image_kv_inference:
        return "caption_fallback"
    for chunk in image_chunks:
        if chunk.get("kv_version") is None or chunk["kv_version"] < current_lora_version:
            return "caption_fallback"
        if chunk.get("kv_cache") is None:
            return "caption_fallback"
    return "image_kv_injection"

def get_image_context(image_chunks: list[dict]) -> str:
    """Format image captions as text context for caption fallback path."""
    return "\n\n".join(
        f"[Image, page {c['page']}, score {c['score']:.3f}]\n{c['caption']}"
        for c in image_chunks
    )
```

**Caption fallback (default, `image_kv_inference: false`):** image captions formatted as text context, appended to text chunks, passed to text LLM. No multimodal LLM needed at query time.

**Image KV injection (Path A, `image_kv_inference: true`):** image KV tensors deserialized and injected into the multimodal LLM (`LLaVALoader`) as `past_key_values` prefix. Text chunks are passed as text-in-context (not text KV injection) to the multimodal LLM. Requires the multimodal LLM loaded at query time. Full simultaneous text KV + image KV injection requires the text LLM and the multimodal LLM backbone to share the same architecture (e.g., both `meta-llama/Llama-2-7b-hf`) — this is a future upgrade path, not required for the initial Path A implementation.

---

### `pipeline/multimodal_query.py`

Thin orchestration layer. No LLM logic.

```python
def multimodal_search(query: str, cfg: dict) -> list[dict]:
    """Parallel search across text and image collections, merged by score."""
    store = get_store(cfg)
    text_embedder = get_embedder(cfg)
    clip = CLIPEmbedder(cfg["clip_model"])
    image_collection = cfg["collection"] + cfg["image_collection_suffix"]

    text_hits  = store.query(cfg["collection"],  text_embedder.encode([query])[0], top_k=cfg["top_k"])
    image_hits = store.query(image_collection, clip.encode_text(query),           top_k=cfg["top_k"])

    text_chunks  = [{"modality": "text",  **_text_hit_to_dict(h)}  for h in text_hits]
    image_chunks = [{"modality": "image", **_image_hit_to_dict(h)} for h in image_hits]

    merged = sorted(text_chunks + image_chunks, key=lambda c: c["score"], reverse=True)
    return merged[:cfg["top_k"]]

def multimodal_answer(query: str, cfg: dict) -> str:
    """Combined text + image retrieval and generation."""
    chunks   = multimodal_search(query, cfg)
    text_ch  = [c for c in chunks if c["modality"] == "text"]
    image_ch = [c for c in chunks if c["modality"] == "image"]

    current_ver = ver.get_lora_version()
    lora_ckpt   = ver.load().get("checkpoint_path")
    model, tokenizer = model_loader.load(lora_ckpt)

    text_mode  = decide_inference_mode(text_ch, current_ver)       # from kv_inference.py
    image_mode = decide_image_inference_mode(image_ch, current_ver, cfg.get("image_kv_inference", False))

    if text_mode == "kv_injection" and image_mode == "caption_fallback":
        # Inject text KV; append image captions as text context prefix
        image_context = get_image_context(image_ch)
        return generate_with_kv(query, text_ch, model, tokenizer, cfg,
                                 extra_context=image_context)

    if text_mode == "kv_injection" and image_mode == "image_kv_injection":
        # Path A: inject image KV via multimodal LLM; text chunks are text-in-context.
        # Text KV injection + image KV injection simultaneously requires the text LLM
        # and multimodal LLM backbone to share the same architecture (e.g., both Llama-2-7B).
        # In this initial Path A implementation, text chunks go in-context and image KV
        # is injected as past_key_values prefix into the multimodal LLM.
        mm_llm = LLaVALoader(cfg)
        text_context = "\n\n".join(
            f"[page {c['page']}, score {c['score']:.3f}]\n{c['text']}" for c in text_ch
        )
        num_layers, num_kv_heads, head_dim = mm_llm.kv_shape
        full_shape = (num_layers, 2, num_kv_heads, head_dim)
        image_kv_arrays = [
            kv_utils.deserialize_kv(c["kv_cache"], shape=full_shape)
            for c in image_ch if c.get("kv_cache")
        ]
        past_kv = kv_utils.stack_past_key_values(
            image_kv_arrays, num_layers=num_layers,
            num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        prompt = (
            f"Using the context and images below, answer: {query}\n\n"
            f"Context:\n{text_context}"
        )
        inputs = mm_llm.tokenizer(prompt, return_tensors="pt").to(mm_llm.model.device)
        with torch.no_grad():
            output = mm_llm.model.generate(
                **inputs, past_key_values=past_kv,
                max_new_tokens=cfg.get("max_new_tokens", 256), do_sample=False,
            )
        return mm_llm.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)

    # Text fallback: combine text chunks and image captions into prompt
    image_context = get_image_context(image_ch)
    return generate_text_in_context(query, text_ch, model, tokenizer,
                                     extra_context=image_context)
```

`generate_with_kv` in `kv_inference.py` gains an optional `extra_context: str = ""` parameter — prepended to the prompt before KV injection. This is the only change to the existing text inference path.

---

## Config Fields

New fields in `DatasourceConfig` (`core/config.py`):

```python
# Multimodal / image support
image_collection_suffix: str = "_images"
image_store_dir: str = ""                        # directory for extracted PNGs; required if indexing images
multimodal_model: str = "llava-hf/llava-1.5-7b-hf"
clip_model: str = "openai/clip-vit-base-patch32"
image_kv_inference: bool = False                 # Path A: load multimodal LLM at query time
```

`image_store_dir` defaults to `""`. `image_indexer.py` raises `ValueError` if it is empty when `cmd_index_images` is called.

---

## Data Flow Summary

### Index time (images)
```
PDF → PDFImageExtractor → [{image_path, page, source_file}]
    → CLIPEmbedder.encode_image()         → 512-dim vector
    → LLaVALoader.encode_image_kv()       → [layers, 2, heads, head_dim] float16
    → LLaVALoader.caption()               → text caption
    → store.upsert(<collection>_images, Point(vector, payload))
```

### Query time
```
text query
    → text_embedder.encode()  → store.query(<collection>)   → text_hits
    → clip.encode_text()      → store.query(<collection>_images) → image_hits
    → merge by score → top-K mixed chunks
    → decide_inference_mode(text_chunks)   → kv_injection | text_fallback
    → decide_image_inference_mode(image_chunks, image_kv_inference)
                                           → caption_fallback | image_kv_injection
    → generate answer
```

---

## New Files

| File | Purpose |
|---|---|
| `embeddings/clip_embedder.py` | CLIP image encoder + text encoder |
| `ingestion/image_extractor.py` | `ImageLoader` protocol + `PDFImageExtractor` |
| `core/multimodal_loader.py` | `MultimodalLLM` protocol + `LLaVALoader` implementation |
| `pipeline/image_indexer.py` | Image chunk indexing + KV recomputation |
| `pipeline/image_inference.py` | `decide_image_inference_mode`, `get_image_context` |
| `pipeline/multimodal_query.py` | Parallel search + merge + `multimodal_answer` |

## Modified Files

| File | Change |
|---|---|
| `core/config.py` | 5 new image config fields |
| `embeddings/registry.py` | Add `"clip"` backend |
| `pipeline/kv_inference.py` | Add optional `extra_context: str = ""` param to `generate_with_kv` and `generate_text_in_context` |

---

## What Does Not Change

- `kv_indexer.py` — text indexing pipeline untouched
- `vectorstore/` — all backends untouched
- `core/kv_utils.py` — KV serialization/deserialization reused as-is
- All existing tests — image pipeline is purely additive

---

## Success Criteria

1. `PDFImageExtractor.load()` extracts at least one image from a PDF containing an embedded figure; images smaller than 32×32 are skipped
2. `CLIPEmbedder.encode_image()` and `encode_text()` return vectors of the same dimension (512), enabling cosine similarity between text queries and image embeddings
3. `LLaVALoader.encode_image_kv()` returns an array of shape `[num_layers, 2, num_kv_heads, head_dim]` matching the LLaVA model's KV shape
4. `multimodal_search()` returns chunks from both collections, sorted by score descending, total ≤ `top_k`
5. With `image_kv_inference: false` (default), `decide_image_inference_mode` always returns `"caption_fallback"` regardless of KV freshness
6. With `image_kv_inference: true` and all image chunks fresh, `decide_image_inference_mode` returns `"image_kv_injection"`
7. The existing text pipeline test suite passes without modification — `kv_indexer.py` and `kv_inference.py` are untouched
