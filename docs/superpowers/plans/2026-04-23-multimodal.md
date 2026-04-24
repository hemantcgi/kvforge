# KVForge Multimodal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend KVForge to index images from PDFs, embed them with CLIP into a separate collection, compute KV tensors via a multimodal LLM (LLaVA), and merge image + text retrieval at query time — with caption-fallback by default and full image-KV-injection as an opt-in path.

**Architecture:** Two parallel pipelines share the vector store but use separate collections (`<collection>` for text, `<collection>_images` for images). Existing text pipeline files are untouched. A thin `multimodal_query.py` orchestrates parallel search and merges by score. Image chunks use stored captions for text-fallback when KV tensors are stale; setting `image_kv_inference: true` enables full image KV injection via LLaVA at query time.

**Tech Stack:** `pdfplumber` (image extraction), `Pillow` (image crop/save), `transformers.CLIPModel` + `CLIPProcessor` (image/text embeddings), `transformers.AutoModelForCausalLM` + `AutoProcessor` (LLaVA multimodal LLM), `torch`, existing `core/kv_utils.py` (KV serialization — reused unchanged).

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `core/config.py` | Modify | Add 5 image config fields |
| `core/multimodal_loader.py` | Create | `MultimodalLLM` protocol + `LLaVALoader` |
| `embeddings/clip_embedder.py` | Create | `CLIPEmbedder` (image + text CLIP encoding) |
| `embeddings/registry.py` | Modify | Add `"clip"` backend |
| `ingestion/image_extractor.py` | Create | `ImageLoader` protocol + `PDFImageExtractor` |
| `pipeline/image_inference.py` | Create | `decide_image_inference_mode`, `get_image_context` |
| `pipeline/kv_inference.py` | Modify | Add `extra_context` param to `generate_with_kv` and `generate_text_in_context` |
| `pipeline/image_indexer.py` | Create | Image indexing + KV recomputation pipeline |
| `pipeline/multimodal_query.py` | Create | Parallel search + merge + `multimodal_answer` |
| `tests/test_multimodal.py` | Create | All tests for new multimodal components |

---

## Task 1: Config fields

**Files:**
- Modify: `core/config.py:78-137`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multimodal.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_datasource_config_has_image_fields():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="test",
        embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384,
        llm_model="meta-llama/Llama-2-7b-hf",
        checkpoint_dir="/tmp/ckpt",
        version_file="/tmp/version.json",
        replay_db="/tmp/replay.db",
    )
    assert cfg.image_collection_suffix == "_images"
    assert cfg.image_store_dir == ""
    assert cfg.multimodal_model == "llava-hf/llava-1.5-7b-hf"
    assert cfg.clip_model == "openai/clip-vit-base-patch32"
    assert cfg.image_kv_inference is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_multimodal.py::test_datasource_config_has_image_fields -v --override-ini="addopts="
```

Expected: FAIL with `AttributeError: 'DatasourceConfig' object has no attribute 'image_collection_suffix'`

- [ ] **Step 3: Add fields to DatasourceConfig**

In `core/config.py`, after the `dashboard_port` field (line ~137), add:

```python
    # Multimodal / image support
    image_collection_suffix: str = "_images"
    image_store_dir: str = ""
    multimodal_model: str = "llava-hf/llava-1.5-7b-hf"
    clip_model: str = "openai/clip-vit-base-patch32"
    image_kv_inference: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_multimodal.py::test_datasource_config_has_image_fields -v --override-ini="addopts="
```

Expected: PASS

- [ ] **Step 5: Verify existing config tests still pass**

```bash
python -m pytest tests/test_config.py -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add core/config.py tests/test_multimodal.py
git commit -m "feat: add image config fields to DatasourceConfig"
```

---

## Task 2: MultimodalLLM Protocol + LLaVALoader

**Files:**
- Create: `core/multimodal_loader.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_multimodal_llm_protocol_shape():
    """LLaVALoader.encode_image_kv returns [layers, 2, heads, head_dim] float16."""
    import numpy as np
    from unittest.mock import MagicMock, patch
    import torch

    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM = 4, 8, 64

    # Build a fake past_key_values: tuple of (k, v) per layer
    fake_k = torch.zeros(1, NUM_KV_HEADS, 10, HEAD_DIM)
    fake_v = torch.zeros(1, NUM_KV_HEADS, 10, HEAD_DIM)
    fake_pkv = tuple((fake_k, fake_v) for _ in range(NUM_LAYERS))

    fake_output = MagicMock()
    fake_output.past_key_values = fake_pkv

    fake_model = MagicMock()
    fake_model.device = torch.device("cpu")
    fake_model.return_value = fake_output
    fake_model.__call__ = MagicMock(return_value=fake_output)
    fake_model.language_model = MagicMock()
    fake_model.language_model.config.num_hidden_layers = NUM_LAYERS
    fake_model.language_model.config.num_key_value_heads = NUM_KV_HEADS
    fake_model.language_model.config.num_attention_heads = NUM_KV_HEADS
    fake_model.language_model.config.hidden_size = NUM_KV_HEADS * HEAD_DIM

    fake_processor = MagicMock()
    fake_processor.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
    }

    with patch("core.multimodal_loader.AutoModelForCausalLM") as mock_auto, \
         patch("core.multimodal_loader.AutoProcessor") as mock_proc, \
         patch("core.multimodal_loader.Image") as mock_pil:
        mock_auto.from_pretrained.return_value = fake_model
        mock_proc.from_pretrained.return_value = fake_processor
        mock_pil.open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_pil.open.return_value.__exit__ = MagicMock(return_value=False)

        import core.multimodal_loader as ml
        ml._model = None  # reset singleton
        ml._processor = None

        loader = ml.LLaVALoader({"multimodal_model": "fake-llava"})
        result = loader.encode_image_kv("/fake/image.png")

    assert result.dtype == np.float16
    assert result.shape == (NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM)


def test_multimodal_llm_kv_shape_property():
    """kv_shape returns (num_layers, num_kv_heads, head_dim) 3-tuple."""
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_model.language_model.config.num_hidden_layers = 32
    fake_model.language_model.config.num_key_value_heads = 32
    fake_model.language_model.config.num_attention_heads = 32
    fake_model.language_model.config.hidden_size = 32 * 128

    with patch("core.multimodal_loader.AutoModelForCausalLM") as mock_auto, \
         patch("core.multimodal_loader.AutoProcessor"):
        mock_auto.from_pretrained.return_value = fake_model
        import core.multimodal_loader as ml
        ml._model = None
        ml._processor = None
        loader = ml.LLaVALoader({"multimodal_model": "fake-llava"})
        assert loader.kv_shape == (32, 32, 128)


def test_multimodal_llm_caption_returns_string():
    """caption() returns a non-empty string."""
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_model.device = torch.device("cpu")
    fake_model.language_model.config.num_hidden_layers = 4
    fake_model.language_model.config.num_key_value_heads = 4
    fake_model.language_model.config.num_attention_heads = 4
    fake_model.language_model.config.hidden_size = 4 * 64

    # generate() returns token ids
    fake_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    fake_processor = MagicMock()
    fake_processor.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "pixel_values": torch.zeros(1, 3, 224, 224),
    }
    fake_processor.decode.return_value = "A bar chart showing sales data."

    with patch("core.multimodal_loader.AutoModelForCausalLM") as mock_auto, \
         patch("core.multimodal_loader.AutoProcessor") as mock_proc, \
         patch("core.multimodal_loader.Image"):
        mock_auto.from_pretrained.return_value = fake_model
        mock_proc.from_pretrained.return_value = fake_processor

        import core.multimodal_loader as ml
        ml._model = None
        ml._processor = None
        loader = ml.LLaVALoader({"multimodal_model": "fake-llava"})
        caption = loader.caption("/fake/image.png")

    assert isinstance(caption, str)
    assert len(caption) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_multimodal_llm_protocol_shape tests/test_multimodal.py::test_multimodal_llm_kv_shape_property tests/test_multimodal.py::test_multimodal_llm_caption_returns_string -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'core.multimodal_loader'`

- [ ] **Step 3: Create `core/multimodal_loader.py`**

```python
"""MultimodalLLM protocol and LLaVALoader implementation.

LLaVALoader is a singleton — call LLaVALoader(cfg) and the model loads once.
"""
import numpy as np
from typing import Protocol, runtime_checkable

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import core.kv_utils as kv_utils

_model = None
_processor = None
_loaded_model_name = None


@runtime_checkable
class MultimodalLLM(Protocol):
    def encode_image_kv(self, image_path: str) -> np.ndarray: ...
    def caption(self, image_path: str) -> str: ...

    @property
    def kv_shape(self) -> tuple[int, int, int]: ...


class LLaVALoader:
    """Loads a LLaVA-style multimodal LLM and computes image KV tensors.

    Uses a module-level singleton so the model loads once per process.
    """

    def __init__(self, cfg: dict) -> None:
        global _model, _processor, _loaded_model_name
        model_name = cfg.get("multimodal_model", "llava-hf/llava-1.5-7b-hf")
        if _model is None or _loaded_model_name != model_name:
            _model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            _model.eval()
            _processor = AutoProcessor.from_pretrained(model_name)
            _loaded_model_name = model_name
        self._model = _model
        self._processor = _processor

    @property
    def kv_shape(self) -> tuple[int, int, int]:
        lm_cfg = self._model.language_model.config
        num_layers = lm_cfg.num_hidden_layers
        num_kv_heads = getattr(lm_cfg, "num_key_value_heads", lm_cfg.num_attention_heads)
        head_dim = lm_cfg.hidden_size // lm_cfg.num_attention_heads
        return (num_layers, num_kv_heads, head_dim)

    def encode_image_kv(self, image_path: str) -> np.ndarray:
        """Run image through LLaVA and mean-pool the KV tensors.

        Returns float16 array of shape [num_layers, 2, num_kv_heads, head_dim].
        """
        with Image.open(image_path) as img:
            inputs = self._processor(
                text="<image>",
                images=img,
                return_tensors="pt",
            )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, use_cache=True)
        return kv_utils.mean_pool_kv(outputs.past_key_values)

    def caption(self, image_path: str) -> str:
        """Generate a text caption for an image."""
        with Image.open(image_path) as img:
            inputs = self._processor(
                text="<image>\nDescribe this image concisely.",
                images=img,
                return_tensors="pt",
            )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        return self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_multimodal_llm_protocol_shape tests/test_multimodal.py::test_multimodal_llm_kv_shape_property tests/test_multimodal.py::test_multimodal_llm_caption_returns_string -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add core/multimodal_loader.py tests/test_multimodal.py
git commit -m "feat: add MultimodalLLM protocol and LLaVALoader"
```

---

## Task 3: CLIPEmbedder + registry

**Files:**
- Create: `embeddings/clip_embedder.py`
- Modify: `embeddings/registry.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_clip_embedder_encode_image_returns_512_dim_vector():
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_model.get_image_features.return_value = torch.zeros(1, 512)

    fake_processor = MagicMock()
    fake_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    with patch("embeddings.clip_embedder.CLIPModel") as mock_model_cls, \
         patch("embeddings.clip_embedder.CLIPProcessor") as mock_proc_cls, \
         patch("embeddings.clip_embedder.Image") as mock_pil:
        mock_model_cls.from_pretrained.return_value = fake_model
        mock_proc_cls.from_pretrained.return_value = fake_processor
        mock_pil.open.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_pil.open.return_value.__exit__ = MagicMock(return_value=False)

        from embeddings.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder("openai/clip-vit-base-patch32")
        vec = embedder.encode_image("/fake/img.png")

    assert isinstance(vec, list)
    assert len(vec) == 512


def test_clip_embedder_encode_text_returns_512_dim_vector():
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_model.get_text_features.return_value = torch.zeros(1, 512)

    fake_processor = MagicMock()
    fake_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

    with patch("embeddings.clip_embedder.CLIPModel") as mock_model_cls, \
         patch("embeddings.clip_embedder.CLIPProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        mock_proc_cls.from_pretrained.return_value = fake_processor

        from embeddings.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder("openai/clip-vit-base-patch32")
        vec = embedder.encode_text("a chart showing revenue growth")

    assert isinstance(vec, list)
    assert len(vec) == 512


def test_clip_embedder_image_and_text_same_dim():
    """encode_image and encode_text must return same-length vectors for cosine sim."""
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_model.get_image_features.return_value = torch.zeros(1, 512)
    fake_model.get_text_features.return_value = torch.zeros(1, 512)

    fake_processor = MagicMock()
    fake_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224),
                                    "input_ids": torch.zeros(1, 10, dtype=torch.long)}

    with patch("embeddings.clip_embedder.CLIPModel") as mock_model_cls, \
         patch("embeddings.clip_embedder.CLIPProcessor") as mock_proc_cls, \
         patch("embeddings.clip_embedder.Image"):
        mock_model_cls.from_pretrained.return_value = fake_model
        mock_proc_cls.from_pretrained.return_value = fake_processor

        from embeddings.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder("openai/clip-vit-base-patch32")
        assert embedder.dim == len(embedder.encode_text("test query"))


def test_embedder_registry_returns_clip_embedder():
    from unittest.mock import MagicMock, patch
    import torch

    fake_model = MagicMock()
    fake_processor = MagicMock()

    with patch("embeddings.clip_embedder.CLIPModel") as mock_model_cls, \
         patch("embeddings.clip_embedder.CLIPProcessor") as mock_proc_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        mock_proc_cls.from_pretrained.return_value = fake_processor

        from embeddings.registry import get_embedder
        cfg = {"embedder_backend": "clip", "clip_model": "openai/clip-vit-base-patch32"}
        embedder = get_embedder(cfg)

    from embeddings.clip_embedder import CLIPEmbedder
    assert isinstance(embedder, CLIPEmbedder)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_clip_embedder_encode_image_returns_512_dim_vector tests/test_multimodal.py::test_clip_embedder_encode_text_returns_512_dim_vector tests/test_multimodal.py::test_clip_embedder_image_and_text_same_dim tests/test_multimodal.py::test_embedder_registry_returns_clip_embedder -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'embeddings.clip_embedder'`

- [ ] **Step 3: Create `embeddings/clip_embedder.py`**

```python
"""CLIP-based embedder for images and text queries.

Used for the separate image collection. Both encode_image and encode_text
return 512-dim vectors (CLIP ViT-B/32), enabling cosine similarity between
text queries and image embeddings in the same vector space.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        # Infer dim from model config
        self._dim: int = self._model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode_image(self, image_path: str) -> list[float]:
        with Image.open(image_path) as img:
            inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()

    def encode_text(self, text: str) -> list[float]:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()
```

- [ ] **Step 4: Add `"clip"` to `embeddings/registry.py`**

In `embeddings/registry.py`, add before the final `raise ValueError`:

```python
    if backend == "clip":
        from embeddings.clip_embedder import CLIPEmbedder
        return CLIPEmbedder(model_name=cfg.get("clip_model", "openai/clip-vit-base-patch32"))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_clip_embedder_encode_image_returns_512_dim_vector tests/test_multimodal.py::test_clip_embedder_encode_text_returns_512_dim_vector tests/test_multimodal.py::test_clip_embedder_image_and_text_same_dim tests/test_multimodal.py::test_embedder_registry_returns_clip_embedder -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 6: Verify existing embedder tests still pass**

```bash
python -m pytest tests/test_embeddings.py -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add embeddings/clip_embedder.py embeddings/registry.py tests/test_multimodal.py
git commit -m "feat: add CLIPEmbedder and register 'clip' backend in embeddings registry"
```

---

## Task 4: ImageLoader Protocol + PDFImageExtractor

**Files:**
- Create: `ingestion/image_extractor.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_pdf_image_extractor_returns_image_dicts(tmp_path):
    """PDFImageExtractor.load() returns one dict per extracted image."""
    from unittest.mock import MagicMock, patch, call
    from PIL import Image as PILImage
    import io

    # Create a small real PNG in memory so PIL can open it
    real_img = PILImage.new("RGB", (100, 100), color=(128, 0, 0))
    img_bytes = io.BytesIO()
    real_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    fake_pdf_image = {
        "x0": 10, "top": 10, "x1": 110, "bottom": 110,
        "width": 100, "height": 100,
    }

    fake_page = MagicMock()
    fake_page.page_number = 1
    fake_page.images = [fake_pdf_image]
    fake_page.to_image.return_value.original = real_img

    fake_pdf = MagicMock()
    fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
    fake_pdf.__exit__ = MagicMock(return_value=False)
    fake_pdf.pages = [fake_page]

    image_store = tmp_path / "images"
    image_store.mkdir()
    cfg = {
        "collection": "test-col",
        "image_store_dir": str(image_store),
    }

    with patch("ingestion.image_extractor.pdfplumber") as mock_pdf:
        mock_pdf.open.return_value = fake_pdf

        from ingestion.image_extractor import PDFImageExtractor
        extractor = PDFImageExtractor(cfg)
        results = extractor.load(str(tmp_path / "fake.pdf"))

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert results[0]["source_file"] == "fake.pdf"
    assert results[0]["image_path"].endswith(".png")


def test_pdf_image_extractor_skips_small_images(tmp_path):
    """Images smaller than 32x32 are skipped."""
    from unittest.mock import MagicMock, patch
    from PIL import Image as PILImage

    tiny_img = PILImage.new("RGB", (10, 10))

    fake_pdf_image = {"x0": 0, "top": 0, "x1": 10, "bottom": 10, "width": 10, "height": 10}
    fake_page = MagicMock()
    fake_page.page_number = 1
    fake_page.images = [fake_pdf_image]
    fake_page.to_image.return_value.original = tiny_img

    fake_pdf = MagicMock()
    fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
    fake_pdf.__exit__ = MagicMock(return_value=False)
    fake_pdf.pages = [fake_page]

    image_store = tmp_path / "images"
    image_store.mkdir()
    cfg = {"collection": "test-col", "image_store_dir": str(image_store)}

    with patch("ingestion.image_extractor.pdfplumber") as mock_pdf:
        mock_pdf.open.return_value = fake_pdf

        from ingestion.image_extractor import PDFImageExtractor
        extractor = PDFImageExtractor(cfg)
        results = extractor.load(str(tmp_path / "fake.pdf"))

    assert results == []


def test_pdf_image_extractor_raises_without_image_store_dir():
    """PDFImageExtractor raises ValueError if image_store_dir is empty."""
    import pytest
    from ingestion.image_extractor import PDFImageExtractor
    with pytest.raises(ValueError, match="image_store_dir"):
        PDFImageExtractor({"collection": "col", "image_store_dir": ""})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_pdf_image_extractor_returns_image_dicts tests/test_multimodal.py::test_pdf_image_extractor_skips_small_images tests/test_multimodal.py::test_pdf_image_extractor_raises_without_image_store_dir -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion.image_extractor'`

- [ ] **Step 3: Create `ingestion/image_extractor.py`**

```python
"""ImageLoader protocol and PDFImageExtractor implementation.

Extracts raster images from PDF pages using pdfplumber, saves each as a PNG,
and returns structured dicts for downstream embedding and KV computation.
"""
import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import pdfplumber


_MIN_IMAGE_PX = 32   # images smaller than this in either dimension are skipped


@runtime_checkable
class ImageLoader(Protocol):
    def load(self, source: str) -> list[dict]: ...
    # Returns: [{"image_path": str, "page": int, "source_file": str}, ...]


class PDFImageExtractor:
    """Extracts raster images from PDF files.

    Each extracted image is saved as a PNG file in
    ``<image_store_dir>/<collection>/`` and represented as a dict with keys
    ``image_path``, ``page`` (1-indexed), and ``source_file``.
    """

    def __init__(self, cfg: dict) -> None:
        image_store_dir = cfg.get("image_store_dir", "")
        if not image_store_dir:
            raise ValueError(
                "image_store_dir must be set in config to use PDFImageExtractor"
            )
        self._out_dir = Path(image_store_dir) / cfg.get("collection", "default")
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def load(self, source: str) -> list[dict]:
        source_path = Path(source)
        source_stem = source_path.stem
        source_file = source_path.name
        results = []

        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                page_num = page.page_number  # 1-indexed
                for idx, img_obj in enumerate(page.images):
                    w = img_obj.get("width", 0)
                    h = img_obj.get("height", 0)
                    if w < _MIN_IMAGE_PX or h < _MIN_IMAGE_PX:
                        continue

                    pil_image = page.to_image(resolution=150).original
                    fname = f"{source_stem}_p{page_num}_{idx}.png"
                    out_path = self._out_dir / fname
                    pil_image.save(str(out_path), format="PNG")

                    results.append({
                        "image_path": str(out_path),
                        "page": page_num,
                        "source_file": source_file,
                    })

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_pdf_image_extractor_returns_image_dicts tests/test_multimodal.py::test_pdf_image_extractor_skips_small_images tests/test_multimodal.py::test_pdf_image_extractor_raises_without_image_store_dir -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add ingestion/image_extractor.py tests/test_multimodal.py
git commit -m "feat: add ImageLoader protocol and PDFImageExtractor"
```

---

## Task 5: image_inference.py (pure decision functions)

**Files:**
- Create: `pipeline/image_inference.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
import core.kv_utils as kv_utils
import numpy as np


def _fake_image_chunk(kv_version, chunk_id=100):
    fake_kv = np.zeros((4, 2, 8, 64), dtype=np.float16)
    return {
        "chunk_id": chunk_id,
        "modality": "image",
        "image_path": "/fake/img.png",
        "caption": "A bar chart showing quarterly revenue.",
        "page": 3,
        "score": 0.85,
        "kv_cache": kv_utils.serialize_kv(fake_kv),
        "kv_version": kv_version,
    }


def test_image_inference_always_caption_fallback_when_flag_false():
    """When image_kv_inference=False, always returns caption_fallback."""
    from pipeline.image_inference import decide_image_inference_mode
    chunks = [_fake_image_chunk(5), _fake_image_chunk(5)]
    mode = decide_image_inference_mode(chunks, current_lora_version=5,
                                        image_kv_inference=False)
    assert mode == "caption_fallback"


def test_image_inference_caption_fallback_when_kv_stale():
    """Stale kv_version triggers caption_fallback even with flag=True."""
    from pipeline.image_inference import decide_image_inference_mode
    chunks = [_fake_image_chunk(3)]  # version 3, current is 5
    mode = decide_image_inference_mode(chunks, current_lora_version=5,
                                        image_kv_inference=True)
    assert mode == "caption_fallback"


def test_image_inference_caption_fallback_when_kv_null():
    """Missing kv_cache triggers caption_fallback even with flag=True."""
    from pipeline.image_inference import decide_image_inference_mode
    chunk = _fake_image_chunk(5)
    chunk["kv_cache"] = None
    mode = decide_image_inference_mode([chunk], current_lora_version=5,
                                        image_kv_inference=True)
    assert mode == "caption_fallback"


def test_image_inference_kv_injection_when_all_fresh_and_flag_true():
    """All fresh + flag=True → image_kv_injection."""
    from pipeline.image_inference import decide_image_inference_mode
    chunks = [_fake_image_chunk(5, i) for i in range(3)]
    mode = decide_image_inference_mode(chunks, current_lora_version=5,
                                        image_kv_inference=True)
    assert mode == "image_kv_injection"


def test_get_image_context_formats_captions():
    from pipeline.image_inference import get_image_context
    chunks = [
        {"page": 2, "score": 0.91, "caption": "A scatter plot of user retention."},
        {"page": 5, "score": 0.78, "caption": "Architecture diagram of the system."},
    ]
    ctx = get_image_context(chunks)
    assert "scatter plot" in ctx
    assert "Architecture diagram" in ctx
    assert "page 2" in ctx
    assert "page 5" in ctx
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_image_inference_always_caption_fallback_when_flag_false tests/test_multimodal.py::test_image_inference_caption_fallback_when_kv_stale tests/test_multimodal.py::test_image_inference_caption_fallback_when_kv_null tests/test_multimodal.py::test_image_inference_kv_injection_when_all_fresh_and_flag_true tests/test_multimodal.py::test_get_image_context_formats_captions -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.image_inference'`

- [ ] **Step 3: Create `pipeline/image_inference.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_image_inference_always_caption_fallback_when_flag_false tests/test_multimodal.py::test_image_inference_caption_fallback_when_kv_stale tests/test_multimodal.py::test_image_inference_caption_fallback_when_kv_null tests/test_multimodal.py::test_image_inference_kv_injection_when_all_fresh_and_flag_true tests/test_multimodal.py::test_get_image_context_formats_captions -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/image_inference.py tests/test_multimodal.py
git commit -m "feat: add image_inference decision logic and caption formatter"
```

---

## Task 6: Add extra_context to kv_inference.py

**Files:**
- Modify: `pipeline/kv_inference.py:56-141`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_generate_with_kv_accepts_extra_context():
    """generate_with_kv should accept extra_context kwarg without error."""
    import inspect
    from pipeline.kv_inference import generate_with_kv
    sig = inspect.signature(generate_with_kv)
    assert "extra_context" in sig.parameters
    assert sig.parameters["extra_context"].default == ""


def test_generate_text_in_context_accepts_extra_context():
    """generate_text_in_context should accept extra_context kwarg without error."""
    import inspect
    from pipeline.kv_inference import generate_text_in_context
    sig = inspect.signature(generate_text_in_context)
    assert "extra_context" in sig.parameters
    assert sig.parameters["extra_context"].default == ""
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_generate_with_kv_accepts_extra_context tests/test_multimodal.py::test_generate_text_in_context_accepts_extra_context -v --override-ini="addopts="
```

Expected: FAIL with `AssertionError`

- [ ] **Step 3: Modify `pipeline/kv_inference.py`**

Change the signature of `generate_with_kv` (line ~57):

```python
def generate_with_kv(query: str, chunks: list[dict],
                      model, tokenizer, cfg: dict,
                      extra_context: str = "") -> str:
```

Inside `generate_with_kv`, change the prompt line:

```python
    context_prefix = f"Additional context:\n{extra_context}\n\n" if extra_context else ""
    prompt = f"{context_prefix}Based on the context provided, answer: {query}"
```

Change the signature of `generate_text_in_context` (line ~103):

```python
def generate_text_in_context(query: str, chunks: list[dict],
                               model, tokenizer,
                               max_new_tokens: int = 256,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               repetition_penalty: float = 1.2,
                               extra_context: str = "") -> str:
```

Inside `generate_text_in_context`, after building `context`, add:

```python
    if extra_context:
        context += f"\n\n---\n\n{extra_context}"
```

(Add this line immediately after the `context = "\n\n---\n\n".join(...)` assignment.)

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest tests/test_multimodal.py::test_generate_with_kv_accepts_extra_context tests/test_multimodal.py::test_generate_text_in_context_accepts_extra_context -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Verify existing kv_inference tests still pass**

```bash
python -m pytest tests/test_kv_inference.py -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/kv_inference.py tests/test_multimodal.py
git commit -m "feat: add extra_context param to generate_with_kv and generate_text_in_context"
```

---

## Task 7: image_indexer.py

**Files:**
- Create: `pipeline/image_indexer.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_build_image_payload_structure():
    """build_image_payload returns a dict with all required keys."""
    from pipeline.image_indexer import build_image_payload
    import numpy as np
    fake_kv = np.zeros((4, 2, 8, 64), dtype=np.float16)
    payload = build_image_payload(
        image_path="/data/imgs/doc_p1_0.png",
        page=1,
        source_file="doc.pdf",
        kv_array=fake_kv,
        caption="A revenue chart.",
    )
    assert payload["image_path"] == "/data/imgs/doc_p1_0.png"
    assert payload["page"] == 1
    assert payload["source_file"] == "doc.pdf"
    assert payload["caption"] == "A revenue chart."
    assert isinstance(payload["kv_cache"], str)  # base64-encoded
    assert payload["kv_version"] is None
    assert payload["access_count"] == 0
    assert payload["tier"] == "frozen"


def test_image_chunk_id_is_deterministic():
    """The same source+page+idx always yields the same integer ID."""
    from pipeline.image_indexer import image_chunk_id
    id1 = image_chunk_id("report.pdf", 3, 1)
    id2 = image_chunk_id("report.pdf", 3, 1)
    id3 = image_chunk_id("report.pdf", 3, 2)
    assert id1 == id2
    assert id1 != id3
    assert isinstance(id1, int)
    assert 0 <= id1 < 2**62
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_build_image_payload_structure tests/test_multimodal.py::test_image_chunk_id_is_deterministic -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.image_indexer'`

- [ ] **Step 3: Create `pipeline/image_indexer.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_build_image_payload_structure tests/test_multimodal.py::test_image_chunk_id_is_deterministic -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/image_indexer.py tests/test_multimodal.py
git commit -m "feat: add image_indexer pipeline (CLIP embed + LLaVA KV + upsert)"
```

---

## Task 8: multimodal_query.py

**Files:**
- Create: `pipeline/multimodal_query.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_multimodal_search_merges_and_sorts_by_score():
    """multimodal_search returns chunks from both collections sorted by score."""
    from unittest.mock import MagicMock, patch
    import numpy as np

    # Fake scored points
    def _make_hit(id_, score, modality):
        h = MagicMock()
        h.id = id_
        h.score = score
        if modality == "text":
            h.payload = {"text": "some text", "page": 1, "kv_cache": None, "kv_version": None}
        else:
            h.payload = {"caption": "a chart", "image_path": "/img.png",
                          "page": 2, "kv_cache": None, "kv_version": None}
        return h

    text_hits  = [_make_hit(1, 0.92, "text"), _make_hit(2, 0.80, "text")]
    image_hits = [_make_hit(101, 0.88, "image"), _make_hit(102, 0.70, "image")]

    fake_store = MagicMock()
    fake_store.query.side_effect = [text_hits, image_hits]

    fake_text_embedder = MagicMock()
    fake_text_embedder.encode.return_value = [[0.0] * 384]

    fake_clip = MagicMock()
    fake_clip.encode_text.return_value = [0.0] * 512

    cfg = {
        "collection": "mycol",
        "image_collection_suffix": "_images",
        "top_k": 4,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    with patch("pipeline.multimodal_query.get_store", return_value=fake_store), \
         patch("pipeline.multimodal_query.get_embedder", return_value=fake_text_embedder), \
         patch("pipeline.multimodal_query.CLIPEmbedder", return_value=fake_clip):
        from pipeline.multimodal_query import multimodal_search
        results = multimodal_search("how did revenue grow?", cfg)

    assert len(results) == 4
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    modalities = [r["modality"] for r in results]
    assert "text" in modalities
    assert "image" in modalities


def test_multimodal_search_caps_at_top_k():
    """multimodal_search never returns more than top_k results."""
    from unittest.mock import MagicMock, patch

    def _hit(id_, score):
        h = MagicMock()
        h.id = id_
        h.score = score
        h.payload = {"text": "t", "page": 1, "kv_cache": None, "kv_version": None}
        return h

    text_hits  = [_hit(i, 0.9 - i * 0.05) for i in range(5)]
    image_hits = [_hit(i + 100, 0.85 - i * 0.05) for i in range(5)]

    fake_store = MagicMock()
    fake_store.query.side_effect = [text_hits, image_hits]

    fake_text_embedder = MagicMock()
    fake_text_embedder.encode.return_value = [[0.0] * 384]

    fake_clip = MagicMock()
    fake_clip.encode_text.return_value = [0.0] * 512

    cfg = {
        "collection": "col",
        "image_collection_suffix": "_images",
        "top_k": 3,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    with patch("pipeline.multimodal_query.get_store", return_value=fake_store), \
         patch("pipeline.multimodal_query.get_embedder", return_value=fake_text_embedder), \
         patch("pipeline.multimodal_query.CLIPEmbedder", return_value=fake_clip):
        from pipeline.multimodal_query import multimodal_search
        results = multimodal_search("query", cfg)

    assert len(results) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_multimodal_search_merges_and_sorts_by_score tests/test_multimodal.py::test_multimodal_search_caps_at_top_k -v --override-ini="addopts="
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.multimodal_query'`

- [ ] **Step 3: Create `pipeline/multimodal_query.py`**

```python
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
from pipeline.image_inference import decide_image_inference_mode, get_image_context
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
    text_mode = decide_inference_mode(text_ch, current_ver)

    stale = get_stale_chunk_ids(text_ch, current_ver)
    if stale:
        import pipeline.kv_background as kv_background
        kv_background.enqueue_kv_recompute(stale)

    if text_mode == "kv_injection":
        return generate_with_kv(query, text_ch, model, tokenizer, cfg,
                                  extra_context=image_context)
    return generate_text_in_context(query, text_ch, model, tokenizer,
                                     extra_context=image_context)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_multimodal.py::test_multimodal_search_merges_and_sorts_by_score tests/test_multimodal.py::test_multimodal_search_caps_at_top_k -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/multimodal_query.py tests/test_multimodal.py
git commit -m "feat: add multimodal_query — parallel search + merge + multimodal_answer"
```

---

## Task 10: Background image KV recomputation

When stale image chunks are retrieved at query time, enqueue them for
background recomputation — mirroring the text path in `kv_inference.py`.
Requires a separate queue and worker in `kv_background.py` because image
recomputation uses `LLaVALoader`, not the text LLM.

**Files:**
- Modify: `pipeline/image_inference.py`
- Modify: `pipeline/kv_background.py`
- Modify: `pipeline/multimodal_query.py`
- Test: `tests/test_multimodal.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multimodal.py`:

```python
def test_get_stale_image_chunk_ids_returns_null_and_old_versions():
    from pipeline.image_inference import get_stale_image_chunk_ids
    chunks = [
        {**_fake_image_chunk(5,  100), "chunk_id": 100},   # fresh
        {**_fake_image_chunk(None, 101), "chunk_id": 101},  # null → stale
        {**_fake_image_chunk(3,  102), "chunk_id": 102},   # old version → stale
    ]
    stale = get_stale_image_chunk_ids(chunks, current_lora_version=5)
    assert set(stale) == {101, 102}


def test_enqueue_image_kv_recompute_puts_ids_on_queue():
    import pipeline.kv_background as bg
    bg._image_kv_queue.queue.clear()
    bg.enqueue_image_kv_recompute([201, 202, 203])
    assert bg._image_kv_queue.qsize() == 3


def test_multimodal_search_enqueues_stale_image_chunks():
    """multimodal_search must enqueue stale image chunk IDs for background healing."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    import core.kv_utils as kv_utils

    stale_kv = np.zeros((4, 2, 8, 64), dtype=np.float16)

    def _hit(id_, score, modality, kv_ver):
        h = MagicMock()
        h.id = id_
        h.score = score
        if modality == "text":
            h.payload = {"text": "t", "page": 1,
                          "kv_cache": kv_utils.serialize_kv(stale_kv),
                          "kv_version": kv_ver}
        else:
            h.payload = {"caption": "c", "image_path": "/img.png", "page": 2,
                          "kv_cache": kv_utils.serialize_kv(stale_kv),
                          "kv_version": kv_ver}
        return h

    text_hits  = [_hit(1, 0.9, "text",  5)]
    image_hits = [_hit(101, 0.85, "image", 2)]   # stale (current is 5)

    fake_store = MagicMock()
    fake_store.query.side_effect = [text_hits, image_hits]
    fake_text_embedder = MagicMock()
    fake_text_embedder.encode.return_value = [[0.0] * 384]
    fake_clip = MagicMock()
    fake_clip.encode_text.return_value = [0.0] * 512

    import pipeline.kv_background as bg
    bg._image_kv_queue.queue.clear()

    cfg = {
        "collection": "col",
        "image_collection_suffix": "_images",
        "top_k": 5,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    with patch("pipeline.multimodal_query.get_store", return_value=fake_store), \
         patch("pipeline.multimodal_query.get_embedder", return_value=fake_text_embedder), \
         patch("pipeline.multimodal_query.CLIPEmbedder", return_value=fake_clip), \
         patch("pipeline.multimodal_query.ver.get_lora_version", return_value=5):
        from pipeline.multimodal_query import multimodal_search
        multimodal_search("query", cfg)

    assert bg._image_kv_queue.qsize() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_multimodal.py::test_get_stale_image_chunk_ids_returns_null_and_old_versions tests/test_multimodal.py::test_enqueue_image_kv_recompute_puts_ids_on_queue tests/test_multimodal.py::test_multimodal_search_enqueues_stale_image_chunks -v --override-ini="addopts="
```

Expected: FAIL — `get_stale_image_chunk_ids` not defined, `_image_kv_queue` not defined.

- [ ] **Step 3: Add `get_stale_image_chunk_ids` to `pipeline/image_inference.py`**

Append to `pipeline/image_inference.py`:

```python
def get_stale_image_chunk_ids(
    image_chunks: list[dict], current_lora_version: int
) -> list[int]:
    """Return chunk IDs whose KV tensors are missing or behind the current LoRA version."""
    return [
        c["chunk_id"]
        for c in image_chunks
        if c.get("kv_version") is None or c["kv_version"] < current_lora_version
    ]
```

- [ ] **Step 4: Add image KV queue + worker + enqueue function to `pipeline/kv_background.py`**

After the existing `_kv_queue` declaration (around line 40), add:

```python
_image_kv_queue: queue.Queue = queue.Queue()
```

After the existing `enqueue_kv_recompute` function, add:

```python
def enqueue_image_kv_recompute(chunk_ids: list[int]) -> None:
    """Schedule image KV tensor recomputation for the given chunk IDs.

    Puts each ID onto the internal image queue consumed by ``_image_kv_worker``.
    Returns immediately; does not block the inference thread.

    Args:
        chunk_ids: List of image chunk identifiers whose KV cache needs refreshing.
    """
    for cid in chunk_ids:
        _image_kv_queue.put(cid)
```

After the existing `_kv_worker` function, add:

```python
def _image_kv_worker(cfg: dict) -> None:
    """Background thread: drain the image KV recompute queue.

    Uses LLaVALoader (multimodal LLM) to recompute image KV tensors.
    Runs independently of the text KV worker — separate queue, separate model.

    Args:
        cfg: Datasource configuration dict.
    """
    from core.multimodal_loader import LLaVALoader
    import core.kv_utils as kv_utils

    client = get_store(cfg)
    image_collection = cfg["collection"] + cfg.get("image_collection_suffix", "_images")
    mm_llm = LLaVALoader(cfg)

    while True:
        chunk_id = _image_kv_queue.get()
        try:
            current_ver = ver.get_lora_version()
            results, _ = client.scroll(
                image_collection, limit=1, with_payload=True,
            )
            results = [r for r in results if r.id == chunk_id]
            if not results:
                _image_kv_queue.task_done()
                continue
            image_path = results[0].payload.get("image_path", "")
            if not image_path:
                _image_kv_queue.task_done()
                continue
            kv_arr = mm_llm.encode_image_kv(image_path)
            client.set_payload(
                image_collection,
                chunk_id,
                {"kv_cache": kv_utils.serialize_kv(kv_arr), "kv_version": current_ver},
            )
        except Exception as e:
            print(f"[kv_background] image KV recompute error for chunk {chunk_id}: {e}",
                  flush=True)
        finally:
            _image_kv_queue.task_done()
```

- [ ] **Step 5: Update `start()` in `kv_background.py` to launch the image worker**

Find the `start(cfg)` function in `kv_background.py`. It currently starts two threads (kv_worker and access_worker). Add a third thread for the image KV worker. The existing start function likely looks like:

```python
def start(cfg: dict) -> None:
    ...
    t = threading.Thread(target=_kv_worker, args=(cfg,), daemon=True)
    t.start()
    ...
```

Add immediately after the existing `_kv_worker` thread start:

```python
    img_t = threading.Thread(target=_image_kv_worker, args=(cfg,), daemon=True,
                              name="image-kv-worker")
    img_t.start()
```

- [ ] **Step 6: Add stale-image enqueue to `multimodal_search` in `pipeline/multimodal_query.py`**

In `multimodal_query.py`, add the import at the top:

```python
from pipeline.image_inference import (
    decide_image_inference_mode,
    get_image_context,
    get_stale_image_chunk_ids,
)
```

In `multimodal_search`, after `merged = sorted(...)`, add:

```python
    # Enqueue stale image chunks for background KV recomputation
    import pipeline.kv_background as kv_background
    current_ver = ver.get_lora_version()
    stale_image_ids = get_stale_image_chunk_ids(
        [c for c in merged if c["modality"] == "image"], current_ver
    )
    if stale_image_ids:
        kv_background.enqueue_image_kv_recompute(stale_image_ids)

    return merged[:top_k]
```

Remove the existing `return merged[:top_k]` line that was there before (it now moves inside the block above).

- [ ] **Step 7: Run all three new tests**

```bash
python -m pytest tests/test_multimodal.py::test_get_stale_image_chunk_ids_returns_null_and_old_versions tests/test_multimodal.py::test_enqueue_image_kv_recompute_puts_ids_on_queue tests/test_multimodal.py::test_multimodal_search_enqueues_stale_image_chunks -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 8: Run full multimodal suite**

```bash
python -m pytest tests/test_multimodal.py -v --override-ini="addopts="
```

Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add pipeline/image_inference.py pipeline/kv_background.py pipeline/multimodal_query.py tests/test_multimodal.py
git commit -m "feat: background image KV recomputation — enqueue stale image chunks at query time"
```

---

## Task 9: Full suite verification

**Files:**
- Test: all existing test files

- [ ] **Step 1: Run the complete multimodal test suite**

```bash
python -m pytest tests/test_multimodal.py -v --override-ini="addopts="
```

Expected: All tests PASS

- [ ] **Step 2: Run the full existing test suite to confirm nothing regressed**

```bash
python -m pytest tests/ -v --override-ini="addopts=" \
  --ignore=tests/test_multimodal.py \
  -k "not integration"
```

Expected: All tests PASS. Pay special attention to `test_kv_inference.py`, `test_ingestion.py`, `test_embeddings.py`, `test_config.py`.

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "test: verify full suite passes after multimodal addition"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|---|---|
| `MultimodalLLM` Protocol with `encode_image_kv`, `caption`, `kv_shape` | Task 2 |
| `LLaVALoader` singleton with correct KV shape output | Task 2 |
| `CLIPEmbedder` image + text encoding (same 512-dim) | Task 3 |
| `"clip"` backend in embeddings registry | Task 3 |
| `ImageLoader` protocol + `PDFImageExtractor` | Task 4 |
| Skip images smaller than 32×32 | Task 4 |
| Raise `ValueError` if `image_store_dir` empty | Task 4 |
| `decide_image_inference_mode` — all four cases | Task 5 |
| `get_image_context` caption formatter | Task 5 |
| `extra_context` param on both generate functions | Task 6 |
| `image_indexer.py` — deterministic ID, build_payload, full pipeline | Task 7 |
| `multimodal_search` — parallel search, merge by score, cap at top_k | Task 8 |
| `multimodal_answer` — default path + Path A | Task 8 |
| 5 new config fields | Task 1 |
| Background image KV recomputation — stale chunks enqueued at query time | Task 10 |
| `_image_kv_worker` uses LLaVALoader (separate from text KV worker) | Task 10 |
| Existing text pipeline tests unaffected | Task 9 |

All 7 success criteria from the spec are covered by tests in Tasks 3, 5, 8, and 9.
The background recomputation gap (Task 10) adds 3 additional tests.
