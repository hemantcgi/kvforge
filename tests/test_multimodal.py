import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_datasource_config_has_image_fields():
    from addons.multimodal.config import MultimodalConfig
    cfg = MultimodalConfig(
        image_kv_inference=False,
        multimodal_model="llava-hf/llava-1.5-7b-hf",
    )
    assert cfg.image_collection_suffix == "_images"
    assert cfg.image_store_dir == ""
    assert cfg.multimodal_model == "llava-hf/llava-1.5-7b-hf"
    assert cfg.clip_model == "openai/clip-vit-base-patch32"
    assert cfg.image_kv_inference is False


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

    with patch("core.multimodal_loader.LlavaForConditionalGeneration") as mock_auto, \
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

    with patch("core.multimodal_loader.LlavaForConditionalGeneration") as mock_auto, \
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

    with patch("core.multimodal_loader.LlavaForConditionalGeneration") as mock_auto, \
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
    fake_model.config.projection_dim = 512

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
