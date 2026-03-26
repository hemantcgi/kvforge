# KVForge Generalization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KVForge work with any dataset, embedding model, small language model, and vector database — with no hardcoded Bedrock or Qdrant dependencies.

**Architecture:** Seven sequential phases, each independently shippable. Phases 1–2 are pure additive changes (no breakage). Phases 3–5 introduce abstraction layers. Phases 6–7 add config validation, CLI, and multi-collection support. Every phase leaves the system fully functional with the existing Bedrock dataset.

**Tech Stack:** Python 3.10+, fastembed, sentence-transformers, qdrant-client, chromadb, pydantic v2, pypdf, pytest, unittest.mock

---

## File Structure

### New files (created across all phases)
```
ingestion/
  __init__.py
  base.py            — DocumentLoader Protocol (typed duck-typing interface)
  pdf_loader.py      — PDF → chunks (extracted from bedrock_rag.py)
  markdown_loader.py — Markdown → chunks (split by headings)
  jsonl_loader.py    — JSONL → chunks (one doc per line)
  html_loader.py     — HTML → chunks (BeautifulSoup, split by section)
  directory_loader.py — Walk dir, dispatch by extension
  registry.py        — get_loader(name) factory

embeddings/
  __init__.py
  base.py            — Embedder Protocol
  fastembed_embedder.py  — wraps fastembed.TextEmbedding
  sentence_transformer_embedder.py — wraps sentence-transformers
  openai_embedder.py     — wraps openai.embeddings API
  registry.py        — get_embedder(cfg) factory

vectorstore/
  __init__.py
  base.py            — VectorStore Protocol + Point/ScoredPoint dataclasses
  qdrant_store.py    — QdrantStore (wraps qdrant_client)
  chroma_store.py    — ChromaStore (wraps chromadb)
  registry.py        — get_store(cfg) factory

tools/
  generate_faqs.py   — Auto-generate FAQs from indexed corpus

config.py            — Pydantic DatasourceConfig model
kvforge.py       — CLI: init / index / search / train / evaluate

tests/
  test_ingestion.py
  test_embeddings.py
  test_vectorstore.py
  test_prs_evaluator.py
  test_model_loader.py
  test_config.py
  test_generate_faqs.py
```

### Modified files (existing)
```
bedrock_rag.py       — use DocumentLoader + Embedder + VectorStore abstractions
kv_indexer.py        — same; remove direct qdrant_client import
kv_inference.py      — use VectorStore abstraction
kv_background.py     — use VectorStore abstraction
access_tracker.py    — use VectorStore abstraction
monitoring_dashboard.py — use VectorStore abstraction
model_loader.py      — add KV shape auto-discovery + LoRA target detection
prs_evaluator.py     — flexible FAQ schema + configurable PRS weights
lora_trainer.py      — use flexible FAQ schema keys from config
datasource_bedrock.json  — add prs_threshold, prs_weights, loader, vector_store fields
datasource_template.json — update to match full DatasourceConfig schema
```

---

## Chunk 1: Phase 1 — Safety Net (Quick Wins, No Architecture Changes)

### Overview
These are purely additive or small targeted changes. No interfaces are created yet. Everything still works identically after each task. Total risk: near zero.

Tasks: T1.1 (rename defaults), T1.2 (embed dim validation), T1.3 (KV auto-discovery), T1.4 (LoRA target detection), T1.5 (flexible FAQ schema).

---

### Task 1: Rename Bedrock defaults in `bedrock_rag.py`

**Files:**
- Modify: `bedrock_rag.py:44-54`
- No test needed (this is a string default change — covered by existing behavior)

- [ ] **Step 1: Change the default collection name**

In `bedrock_rag.py`, change the `Config` dataclass default:
```python
# Before
collection: str = "bedrock-user-guide"

# After
collection: str = "my-collection"
```

- [ ] **Step 2: Verify the help text still makes sense**

Run: `python bedrock_rag.py --help`
Expected: Shows `my-collection` as the default, no "bedrock" in the output.

- [ ] **Step 3: Commit**
```bash
git add bedrock_rag.py
git commit -m "chore: remove hardcoded bedrock-user-guide as default collection name"
```

---

### Task 2: Add embedding dimension validation

**Files:**
- Modify: `bedrock_rag.py` (in `cmd_index`)
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_embeddings.py`:
```python
"""Tests for embedding dimension validation."""
import pytest
from unittest.mock import MagicMock, patch


def _make_cfg(model: str, dim: int) -> dict:
    return {"embed_model": model, "vector_dim": dim}


def validate_embed_dim(embedder, cfg: dict) -> None:
    """Validate that the embedder's actual output dimension matches cfg['vector_dim']."""
    test_vec = next(iter(embedder.embed(["dimension check"])))
    actual = len(test_vec)
    expected = cfg["vector_dim"]
    if actual != expected:
        raise ValueError(
            f"Embedding model '{cfg['embed_model']}' produces {actual}-dim vectors "
            f"but config declares vector_dim={expected}. "
            f"Update vector_dim in your datasource config."
        )


def test_validation_passes_when_dims_match():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 1024])
    cfg = _make_cfg("some-model", 1024)
    validate_embed_dim(mock_embedder, cfg)  # should not raise


def test_validation_fails_when_dims_mismatch():
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = iter([[0.1] * 384])
    cfg = _make_cfg("some-model", 1024)
    with pytest.raises(ValueError, match="produces 384-dim"):
        validate_embed_dim(mock_embedder, cfg)
```

- [ ] **Step 2: Run tests to see them fail**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: `ImportError` or `NameError` — `validate_embed_dim` not yet in production code.

- [ ] **Step 3: Add `validate_embed_dim` to `bedrock_rag.py`**

Add this function after the `embed_chunks` function (after line ~176):
```python
def validate_embed_dim(embedder: TextEmbedding, cfg: "Config") -> None:
    """Fail fast if embedder output dim doesn't match cfg.vector_dim."""
    test_vec = next(iter(embedder.embed(["dimension check"])))
    actual = len(test_vec)
    if actual != cfg.vector_dim:
        raise ValueError(
            f"Embedding model '{cfg.embed_model}' produces {actual}-dim vectors "
            f"but config declares vector_dim={cfg.vector_dim}. "
            f"Update vector_dim in your datasource config."
        )
```

Call it in `cmd_index`, right after the embedder is created (after `TextEmbedding(...)` line ~314):
```python
embedder = TextEmbedding(model_name=cfg.embed_model, show_download_progress=False)
validate_embed_dim(embedder, cfg)   # ← add this line
```

- [ ] **Step 4: Update test to import from bedrock_rag**

Update `tests/test_embeddings.py` — replace the inline function definition with an import:
```python
from bedrock_rag import validate_embed_dim
```
Remove the local `validate_embed_dim` definition.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: 2 tests PASS.

- [ ] **Step 6: Commit**
```bash
git add bedrock_rag.py tests/test_embeddings.py tests/__init__.py
git commit -m "feat: fail fast when embedding model dim mismatches vector_dim config"
```

---

### Task 3: KV shape auto-discovery from HuggingFace model config

**Files:**
- Modify: `model_loader.py:102-112`
- Create: `tests/test_model_loader.py`

Background: `get_kv_shape()` currently only looks up a manual registry. HuggingFace model configs always expose `num_hidden_layers`, `num_key_value_heads`, and `head_dim` (or derivable as `hidden_size // num_attention_heads`). We auto-discover these first, fall back to registry.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_loader.py`:
```python
"""Tests for KV shape auto-discovery."""
import pytest
from unittest.mock import MagicMock


def _mock_model_config(num_hidden_layers=28, num_key_value_heads=8,
                        hidden_size=4096, num_attention_heads=32,
                        head_dim=None):
    cfg = MagicMock()
    cfg.num_hidden_layers = num_hidden_layers
    cfg.num_key_value_heads = num_key_value_heads
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attention_heads
    # head_dim may or may not be present on the config object
    if head_dim is not None:
        cfg.head_dim = head_dim
    else:
        del cfg.head_dim  # simulate attribute absence
    return cfg


def test_kv_shape_auto_discovery_with_head_dim():
    """When config has head_dim, use it directly."""
    from model_loader import _kv_shape_from_hf_config
    hf_cfg = _mock_model_config(num_hidden_layers=28, num_key_value_heads=8, head_dim=128)
    layers, heads, dim = _kv_shape_from_hf_config(hf_cfg)
    assert layers == 28
    assert heads == 8
    assert dim == 128


def test_kv_shape_auto_discovery_without_head_dim():
    """When config lacks head_dim, derive as hidden_size // num_attention_heads."""
    from model_loader import _kv_shape_from_hf_config
    hf_cfg = _mock_model_config(
        num_hidden_layers=32, num_key_value_heads=8,
        hidden_size=4096, num_attention_heads=32, head_dim=None
    )
    layers, heads, dim = _kv_shape_from_hf_config(hf_cfg)
    assert layers == 32
    assert heads == 8
    assert dim == 128  # 4096 // 32


def test_get_kv_shape_prefers_registry_over_auto():
    """Registry entry takes priority over auto-discovery."""
    from model_loader import get_kv_shape
    cfg = {
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "model_library": {
            "meta-llama/Llama-3.2-3B-Instruct": {
                "kv_num_layers": 28, "kv_num_heads": 8, "kv_head_dim": 128
            }
        }
    }
    layers, heads, dim = get_kv_shape(cfg)
    assert (layers, heads, dim) == (28, 8, 128)


def test_get_kv_shape_falls_back_to_auto_when_not_in_registry():
    """When model not in registry, auto-discover from loaded model."""
    from model_loader import get_kv_shape
    import model_loader
    # Simulate a loaded model with a specific HF config
    mock_model = MagicMock()
    mock_model.config.num_hidden_layers = 24
    mock_model.config.num_key_value_heads = 4
    mock_model.config.head_dim = 64
    original = model_loader._model
    model_loader._model = mock_model
    try:
        cfg = {"llm_model": "some/new-model", "model_library": {}}
        layers, heads, dim = get_kv_shape(cfg)
        assert (layers, heads, dim) == (24, 4, 64)
    finally:
        model_loader._model = original
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: `ImportError: cannot import name '_kv_shape_from_hf_config'`

- [ ] **Step 3: Add `_kv_shape_from_hf_config` and update `get_kv_shape` in `model_loader.py`**

Add after line 100 (before existing `get_kv_shape`), then replace `get_kv_shape`:
```python
def _kv_shape_from_hf_config(hf_cfg) -> tuple[int, int, int]:
    """Extract (num_layers, num_kv_heads, head_dim) from a HuggingFace model config."""
    num_layers = hf_cfg.num_hidden_layers
    num_kv_heads = hf_cfg.num_key_value_heads
    if hasattr(hf_cfg, "head_dim") and hf_cfg.head_dim is not None:
        head_dim = hf_cfg.head_dim
    else:
        head_dim = hf_cfg.hidden_size // hf_cfg.num_attention_heads
    return num_layers, num_kv_heads, head_dim


def get_kv_shape(cfg: dict) -> tuple[int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim).

    Priority:
      1. cfg['model_library'] registry entry (explicit override, backwards compat)
      2. Auto-discovery from loaded model's HF config
      3. Explicit cfg['kv_num_layers'] / 'kv_num_heads' / 'kv_head_dim' (legacy)
    """
    model_id = cfg.get("llm_model", MODEL_ID)
    entry = cfg.get("model_library", {}).get(model_id)
    if entry:
        return entry["kv_num_layers"], entry["kv_num_heads"], entry["kv_head_dim"]

    # Auto-discover from loaded model's config
    if _model is not None and hasattr(_model, "config"):
        try:
            return _kv_shape_from_hf_config(_model.config)
        except AttributeError:
            pass

    # Legacy explicit fields
    return cfg["kv_num_layers"], cfg["kv_num_heads"], cfg["kv_head_dim"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**
```bash
git add model_loader.py tests/test_model_loader.py
git commit -m "feat: auto-discover KV tensor shape from HuggingFace model config"
```

---

### Task 4: LoRA target module auto-detection

**Files:**
- Modify: `model_loader.py` (add new exported function)
- Modify: `lora_trainer.py` (use the function before applying LoRA)
- Add test to: `tests/test_model_loader.py`

- [ ] **Step 1: Add tests for LoRA target detection**

Append to `tests/test_model_loader.py`:
```python
def test_detect_lora_targets_finds_standard_projections():
    from model_loader import detect_lora_targets
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.self_attn.q_proj", MagicMock()),
        ("model.layers.0.self_attn.k_proj", MagicMock()),
        ("model.layers.0.self_attn.v_proj", MagicMock()),
        ("model.layers.0.mlp.gate_proj",   MagicMock()),
    ]
    targets = detect_lora_targets(mock_model, ["q_proj", "k_proj", "v_proj"])
    assert set(targets) == {"q_proj", "k_proj", "v_proj"}


def test_detect_lora_targets_warns_when_none_match():
    from model_loader import detect_lora_targets
    mock_model = MagicMock()
    mock_model.named_modules.return_value = [
        ("model.layers.0.self_attn.query_key_value", MagicMock()),
    ]
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        targets = detect_lora_targets(mock_model, ["q_proj", "k_proj", "v_proj"])
        assert any("query_key_value" in str(warning.message) for warning in w)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_model_loader.py::test_detect_lora_targets_finds_standard_projections -v`
Expected: `ImportError: cannot import name 'detect_lora_targets'`

- [ ] **Step 3: Add `detect_lora_targets` to `model_loader.py`**

Add after `_kv_shape_from_hf_config`:
```python
def detect_lora_targets(model, configured_targets: list[str]) -> list[str]:
    """Verify that configured LoRA target module names exist in the model.

    Returns configured_targets if all are found. If none match, issues a warning
    listing the actual module names so the user can correct their config.
    """
    import warnings
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [t for t in configured_targets if t in module_names]
    if not matched:
        all_linear = sorted(
            {name.split(".")[-1] for name, mod in model.named_modules()
             if hasattr(mod, "weight") and len(getattr(mod, "weight", None).shape or []) == 2}
        )
        warnings.warn(
            f"None of the configured lora_target_modules {configured_targets} were found "
            f"in the model. Available linear layer names: {all_linear[:10]}. "
            f"Check 'lora_target_modules' in your datasource config.",
            UserWarning, stacklevel=2
        )
        return configured_targets  # return as-is; let peft raise a clear error
    return matched
```

- [ ] **Step 4: Use `detect_lora_targets` in `lora_trainer.py`**

Find the section in `lora_trainer.py` where `LoraConfig` is created (after `from peft import LoraConfig ...`). Add detection before it:
```python
import model_loader as _ml
lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"])
lora_target_modules = _ml.detect_lora_targets(model, lora_target_modules)
```

Then use `lora_target_modules` (the variable) instead of `cfg.get("lora_target_modules", ...)` in the `LoraConfig` call.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_model_loader.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**
```bash
git add model_loader.py lora_trainer.py tests/test_model_loader.py
git commit -m "feat: warn when LoRA target modules don't match model architecture"
```

---

### Task 5: Flexible FAQ schema in `prs_evaluator.py`

**Files:**
- Modify: `prs_evaluator.py:97-98` and `prs_evaluator.py:119`
- Create: `tests/test_prs_evaluator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_prs_evaluator.py`:
```python
"""Tests for flexible FAQ schema support."""
import pytest


def _extract_qa(faq: dict, q_key: str = "question", a_key: str = "answer") -> tuple[str, str]:
    """Extract question and answer from a FAQ dict using configurable key names."""
    if q_key not in faq:
        raise KeyError(f"FAQ missing key '{q_key}'. Available keys: {list(faq.keys())}")
    if a_key not in faq:
        raise KeyError(f"FAQ missing key '{a_key}'. Available keys: {list(faq.keys())}")
    return faq[q_key], faq[a_key]


def test_standard_schema():
    faq = {"question": "What is X?", "answer": "X is Y."}
    q, a = _extract_qa(faq)
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_q_a():
    faq = {"q": "What is X?", "a": "X is Y."}
    q, a = _extract_qa(faq, q_key="q", a_key="a")
    assert q == "What is X?"
    assert a == "X is Y."


def test_custom_schema_query_ground_truth():
    faq = {"query": "What is X?", "ground_truth": "X is Y."}
    q, a = _extract_qa(faq, q_key="query", a_key="ground_truth")
    assert q == "What is X?"


def test_missing_key_raises_clear_error():
    faq = {"q": "What?", "a": "This."}
    with pytest.raises(KeyError, match="FAQ missing key 'question'"):
        _extract_qa(faq, q_key="question", a_key="answer")
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_prs_evaluator.py -v`
Expected: Tests FAIL because `_extract_qa` is not yet imported from production code.

- [ ] **Step 3: Add `_extract_qa` to `prs_evaluator.py`**

Add after the imports section (after line 21):
```python
def _extract_qa(faq: dict, q_key: str = "question", a_key: str = "answer") -> tuple[str, str]:
    """Extract question and answer using configurable key names."""
    if q_key not in faq:
        raise KeyError(f"FAQ missing key '{q_key}'. Available keys: {list(faq.keys())}")
    if a_key not in faq:
        raise KeyError(f"FAQ missing key '{a_key}'. Available keys: {list(faq.keys())}")
    return faq[q_key], faq[a_key]
```

- [ ] **Step 4: Use `_extract_qa` in the `evaluate` function**

In `evaluate()`, replace line 98:
```python
# Before
q, gt = faq["question"], faq["answer"]

# After
q_key = cfg.get("faq_question_key", "question")
a_key = cfg.get("faq_answer_key", "answer")
q, gt = _extract_qa(faq, q_key=q_key, a_key=a_key)
```

Also replace line 119 (good_queries extraction):
```python
# Before
good_queries = [faqs[i]["question"] for i, r in enumerate(accuracy_ratios) if r >= 0.85]

# After
good_queries = [faqs[i].get(q_key, faqs[i].get("question", ""))
                for i, r in enumerate(accuracy_ratios) if r >= 0.85]
```

Note: `q_key` is defined inside the loop but needed outside. Refactor to define it once before the loop by moving those two lines above the `for faq in faqs:` loop.

- [ ] **Step 5: Update test to import from prs_evaluator**

Update `tests/test_prs_evaluator.py`:
```python
from prs_evaluator import _extract_qa
```
Remove the local `_extract_qa` definition.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_prs_evaluator.py -v`
Expected: 4 tests PASS.

- [ ] **Step 7: Commit**
```bash
git add prs_evaluator.py tests/test_prs_evaluator.py
git commit -m "feat: support configurable FAQ schema keys (faq_question_key / faq_answer_key)"
```

---

## Chunk 2: Phase 2 — Document Loader Abstraction

### Overview
Extract PDF reading from `bedrock_rag.py` into a `DocumentLoader` protocol with implementations for PDF, Markdown, JSONL, HTML, and Directory. Wire via `"loader"` key in datasource config.

---

### Task 6: `DocumentLoader` protocol and `PDFLoader`

**Files:**
- Create: `ingestion/__init__.py`
- Create: `ingestion/base.py`
- Create: `ingestion/pdf_loader.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests for PDFLoader**

Create `ingestion/__init__.py` (empty).

Create `tests/test_ingestion.py`:
```python
"""Tests for document loader abstraction."""
import pytest
from unittest.mock import patch, MagicMock


def test_document_loader_protocol_is_satisfied_by_pdf_loader():
    """PDFLoader must implement the DocumentLoader protocol."""
    from ingestion.pdf_loader import PDFLoader
    from ingestion.base import DocumentLoader
    import typing
    # Protocol compliance: PDFLoader must have a `load` method
    assert hasattr(PDFLoader, "load")


def test_pdf_loader_returns_list_of_dicts_with_text_and_metadata(tmp_path):
    """PDFLoader.load() returns [{text: str, metadata: {page: int, source: str}}]."""
    from ingestion.pdf_loader import PDFLoader
    # Create a minimal fake PDF using pypdf mock
    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello world from page one."
        mock_reader.return_value.pages = [mock_page]
        loader = PDFLoader(chunk_size=5, chunk_overlap=1)
        docs = loader.load(str(tmp_path / "fake.pdf"))
    assert isinstance(docs, list)
    assert len(docs) > 0
    assert "text" in docs[0]
    assert "metadata" in docs[0]
    assert "page" in docs[0]["metadata"]


def test_pdf_loader_skips_short_chunks(tmp_path):
    """Chunks with fewer than 30 words are dropped."""
    from ingestion.pdf_loader import PDFLoader
    with patch("ingestion.pdf_loader.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Short text."
        mock_reader.return_value.pages = [mock_page]
        loader = PDFLoader(chunk_size=600, chunk_overlap=60)
        docs = loader.load("fake.pdf")
    assert docs == []  # 2 words < 30 word minimum
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_ingestion.py -v`
Expected: `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Create `ingestion/base.py`**

```python
"""ingestion/base.py — DocumentLoader Protocol."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, source: str) -> list[dict]:
        """Load documents from source.

        Returns:
            List of dicts: [{"text": str, "metadata": {"source": str, ...}}, ...]
        """
        ...
```

- [ ] **Step 4: Create `ingestion/pdf_loader.py`**

```python
"""ingestion/pdf_loader.py — Load and chunk PDF files."""
from pathlib import Path
from pypdf import PdfReader


class PDFLoader:
    """Load a PDF and split it into overlapping word-based chunks."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60,
                 min_chunk_words: int = 30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        """Read a PDF file and return chunks as document dicts.

        Each dict: {"text": str, "metadata": {"page": int, "source": str}}
        """
        path = Path(source)
        reader = PdfReader(str(path))
        docs = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunk_id = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            words = text.split()
            for start in range(0, len(words), step):
                chunk_words = words[start: start + self.chunk_size]
                if len(chunk_words) < self.min_chunk_words:
                    continue
                docs.append({
                    "text": " ".join(chunk_words),
                    "metadata": {
                        "page": page_num,
                        "source": path.name,
                        "chunk_id": chunk_id,
                    },
                })
                chunk_id += 1
        return docs
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ingestion.py -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**
```bash
git add ingestion/ tests/test_ingestion.py
git commit -m "feat: add DocumentLoader protocol and PDFLoader"
```

---

### Task 7: `MarkdownLoader`, `JSONLLoader`, `HTMLLoader`, `DirectoryLoader`

**Files:**
- Create: `ingestion/markdown_loader.py`
- Create: `ingestion/jsonl_loader.py`
- Create: `ingestion/html_loader.py`
- Create: `ingestion/directory_loader.py`
- Create: `ingestion/registry.py`
- Add tests to: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests for all loaders**

Append to `tests/test_ingestion.py`:
```python
def test_markdown_loader_splits_by_heading(tmp_path):
    from ingestion.markdown_loader import MarkdownLoader
    md = tmp_path / "doc.md"
    md.write_text("# Section One\n\nThis is section one content with many words to fill the chunk.\n\n"
                  "# Section Two\n\nThis is section two content with different words here now.")
    loader = MarkdownLoader()
    docs = loader.load(str(md))
    assert len(docs) >= 1
    assert all("text" in d and "metadata" in d for d in docs)


def test_jsonl_loader_reads_one_doc_per_line(tmp_path):
    from ingestion.jsonl_loader import JSONLLoader
    jl = tmp_path / "data.jsonl"
    jl.write_text(
        '{"text": "First document with enough words to be a real chunk."}\n'
        '{"text": "Second document also has enough words to pass the minimum check."}\n'
    )
    loader = JSONLLoader(text_key="text")
    docs = loader.load(str(jl))
    assert len(docs) == 2
    assert docs[0]["text"].startswith("First")


def test_jsonl_loader_custom_text_key(tmp_path):
    from ingestion.jsonl_loader import JSONLLoader
    jl = tmp_path / "data.jsonl"
    jl.write_text('{"content": "Document content here with enough words."}\n')
    loader = JSONLLoader(text_key="content")
    docs = loader.load(str(jl))
    assert len(docs) == 1


def test_html_loader_strips_tags(tmp_path):
    from ingestion.html_loader import HTMLLoader
    html = tmp_path / "page.html"
    html.write_text("<html><body><h1>Title</h1><p>Body text content with enough words here.</p></body></html>")
    loader = HTMLLoader()
    docs = loader.load(str(html))
    assert len(docs) >= 1
    assert "<" not in docs[0]["text"]  # tags stripped


def test_directory_loader_dispatches_by_extension(tmp_path):
    from ingestion.directory_loader import DirectoryLoader
    (tmp_path / "a.md").write_text("# Doc A\n\nContent for document A with many words here.")
    (tmp_path / "b.jsonl").write_text('{"text": "Document B content with sufficient words."}\n')
    loader = DirectoryLoader()
    docs = loader.load(str(tmp_path))
    assert len(docs) >= 2


def test_registry_returns_correct_loader():
    from ingestion.registry import get_loader
    from ingestion.pdf_loader import PDFLoader
    from ingestion.markdown_loader import MarkdownLoader
    from ingestion.jsonl_loader import JSONLLoader
    assert isinstance(get_loader({"loader": "pdf"}), PDFLoader)
    assert isinstance(get_loader({"loader": "markdown"}), MarkdownLoader)
    assert isinstance(get_loader({"loader": "jsonl"}), JSONLLoader)
    assert isinstance(get_loader({}), PDFLoader)  # default is pdf
```

- [ ] **Step 2: Create `ingestion/markdown_loader.py`**

```python
"""ingestion/markdown_loader.py — Load Markdown files, split by heading."""
import re
from pathlib import Path


class MarkdownLoader:
    """Split a Markdown file into sections at each heading (# / ## / ###)."""

    def __init__(self, min_chunk_words: int = 30):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"(?m)^#{1,3}\s+", text)
        docs = []
        for i, section in enumerate(sections):
            clean = section.strip()
            if not clean or len(clean.split()) < self.min_chunk_words:
                continue
            docs.append({
                "text": clean,
                "metadata": {"source": path.name, "section": i, "chunk_id": i},
            })
        return docs
```

- [ ] **Step 3: Create `ingestion/jsonl_loader.py`**

```python
"""ingestion/jsonl_loader.py — Load JSONL files (one JSON object per line)."""
import json
from pathlib import Path


class JSONLLoader:
    """Each line of the JSONL file becomes one document."""

    def __init__(self, text_key: str = "text", min_chunk_words: int = 5):
        self.text_key = text_key
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        path = Path(source)
        docs = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(self.text_key, "")
                if not text or len(text.split()) < self.min_chunk_words:
                    continue
                metadata = {k: v for k, v in obj.items() if k != self.text_key}
                metadata.update({"source": path.name, "chunk_id": line_num - 1})
                docs.append({"text": text, "metadata": metadata})
        return docs
```

- [ ] **Step 4: Create `ingestion/html_loader.py`**

```python
"""ingestion/html_loader.py — Load HTML files, strip tags, split by section."""
from pathlib import Path


class HTMLLoader:
    """Strip HTML tags and return text content as chunks."""

    def __init__(self, min_chunk_words: int = 30):
        self.min_chunk_words = min_chunk_words

    def load(self, source: str) -> list[dict]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("HTMLLoader requires beautifulsoup4: pip install beautifulsoup4")
        path = Path(source)
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        # Split at block-level elements
        sections = []
        current = []
        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue
            if tag.name in ("h1", "h2", "h3") and current:
                sections.append(" ".join(current))
                current = [text]
            else:
                current.append(text)
        if current:
            sections.append(" ".join(current))

        docs = []
        for i, section in enumerate(sections):
            if len(section.split()) < self.min_chunk_words:
                continue
            docs.append({
                "text": section,
                "metadata": {"source": path.name, "section": i, "chunk_id": i},
            })
        return docs
```

- [ ] **Step 5: Create `ingestion/directory_loader.py`**

```python
"""ingestion/directory_loader.py — Recursively load all supported docs in a dir."""
from pathlib import Path


EXTENSION_MAP = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".jsonl": "jsonl",
    ".html": "html",
    ".htm": "html",
}


class DirectoryLoader:
    """Walk a directory and load all supported document types."""

    def __init__(self, recursive: bool = True, **loader_kwargs):
        self.recursive = recursive
        self.loader_kwargs = loader_kwargs

    def load(self, source: str) -> list[dict]:
        from ingestion.registry import get_loader
        path = Path(source)
        pattern = "**/*" if self.recursive else "*"
        docs = []
        for file_path in sorted(path.glob(pattern)):
            if not file_path.is_file():
                continue
            loader_name = EXTENSION_MAP.get(file_path.suffix.lower())
            if not loader_name:
                continue
            loader = get_loader({"loader": loader_name, **self.loader_kwargs})
            docs.extend(loader.load(str(file_path)))
        return docs
```

- [ ] **Step 6: Create `ingestion/registry.py`**

```python
"""ingestion/registry.py — Factory for DocumentLoader implementations."""


def get_loader(cfg: dict):
    """Return the appropriate DocumentLoader for the given config.

    Dispatches on cfg['loader'] (default: 'pdf').
    Passes chunk_size, chunk_overlap, and other cfg keys as kwargs where applicable.
    """
    name = cfg.get("loader", "pdf")
    chunk_size = cfg.get("chunk_size", 600)
    chunk_overlap = cfg.get("chunk_overlap", 60)

    if name == "pdf":
        from ingestion.pdf_loader import PDFLoader
        return PDFLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if name == "markdown":
        from ingestion.markdown_loader import MarkdownLoader
        return MarkdownLoader()
    if name == "jsonl":
        text_key = cfg.get("jsonl_text_key", "text")
        from ingestion.jsonl_loader import JSONLLoader
        return JSONLLoader(text_key=text_key)
    if name == "html":
        from ingestion.html_loader import HTMLLoader
        return HTMLLoader()
    if name == "directory":
        from ingestion.directory_loader import DirectoryLoader
        return DirectoryLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unknown loader '{name}'. Choose: pdf, markdown, jsonl, html, directory")
```

- [ ] **Step 7: Run all ingestion tests**

Run: `python -m pytest tests/test_ingestion.py -v`
Expected: All tests PASS. (HTMLLoader test requires `pip install beautifulsoup4` first.)

Install if needed: `pip install beautifulsoup4`

- [ ] **Step 8: Commit**
```bash
git add ingestion/ tests/test_ingestion.py
git commit -m "feat: add MarkdownLoader, JSONLLoader, HTMLLoader, DirectoryLoader, and registry"
```

---

### Task 8: Wire `DocumentLoader` into `bedrock_rag.py` and `kv_indexer.py`

**Files:**
- Modify: `bedrock_rag.py` (use `get_loader` in `cmd_index`)
- Modify: `kv_indexer.py` (use `get_loader` in its index path)

- [ ] **Step 1: Update `bedrock_rag.py`**

In `cmd_index`, replace:
```python
pages = read_pdf(pdf_path)
chunks = chunk_pages(pages, cfg.chunk_size, cfg.chunk_overlap)
```
with:
```python
from ingestion.registry import get_loader
loader = get_loader(vars(cfg))
docs = loader.load(str(pdf_path))
# Convert to bedrock_rag's internal chunk format for backwards compatibility
chunks = [{"chunk_id": d["metadata"]["chunk_id"],
           "page": d["metadata"].get("page", 0),
           "text": d["text"]} for d in docs]
```

The `read_pdf` and `chunk_pages` functions can stay (they are used by `kv_indexer.py` still), but `cmd_index` no longer calls them directly.

- [ ] **Step 2: Verify existing Bedrock dataset still works**

Run: `python bedrock_rag.py --config datasource_bedrock.json index "examples/Amazon Bedrock Dataset.pdf"`
Expected: Indexes successfully — same number of chunks as before.

- [ ] **Step 3: Commit**
```bash
git add bedrock_rag.py kv_indexer.py
git commit -m "feat: wire DocumentLoader into ingestion pipeline; PDF remains default"
```

---

## Chunk 3: Phase 3 — Embedding Model Abstraction

### Task 9: `Embedder` protocol and `FastEmbedEmbedder`

**Files:**
- Create: `embeddings/__init__.py`
- Create: `embeddings/base.py`
- Create: `embeddings/fastembed_embedder.py`
- Create: `embeddings/sentence_transformer_embedder.py`
- Create: `embeddings/openai_embedder.py`
- Create: `embeddings/registry.py`
- Add tests to: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_embeddings.py`:
```python
def test_embedder_protocol_satisfied_by_fastembed():
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    from embeddings.base import Embedder
    embedder = FastEmbedEmbedder.__new__(FastEmbedEmbedder)
    assert hasattr(embedder, "encode")
    assert hasattr(embedder, "dim")


def test_fastembed_embedder_encode_returns_correct_shape():
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    from unittest.mock import patch, MagicMock
    with patch("embeddings.fastembed_embedder.TextEmbedding") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed.return_value = iter([[0.1] * 384, [0.2] * 384])
        mock_cls.return_value = mock_instance
        embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", dim=384)
        result = embedder.encode(["text one", "text two"])
    assert len(result) == 2
    assert len(result[0]) == 384


def test_registry_returns_fastembed_by_default():
    from embeddings.registry import get_embedder
    from embeddings.fastembed_embedder import FastEmbedEmbedder
    from unittest.mock import patch
    with patch("embeddings.fastembed_embedder.TextEmbedding"):
        cfg = {"embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384}
        embedder = get_embedder(cfg)
    assert isinstance(embedder, FastEmbedEmbedder)
```

- [ ] **Step 2: Create `embeddings/base.py`**

```python
"""embeddings/base.py — Embedder Protocol."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings; returns list of float vectors."""
        ...

    @property
    def dim(self) -> int:
        """The dimensionality of produced vectors."""
        ...
```

- [ ] **Step 3: Create `embeddings/fastembed_embedder.py`**

```python
"""embeddings/fastembed_embedder.py — Wraps fastembed.TextEmbedding."""
from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int, show_progress: bool = False):
        self._model_name = model_name
        self._dim = dim
        self._embedder = TextEmbedding(model_name=model_name,
                                        show_download_progress=show_progress)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    @property
    def dim(self) -> int:
        return self._dim
```

- [ ] **Step 4: Create `embeddings/sentence_transformer_embedder.py`**

```python
"""embeddings/sentence_transformer_embedder.py — Wraps sentence-transformers."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("SentenceTransformerEmbedder requires: pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._dim = dim or self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    @property
    def dim(self) -> int:
        return self._dim
```

- [ ] **Step 5: Create `embeddings/openai_embedder.py`**

```python
"""embeddings/openai_embedder.py — Wraps OpenAI Embeddings API."""
import os


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small", dim: int = 1536,
                 api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAIEmbedder requires: pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._model_name = model_name
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model_name)
        return [item.embedding for item in resp.data]

    @property
    def dim(self) -> int:
        return self._dim
```

- [ ] **Step 6: Create `embeddings/registry.py`**

```python
"""embeddings/registry.py — Factory for Embedder implementations."""


def get_embedder(cfg: dict):
    """Return the configured Embedder instance.

    Dispatches on cfg['embedder_backend'] (default: 'fastembed').
    """
    backend = cfg.get("embedder_backend", "fastembed")
    model_name = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    dim = cfg.get("vector_dim", 384)

    if backend == "fastembed":
        from embeddings.fastembed_embedder import FastEmbedEmbedder
        return FastEmbedEmbedder(model_name=model_name, dim=dim)
    if backend == "sentence_transformers":
        from embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=model_name, dim=dim)
    if backend == "openai":
        from embeddings.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(model_name=model_name, dim=dim,
                               api_key=cfg.get("openai_api_key"))
    raise ValueError(f"Unknown embedder_backend '{backend}'. "
                     f"Choose: fastembed, sentence_transformers, openai")
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_embeddings.py -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**
```bash
git add embeddings/ tests/test_embeddings.py
git commit -m "feat: add Embedder abstraction with FastEmbed, SentenceTransformers, OpenAI backends"
```

---

## Chunk 4: Phase 4 — Auto-FAQ Generation + PRS Enhancements

### Task 10: `tools/generate_faqs.py`

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/generate_faqs.py`
- Create: `tests/test_generate_faqs.py`

Purpose: Given an indexed collection, sample chunks, prompt the LLM to generate a Q&A pair per chunk, validate each pair by re-retrieving, and save to JSON.

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate_faqs.py`:
```python
"""Tests for automatic FAQ generation."""
import pytest
from unittest.mock import MagicMock, patch


def test_parse_qa_from_llm_output_standard_format():
    from tools.generate_faqs import _parse_qa
    text = "Q: What is Qdrant?\nA: Qdrant is a vector database."
    q, a = _parse_qa(text)
    assert q == "What is Qdrant?"
    assert a == "Qdrant is a vector database."


def test_parse_qa_from_llm_output_question_answer_format():
    from tools.generate_faqs import _parse_qa
    text = "Question: How does KV injection work?\nAnswer: It injects cached tensors."
    q, a = _parse_qa(text)
    assert q == "How does KV injection work?"
    assert a == "It injects cached tensors."


def test_parse_qa_returns_none_when_format_unrecognized():
    from tools.generate_faqs import _parse_qa
    result = _parse_qa("This is just some random text without QA structure.")
    assert result is None


def test_sample_chunks_returns_n_items():
    from tools.generate_faqs import _sample_chunks
    mock_store = MagicMock()
    mock_store.scroll.return_value = [
        MagicMock(payload={"text": f"Chunk {i} text"}) for i in range(10)
    ]
    chunks = _sample_chunks(mock_store, "my-collection", n=5)
    assert len(chunks) == 5
```

- [ ] **Step 2: Create `tools/generate_faqs.py`**

```python
"""tools/generate_faqs.py — Auto-generate FAQs from an indexed corpus.

Usage:
  python tools/generate_faqs.py --config datasource_example.json --count 50 --output faqs.json

Algorithm:
  1. Sample N chunks from the Qdrant collection
  2. For each chunk, prompt the LLM: "Generate one factual Q&A pair..."
  3. Parse Q and A from the LLM output
  4. Save to output JSON file
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


QA_PROMPT_TEMPLATE = """Read the following passage and generate exactly one factual question and answer pair about it.

Passage:
{chunk}

Respond in this exact format:
Q: <question>
A: <answer>"""


def _parse_qa(text: str) -> tuple[str, str] | None:
    """Parse a Q&A pair from LLM output. Returns (question, answer) or None."""
    # Match Q:/Question: and A:/Answer: patterns
    q_match = re.search(r"(?:Q|Question):\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    a_match = re.search(r"(?:A|Answer):\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if not q_match or not a_match:
        return None
    q = q_match.group(1).strip()
    a = a_match.group(1).strip()
    if not q or not a:
        return None
    return q, a


def _sample_chunks(store, collection: str, n: int) -> list[str]:
    """Sample up to n chunks from the vector store."""
    results = store.scroll(collection, limit=500, with_payload=True, with_vectors=False)
    texts = [r.payload.get("text", "") for r in results if r.payload.get("text")]
    return random.sample(texts, min(n, len(texts)))


def generate(cfg: dict, count: int, output_path: str) -> None:
    """Main generation loop."""
    from qdrant_client import QdrantClient
    import model_loader

    client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    model_loader.init(cfg)
    model, tokenizer = model_loader.load()

    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=128, do_sample=False)

    # Sample chunks
    results, _ = client.scroll(
        collection_name=cfg["collection"], limit=500,
        with_payload=True, with_vectors=False
    )
    texts = [r.payload["text"] for r in results if r.payload.get("text")]
    sampled = random.sample(texts, min(count * 2, len(texts)))  # oversample to account for parse failures

    faqs = []
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")

    for chunk in sampled:
        if len(faqs) >= count:
            break
        prompt = QA_PROMPT_TEMPLATE.format(chunk=chunk[:1000])
        out = pipe(prompt)[0]["generated_text"][len(prompt):].strip()
        parsed = _parse_qa(out)
        if parsed is None:
            continue
        q, a = parsed
        faqs.append({q_key: q, a_key: a})
        print(f"  [{len(faqs)}/{count}] Q: {q[:80]}")

    with open(output_path, "w") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(faqs)} FAQs to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Auto-generate FAQs from an indexed corpus")
    p.add_argument("--config", required=True, help="Datasource config JSON")
    p.add_argument("--count", type=int, default=50, help="Number of FAQs to generate")
    p.add_argument("--output", default="generated_faqs.json", help="Output JSON path")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    generate(cfg, args.count, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_generate_faqs.py -v`
Expected: 4 tests PASS.

- [ ] **Step 4: Commit**
```bash
git add tools/ tests/test_generate_faqs.py
git commit -m "feat: add auto-FAQ generation tool for any indexed corpus"
```

---

### Task 11: Configurable PRS weights

**Files:**
- Modify: `prs_evaluator.py` (make weights read from config)
- Add tests to: `tests/test_prs_evaluator.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_prs_evaluator.py`:
```python
def test_prs_weights_from_config():
    """PRS formula respects configurable weights."""
    from prs_evaluator import _compute_prs
    # All-accuracy config
    prs = _compute_prs(
        accuracy_ratios=[1.0, 1.0],
        calibrations=[0.0, 0.0],
        consistencies=[0.0, 0.0],
        weights={"accuracy": 1.0, "calibration": 0.0, "consistency": 0.0}
    )
    assert abs(prs - 1.0) < 0.001


def test_prs_uses_default_weights_when_not_in_config():
    from prs_evaluator import _compute_prs
    prs = _compute_prs([0.8], [0.9], [0.7], weights=None)
    expected = 0.5 * 0.8 + 0.3 * 0.9 + 0.2 * 0.7
    assert abs(prs - expected) < 0.001
```

- [ ] **Step 2: Add `_compute_prs` to `prs_evaluator.py`**

Add before `evaluate()`:
```python
_DEFAULT_PRS_WEIGHTS = {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2}


def _compute_prs(accuracy_ratios: list, calibrations: list, consistencies: list,
                 weights: dict | None) -> float:
    """Compute weighted PRS from component lists."""
    import numpy as np
    w = weights or _DEFAULT_PRS_WEIGHTS
    return float(np.clip(
        w.get("accuracy", 0.5) * np.mean(accuracy_ratios)
        + w.get("calibration", 0.3) * np.mean(calibrations)
        + w.get("consistency", 0.2) * np.mean(consistencies),
        0.0, 1.0
    ))
```

- [ ] **Step 3: Use `_compute_prs` in `evaluate()`**

Replace the inline PRS computation (lines 113–115) with:
```python
weights = cfg.get("prs_weights", None)
prs = _compute_prs(accuracy_ratios, calibrations, consistencies, weights)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_prs_evaluator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Update `datasource_bedrock.json` with new optional fields**

Add at the end of `datasource_bedrock.json`:
```json
  "prs_threshold": 0.75,
  "prs_weights": {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2},
  "faq_question_key": "question",
  "faq_answer_key": "answer",
  "loader": "pdf",
  "embedder_backend": "fastembed"
```

- [ ] **Step 6: Commit**
```bash
git add prs_evaluator.py datasource_bedrock.json tests/test_prs_evaluator.py
git commit -m "feat: configurable PRS weights via prs_weights config key"
```

---

## Chunk 5: Phase 5 — Config Validation (Pydantic)

### Task 12: `DatasourceConfig` Pydantic model

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_config.py`:
```python
"""Tests for Pydantic DatasourceConfig."""
import pytest


def test_config_loads_from_dict_with_defaults():
    from config import DatasourceConfig
    cfg = DatasourceConfig(collection="test-col", embed_model="BAAI/bge-small-en-v1.5",
                            vector_dim=384, llm_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            checkpoint_dir="checkpoints/", version_file="v.json",
                            replay_db="r.db")
    assert cfg.loader == "pdf"
    assert cfg.vector_store == "qdrant"
    assert cfg.gate_threshold == 0.75
    assert cfg.prs_threshold == 0.75
    assert cfg.faq_question_key == "question"


def test_config_rejects_unknown_loader():
    from config import DatasourceConfig
    import pydantic
    with pytest.raises(pydantic.ValidationError, match="loader"):
        DatasourceConfig(collection="x", embed_model="x", vector_dim=1,
                          llm_model="x", checkpoint_dir="x", version_file="x",
                          replay_db="x", loader="excel")  # not valid


def test_config_loads_from_json_file(tmp_path):
    from config import load_config
    import json
    cfg_data = {
        "collection": "my-docs", "embed_model": "BAAI/bge-small-en-v1.5",
        "vector_dim": 384, "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "checkpoint_dir": "ckpt/", "version_file": "v.json", "replay_db": "r.db"
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_data))
    cfg = load_config(str(p))
    assert cfg.collection == "my-docs"
    assert isinstance(cfg.lora_target_modules, list)


def test_config_as_dict_is_compatible_with_existing_code():
    """cfg.model_dump() must produce keys that existing code reads via cfg['key']."""
    from config import DatasourceConfig
    cfg = DatasourceConfig(collection="x", embed_model="x", vector_dim=1,
                            llm_model="x", checkpoint_dir="x", version_file="x",
                            replay_db="x")
    d = cfg.model_dump()
    for key in ["collection", "embed_model", "vector_dim", "llm_model",
                 "lora_rank", "gate_threshold"]:
        assert key in d, f"Missing key: {key}"
```

- [ ] **Step 2: Create `config.py`**

```python
"""config.py — Pydantic model for KVForge datasource configuration."""
import json
from typing import Literal
from pydantic import BaseModel, Field


class DatasourceConfig(BaseModel):
    # Vector store connection
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    vector_store: Literal["qdrant", "chroma"] = "qdrant"

    # Collection & ingestion
    collection: str
    loader: Literal["pdf", "markdown", "jsonl", "html", "directory"] = "pdf"
    chunk_size: int = 600
    chunk_overlap: int = 60
    embed_batch: int = 64
    upsert_batch: int = 128
    top_k: int = 5
    jsonl_text_key: str = "text"

    # Embedding
    embedder_backend: Literal["fastembed", "sentence_transformers", "openai"] = "fastembed"
    embed_model: str
    vector_dim: int

    # Language model
    llm_model: str
    hf_token: str | None = None
    max_new_tokens: int = 256
    model_library: dict = Field(default_factory=dict)

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    lora_dropout: float = 0.05
    lora_epochs: int = 3
    lora_lr: float = 0.0002

    # State files
    checkpoint_dir: str
    version_file: str
    replay_db: str

    # Phase gating
    gate_threshold: float = 0.75
    prs_threshold: float = 0.75
    prs_weights: dict = Field(default_factory=lambda: {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2})

    # FAQ schema
    faq_question_key: str = "question"
    faq_answer_key: str = "answer"

    # Dashboard
    access_flush_seconds: int = 300
    access_flush_queries: int = 50
    dashboard_port: int = 8080


def load_config(path: str) -> DatasourceConfig:
    """Load and validate a datasource config from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return DatasourceConfig(**data)
```

- [ ] **Step 3: Install pydantic if needed**

Run: `pip install "pydantic>=2.0"`

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_config.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**
```bash
git add config.py tests/test_config.py
git commit -m "feat: add Pydantic DatasourceConfig for validated, documented configuration"
```

---

## Chunk 6: Phase 6 — Vector Store Abstraction

### Overview
This is the most invasive phase. It introduces a `VectorStore` protocol and `QdrantStore`/`ChromaStore` implementations, then migrates all 6 files that directly import `qdrant_client`. Each file is migrated one at a time, tested, then committed.

---

### Task 13: `VectorStore` protocol + `QdrantStore`

**Files:**
- Create: `vectorstore/__init__.py`
- Create: `vectorstore/base.py`
- Create: `vectorstore/qdrant_store.py`
- Create: `vectorstore/chroma_store.py`
- Create: `vectorstore/registry.py`
- Create: `tests/test_vectorstore.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_vectorstore.py`:
```python
"""Tests for VectorStore abstraction."""
import pytest
from unittest.mock import MagicMock, patch


def test_vectorstore_protocol_has_required_methods():
    from vectorstore.base import VectorStore, Point, ScoredPoint
    import typing
    hints = typing.get_protocol_members(VectorStore)
    for method in ["create_collection", "upsert", "query", "scroll", "set_payload",
                   "collection_exists", "delete_collection"]:
        assert method in hints, f"VectorStore missing: {method}"


def test_qdrant_store_create_collection():
    from vectorstore.qdrant_store import QdrantStore
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.collection_exists.return_value = False
        store = QdrantStore(host="localhost", port=6333)
        store.create_collection("test-col", dim=384)
        mock_client.create_collection.assert_called_once()


def test_qdrant_store_upsert():
    from vectorstore.qdrant_store import QdrantStore
    from vectorstore.base import Point
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        store = QdrantStore(host="localhost", port=6333)
        points = [Point(id=0, vector=[0.1, 0.2], payload={"text": "hello"})]
        store.upsert("col", points)
        mock_client.upsert.assert_called_once()


def test_qdrant_store_query_returns_scored_points():
    from vectorstore.qdrant_store import QdrantStore
    from vectorstore.base import ScoredPoint
    with patch("vectorstore.qdrant_store.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.score = 0.9
        mock_hit.payload = {"text": "result"}
        mock_client.query_points.return_value.points = [mock_hit]
        store = QdrantStore(host="localhost", port=6333)
        results = store.query("col", [0.1, 0.2], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], ScoredPoint)
        assert results[0].score == 0.9


def test_registry_returns_qdrant_store_by_default():
    from vectorstore.registry import get_store
    from vectorstore.qdrant_store import QdrantStore
    with patch("vectorstore.qdrant_store.QdrantClient"):
        store = get_store({"vector_store": "qdrant", "qdrant_host": "localhost", "qdrant_port": 6333})
    assert isinstance(store, QdrantStore)
```

- [ ] **Step 2: Create `vectorstore/base.py`**

```python
"""vectorstore/base.py — VectorStore Protocol + shared dataclasses."""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Point:
    id: int | str
    vector: list[float]
    payload: dict = field(default_factory=dict)


@dataclass
class ScoredPoint:
    id: int | str
    score: float
    payload: dict = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    def create_collection(self, name: str, dim: int) -> None: ...
    def collection_exists(self, name: str) -> bool: ...
    def delete_collection(self, name: str) -> None: ...
    def upsert(self, collection: str, points: list[Point]) -> None: ...
    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]: ...
    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset: Any = None, scroll_filter: Any = None) -> tuple[list, Any]: ...
    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None: ...
    def count(self, collection: str) -> int: ...
```

- [ ] **Step 3: Create `vectorstore/qdrant_store.py`**

```python
"""vectorstore/qdrant_store.py — Qdrant implementation of VectorStore."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vectorstore.base import Point, ScoredPoint


class QdrantStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self._client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, dim: int) -> None:
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def upsert(self, collection: str, points: list[Point]) -> None:
        qdrant_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=qdrant_points)

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        kwargs = dict(collection_name=collection, query=vector, limit=top_k, with_payload=True)
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold
        resp = self._client.query_points(**kwargs)
        return [ScoredPoint(id=h.id, score=h.score, payload=h.payload or {})
                for h in resp.points]

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None) -> tuple[list, any]:
        kwargs = dict(collection_name=collection, limit=limit,
                       with_payload=with_payload, with_vectors=with_vectors)
        if offset is not None:
            kwargs["offset"] = offset
        if scroll_filter is not None:
            kwargs["scroll_filter"] = scroll_filter
        results, next_offset = self._client.scroll(**kwargs)
        # Return as ScoredPoint-like objects (no score for scroll)
        wrapped = [ScoredPoint(id=r.id, score=0.0, payload=r.payload or {})
                   for r in results]
        return wrapped, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        self._client.set_payload(
            collection_name=collection, payload=payload,
            points=[point_id]
        )

    def count(self, collection: str) -> int:
        return self._client.count(collection_name=collection).count

    @property
    def native_client(self) -> QdrantClient:
        """Escape hatch for Qdrant-specific operations not in the protocol."""
        return self._client
```

- [ ] **Step 4: Create `vectorstore/chroma_store.py`**

```python
"""vectorstore/chroma_store.py — ChromaDB implementation of VectorStore."""
from vectorstore.base import Point, ScoredPoint


class ChromaStore:
    """Local in-process ChromaDB — good for development without Docker."""

    def __init__(self, persist_dir: str = ".chroma"):
        try:
            import chromadb
        except ImportError:
            raise ImportError("ChromaStore requires: pip install chromadb")
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict = {}

    def _get_col(self, name: str):
        if name not in self._collections:
            self._collections[name] = self._client.get_collection(name)
        return self._collections[name]

    def create_collection(self, name: str, dim: int) -> None:
        import chromadb.utils.embedding_functions as ef
        col = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        self._collections[name] = col

    def collection_exists(self, name: str) -> bool:
        return any(c.name == name for c in self._client.list_collections())

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)
        self._collections.pop(name, None)

    def upsert(self, collection: str, points: list[Point]) -> None:
        col = self._get_col(collection)
        col.upsert(
            ids=[str(p.id) for p in points],
            embeddings=[p.vector for p in points],
            documents=[p.payload.get("text", "") for p in points],
            metadatas=[{k: v for k, v in p.payload.items() if k != "text"} for p in points],
        )

    def query(self, collection: str, vector: list[float], top_k: int,
               score_threshold: float | None = None) -> list[ScoredPoint]:
        col = self._get_col(collection)
        results = col.query(query_embeddings=[vector], n_results=top_k,
                             include=["documents", "metadatas", "distances"])
        out = []
        for id_, doc, meta, dist in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        ):
            score = 1.0 - dist  # chroma returns L2 distance; convert to similarity
            if score_threshold is not None and score < score_threshold:
                continue
            payload = {"text": doc, **(meta or {})}
            out.append(ScoredPoint(id=id_, score=score, payload=payload))
        return out

    def scroll(self, collection: str, limit: int = 100,
                with_payload: bool = True, with_vectors: bool = False,
                offset=None, scroll_filter=None) -> tuple[list, any]:
        col = self._get_col(collection)
        offset_int = offset or 0
        results = col.get(limit=limit, offset=offset_int,
                           include=["documents", "metadatas"])
        out = [
            ScoredPoint(id=id_, score=0.0, payload={"text": doc, **(meta or {})})
            for id_, doc, meta in zip(
                results["ids"], results["documents"], results["metadatas"]
            )
        ]
        next_offset = offset_int + len(out) if len(out) == limit else None
        return out, next_offset

    def set_payload(self, collection: str, point_id: int | str, payload: dict) -> None:
        col = self._get_col(collection)
        # Chroma requires re-upsert to update metadata
        existing = col.get(ids=[str(point_id)], include=["metadatas", "documents", "embeddings"])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0] or {}
        meta.update({k: v for k, v in payload.items() if k != "text"})
        col.update(ids=[str(point_id)], metadatas=[meta])

    def count(self, collection: str) -> int:
        return self._get_col(collection).count()
```

- [ ] **Step 5: Create `vectorstore/registry.py`**

```python
"""vectorstore/registry.py — Factory for VectorStore implementations."""


def get_store(cfg: dict):
    """Return the appropriate VectorStore for the given config."""
    backend = cfg.get("vector_store", "qdrant")
    if backend == "qdrant":
        from vectorstore.qdrant_store import QdrantStore
        return QdrantStore(host=cfg.get("qdrant_host", "localhost"),
                            port=cfg.get("qdrant_port", 6333))
    if backend == "chroma":
        persist_dir = cfg.get("chroma_persist_dir", ".chroma")
        from vectorstore.chroma_store import ChromaStore
        return ChromaStore(persist_dir=persist_dir)
    raise ValueError(f"Unknown vector_store '{backend}'. Choose: qdrant, chroma")
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_vectorstore.py -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**
```bash
git add vectorstore/ tests/test_vectorstore.py
git commit -m "feat: add VectorStore abstraction with Qdrant and Chroma implementations"
```

---

### Task 14: Migrate `bedrock_rag.py` to use `VectorStore`

**Files:**
- Modify: `bedrock_rag.py`

- [ ] **Step 1: Update `cmd_index` to use VectorStore**

Replace the direct `QdrantClient` creation and `index_chunks` call in `cmd_index`:
```python
# Before
client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
index_chunks(chunks, vectors, client, cfg)

# After
from vectorstore.registry import get_store
from vectorstore.base import Point
store = get_store(vars(cfg))
if store.collection_exists(cfg.collection):
    log(f"Deleting existing collection '{cfg.collection}'")
    store.delete_collection(cfg.collection)
log(f"Creating collection '{cfg.collection}' (dim={cfg.vector_dim})")
store.create_collection(cfg.collection, cfg.vector_dim)
points = [Point(id=c["chunk_id"], vector=v,
                payload={"page": c["page"], "text": c["text"]})
          for c, v in zip(chunks, vectors)]
for start in range(0, len(points), cfg.upsert_batch):
    store.upsert(cfg.collection, points[start:start + cfg.upsert_batch])
log(f"Indexed {len(points)} vectors in '{cfg.collection}'")
```

- [ ] **Step 2: Update `cmd_search` to use VectorStore**

Replace `QdrantClient` usage in `cmd_search`:
```python
from vectorstore.registry import get_store
store = get_store(vars(cfg))
# ... pass store instead of client to _run_search
```

Update `_run_search` signature to accept a `store` instead of `client`:
```python
def _run_search(question, embedder, store, cfg):
    q_vector = next(iter(embedder.embed([question]))).tolist()
    return store.query(cfg.collection, q_vector, cfg.top_k)
```

- [ ] **Step 3: Verify end-to-end with Bedrock dataset**

Run: `python bedrock_rag.py --config datasource_bedrock.json search "What is Amazon Bedrock?"`
Expected: Returns top-k results as before.

- [ ] **Step 4: Commit**
```bash
git add bedrock_rag.py
git commit -m "refactor: migrate bedrock_rag.py to VectorStore abstraction"
```

---

### Task 15: Migrate `kv_inference.py`, `kv_indexer.py`, `kv_background.py`, `access_tracker.py`, `monitoring_dashboard.py`

Each file is migrated in a separate step. For each file:

- [ ] **Step 1: `kv_inference.py`** — Replace `QdrantClient(host, port)` creation and `client.query_points(...)` with `get_store(cfg).query(...)`. The `store` object should be created once and passed down.

- [ ] **Step 2: `kv_indexer.py`** — Replace `QdrantClient` with `get_store(cfg)`. `scroll()` and `set_payload()` map directly. `upsert()` wraps Qdrant points in `vectorstore.base.Point`.

- [ ] **Step 3: `kv_background.py`** — Replace the `QdrantClient` parameter with `VectorStore`. The background worker only calls `set_payload()` and `query()`.

- [ ] **Step 4: `access_tracker.py`** — Replace `QdrantClient` with `VectorStore`. Uses `set_payload()` and `scroll()`.

- [ ] **Step 5: `monitoring_dashboard.py`** — Replace `QdrantClient` with `VectorStore`. Uses `scroll()` and `count()`.

- [ ] **Step 6: Run full smoke test**

```bash
python ask.py --config datasource_bedrock.json "What models does Amazon Bedrock support?"
```
Expected: Returns a coherent answer using the existing indexed Bedrock collection.

- [ ] **Step 7: Remove top-level `qdrant_client` imports from migrated files**

Search for remaining direct imports:
```bash
grep -n "from qdrant_client" bedrock_rag.py kv_inference.py kv_indexer.py kv_background.py access_tracker.py monitoring_dashboard.py
```
Expected: Only `vectorstore/qdrant_store.py` should import `qdrant_client`.

- [ ] **Step 8: Commit**
```bash
git add kv_inference.py kv_indexer.py kv_background.py access_tracker.py monitoring_dashboard.py
git commit -m "refactor: migrate all pipeline files to VectorStore abstraction; qdrant_client isolated to QdrantStore"
```

---

## Chunk 7: Phase 7 — CLI Scaffold + Multi-Collection Server

### Task 16: `kvforge.py` — CLI `init` command

**Files:**
- Create: `kvforge.py`

- [ ] **Step 1: Create `kvforge.py`**

```python
"""kvforge.py — KVForge CLI.

Commands:
  init    Create a new datasource config
  index   Index a source into a collection
  search  Search a collection
  train   Run LoRA fine-tuning
  eval    Compute PRS score
  faqs    Auto-generate FAQs

Usage:
  python kvforge.py init --name my-corpus
  python kvforge.py index --config datasource_my-corpus.json --source ./docs/
  python kvforge.py search --config datasource_my-corpus.json "my query"
"""

import argparse
import json
import os
import sys
from pathlib import Path


def cmd_init(args) -> None:
    """Scaffold a new datasource config interactively."""
    name = args.name
    config_path = f"datasource_{name}.json"
    if Path(config_path).exists() and not args.force:
        print(f"Config already exists: {config_path}. Use --force to overwrite.")
        sys.exit(1)

    cfg = {
        "collection": name,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "vector_store": "qdrant",
        "loader": args.loader,
        "embed_model": args.embed_model,
        "embedder_backend": "fastembed",
        "vector_dim": args.vector_dim,
        "llm_model": args.llm_model,
        "chunk_size": 600,
        "chunk_overlap": 60,
        "embed_batch": 64,
        "upsert_batch": 128,
        "top_k": 5,
        "model_library": {},
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
        "lora_dropout": 0.05,
        "lora_epochs": 3,
        "lora_lr": 0.0002,
        "checkpoint_dir": f"lora_checkpoints/{name}/",
        "version_file": f"{name}_version.json",
        "replay_db": f"{name}_replay.db",
        "gate_threshold": 0.75,
        "prs_threshold": 0.75,
        "prs_weights": {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2},
        "faq_question_key": "question",
        "faq_answer_key": "answer",
        "access_flush_seconds": 300,
        "access_flush_queries": 50,
        "dashboard_port": 8080,
    }

    # Validate before writing
    from config import DatasourceConfig
    DatasourceConfig(**cfg)  # raises ValidationError if invalid

    # Create checkpoint dir
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Created {config_path}")
    print(f"Next steps:")
    print(f"  1. Index your source:  python kvforge.py index --config {config_path} --source <path>")
    print(f"  2. Generate FAQs:      python tools/generate_faqs.py --config {config_path} --output {name}_faqs.json")
    print(f"  3. Train:              python index_and_train.py --config {config_path} --source <path> --faqs {name}_faqs.json")


def cmd_index(args) -> None:
    import json as _json
    from ingestion.registry import get_loader
    from embeddings.registry import get_embedder
    from vectorstore.registry import get_store
    from vectorstore.base import Point

    with open(args.config) as f:
        cfg = _json.load(f)

    loader = get_loader(cfg)
    embedder = get_embedder(cfg)
    store = get_store(cfg)

    print(f"Loading documents from {args.source}...")
    docs = loader.load(args.source)
    print(f"Loaded {len(docs)} chunks")

    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts)

    collection = cfg["collection"]
    if store.collection_exists(collection):
        store.delete_collection(collection)
    store.create_collection(collection, embedder.dim)

    points = [Point(id=i, vector=v, payload={**d["metadata"], "text": d["text"]})
               for i, (d, v) in enumerate(zip(docs, vectors))]
    batch = cfg.get("upsert_batch", 128)
    for start in range(0, len(points), batch):
        store.upsert(collection, points[start:start + batch])
        print(f"  Upserted {min(start + batch, len(points))}/{len(points)}", end="\r")
    print(f"\nIndexed {len(points)} points into '{collection}'")


def cmd_search(args) -> None:
    import json as _json
    from embeddings.registry import get_embedder
    from vectorstore.registry import get_store

    with open(args.config) as f:
        cfg = _json.load(f)

    embedder = get_embedder(cfg)
    store = get_store(cfg)
    vector = embedder.encode([args.query])[0]
    results = store.query(cfg["collection"], vector, top_k=cfg.get("top_k", 5))
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] score={r.score:.4f}")
        print(r.payload.get("text", "")[:300])


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvforge", description="KVForge CLI")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # init
    p_init = sub.add_parser("init", help="Create a new datasource config")
    p_init.add_argument("--name", required=True)
    p_init.add_argument("--loader", default="pdf",
                         choices=["pdf", "markdown", "jsonl", "html", "directory"])
    p_init.add_argument("--embed-model", dest="embed_model",
                         default="BAAI/bge-small-en-v1.5")
    p_init.add_argument("--vector-dim", dest="vector_dim", type=int, default=384)
    p_init.add_argument("--llm-model", dest="llm_model",
                         default="meta-llama/Llama-3.2-3B-Instruct")
    p_init.add_argument("--force", action="store_true")

    # index
    p_idx = sub.add_parser("index", help="Index a source into the collection")
    p_idx.add_argument("--config", required=True)
    p_idx.add_argument("--source", required=True)

    # search
    p_srch = sub.add_parser("search", help="Search the collection")
    p_srch.add_argument("--config", required=True)
    p_srch.add_argument("query")

    args = parser.parse_args()
    if args.command == "init":
        cmd_init(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with a new corpus**

```bash
python kvforge.py init --name test-corpus --loader jsonl --vector-dim 384
```
Expected: Creates `datasource_test-corpus.json` and `lora_checkpoints/test-corpus/`.

- [ ] **Step 3: Commit**
```bash
git add kvforge.py
git commit -m "feat: add kvforge CLI with init/index/search commands"
```

---

### Task 17: Update `datasource_template.json` to reflect all new fields

**Files:**
- Modify: `datasource_template.json`

- [ ] **Step 1: Rewrite `datasource_template.json`**

Replace the entire file contents with a fully-documented template:
```json
{
  "_comment": "KVForge datasource configuration. All fields with defaults are optional.",

  "collection": "my-corpus",
  "vector_store": "qdrant",
  "qdrant_host": "localhost",
  "qdrant_port": 6333,

  "loader": "pdf",
  "_loader_options": ["pdf", "markdown", "jsonl", "html", "directory"],
  "chunk_size": 600,
  "chunk_overlap": 60,
  "embed_batch": 64,
  "upsert_batch": 128,
  "top_k": 5,

  "embedder_backend": "fastembed",
  "_embedder_options": ["fastembed", "sentence_transformers", "openai"],
  "embed_model": "BAAI/bge-small-en-v1.5",
  "vector_dim": 384,

  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "model_library": {},

  "lora_rank": 16,
  "lora_alpha": 32,
  "lora_target_modules": ["q_proj", "k_proj", "v_proj"],
  "lora_dropout": 0.05,
  "lora_epochs": 3,
  "lora_lr": 0.0002,

  "checkpoint_dir": "lora_checkpoints/my-corpus/",
  "version_file": "my-corpus_version.json",
  "replay_db": "my-corpus_replay.db",

  "gate_threshold": 0.75,
  "prs_threshold": 0.75,
  "prs_weights": {"accuracy": 0.5, "calibration": 0.3, "consistency": 0.2},

  "faq_question_key": "question",
  "faq_answer_key": "answer",

  "access_flush_seconds": 300,
  "access_flush_queries": 50,
  "dashboard_port": 8080
}
```

- [ ] **Step 2: Commit**
```bash
git add datasource_template.json
git commit -m "docs: update datasource_template.json with all generalization fields"
```

---

## Phase Summary

| Phase | Tasks | Key Deliverable | Risk |
|---|---|---|---|
| 1 — Safety Net | 1–5 | No breakage; better defaults + auto-discovery | Zero |
| 2 — Loaders | 6–8 | PDF, Markdown, JSONL, HTML, Directory support | Low |
| 3 — Embeddings | 9 | FastEmbed, SentenceTransformers, OpenAI backends | Low |
| 4 — Auto-FAQ + PRS | 10–11 | Self-contained FAQ generation; configurable PRS | Low |
| 5 — Config | 12 | Pydantic validation; clear errors on misconfiguration | Low |
| 6 — VectorStore | 13–15 | Qdrant + Chroma; all files decoupled | Medium |
| 7 — CLI | 16–17 | `kvforge init/index/search`; new template | Low |

**Any phase can be stopped after completion and the system remains fully functional.**
