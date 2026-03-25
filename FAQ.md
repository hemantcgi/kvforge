# SmartQdrant — Frequently Asked Questions

---

## Table of Contents

**Vector Stores**
- [How do I use this with ChromaDB instead of Qdrant?](#how-do-i-use-this-with-chromadb-instead-of-qdrant)
- [How do I add support for Pinecone, Weaviate, or another vector database?](#how-do-i-add-support-for-pinecone-weaviate-or-another-vector-database)
- [Can I use an existing Qdrant collection I already have?](#can-i-use-an-existing-qdrant-collection-i-already-have)

**Language Models**
- [How do I use my own LLM for KV computation?](#how-do-i-use-my-own-llm-for-kv-computation)
- [How do I use a gated model like Llama 3 that requires a HuggingFace token?](#how-do-i-use-a-gated-model-like-llama-3-that-requires-a-huggingface-token)
- [Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?](#can-i-use-an-api-hosted-llm-openai-anthropic-gemini)
- [Can I run this without a GPU?](#can-i-run-this-without-a-gpu)

**Embedding Models**
- [How do I use OpenAI embeddings instead of FastEmbed?](#how-do-i-use-openai-embeddings-instead-of-fastembed)
- [How do I use sentence-transformers embeddings?](#how-do-i-use-sentence-transformers-embeddings)
- [How do I add a custom embedding model?](#how-do-i-add-a-custom-embedding-model)
- [Can I use different embedding models for different collections?](#can-i-use-different-embedding-models-for-different-collections)

**Document Ingestion**
- [How do I index Markdown documentation?](#how-do-i-index-markdown-documentation)
- [How do I index a JSONL dataset?](#how-do-i-index-a-jsonl-dataset)
- [How do I index HTML pages or web content?](#how-do-i-index-html-pages-or-web-content)
- [How do I index an entire directory of files?](#how-do-i-index-an-entire-directory-of-files)
- [How do I add support for a custom document format?](#how-do-i-add-support-for-a-custom-document-format)

**KV Cache & Phases**
- [What exactly is stored in the KV cache payload?](#what-exactly-is-stored-in-the-kv-cache-payload)
- [How does KV injection work under the hood?](#how-does-kv-injection-work-under-the-hood)
- [Why do some queries fall back to text-in-context even in Phase 2?](#why-do-some-queries-fall-back-to-text-in-context-even-in-phase-2)
- [How do I manually advance or roll back the phase?](#how-do-i-manually-advance-or-roll-back-the-phase)

**Training & PRS**
- [How do I tune the PRS threshold?](#how-do-i-tune-the-prs-threshold)
- [My PRS is not improving across training rounds. What do I do?](#my-prs-is-not-improving-across-training-rounds-what-do-i-do)
- [How do I bring my own FAQs for PRS evaluation?](#how-do-i-bring-my-own-faqs-for-prs-evaluation)
- [How do I change the PRS scoring weights?](#how-do-i-change-the-prs-scoring-weights)

**Multi-Corpus & Production**
- [Can I run multiple independent corpora on the same instance?](#can-i-run-multiple-independent-corpora-on-the-same-instance)
- [How do I keep KV tensors fresh when I update my documents?](#how-do-i-keep-kv-tensors-fresh-when-i-update-my-documents)
- [How do I monitor what is happening at runtime?](#how-do-i-monitor-what-is-happening-at-runtime)
- [How do I reset everything and start over?](#how-do-i-reset-everything-and-start-over)
- [What are the GPU memory requirements?](#what-are-the-gpu-memory-requirements)

---

## Vector Stores

### How do I use this with ChromaDB instead of Qdrant?

Set `vector_store` to `"chroma"` in your config and optionally set `chroma_persist_dir`:

```bash
python smartqdrant.py init --name my-corpus
```

Then edit `datasource_my-corpus.json`:

```json
{
  "vector_store": "chroma",
  "chroma_persist_dir": ".chroma/my-corpus"
}
```

Install ChromaDB if you haven't:

```bash
pip install chromadb
```

That's it — `smartqdrant.py index` and `search` work identically. ChromaDB runs in-process with no Docker required, which makes it convenient for local development. Qdrant is recommended for production because it supports filtering on payload fields (used by `kv_indexer.py compute-kv`).

> **Note:** KV cache computation (`kv_indexer.py compute-kv`) uses Qdrant-specific scroll filters when `vector_store = "qdrant"`. With ChromaDB, full-collection scrolls are used instead, which is slower for large collections.

---

### How do I add support for Pinecone, Weaviate, or another vector database?

Implement the `VectorStore` protocol from `vectorstore/base.py`:

```python
# vectorstore/pinecone_store.py
from vectorstore.base import Point, ScoredPoint


class PineconeStore:
    def __init__(self, api_key: str, index_name: str, environment: str):
        from pinecone import Pinecone
        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

    def create_collection(self, name: str, dim: int) -> None:
        # Pinecone indexes are pre-created — no-op or raise
        pass

    def collection_exists(self, name: str) -> bool:
        return True  # assume index exists

    def delete_collection(self, name: str) -> None:
        self._index.delete(delete_all=True)

    def upsert(self, collection: str, points: list[Point]) -> None:
        vectors = [{"id": str(p.id), "values": p.vector, "metadata": p.payload}
                   for p in points]
        self._index.upsert(vectors=vectors)

    def query(self, collection: str, vector: list[float], top_k: int,
              score_threshold=None) -> list[ScoredPoint]:
        results = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [ScoredPoint(id=m.id, score=m.score, payload=m.metadata or {})
                for m in results.matches
                if score_threshold is None or m.score >= score_threshold]

    def scroll(self, collection, limit=100, with_payload=True,
               with_vectors=False, offset=None, scroll_filter=None):
        # Pinecone does not support full-collection scan — return empty
        return [], None

    def set_payload(self, collection: str, point_id, payload: dict) -> None:
        self._index.update(id=str(point_id), set_metadata=payload)

    def count(self, collection: str) -> int:
        return self._index.describe_index_stats()["total_vector_count"]
```

Then register it in `vectorstore/registry.py`:

```python
if backend == "pinecone":
    from vectorstore.pinecone_store import PineconeStore
    return PineconeStore(
        api_key=cfg["pinecone_api_key"],
        index_name=cfg["collection"],
        environment=cfg.get("pinecone_environment", "us-east1-gcp"),
    )
```

And add `"vector_store": "pinecone"` to your datasource config.

---

### Can I use an existing Qdrant collection I already have?

Yes. Point the config at your existing collection and host:

```json
{
  "collection":   "your-existing-collection",
  "qdrant_host":  "your-qdrant-host",
  "qdrant_port":  6333,
  "embed_model":  "BAAI/bge-small-en-v1.5",
  "vector_dim":   384
}
```

The `embed_model` and `vector_dim` must match whatever was used to embed your existing vectors.

Then backfill KV tensors without re-indexing your content:

```bash
# Compute KV tensors for all points that don't have one yet
python kv_indexer.py --config my_existing.json compute-kv

# Or only for a specific source file
python kv_indexer.py --config my_existing.json compute-kv --source-file mydoc.pdf
```

Your vectors and all existing payload fields are untouched. SmartQdrant only adds the fields listed in [What exactly is stored in the KV cache payload?](#what-exactly-is-stored-in-the-kv-cache-payload).

---

## Language Models

### How do I use my own LLM for KV computation?

Set `llm_model` to any HuggingFace causal LM model ID in your config:

```json
{
  "llm_model": "mistralai/Mistral-7B-Instruct-v0.3"
}
```

```json
{
  "llm_model": "google/gemma-2-2b-it"
}
```

```json
{
  "llm_model": "Qwen/Qwen2.5-3B-Instruct"
}
```

SmartQdrant auto-discovers the KV tensor shape (`[num_layers, 2, num_kv_heads, head_dim]`) from the HuggingFace model config — no manual shape configuration needed. You can verify what shape will be used:

```python
import model_loader

cfg = {"llm_model": "mistralai/Mistral-7B-Instruct-v0.3"}
model_loader.init(cfg)
num_layers, num_kv_heads, head_dim = model_loader.get_kv_shape(cfg)
print(f"KV shape: [{num_layers}, 2, {num_kv_heads}, {head_dim}]")
```

**Requirements:** The model must be a standard HuggingFace `AutoModelForCausalLM` with `use_cache=True` support (all decoder-only transformers support this).

**LoRA target modules:** SmartQdrant auto-detects which attention projection names exist in the model. If your model uses `q_proj`/`k_proj`/`v_proj` (Llama family) or `query_key_value` (Falcon, StarCoder), no config change is needed. For unusual architectures, set explicitly:

```json
{
  "lora_target_modules": ["c_attn"]
}
```

---

### How do I use a gated model like Llama 3 that requires a HuggingFace token?

Add `hf_token` to your config:

```json
{
  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "hf_token":  "hf_xxxxxxxxxxxxxxxxxxxx"
}
```

Or set the environment variable before running:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
python index_and_train.py document.pdf --config my_config.json
```

You must have accepted the model's license on [huggingface.co](https://huggingface.co) before the token will grant access.

---

### Can I use an API-hosted LLM (OpenAI, Anthropic, Gemini)?

**For KV computation and LoRA training: No.** These phases require a local HuggingFace model because they need access to internal attention tensors (`past_key_values`) and gradient computation. API providers do not expose these.

**For the text-in-context fallback path (Phase 1 only):** You can adapt `kv_inference.generate_text_in_context()` to call an API instead. This is a one-function change and gives you a usable Phase 1 RAG system without any GPU. KV injection and LoRA fine-tuning (Phases 2–3) require a local model.

A practical hybrid: use OpenAI embeddings for retrieval quality, a local small model for KV computation and LoRA, and the API LLM only for the final text-in-context answer generation.

---

### Can I run this without a GPU?

**Indexing and search:** Fully CPU-compatible. `smartqdrant.py index` and `search` work without a GPU.

**KV tensor computation:** Requires a GPU. Each chunk needs one LLM forward pass. On CPU this would take hours for a large corpus.

**LoRA training:** Requires a GPU.

**PRS evaluation and monitoring:** CPU-compatible.

For development or testing, you can run the full test suite (76 tests) on CPU in under 30 seconds — all GPU-dependent modules are mocked.

---

## Embedding Models

### How do I use OpenAI embeddings instead of FastEmbed?

```json
{
  "embedder_backend": "openai",
  "embed_model":      "text-embedding-3-small",
  "vector_dim":       1536,
  "openai_api_key":   "sk-..."
}
```

Or omit `openai_api_key` and set `OPENAI_API_KEY` in your environment. Install the dependency:

```bash
pip install openai
```

For `text-embedding-3-large`, set `vector_dim: 3072`. For `text-embedding-ada-002`, set `vector_dim: 1536`.

> **Cost note:** OpenAI embeddings are billed per token. For large corpora, FastEmbed (`BAAI/bge-*` models) runs locally for free and performs competitively on most retrieval benchmarks.

---

### How do I use sentence-transformers embeddings?

```json
{
  "embedder_backend": "sentence_transformers",
  "embed_model":      "sentence-transformers/all-mpnet-base-v2",
  "vector_dim":       768
}
```

Install the dependency:

```bash
pip install sentence-transformers
```

Any model on the HuggingFace Hub that works with `SentenceTransformer(model_name)` is supported. Common choices:

| Model | `vector_dim` | Notes |
|-------|:------------:|-------|
| `BAAI/bge-small-en-v1.5` | 384 | Fast, good quality |
| `BAAI/bge-base-en-v1.5` | 768 | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | Best quality |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Good general purpose |
| `intfloat/e5-large-v2` | 1024 | Strong retrieval model |

---

### How do I add a custom embedding model?

Implement the `Embedder` protocol (`embeddings/base.py`) — two methods, that's it:

```python
# embeddings/my_embedder.py
class MyEmbedder:
    def __init__(self, model_name: str, dim: int):
        self._model = load_my_model(model_name)
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed(texts)  # must return list[list[float]]

    @property
    def dim(self) -> int:
        return self._dim
```

Register it in `embeddings/registry.py`:

```python
if backend == "my_backend":
    from embeddings.my_embedder import MyEmbedder
    return MyEmbedder(model_name=model_name, dim=dim)
```

Set in your config:

```json
{
  "embedder_backend": "my_backend",
  "embed_model":      "my-model-name",
  "vector_dim":       512
}
```

---

### Can I use different embedding models for different collections?

Yes — each datasource config is fully independent. You can have:

```
datasource_legal.json       → embedder_backend: sentence_transformers, embed_model: legal-bert
datasource_code.json        → embedder_backend: openai, embed_model: text-embedding-3-small
datasource_internal.json    → embedder_backend: fastembed, embed_model: BAAI/bge-small-en-v1.5
```

Each config also gets its own collection, version file, replay DB, and checkpoint directory — no state is shared between them.

---

## Document Ingestion

### How do I index Markdown documentation?

```bash
python smartqdrant.py init --name docs --loader markdown
python smartqdrant.py index --config datasource_docs.json --source ./docs/guide.md
```

The Markdown loader splits on `#`, `##`, and `###` headings. Sections with fewer than 10 words are skipped. To index a whole directory of `.md` files, use `loader: directory` instead:

```bash
python smartqdrant.py init --name docs --loader directory
python smartqdrant.py index --config datasource_docs.json --source ./docs/
```

The directory loader dispatches to the right loader per file extension automatically.

---

### How do I index a JSONL dataset?

```json
{
  "loader":        "jsonl",
  "jsonl_text_key": "content"
}
```

Each line of the JSONL file must be a JSON object. The loader reads the field named by `jsonl_text_key` as the chunk text (default `"text"`). All other fields are stored in metadata.

Example JSONL file:

```jsonl
{"content": "First document chunk text here.", "id": 1, "category": "science"}
{"content": "Second document chunk text here.", "id": 2, "category": "history"}
```

---

### How do I index HTML pages or web content?

```json
{
  "loader": "html"
}
```

```bash
python smartqdrant.py index --config datasource_web.json --source ./pages/article.html
```

The HTML loader strips all tags using `BeautifulSoup` and splits the cleaned text into overlapping word-level chunks. Install the dependency:

```bash
pip install beautifulsoup4
```

For web crawling (fetching URLs), pre-download the HTML files first and then index the directory with `loader: directory`.

---

### How do I index an entire directory of files?

```json
{
  "loader": "directory"
}
```

```bash
python smartqdrant.py index --config datasource_mixed.json --source ./corpus/
```

The directory loader recursively walks the directory and dispatches each file to the right loader by extension:

| Extension | Loader used |
|-----------|-------------|
| `.pdf` | PDFLoader |
| `.md`, `.markdown` | MarkdownLoader |
| `.jsonl` | JSONLLoader |
| `.html`, `.htm` | HTMLLoader |
| Other | Skipped |

---

### How do I add support for a custom document format?

Implement the `DocumentLoader` protocol (`ingestion/base.py`):

```python
# ingestion/csv_loader.py
import csv
from pathlib import Path


class CSVLoader:
    """Load rows from a CSV file as individual chunks."""

    def __init__(self, text_column: str = "text", min_words: int = 5):
        self.text_column = text_column
        self.min_words = min_words

    def load(self, source: str) -> list[dict]:
        docs = []
        with open(source, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = row.get(self.text_column, "")
                if len(text.split()) < self.min_words:
                    continue
                docs.append({
                    "text": text,
                    "metadata": {"source": Path(source).name, "row": i, "chunk_id": i},
                })
        return docs
```

Register it in `ingestion/registry.py`:

```python
if loader_type == "csv":
    from ingestion.csv_loader import CSVLoader
    return CSVLoader(text_column=cfg.get("csv_text_column", "text"))
```

Add `"loader": "csv"` to your datasource config.

---

## KV Cache & Phases

### What exactly is stored in the KV cache payload?

For each chunk, SmartQdrant runs one LLM forward pass and stores the mean-pooled key-value tensors. Concretely:

- **Raw shape from model:** `[num_layers, 2, num_heads, seq_len, head_dim]`
- **After mean pooling over `seq_len`:** `[num_layers, 2, num_kv_heads, head_dim]`
- **Storage format:** float16, serialized to base64 string

For Llama 3.2 3B (28 layers, 8 KV heads, head_dim 128) a single chunk's KV tensor is approximately:
`28 × 2 × 8 × 128 × 2 bytes = ~115 KB`

This is stored in Qdrant's payload alongside the text. Qdrant payloads are stored in memory by default, so plan for roughly `num_chunks × 115 KB` of additional RAM for a Llama 3.2 3B backend.

---

### How does KV injection work under the hood?

At query time, for each retrieved chunk:

1. Deserialize the base64 KV tensor from the payload
2. Stack all chunk KV tensors into a single `past_key_values` cache
3. Pass the stacked cache as `past_key_values` to `model.generate()`

The LLM receives the chunk context without re-tokenizing or re-encoding it — the attention computation over the chunks is skipped entirely. The model only processes the query tokens against the pre-loaded KV state.

```
Normal RAG:  query tokens + context tokens → attention → generation
KV inject:   query tokens → attention against pre-loaded KV → generation
             (context tokens never re-processed)
```

This is the same principle as prefix KV caching used in inference engines like vLLM, applied here at the retrieval-augmentation layer. See PromptCache [8] in the references.

---

### Why do some queries fall back to text-in-context even in Phase 2?

A fallback occurs when any retrieved chunk has a stale or missing KV tensor:

- **`kv_version` is null:** The chunk was indexed before KV computation ran. Run `python kv_indexer.py compute-kv` to backfill.
- **`kv_version` < current LoRA version:** A new LoRA adapter was trained and the chunk's KV tensor was computed with an older one. Background workers heal these automatically after retrieval — the next query for the same chunks will use KV injection.
- **`kv_cache` field missing:** The chunk was added directly to Qdrant without going through SmartQdrant's indexer. Backfill with `compute-kv`.

You can check which chunks are stale:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsNullCondition

client = QdrantClient("localhost", port=6333)
results, _ = client.scroll(
    collection_name="my-corpus",
    scroll_filter=Filter(must=[IsNullCondition(is_null={"key": "kv_version"})]),
    limit=10,
)
print(f"{len(results)} chunks with null kv_version")
```

---

### How do I manually advance or roll back the phase?

```python
import json, version as ver

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)
ver.init(cfg)

# Check current phase
print("Current phase:", ver.get_phase())

# Advance to Phase 2
ver.activate_phase_2()

# Advance to Phase 3
ver.activate_phase_3()

# Roll back to Phase 1 (disables KV injection and confidence gate)
ver.set_phase(1)
```

Rollbacks are useful when you want to A/B test phases or if you notice quality regression after a phase transition.

---

## Training & PRS

### How do I tune the PRS threshold?

The default threshold is `0.75`. Lower it if Phase 2 never activates; raise it to require higher model quality before enabling KV injection.

```json
{
  "prs_threshold": 0.70
}
```

Typical PRS ranges:
| Range | Interpretation |
|-------|---------------|
| < 0.60 | Model hasn't learned the corpus well — more data or training epochs needed |
| 0.60–0.75 | Reasonable but below threshold — consider more LoRA epochs or a better base model |
| 0.75–0.85 | Good — Phase 2 activates; KV injection is reliable |
| > 0.85 | Excellent — Phase 3 is appropriate; model answers many queries from weights |

---

### My PRS is not improving across training rounds. What do I do?

Try in order:

1. **More FAQs.** PRS is evaluated against FAQs. Auto-generate more with `tools/generate_faqs.py --n 100`.

2. **More training epochs.** Increase `lora_epochs` (default 3):
   ```json
   { "lora_epochs": 6 }
   ```

3. **Higher LoRA rank.** Increases model capacity at the cost of memory:
   ```json
   { "lora_rank": 32, "lora_alpha": 64 }
   ```

4. **Better base model.** Larger models (7B+) have higher capacity. A stronger starting point learns the corpus faster.

5. **Check chunk quality.** If chunks are too short (< 30 words) or too long (> 800 words), the model learns less efficiently. Tune `chunk_size` and `chunk_overlap`.

6. **Check the FAQ quality.** If auto-generated FAQs contain hallucinations, PRS accuracy will be artificially low. Review the FAQ file before running evaluation.

---

### How do I bring my own FAQs for PRS evaluation?

Any JSON file where each object has a question and answer field works:

```json
[
  {"question": "What is the return policy?", "answer": "30 days"},
  {"question": "How do I contact support?", "answer": "Email support@example.com"}
]
```

If your fields are named differently (e.g. `"query"` and `"ground_truth"`), set:

```json
{
  "faq_question_key": "query",
  "faq_answer_key":   "ground_truth"
}
```

Run evaluation:

```bash
python prs_evaluator.py --config datasource_my-corpus.json --faqs my_faqs.json
```

---

### How do I change the PRS scoring weights?

The default formula is `0.5 × accuracy + 0.3 × calibration + 0.2 × consistency`. Change the weights in your config:

```json
{
  "prs_weights": {
    "accuracy":    0.7,
    "calibration": 0.2,
    "consistency": 0.1
  }
}
```

Weights must sum to 1.0. To disable a component entirely, set its weight to 0.

---

## Multi-Corpus & Production

### Can I run multiple independent corpora on the same instance?

Yes. Each corpus gets its own datasource config and its own Qdrant collection. State (version file, replay buffer, LoRA checkpoints) is completely isolated:

```
datasource_legal.json       → collection: legal-docs,  version_file: legal_version.json
datasource_hr.json          → collection: hr-docs,     version_file: hr_version.json
datasource_engineering.json → collection: eng-docs,    version_file: eng_version.json
```

They share the same Qdrant instance and, if desired, the same base LLM (the singleton in `model_loader.py` caches by checkpoint path — different LoRA adapters are hot-swapped as needed).

---

### How do I keep KV tensors fresh when I update my documents?

When you add or change documents, re-index the changed files and then recompute KV tensors for the new or updated chunks:

```bash
# Re-index a specific updated file (replaces its chunks)
python smartqdrant.py index --config datasource_my-corpus.json --source ./updated_doc.pdf

# Backfill KV tensors for chunks that don't have them (new chunks)
python kv_indexer.py --config datasource_my-corpus.json compute-kv

# After the next training round, recompute KV tensors for stale chunks
python kv_indexer.py --config datasource_my-corpus.json compute-kv --stale-version <new_lora_version>
```

Background workers also heal stale chunks lazily — any chunk retrieved in a query is automatically queued for KV recomputation if its `kv_version` is outdated.

---

### How do I monitor what is happening at runtime?

**Dashboard** (recommended):
```bash
python monitoring_dashboard.py --config datasource_my-corpus.json
# Open http://localhost:8080
```

Shows phase, LoRA version, tier distribution, and top accessed chunks in real time.

**Access reports:** Generated periodically in `access_report.json`. Contains tier counts, most-accessed pages, and parametric answer rate.

**Version file:** `cat my-corpus_version.json` shows the current phase, PRS history, and LoRA checkpoint path.

**Logs:** `kv_background.py` prints to stdout. Run with `nohup` and redirect output:
```bash
nohup python kv_background.py --config datasource_my-corpus.json > kv_background.log 2>&1 &
```

---

### How do I reset everything and start over?

```bash
# Delete the Qdrant collection
python -c "
from qdrant_client import QdrantClient
c = QdrantClient('localhost', port=6333)
c.delete_collection('my-corpus')
"

# Remove local state files
rm -f my-corpus_version.json my-corpus_replay.db
rm -rf lora_checkpoints/my-corpus/

# Re-index from scratch
python smartqdrant.py index --config datasource_my-corpus.json --source ./my_document.pdf
```

---

### What are the GPU memory requirements?

Memory depends on the model size. Benchmarked on AWS g5.xlarge (NVIDIA A10G, 24 GB VRAM):

| Model | Parameters | VRAM (inference) | VRAM (LoRA training) |
|-------|:----------:|:----------------:|:--------------------:|
| TinyLlama-1.1B | 1.1B | ~2 GB | ~4 GB |
| Llama-3.2-3B | 3B | ~6 GB | ~10 GB |
| Mistral-7B | 7B | ~14 GB | ~20 GB |
| Llama-3.1-8B | 8B | ~16 GB | ~24 GB |

SmartQdrant loads the model in `float16` by default. For larger models on smaller GPUs, enable 4-bit quantization by setting `load_in_4bit: true` in your config (requires `bitsandbytes`). The A10G (24 GB) comfortably handles Llama 3.2 3B for both KV computation and LoRA training.

---

*Have a question not covered here? Open an issue at [github.com/hemantcgi/smartqdrant](https://github.com/hemantcgi/smartqdrant/issues).*
