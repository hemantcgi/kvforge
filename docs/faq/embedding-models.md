# Embedding Models

← [Back to FAQ index](../../FAQ.md)

---

### How do I use OpenAI embeddings instead of FastEmbed?

#### Step 1 — Install the OpenAI SDK

```bash
pip install openai
```

#### Step 2 — Update your datasource config

```json
{
  "embedder_backend": "openai",
  "embed_model":      "text-embedding-3-small",
  "vector_dim":       1536,
  "openai_api_key":   "sk-..."
}
```

Or use the environment variable (preferred — keeps secrets out of config files):

```bash
export OPENAI_API_KEY=sk-...
```

OpenAI embedding model reference:

| Model | `vector_dim` | Cost (per 1M tokens) | Notes |
|-------|:------------:|:--------------------:|-------|
| `text-embedding-3-small` | 1536 | ~$0.02 | Best value — strong performance |
| `text-embedding-3-large` | 3072 | ~$0.13 | Highest quality |
| `text-embedding-ada-002` | 1536 | ~$0.10 | Legacy; prefer 3-small |

> **Important:** The `vector_dim` in your config must exactly match the model's output. A mismatch causes an error at index time. KVForge validates this with `validate_embed_dim()` before writing any data.

#### Step 3 — Index and search as normal

```bash
python kvforge.py index --config datasource_my-corpus.json --source ./docs/
python kvforge.py search --config datasource_my-corpus.json "your query"
```

#### Step 4 — Estimate cost before indexing

```python
# Rough cost estimate for a corpus
num_chunks = 5000
avg_words_per_chunk = 120
avg_tokens_per_word = 1.3
total_tokens = num_chunks * avg_words_per_chunk * avg_tokens_per_word
cost_usd = total_tokens / 1_000_000 * 0.02   # text-embedding-3-small rate
print(f"Estimated indexing cost: ${cost_usd:.4f}")
# 5000 chunks → ~$0.016
```

---

### How do I use sentence-transformers embeddings?

#### Step 1 — Install sentence-transformers

```bash
pip install sentence-transformers
```

#### Step 2 — Update your config

```json
{
  "embedder_backend": "sentence_transformers",
  "embed_model":      "BAAI/bge-base-en-v1.5",
  "vector_dim":       768
}
```

#### Step 3 — Verify the model dimension before indexing

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
test_vec = model.encode(["hello world"])
print(f"Dimension: {len(test_vec[0])}")   # must match vector_dim in config
```

KVForge will raise a `ValueError` at index time if the actual dimension does not match `vector_dim`, so misconfiguration is caught before any data is written.

#### Common sentence-transformers models

| Model | `vector_dim` | Speed | Quality | Notes |
|-------|:------------:|:-----:|:-------:|-------|
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Default; best for resource-limited setups |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Better | Good general-purpose choice |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best BAAI | Use when retrieval quality matters most |
| `intfloat/e5-large-v2` | 1024 | Slow | Excellent | Strong on MTEB benchmarks |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Medium | Good | Well-tested, widely used |
| `thenlper/gte-large` | 1024 | Slow | Excellent | Strong on passage retrieval tasks |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | Slow | Excellent | Top performer on many benchmarks |

> **Multilingual corpora:** Use `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (dim 768) or `intfloat/multilingual-e5-large` (dim 1024).

---

### How do I add a custom embedding model?

The `Embedder` protocol requires exactly two things: an `encode(texts)` method that returns `list[list[float]]`, and a `dim` property that returns the vector dimension.

#### Step 1 — Create the embedder class

```python
# embeddings/cohere_embedder.py
import os


class CohereEmbedder:
    """Cohere Embed API backend."""

    def __init__(self, model_name: str = "embed-english-v3.0",
                 dim: int = 1024,
                 api_key: str | None = None,
                 input_type: str = "search_document"):
        try:
            import cohere
        except ImportError:
            raise ImportError("CohereEmbedder requires: pip install cohere")
        self._client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self._model_name = model_name
        self._dim = dim
        self._input_type = input_type

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Cohere has a batch limit of 96 texts per call
        all_vectors = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embed(
                texts=batch,
                model=self._model_name,
                input_type=self._input_type,
            )
            all_vectors.extend(resp.embeddings)
        return all_vectors

    @property
    def dim(self) -> int:
        return self._dim
```

#### Step 2 — Register in the factory

Open `embeddings/registry.py` and add before the final `raise ValueError`:

```python
if backend == "cohere":
    from embeddings.cohere_embedder import CohereEmbedder
    return CohereEmbedder(
        model_name=model_name,
        dim=dim,
        api_key=cfg.get("cohere_api_key"),
        input_type=cfg.get("cohere_input_type", "search_document"),
    )
```

#### Step 3 — Update your config

```json
{
  "embedder_backend": "cohere",
  "embed_model":      "embed-english-v3.0",
  "vector_dim":       1024,
  "cohere_api_key":   "..."
}
```

#### Step 4 — Validate before indexing

```python
from embeddings.registry import get_embedder

cfg = {"embedder_backend": "cohere", "embed_model": "embed-english-v3.0",
       "vector_dim": 1024, "cohere_api_key": "..."}
embedder = get_embedder(cfg)
vecs = embedder.encode(["test sentence"])
assert len(vecs) == 1
assert len(vecs[0]) == embedder.dim == 1024
print("Embedder OK")
```

---

### Can I use different embedding models for different collections?

Yes — each datasource config is completely independent. The embedding model is baked into the collection at index time: the vector dimension stored in Qdrant must match the model output. You cannot mix embedding models within a single collection.

```
datasource_legal.json
  embedder_backend: sentence_transformers
  embed_model:      legal-bert-base-uncased
  vector_dim:       768
  collection:       legal-docs

datasource_code.json
  embedder_backend: openai
  embed_model:      text-embedding-3-small
  vector_dim:       1536
  collection:       code-docs

datasource_general.json
  embedder_backend: fastembed
  embed_model:      BAAI/bge-small-en-v1.5
  vector_dim:       384
  collection:       general-docs
```

Each corpus runs completely independently — separate collections, separate version files, separate replay databases, separate LoRA checkpoints. They can share the same Qdrant instance and the same GPU server.

---

← [Back to FAQ index](../../FAQ.md)
