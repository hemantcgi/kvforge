# Embeddings API Reference

## Protocol

All embedder backends implement the `Embedder` Protocol from `embeddings/base.py`.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `(texts: list[str]) → list[list[float]]` | Encode texts to embedding vectors |

### Property

| Property | Type | Description |
|----------|------|-------------|
| `dim` | `int` | Embedding vector dimension |

## Backends

### FastEmbed (`embedder_backend: "fastembed"`)

- **Install:** `pip install fastembed`
- **Config fields:** `embed_model` (default: `"BAAI/bge-small-en-v1.5"`)
- **Notes:** Fast, no GPU needed, models downloaded automatically

```json
{ "embedder_backend": "fastembed", "embed_model": "BAAI/bge-small-en-v1.5", "vector_dim": 384 }
```

### SentenceTransformers (`embedder_backend: "sentence_transformers"`)

- **Install:** `pip install sentence-transformers`
- **Config fields:** `embed_model` (default: `"BAAI/bge-small-en-v1.5"`)

```json
{ "embedder_backend": "sentence_transformers", "embed_model": "all-MiniLM-L6-v2", "vector_dim": 384 }
```

### OpenAI (`embedder_backend: "openai"`)

- **Install:** `pip install openai`
- **Config fields:** `embed_model` (default: `"text-embedding-3-small"`), `openai_api_key`
- **Requires:** `OPENAI_API_KEY` environment variable or `openai_api_key` in config

```json
{ "embedder_backend": "openai", "embed_model": "text-embedding-3-small", "vector_dim": 1536 }
```
