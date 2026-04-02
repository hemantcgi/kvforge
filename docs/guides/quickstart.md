# KVForge — Quick Start

Get from zero to a working question-answering system in 5 minutes (retrieval-only, no GPU needed).

## Prerequisites

- Python 3.10+
- [Docker](https://docs.docker.com/get-docker/) (for Qdrant vector store)
- Install dependencies:

```bash
pip install qdrant-client fastembed pydantic
```

## 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 2. Initialise a datasource

```bash
python kvforge.py init --name my-corpus --loader markdown
```

This creates `datasource_my-corpus.json` with sensible defaults and a `lora_checkpoints/my-corpus/` directory.

## 3. Index your documents

```bash
./scripts/index.sh datasource_my-corpus.json ./my-docs/
# or equivalently:
python kvforge.py index --config datasource_my-corpus.json --source ./my-docs/
```

## 4. Ask a question

```bash
./scripts/ask.sh datasource_my-corpus.json "What is the return policy?"
# or equivalently:
python ask.py --config datasource_my-corpus.json "What is the return policy?"
```

## 5. Full pipeline (GPU required)

To advance beyond Phase 1 retrieval to Phase 2 (KV cache) and Phase 3 (parametric answering):

```bash
./scripts/run_pipeline.sh datasource_my-corpus.json ./my-docs/ my-corpus_faqs.json
```

See `scripts/README.md` for all individual pipeline steps.

## 6. Using KVForge Studio

KVForge Studio gives you a browser UI to run and monitor the full pipeline without any command-line pipeline scripts.

**Start the portal:**

```bash
python kvforge_portal.py --port 8080
# Open http://localhost:8080
```

From the Studio you can:

1. **Index** — point to a document directory; watch chunk/embed/upsert progress stream live.
2. **LLM Config** — set the local model path, quantization (4bit/8bit), and vLLM endpoint URL.
3. **Sleep-time FAQ Gen** — configure a cloud LLM provider and generate Q&A pairs offline; the generated `faqs.json` is used automatically in the next training step.
4. **Training** — kick off LoRA fine-tuning; GPU availability is checked before the job starts.
5. **KV Recompute** — refresh KV tensors with the updated adapter.
6. **PRS Eval** — score the model and advance the phase if the threshold is met.

Per-use-case settings (GPU ID, model, FAQ count) are stored in `uc_config.json` inside each use-case directory and editable directly from the Studio UI.

## Supported vector stores

| Backend | Config value | When to use |
|---------|-------------|-------------|
| Qdrant (Docker) | `"qdrant"` | Production, multi-corpus |
| ChromaDB | `"chroma"` | Local dev, no Docker |
| FAISS | `"faiss"` | Fully offline, no services |

See `FAQ.md` → Vector Stores for full setup instructions for each backend.

## Next steps

- `docs/guides/architecture.md` — understand the 3-phase pipeline
- `docs/api/config.md` — all config fields and their effects
- `examples/` — three complete end-to-end examples with different datasets and vector stores
- `FAQ.md` — detailed how-to answers for common tasks
