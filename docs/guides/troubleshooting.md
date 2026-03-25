# Troubleshooting

## GPU / Memory

### `CUDA out of memory` during KV computation or LoRA training

Reduce batch sizes in the config:
```json
{
  "embed_batch": 16,
  "upsert_batch": 32
}
```

Or use 8-bit quantization (requires `bitsandbytes`):
```json
{
  "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
  "load_in_8bit": true
}
```

### `No CUDA device found` / falls back to CPU

KV computation and LoRA training work on CPU but are very slow. For production, a GPU is required.
For development / testing, CPU is fine for indexing and retrieval (Phase 1 only).

---

## Vector Store

### Qdrant: `Connection refused` / `Failed to connect to localhost:6333`

Qdrant is not running. Start it:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Check your config has the correct host/port:
```json
{ "qdrant_host": "localhost", "qdrant_port": 6333 }
```

### ChromaDB: `ValueError: Could not connect to tenant` or `tenant does not exist`

The `chroma_persist_dir` in your config does not match where data was previously stored.
Set it consistently:
```json
{ "chroma_persist_dir": ".chroma/my-corpus" }
```

### FAISS: `FileNotFoundError: .faiss/my-corpus.index not found`

The FAISS index files were deleted or the `faiss_persist_dir` changed. Re-index:
```bash
./scripts/index.sh datasource_my-corpus.json ./my-docs/
```

---

## PRS / Training

### PRS is not improving across training rounds

1. **Check FAQ quality** — FAQs that are too easy (single-word answers) or too vague don't train well.
   Generate better FAQs:
   ```bash
   ./scripts/generate_faqs.sh datasource_my-corpus.json my-corpus_faqs.json 100
   ```

2. **Lower the threshold** — In your config: `"prs_threshold": 0.65`

3. **Increase training epochs** — `"lora_epochs": 5`

4. **Check model size** — Small models (1B) plateau quickly. Try `meta-llama/Llama-3.2-3B-Instruct`.

### `ValidationError: prs_threshold must be between 0 and 1`

The PRS threshold must be a float between 0 and 1. Default is 0.75.

---

## KV Cache

### KV shape mismatch: `Expected kv shape ... got ...`

The LLM model was changed after KV tensors were computed. Recompute:
```bash
./scripts/compute_kv.sh datasource_my-corpus.json
```

### `KeyError: 'kv_tensors'` in payload

KV tensors have not been computed for this chunk yet. Run:
```bash
./scripts/compute_kv.sh datasource_my-corpus.json
```

---

## Import / Module Errors

### `ModuleNotFoundError: No module named 'pipeline'`

The `pipeline/` package was not found. Run scripts from the repo root:
```bash
cd /path/to/smartqdrant
python -m pipeline.kv_indexer ...
```

### `ModuleNotFoundError: No module named 'fastembed'`

Install dependencies:
```bash
pip install -r requirements_gpu.txt
```

---

## HuggingFace / Model Loading

### `OSError: You are trying to access a gated repo` (Llama 3)

You need a HuggingFace token:
```bash
huggingface-cli login
# or set environment variable:
export HF_TOKEN=hf_...
```

### Slow model download on first run

Models are cached in `~/.cache/huggingface/`. Subsequent runs are fast.
For air-gapped environments, pre-download the model and set:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```
