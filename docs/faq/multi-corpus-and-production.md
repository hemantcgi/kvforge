# Multi-Corpus & Production

← [Back to FAQ index](../../FAQ.md)

---

### Can I run multiple independent corpora on the same instance?

Yes. Each corpus is fully isolated by its datasource config. The only shared resource is the Qdrant instance (or ChromaDB process) and the GPU.

#### Typical multi-corpus layout

```
project/
├── datasource_legal.json          ← legal document corpus
├── datasource_hr.json             ← HR policy corpus
├── datasource_engineering.json    ← technical documentation
│
├── legal_version.json             ← phase/PRS state for legal
├── hr_version.json
├── engineering_version.json
│
├── legal_replay.db                ← LoRA training replay buffer
├── hr_replay.db
├── engineering_replay.db
│
└── lora_checkpoints/
    ├── legal/                     ← LoRA adapter for legal
    ├── hr/
    └── engineering/
```

#### Running operations per corpus

```bash
# Index each corpus independently
python kvforge.py index --config datasource_legal.json --source ./legal_docs/
python kvforge.py index --config datasource_hr.json    --source ./hr_policies/

# Train each corpus — these are sequential (one GPU)
python index_and_train.py dummy.pdf --config datasource_legal.json       --faqs legal_faqs.json --skip-index
python index_and_train.py dummy.pdf --config datasource_engineering.json --faqs eng_faqs.json   --skip-index

# Query from the right corpus
python kvforge.py search --config datasource_legal.json       "What is the arbitration clause?"
python kvforge.py search --config datasource_engineering.json "How do I configure OAuth?"
```

#### Shared base model, separate LoRA adapters

`model_loader.py` caches the model by checkpoint path. Different LoRA adapters hot-swap on top of the same base model weights. The base model (e.g. Llama 3.2 3B) is loaded once per process; each corpus's adapter is merged in when that corpus's pipeline runs.

---

### How do I keep KV tensors fresh when I update my documents?

KV tensors are tied to both the document content and the LoRA adapter version. Changes to either require recomputation.

#### When you add new documents

```bash
# 1. Index the new documents (adds chunks, no KV yet)
python kvforge.py index --config datasource_my-corpus.json --source ./new_docs/

# 2. Compute KV tensors for the new chunks (they have kv_version=null)
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

#### When you update existing documents

There is no partial update — re-index the changed file (which replaces its chunks) then recompute:

```bash
# Re-index deletes old chunks and creates new ones
python kvforge.py index --config datasource_my-corpus.json --source ./updated_file.pdf

# Compute KV for the new chunks
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

#### After a LoRA training round

All existing KV tensors become stale because the model weights changed. Background workers heal them lazily (each chunk is recomputed the first time it is retrieved), but you can pre-heal the entire collection:

```bash
# Get the current LoRA version number
python -c "import json, version as ver; ver.init(json.load(open('datasource_my-corpus.json'))); print(ver.get_lora_version())"

# Recompute KV for all chunks with an outdated version
python kv_indexer.py --config datasource_my-corpus.json compute-kv \
  --stale-version <current_lora_version>
```

#### Continuous update strategy

For corpora that are updated frequently (e.g. daily news ingestion), run this after each update batch:

```bash
#!/bin/bash
# daily_update.sh
python kvforge.py index --config $CONFIG --source $NEW_DOCS_DIR
python kv_indexer.py --config $CONFIG compute-kv
echo "Update complete at $(date)"
```

---

### How do I monitor what is happening at runtime?

#### Live dashboard (recommended)

```bash
python monitoring_dashboard.py --config datasource_my-corpus.json
```

Open [http://localhost:8080](http://localhost:8080). The dashboard shows:

- **Phase and LoRA version** — current system state
- **Tier distribution** — how many chunks are hot / warm / cold / frozen
- **Top 10 chunks by access count** — which knowledge is used most
- **PRS history** — score trend across training rounds
- **Query rate** — approximate queries per minute

#### Version file inspection

```bash
cat my-corpus_version.json
```

```json
{
  "phase": 2,
  "current_lora_version": 3,
  "checkpoint_path": "lora_checkpoints/my-corpus/v3/",
  "prs_history": [
    {"round": 1, "prs": 0.71, "timestamp": 1742847000},
    {"round": 2, "prs": 0.79, "timestamp": 1742933400},
    {"round": 3, "prs": 0.84, "timestamp": 1743019800}
  ]
}
```

#### Background worker logs

```bash
# Run background workers and capture logs
nohup python kv_background.py --config datasource_my-corpus.json \
  > logs/kv_background.log 2>&1 &

# Tail the log
tail -f logs/kv_background.log
```

Log lines look like:
```
✅ kv_background workers started
[kv_background] Healed chunk 142 → kv_version=3
[kv_background] Access flush: 23 chunks updated
[kv_background] KV recompute error for chunk 99: CUDA out of memory
```

#### Checking collection health

```python
from vectorstore.registry import get_store
import json

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)

store = get_store(cfg)
total = store.count(cfg["collection"])
print(f"Total chunks: {total}")

# For Qdrant — count chunks per KV status
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, IsNullCondition
client = QdrantClient(cfg["qdrant_host"], port=cfg["qdrant_port"])
null_kv, _ = client.scroll(cfg["collection"],
    scroll_filter=Filter(must=[IsNullCondition(is_null={"key": "kv_version"})]),
    limit=1)
print(f"Chunks without KV: {len(null_kv)}")
```

---

### How do I reset everything and start over?

#### Full reset

```bash
# 1. Delete the collection from the vector store
python -c "
import json
from vectorstore.registry import get_store
with open('datasource_my-corpus.json') as f:
    cfg = json.load(f)
store = get_store(cfg)
if store.collection_exists(cfg['collection']):
    store.delete_collection(cfg['collection'])
    print(f'Deleted collection: {cfg[\"collection\"]}')
"

# 2. Remove phase/PRS state
rm -f my-corpus_version.json

# 3. Remove LoRA training state
rm -f my-corpus_replay.db
rm -rf lora_checkpoints/my-corpus/

# 4. For ChromaDB — also remove the persist directory
rm -rf .chroma/my-corpus/

# 5. Re-index from scratch
python kvforge.py index \
  --config datasource_my-corpus.json \
  --source ./my_document.pdf
```

#### Partial reset — keep vectors, remove KV state

If you want to keep your indexed vectors but start KV computation and training over (e.g. after switching to a different LLM):

```bash
# Remove only training state — keep the collection intact
rm -f my-corpus_version.json my-corpus_replay.db
rm -rf lora_checkpoints/my-corpus/

# Recompute KV with the new model (overwrites existing kv_cache fields)
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

---

### What are the GPU memory requirements?

#### Memory by model size

| Model | Parameters | VRAM for KV compute | VRAM for LoRA training | Recommended GPU |
|-------|:----------:|:-------------------:|:----------------------:|----------------|
| TinyLlama-1.1B | 1.1B | ~2 GB | ~4 GB | Any GPU with ≥ 6 GB VRAM |
| Llama-3.2-3B | 3B | ~6 GB | ~10 GB | RTX 3080, A10G, RTX 4080 |
| Mistral-7B | 7B | ~14 GB | ~20 GB | A10G (24 GB), RTX 3090, RTX 4090 |
| Llama-3.1-8B | 8B | ~16 GB | ~24 GB | A10G (24 GB), RTX 4090 |
| Mistral-22B | 22B | ~44 GB | OOM | A100 80GB or multi-GPU |

All sizes assume float16 precision. Batch size 1 during KV computation, batch size 4–8 during LoRA training.

#### Reducing memory usage

**4-bit quantization** (reduces inference VRAM by ~60%):

```bash
pip install bitsandbytes
```

```json
{
  "load_in_4bit": true
}
```

This is supported by `model_loader.py` via the `bitsandbytes` library. Quantized models cannot be used for gradient computation, so LoRA training falls back to a QLoRA approach (`prepare_model_for_kbit_training` from PEFT).

**Gradient checkpointing** (reduces training VRAM by ~30% at the cost of ~20% slower training):

```json
{
  "gradient_checkpointing": true
}
```

**Reducing LoRA rank** for smaller VRAM:

```json
{
  "lora_rank":  8,
  "lora_alpha": 16
}
```

#### AWS instance guide

| Instance | GPU | VRAM | Max model (training) | Monthly cost (on-demand) |
|----------|-----|:----:|:--------------------:|:------------------------:|
| g4dn.xlarge | T4 | 16 GB | Llama-3.2-3B | ~$0.53/hr |
| g5.xlarge | A10G | 24 GB | Mistral-7B | ~$1.01/hr |
| g5.2xlarge | A10G | 24 GB | Mistral-7B (more RAM) | ~$1.21/hr |
| p3.2xlarge | V100 | 16 GB | Llama-3.2-3B | ~$3.06/hr |
| p4d.24xlarge | 8× A100 | 8×40 GB | 70B models | ~$32.77/hr |

KVForge was benchmarked on `g5.xlarge` with Llama 3.2 3B — this is the recommended starting point.

---

← [Back to FAQ index](../../FAQ.md)
