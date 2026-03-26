# KV Cache & Phases

← [Back to FAQ index](../../FAQ.md)

---

### What exactly is stored in the KV cache payload?

#### What gets stored

For each chunk, KVForge runs one LLM forward pass (`model(**inputs, use_cache=True)`) and stores the resulting attention key-value tensors. Before storage, the per-token tensors are mean-pooled over the sequence length dimension, collapsing the variable-length token axis into a fixed-size representation.

```
Raw forward pass output:    [num_layers, 2, num_kv_heads, seq_len, head_dim]
After mean pool over seq:   [num_layers, 2, num_kv_heads, head_dim]
Cast to float16, base64:    stored in Qdrant payload as "kv_cache" string
```

#### Size calculation for common models

| Model | Layers | KV heads | Head dim | Size per chunk |
|-------|:------:|:--------:|:--------:|:--------------:|
| TinyLlama-1.1B | 22 | 4 | 64 | ~45 KB |
| Llama-3.2-3B | 28 | 8 | 128 | ~115 KB |
| Mistral-7B | 32 | 8 | 128 | ~131 KB |
| Llama-3.1-8B | 32 | 8 | 128 | ~131 KB |

Formula: `num_layers × 2 × num_kv_heads × head_dim × 2 bytes`

For 2,520 chunks with Llama 3.2 3B: `2520 × 115 KB ≈ 290 MB` of additional payload data in Qdrant.

#### All payload fields written by KVForge

| Field | Type | Written by | Description |
|-------|------|-----------|-------------|
| `kv_cache` | string (base64) | `kv_indexer.py` | Mean-pooled KV tensor |
| `kv_version` | int or null | `kv_indexer.py` | LoRA version used to compute the tensor |
| `access_count` | int | `kv_background.py` | Total times retrieved by a query |
| `last_accessed_ts` | int | `kv_background.py` | Unix timestamp of most recent retrieval |
| `avg_retrieval_rank` | float | `kv_background.py` | Mean rank position across all retrievals |
| `parametric_hit_count` | int | `kv_background.py` | Times answered directly from model weights (Phase 3) |
| `tier` | string | `access_tracker.py` | `hot` / `warm` / `cold` / `frozen` |
| `text` | string | `kv_indexer.py` | Chunk text (used for KV computation and text-in-context fallback) |
| `page` | int | `kv_indexer.py` | Source page number |
| `source_file` | string | `kv_indexer.py` | Originating filename |
| `indexed_at` | int | `kv_indexer.py` | Unix timestamp of indexing |

---

### How does KV injection work under the hood?

#### The problem with standard RAG

In standard RAG, every query requires the LLM to process both the query and all retrieved chunks through the full attention stack:

```
Prompt = [system] + [chunk 1 text] + [chunk 2 text] + ... + [query]
model.generate(tokenize(Prompt))
  → attention over all tokens → generation
```

For a 5-chunk retrieval with 500 tokens per chunk, the model processes ~2,500 context tokens on every single query. This is slow and scales linearly with the number of retrieved chunks.

#### How KV injection avoids this

Instead of including chunk text in the prompt, KVForge pre-computes what the LLM would have produced when attending over those chunks — the key-value tensors — and injects them directly as `past_key_values`:

```python
# Simplified from kv_inference.py
chunk_kvs = [kv_utils.deserialize_kv(c["kv_cache"], shape=kv_shape) for c in chunks]
past_kv = kv_utils.stack_past_key_values(chunk_kvs, ...)

output = model.generate(
    **tokenize(f"Answer: {query}"),
    past_key_values=past_kv,   # ← pre-loaded context; not re-processed
)
```

The model attends over the pre-loaded KV state when processing the query tokens, but never re-tokenizes or re-encodes the chunk text. The effective prompt for the LLM at query time is just the question — the "context" is already baked into the KV cache.

#### Performance comparison

| Method | Tokens processed at query time | Relative latency |
|--------|:------------------------------:|:----------------:|
| Text-in-context (Phase 1) | query + all chunk text (~2,500 tokens) | 1× (baseline) |
| KV injection (Phase 2) | query only (~20 tokens) | ~5–10× faster |
| Parametric (Phase 3) | query only, no retrieval | ~15–20× faster |

#### Why KV staleness matters

The KV tensors are computed using a specific set of model weights (identified by `kv_version`). After LoRA training, the model weights change. KV tensors computed with the old weights are incompatible with the new model — injecting them causes the model to attend over stale representations and produce worse answers. This is why KVForge tracks `kv_version` and falls back to text-in-context for stale chunks.

---

### Why do some queries fall back to text-in-context even in Phase 2?

KVForge checks every retrieved chunk before deciding on the inference path. A single stale or missing KV tensor causes the entire query to fall back to text-in-context.

#### Diagnosing the cause

```python
import json
import version as ver
from qdrant_client import QdrantClient

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)

ver.init(cfg)
current_ver = ver.get_lora_version()
print(f"Current LoRA version: {current_ver}")

client = QdrantClient(cfg["qdrant_host"], port=cfg["qdrant_port"])

# Count chunks with null kv_version (never had KV computed)
from qdrant_client.models import Filter, IsNullCondition
null_results, _ = client.scroll(
    collection_name=cfg["collection"],
    scroll_filter=Filter(must=[IsNullCondition(is_null={"key": "kv_version"})]),
    limit=1,
)
print(f"Chunks with kv_version=null: needs backfill")

# Count chunks with stale kv_version
from qdrant_client.models import FieldCondition, Range
stale_results, _ = client.scroll(
    collection_name=cfg["collection"],
    scroll_filter=Filter(must=[
        FieldCondition(key="kv_version", range=Range(lt=current_ver))
    ]),
    limit=1,
)
print(f"Chunks with kv_version < {current_ver}: stale, healing in background")
```

#### Fixing each cause

**`kv_version` is null (chunks never had KV computed)**

These were indexed using `kvforge.py index` or `bedrock_rag.py index` which only embed vectors, not KV tensors. Fix by running the KV indexer:

```bash
python kv_indexer.py --config datasource_my-corpus.json compute-kv
```

**`kv_version` < current LoRA version (stale after training)**

Background workers heal these automatically after first retrieval. If you want to pre-heal before the next query:

```bash
python kv_indexer.py --config datasource_my-corpus.json compute-kv \
  --stale-version <current_lora_version>
```

**`kv_cache` field missing entirely**

Chunks were inserted directly into Qdrant (not through KVForge). Same fix — run `compute-kv`.

**Phase is actually 1, not 2**

```python
import json, version as ver
with open("datasource_my-corpus.json") as f: cfg = json.load(f)
ver.init(cfg)
print("Phase:", ver.get_phase())   # if this prints 1, KV injection is disabled
```

---

### How do I manually advance or roll back the phase?

KVForge advances phases automatically when PRS thresholds are met during `index_and_train.py`. For manual control:

```python
import json
import version as ver

with open("datasource_my-corpus.json") as f:
    cfg = json.load(f)
ver.init(cfg)

# Inspect current state
state = ver.load()
print("Phase:", state.get("phase", 1))
print("LoRA version:", state.get("current_lora_version", 0))
print("PRS history:", state.get("prs_history", []))

# Advance to Phase 2 (enables KV injection)
ver.activate_phase_2()
print("Activated Phase 2")

# Advance to Phase 3 (enables confidence gate)
ver.activate_phase_3()
print("Activated Phase 3")

# Roll back to Phase 1 (disables KV injection and confidence gate)
ver.set_phase(1)
print("Rolled back to Phase 1")
```

Phase rollbacks are appropriate when:
- You observe answer quality degradation after a phase transition
- You want to A/B test response quality between phases
- A training run produced a high PRS but the model actually overfit to the FAQ set

To roll back from the command line without writing Python:

```bash
# Edit the version JSON file directly
python -c "
import json
with open('my-corpus_version.json') as f:
    v = json.load(f)
v['phase'] = 1
with open('my-corpus_version.json', 'w') as f:
    json.dump(v, f, indent=2)
print('Rolled back to Phase 1')
"
```

---

← [Back to FAQ index](../../FAQ.md)
