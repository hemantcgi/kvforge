# Pipeline Module API Reference

All pipeline scripts live in the `pipeline/` package and can be run as:
```bash
python -m pipeline.<module> [args]
```

---

## `pipeline.kv_indexer`

**CLI:**
```
python -m pipeline.kv_indexer --config <cfg.json> index <source>
python -m pipeline.kv_indexer --config <cfg.json> compute-kv [--source-file <file>] [--stale-version N]
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `index <source>` | Load, embed, and upsert documents from `<source>` into the vector store |
| `compute-kv` | Compute KV tensors for all chunks and store in payload |
| `compute-kv --source-file <f>` | Compute KV only for chunks from a specific source file |
| `compute-kv --stale-version N` | Recompute KV for all chunks with `kv_version < N` |

**Programmatic API:**
```python
from pipeline.kv_indexer import compute_kv_for_chunk, build_payload
```

---

## `pipeline.kv_inference`

**Programmatic API (not a standalone CLI):**
```python
from pipeline.kv_inference import decide_inference_mode, get_stale_chunk_ids, answer_with_retrieval
```

| Function | Description |
|----------|-------------|
| `decide_inference_mode(cfg, query)` | Returns `"kv"`, `"text"`, or `"parametric"` based on phase and chunk freshness |
| `get_stale_chunk_ids(cfg, chunk_ids)` | Returns subset of chunk IDs whose KV tensors are stale |
| `answer_with_retrieval(cfg, query, top_k)` | Phase 1/2 answer: retrieve chunks, inject KV or use text |

---

## `pipeline.kv_background`

**Programmatic API (daemon, typically not called directly):**
```python
import pipeline.kv_background as kv_background
kv_background.start(cfg)   # starts background KV healing + access flush threads
kv_background.enqueue(chunk_id)  # enqueue a chunk for KV recomputation
```

---

## `pipeline.lora_trainer`

**CLI:**
```
python -m pipeline.lora_trainer --config <cfg.json> --faqs <faqs.json>
python -m pipeline.lora_trainer --config <cfg.json> --source-file <file.pdf> --replay-ratio 0.2
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | _(required)_ | Datasource config JSON |
| `--faqs` | — | FAQ JSON file (array of `{question, answer}`) |
| `--source-file` | — | Train on raw chunks from this source (alternative to --faqs) |
| `--replay-ratio` | `0.2` | Fraction of training examples from replay buffer |

---

## `pipeline.prs_evaluator`

**CLI:**
```
python -m pipeline.prs_evaluator --config <cfg.json> --faqs <faqs.json>
```

**Output:** Prints PRS breakdown (accuracy / calibration / consistency / total) and whether phase transition was triggered.

**Programmatic API:**
```python
from pipeline.prs_evaluator import _extract_qa, _compute_prs
```

---

## `pipeline.monitoring_dashboard`

**CLI:**
```
python -m pipeline.monitoring_dashboard --config <cfg.json> [--port 8080]
```

**REST API endpoints** (once running on `localhost:<port>`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/stats` | GET | Collection stats (count, phase, PRS) |
| `/version` | GET | Current phase and LoRA version |
| `/config` | GET | Active datasource config |
| `/query` | POST | `{"question": "..."}` — KVForge answer |
| `/ab_query` | POST | `{"question": "..."}` — KVForge vs Gemini A/B comparison |

---

## `pipeline.index_and_train`

**CLI:**
```
python -m pipeline.index_and_train <pdf_file> [--config cfg.json] [--faqs faqs.json] [--skip-prs]
```

Orchestrator that chains: `kv_indexer index` → `lora_trainer` → `kv_indexer compute-kv` → `kv_indexer compute-kv --stale-version N` → `prs_evaluator`.
