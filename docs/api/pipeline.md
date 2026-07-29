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
| `decide_inference_mode(chunks, current_lora_version, phase=2, kds_threshold=None, fkds_threshold=None)` | Returns `"kv_injection"` or `"text_fallback"`. KV injection requires `phase >= 2`, all chunks fresh and cached. When `fkds_threshold` is set, every chunk's `fkds` must clear it; otherwise `kds` must clear `kds_threshold`. If both thresholds are `None` it fails closed to text fallback. |
| `get_stale_chunk_ids(chunks, current_lora_version)` | Returns subset of chunk IDs whose KV tensors are stale |
| `answer_with_retrieval(query, cfg)` | Phase 1/2 answer: retrieve chunks, inject KV or use text |

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

**Output:** Prints PRS breakdown (accuracy / calibration / consistency / total), whether phase transition was triggered, and the corpus-level Knowledge Differentiation Score (KDS) history.

**Programmatic API:**
```python
from pipeline.prs_evaluator import _extract_qa, _compute_prs, compute_kds
```

| Function | Description |
|----------|-------------|
| `compute_kds(faqs, cfg, lora_checkpoint=None, sample_cap=300, n=3)` | Computes per-chunk Knowledge Differentiation Score using topic probes, persists `kds` and `last_kds_round` to each measured chunk's vector-store payload, and appends `mean(KDS)` to `version.json["kds_history"]`. |

---

## `pipeline.monitoring_dashboard`

**CLI:**
```
python -m pipeline.monitoring_dashboard --config <cfg.json> [--port 8081]
python -m pipeline.monitoring_dashboard --config <cfg.json> --port 8084 \
    --gemini-key $GEMINI_KEY --gemini-model gemini-2.5-flash
python -m pipeline.monitoring_dashboard --config <cfg.json> \
    --openai-key $OPENAI_KEY --openai-model gpt-4.1
python -m pipeline.monitoring_dashboard --config <cfg.json> \
    --claude-key $ANTHROPIC_KEY --claude-model claude-sonnet-4-6
```

**REST API endpoints** (once running on `localhost:<port>`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness check — returns `{"status":"ok","timestamp":<unix>}` |
| `/api/version` | GET | Current phase and LoRA version from `version.json` |
| `/api/stats` | GET | Tier counts, top-10 most-accessed chunks (with full text, kv_version), PRS history |
| `/api/config` | GET | Display-safe config fields (model names, embed model, collection, top_k) |
| `/api/access-report` | GET | Raw `access_report.json` if present |
| `/api/coverage` | GET | FAQ coverage heatmap: for each FAQ in `faqs.json`, returns top-K matching chunks with score, tier, text, page, access_count, kv_version. Query param: `top_k` (default 5) |
| `/api/query` | POST | A/B comparison query — Model A (KVForge) vs Model B (Gemini/Claude/OpenAI). Both sides record chunk access counts. Body: `QueryRequest` |
| `/api/set_model_b_config` | POST | Hot-swap Model B provider, model name, and API key at runtime |

**`QueryRequest` body schema:**
```json
{
  "query": "What is the return policy?",
  "a_top_k": 5,
  "a_max_new_tokens": 64,
  "a_temperature": 0.7,
  "a_top_p": 0.9,
  "a_repetition_penalty": 1.2,
  "b_top_k": 5,
  "b_max_output_tokens": 4096,
  "b_temperature": 1.0
}
```

---

## `pipeline.index_and_train`

**CLI:**
```
python -m pipeline.index_and_train <pdf_file> [--config cfg.json] [--faqs faqs.json] [--skip-prs]
```

Orchestrator that chains: `kv_indexer index` → `lora_trainer` → `kv_indexer compute-kv` → `kv_indexer compute-kv --stale-version N` → `prs_evaluator`.
