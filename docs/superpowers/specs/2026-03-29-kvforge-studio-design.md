# KVForge Studio — Generalized Pipeline UI Design

## Overview

A generalized web UI (KVForge Studio) that allows any user to configure and run the full KVForge 3-phase progressive RAG pipeline on arbitrary datasets — without being tied to the 4 hardcoded use cases. The studio extends the existing `kvforge_portal.py` at port 8080 via a new `/studio` route prefix.

---

## Goals

- Load data from HuggingFace datasets or PDF files into a configurable vector store
- Select and configure a vector database (Qdrant / ChromaDB / FAISS) with custom dimensions, embedding model, and indexing parameters
- Select a small local LLM (HuggingFace) for fine-tuning and a 3rd-party LLM (Gemini / OpenAI) for comparison
- Trigger the full KVForge training pipeline (KV index → LoRA round 1 → KV recompute → PRS eval → LoRA round 2 → ...) from the UI with live GPU availability checks and log streaming
- Compare fine-tuned model vs 3rd-party LLM on latency and accuracy
- Surface the 4 existing use cases as pre-configured examples visible in the same UI

---

## Architecture

### New files

```
studio/
  __init__.py
  routes.py          — FastAPI router mounted at /studio; thin layer: page serving + UC CRUD
  api.py             — All /studio/api/* endpoints (module config save/read, GPU check, run-step)
  pipeline_runner.py — Subprocess spawning + SSE log streaming
  job_manager.py     — In-memory job registry: {job_id → status, uc_id, step, pid, logs}
  gpu_monitor.py     — nvidia-smi parsing; detects which vLLM PIDs occupy which GPUs + free VRAM

templates/studio/
  hub.html           — Main studio hub page (all UCs)
  uc_detail.html     — Per-UC page with collapsible module panels

kvforge_registry.json   — Ordered list of UC IDs with display names (root level)
examples/*/
  uc_config.json     — Per-UC config (auto-migrated from version.json on first server start)
```

### Integration point

`kvforge_portal.py` gains one new line:

```python
from studio.routes import router as studio_router
app.include_router(studio_router, prefix="/studio")
```

No other changes to existing portal code.

### Routes added

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/studio` | Hub page |
| GET | `/studio/uc/<uc_id>` | UC detail page |
| POST | `/studio/uc/new` | Create new UC (name + description) |
| GET | `/studio/api/registry` | List all UCs from kvforge_registry.json |
| GET | `/studio/api/uc/<uc_id>/config` | Read uc_config.json |
| POST | `/studio/api/uc/<uc_id>/config` | Save module config (data/vectordb/llm) |
| POST | `/studio/api/gpu-check` | Returns per-GPU free VRAM + vLLM occupancy |
| POST | `/studio/api/run-step` | Trigger pipeline step → returns {job_id} |
| GET | `/studio/api/stream/<job_id>` | SSE log stream for running job |
| DELETE | `/studio/api/job/<job_id>` | Stop a running pipeline job |
| POST | `/studio/api/gpu/stop-vllm` | Stop a vLLM server by UC id to free its GPU |

### API key rule

Gemini and OpenAI API keys are **never written to disk**. They live in browser `localStorage` only and are passed per-request as an `X-Api-Key` header.

---

## Data Model

### `kvforge_registry.json`

```json
{
  "use_cases": [
    {"id": "usecase1_customer_support", "display_name": "Customer Support", "type": "example"},
    {"id": "usecase2_pubmedqa",         "display_name": "PubMedQA",         "type": "example"},
    {"id": "usecase3_squad",            "display_name": "SQuAD",            "type": "example"},
    {"id": "usecase4_bedrock_userguide","display_name": "Bedrock User Guide","type": "example"}
  ]
}
```

- `id` is the canonical UC identifier and must match the `examples/<id>/` directory name exactly.
- `type` is `"example"` for the 4 built-in UCs and `"custom"` for user-created ones. The hub renders an "Example" badge on example UCs.
- The portal's short IDs (`uc1`/`uc2`/`uc3`/`uc4`) are internal to `kvforge_portal.py` and are not used by the studio. Studio always uses the long directory-based IDs.
- New UCs are appended on creation. The 4 existing UCs are pre-populated on first server start.

### `examples/<uc_id>/uc_config.json`

The studio reads its configuration from `uc_config.json`. This is a **studio-only overlay** — it does not replace `config.json` (which remains the source of truth for pipeline scripts). On migration, `uc_config.json` is derived from `config.json`.

```json
{
  "id": "usecase3_squad",
  "display_name": "SQuAD",
  "type": "example",
  "created_at": "2026-03-29T00:00:00Z",
  "data": {
    "source_type": "huggingface",
    "dataset_id": "rajpurkar/squad",
    "split": "train",
    "text_column": "context",
    "max_rows": 5000
  },
  "vectordb": {
    "store": "faiss",
    "dimensions": 384,
    "chunk_size": 600,
    "chunk_overlap": 60,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "index_type": "hnsw"
  },
  "llm": {
    "local_model": "meta-llama/Llama-3.2-3B-Instruct",
    "quantization": "4bit",
    "vllm_url": "http://localhost:8093",
    "comparison_provider": "gemini",
    "comparison_model": "gemini-1.5-flash"
  }
}
```

- `vllm_url` matches the field name in `config.json` (full URL, not a bare port).
- Phase, PRS history, and known good queries remain in `version.json` (path specified by `config.json`'s `version_file` field — not duplicated in `uc_config.json`).

---

## UI Layout

### Visual style

- **Color palette**: Deep black (`#020817`) background, violet-to-cyan gradient accents (`#8b5cf6` → `#06b6d4`), cyan highlights (`#22d3ee`), amber for in-progress (`#f59e0b`), green for complete (`#34d399`)
- **Typography**: System UI stack, tight spacing, uppercase labels at 9px
- **Icons**: Unicode symbols (no CDN dependency) for reliability

### Page structure

```
┌─────────────────────────────────────────────────────┐
│ Sidebar (collapsible)  │  Topbar (breadcrumb + GPU) │
│                        ├─────────────────────────────│
│  KV logo + toggle      │  Content area              │
│  ─────────────────     │                            │
│  Studio Hub            │  UC cards                  │
│  KVQ Live Stats        │  (each with module chips   │
│  A/B Reports           │   + collapsible panels)    │
│  Settings              │                            │
│  ─────────────────     │                            │
│  [UC list with dots]   │                            │
│  + New Use Case        │                            │
└────────────────────────┴────────────────────────────┘
```

### Sidebar

- **Expanded** (220px): KVForge logo + gradient text, collapse button (▲ panel-close), nav items with icons, divider, UC list with colored status dots, "+ New Use Case" button
- **Collapsed** (52px): KV logo mark + `›` chevron (both clickable to expand), icon-only nav items with hover tooltips, colored dots per UC
- Transition: smooth 0.22s width animation

### Topbar

- **Left**: Breadcrumb trail — `⌂ › Studio Hub › <uc_id> › <module>` — home segment uses the Unicode house symbol `⌂` (U+2302). Each segment is a clickable link except the current page (shown in cyan).
- **Right**: GPU status pill (live, shows free GPU count + VRAM), API Keys button (opens key management modal), notifications bell

### UC Hub page (`/studio`)

Each use case displayed as a card with:
1. **Header row**: UC name, phase badge, PRS score, journey dots (5 dots with lines, colored by status)
2. **Module chip strip**: Data / VectorDB / LLM / Training / Evaluation chips
   - Done: green border, green text
   - Active/current: violet border + glow
   - Locked: grey, not clickable, shows lock icon + tooltip on hover explaining prerequisite
3. **Collapsible panel**: Expands below the chip strip when chip is clicked; collapses on second click. Multiple panels can be open simultaneously. Chip arrow rotates 180° when open. Panel has its own "▲ Collapse" button at top-right and in the button row.

**Dependency rules (locking)**:
- VectorDB requires Data configured
- LLM can be configured at any time
- Training requires Data + VectorDB + LLM all configured
- Evaluation requires Training complete (at least round 1 PRS computed)

**New use case creation**: Clicking "+ New Use Case" reveals an inline form (name + optional description) in a bar below the topbar. "Create" appends the UC to the registry and shows it as a new card with all modules in "not configured" state.

---

## Module Panel Designs

### 1. Data

Fields: Source type toggle (HuggingFace / PDF).
- **HuggingFace**: dataset ID (e.g. `rajpurkar/squad`), split selector, text column, max rows. Live preview card shown after save (row count, avg text length, estimated chunks).
- **PDF**: local server path input (e.g. `examples/usecase4_bedrock_userguide/data/`) or file upload (uploaded to `uploads/<uc_id>/` on the server, max 200 MB). Re-uploading overwrites the prior file and marks the VectorDB step as stale (requires re-indexing).

### 2. VectorDB

Fields: Store toggle (Qdrant / ChromaDB / FAISS), dimensions, chunk size, chunk overlap, embedding model dropdown, index type dropdown. "Save & Index" button triggers kv_indexer.py as a background job.

### 3. LLM

Two-column layout:
- **Left (Local)**: Base model dropdown (Llama-3.2-3B, Qwen2.5-3B, custom HF path), quantization toggle (fp16 / 4-bit / 8-bit), vLLM port
- **Right (Comparison)**: Provider toggle (Gemini / OpenAI), model dropdown, API key status indicator (key loaded from localStorage or prompt to add)

### 4. Training

Two sections:
- **GPU Status grid** (2×2): Each GPU card shows name, free/busy status, VRAM bar, occupying process name (e.g. "vLLM UC1"). Refreshes on panel open.
- **Pipeline stepper**: Vertical list of steps (KV Index → LoRA R1 → KV Recompute → PRS Eval → LoRA R2 → KV Recompute → PRS Eval). Each step shows: icon (✓ done / ↻ running / ○ pending), title, metadata (timing, PRS score). Running step shows a progress bar + live log panel (last ~5 lines of SSE stream). "■ Stop Pipeline" button (danger style) visible while running.

### 5. Evaluation

Three stat cards: Win Rate (% of queries answered by fine-tuned model at PRS ≥ 0.75), Avg PRS score, Speed gain (KVForge p50 latency vs 3rd-party). Horizontal comparison bars for semantic accuracy and latency side-by-side. "↺ Re-run Eval" and "View Full A/B Report →" (opens existing ab_eval_viewer.html in new tab).

---

## Pipeline Execution

### GPU check flow

`POST /studio/api/gpu-check` calls `gpu_monitor.py` which runs `nvidia-smi` and scans `/proc/<pid>/cmdline` to identify vLLM processes. Returns:

```json
{
  "gpus": [
    {"id": 0, "free_gb": 21.9, "total_gb": 24, "status": "free"},
    {"id": 1, "free_gb": 3.5,  "total_gb": 24, "status": "busy", "process": "vLLM UC1"}
  ],
  "has_free_gpu": true
}
```

If no GPU is free, the UI shows a warning listing which vLLM servers are occupying GPUs and offers to stop one via `POST /studio/api/gpu/stop-vllm` with `{uc_id}`. This sends SIGTERM to the vLLM process (identified by `gpu_monitor.py`) and waits up to 10 seconds for it to exit before reporting success or failure.

### Pipeline step execution

`POST /studio/api/run-step` with `{uc_id, step}` → `job_manager.py` creates a job entry, `pipeline_runner.py` spawns the appropriate subprocess (kv_indexer.py / lora_trainer.py / prs_evaluator.py / ab_evaluator.py) with the UC's config as arguments. Returns `{job_id}`.

The client immediately opens an SSE connection to `GET /studio/api/stream/<job_id>`. `pipeline_runner.py` reads subprocess stdout/stderr line by line and yields each as an SSE event. On process exit, sends a final `{type: "done", exit_code: 0}` or `{type: "error"}` event and closes the stream.

`job_manager.py` maintains `{job_id: {status, uc_id, step, pid, start_time, last_lines[]}}` in memory. A per-UC lock prevents the same UC from running two pipeline steps simultaneously (returns HTTP 409 if attempted). Different UCs may run concurrently (up to 4 jobs, one per GPU).

The hub page begins polling `/studio/api/registry` every 5 seconds when `run-step` returns a `job_id`. Polling stops when the SSE stream for that job sends `{type: "done"}` or `{type: "error"}`. Jobs are not persisted across server restarts.

---

## Migration of Existing Use Cases

On first server start (detected by absence of `kvforge_registry.json`), `studio/routes.py` runs a one-time migration:

1. Reads `examples/usecase*/config.json` (the authoritative per-UC config file)
2. Maps `config.json` fields to `uc_config.json` schema: `vector_store→vectordb.store`, `vector_dim→dimensions`, `chunk_size/overlap`, `embed_model→embedding_model`, `vllm_url`, `llm_model→local_model`, `quantization`
3. Sets `data.source_type` to `"pdf"` if `loader == "pdf"`, else `"huggingface"` — pre-populates known dataset IDs where available
4. Sets `data.source_path` for PDF use cases to the known local file path (e.g. `examples/usecase4_bedrock_userguide/data/`)
5. Writes `examples/<uc_id>/uc_config.json` for each, with `"type": "example"` and `"created_at"` timestamp
6. Creates `kvforge_registry.json` with the 4 existing UCs

---

## Error Handling

- GPU check failure (nvidia-smi not found): show warning banner, allow proceeding without GPU validation
- Pipeline step failure: SSE stream sends `{type: "error", message}`, UI shows red error state on the failed step, "Retry" button appears
- Dataset not found on HuggingFace: show inline error on Data panel after save attempt
- API key missing for comparison model: evaluation panel shows a prompt to add key via the topbar API Keys button

---

## Out of Scope

- Multi-user authentication
- Cloud storage for datasets (local EC2 only)
- Parallel pipeline runs for the same UC (sequential only; different UCs can run simultaneously)
- Persisting job logs across server restarts (in-memory only)
