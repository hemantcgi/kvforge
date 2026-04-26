# KVForge Studio — Guided End-to-End Pipeline UX Design

**Date:** 2026-04-26
**Branch:** smartqdrant-main

---

## Goal

Transform KVForge Studio from a technical dashboard into a guided end-to-end pipeline UI that a non-expert can follow from raw PDF to Phase 3 parametric answering. The UX introduces two new dedicated pages: a 7-step wizard for first-time use-case setup, and a persistent UC operations page for all ongoing monitoring and experimentation.

---

## Architecture Overview

Two new full HTML pages replace the existing single-page hub for per-UC interaction. The hub (`/studio/`) continues to serve as the use-case gallery. New UCs are created via the wizard (`/studio/wizard`). Ongoing operations live on the UC detail page (`/studio/uc/{uc_id}`).

All new pages are server-rendered HTML files in `templates/studio/`, served by FastAPI. They load data via `fetch()` calls to the existing and new `/api/` endpoints. SSE is used for live pipeline log streaming (same pattern as the existing hub).

**Visual theme:** Charcoal/Teal — VS Code Dark+ inspired. Background `#1e1e1e`, surface `#181818`, teal accent `#4ec9b0`, blue accent `#9cdcfe`, rust `#ce9178` for Phase 3 active state. No purple.

---

## Section 1: File Structure

### New files

```
templates/studio/
├── wizard.html              ← 7-step guided wizard (new)
└── uc_detail.html           ← UC operations page (new)

studio/
├── settings_manager.py      ← read/write ~/.kvforge/settings.json; mask API keys
├── vdb_validator.py         ← per-backend connectivity ping functions
├── ab_runner.py             ← async A/B query runner (local vLLM + cloud API)
└── curation_manager.py      ← faqs_curated.json append/read wrapper
```

### Modified files

```
studio/routes.py             ← add /studio/wizard route; swap /studio/uc/{id} to uc_detail.html
studio/api.py                ← add 10 new endpoints
studio/gpu_monitor.py        ← add get_gpu_realtime() with util/temp/power/processes
studio/pipeline_runner.py    ← add faq-gen-cloud step
```

---

## Section 2: Wizard Page — `/studio/wizard`

The wizard walks through use-case creation in 7 sequential steps. Each step validates before allowing progression. The wizard creates the UC on completion and redirects to the UC detail page.

### Step 1 — Data Source

Two mutually exclusive modes toggled by tab:

**PDF Upload tab:**
- Drag-and-drop or file picker (`.pdf` only for MVP, `.md` / `.txt` / `.jsonl` in future)
- On upload: `POST /api/wizard/upload-pdf` (multipart) → returns `{filename, size_mb, estimated_chunks}`
- Displays estimated chunk count under the drop zone
- Uploaded file stored at `tmp/uploads/{session_id}/` and cleaned up after indexing completes

**Connect to Vector DB tab:**
- Dropdown: Qdrant (default) · Chroma · FAISS · Pinecone · Weaviate · Milvus · Generic
- Fields rendered per selection:
  - **Qdrant:** host, port, collection name, API key (optional)
  - **Chroma:** host, port, collection name
  - **FAISS:** index file path (local filesystem)
  - **Pinecone:** API key, environment, index name
  - **Weaviate:** URL, API key (optional), class name
  - **Milvus:** host, port, collection name, token (optional)
  - **Generic:** base URL, auth header key, auth header value, collection/namespace field
- "Test connection" button → `POST /api/wizard/validate-vdb` → shows green check + collection count, or red error message

**Use-case name field** (shown in both modes): free-text, auto-slugified for the `uc_id`.

### Step 2 — FAQ Generation

Two mutually exclusive modes:

**Upload existing dataset tab:**
- File picker for `.json` or `.jsonl`
- Expected format: `[{"question": "...", "answer": "..."}]`
- Validates format on upload, shows record count

**Generate synthetic Q&A tab:**
- Sub-toggle: **Local GPU** or **Cloud API**
- **Local GPU:** count slider (10–500); note displayed: "FAQ generation will run after model selection in the pipeline (Step 7)." No actual generation happens here — this step configures the mode; execution is deferred to the Step 7 pipeline run, after model and GPU are selected.
- **Cloud API:**
  - Provider dropdown: Anthropic · OpenAI · Gemini
  - Model dropdown per provider (e.g., claude-haiku-4-5, gpt-4o-mini, gemini-2.0-flash)
  - API key field: pre-filled from global settings if present; inline "Save to settings" checkbox
  - Count slider (10–500)
  - Calls new `faq-gen-cloud` pipeline step via SSE

### Step 3 — Model Selection

Grid of model cards. Each card shows:
- Model name and parameter count
- Architecture (Llama / Gemma / Qwen / custom)
- VRAM requirement pill: green (fits on 24 GB A10G at 4-bit), yellow (fits with reduced batch), red (too large)
- HuggingFace model ID

Default cards: Llama 3.2 3B · Llama 3.1 8B · Gemma 4 4B · Qwen3 1.7B · Qwen3 4B

Custom model row: text field for any HuggingFace model ID + "Check VRAM" button → `POST /api/wizard/estimate-vram` → renders a card dynamically.

VRAM estimation: `POST /api/wizard/estimate-vram` body `{model_id, lora_rank}` → returns `{vram_required_gb, fits: bool, fits_with_reduced_batch: bool}`. Calculation: 4-bit model size + ~4 GB training overhead.

### Step 4 — GPU Selection

Grid showing all GPUs on the host (from `GET /api/gpu/realtime`). Each card shows:
- GPU index, name, VRAM total
- Current usage (used / total GB)
- Status badge: Free (green) / Busy — vLLM (yellow) / Busy — training (red)

Feasibility banner below the grid:
- Green: "Selected GPU has sufficient free VRAM — ready to proceed"
- Yellow: "GPU is busy with vLLM — you can stop the vLLM worker to free it"
- Red: "Insufficient VRAM for selected model — choose a smaller model or a different GPU"

"Stop vLLM on this GPU" button (shown only when vLLM is running on the selected GPU) → calls existing `POST /api/gpu/stop-vllm`.

### Step 5 — LoRA Parameters

Sliders and inputs for fine-tuning hyperparameters:

| Parameter | Type | Default | Range |
|---|---|---|---|
| LoRA rank | slider | 16 | 4 – 128 |
| LoRA alpha | slider | 32 | 8 – 256 |
| LoRA dropout | slider | 0.05 | 0.0 – 0.3 |
| Training epochs | slider | 3 | 1 – 10 |
| Batch size | slider | 4 | 1 – 16 |
| Learning rate | input (scientific) | 2e-4 | 1e-5 – 1e-3 |
| Passes | radio | Pass 1 + 2 | Pass 1 only · Pass 1 + 2 |

Live VRAM preview: as parameters change, a small indicator updates the estimated VRAM footprint using the formula from Step 3, factoring in batch size.

### Step 6 — Review & Launch

Summary table showing all selections:
- Data source: filename/VDB type + collection
- FAQ dataset: source + record count
- Model: name + HuggingFace ID
- GPU: index + name + VRAM
- LoRA: rank / alpha / dropout / epochs / batch size / lr / passes

"Launch pipeline" button → `POST /api/uc/new` (creates the UC config) then `POST /api/run-step` with step=`index` → redirects to Step 7.

### Step 7 — Live Pipeline Progress

Stepper showing the sequential pipeline steps:
1. Index (chunks → embeddings → KV tensors → Qdrant upsert)
2. Generate FAQs (if cloud/local gen was chosen)
3. Fine-tune LoRA (pass 1 + optional pass 2)
4. Recompute KV tensors with new LoRA weights
5. PRS evaluation
6. Phase auto-advance (if PRS ≥ threshold)

Each step: status dot (pending / running / done / failed) + collapsed log tail. Running step shows a scrolling SSE log via the existing `/stream/{job_id}` endpoint.

On pipeline completion: "Open UC Dashboard →" button navigates to `/studio/uc/{uc_id}`.

---

## Section 3: UC Detail Page — `/studio/uc/{uc_id}`

The persistent operations page for a live use case. Loaded from `templates/studio/uc_detail.html`.

### Topbar

```
[KV] Studio Hub › UC4 · Amazon Bedrock User Guide    [● Phase 3] [PRS 0.8531] [⬛ 4 GPUs · click]  [⚙ Settings]
```

- Breadcrumb links back to hub
- Phase pill: color-coded (blue=P1, teal=P2, rust=P3)
- PRS pill: static display, refreshed on page load
- GPU pill: clickable → opens GPU overlay (see below)
- Settings button: navigates to global settings page

**GPU overlay** (appears on GPU pill click):
- Floating panel anchored below the topbar pill
- Header: `nvidia-smi · GPU Monitor · {hostname} · {driver_version}`
- Per-GPU row: index, name, VRAM bar (used/total), GPU utilization bar (color-coded: green <50%, yellow 50–80%, red >80%), temperature, power draw
- Expandable process list per GPU (click row): PID, type (C/G), process name, memory used
- Auto-refreshes every 3 seconds via `GET /api/gpu/realtime`
- Close by clicking X or backdrop
- Footer: total VRAM used, average utilization, max temperature

### Left Rail

**Pipeline section** (step-by-step with connector lines):
1. Data Source — chunk count + backend type
2. FAQ Generation — Q&A pair count
3. Model — model name + LoRA version
4. GPU — GPU index + name
5. Fine-tuning — LoRA version + passes
6. Phases & PRS — current phase, active/inactive
7. A/B Comparison — live query test

Done steps: teal check circle. Active steps: blue pulsing circle.

**Actions section:**
- ⟳ Re-generate FAQs
- ↑ Re-train LoRA
- 🎯 Re-run PRS eval
- 🔄 Recompute KV tensors
- 📋 Export config

### Main Area — Phase Cards

Three cards side by side:
- **Phase 1 — Text RAG:** latency + "baseline" note
- **Phase 2 — KV Injection:** latency + speedup multiplier
- **Phase 3 — Parametric:** latency + speedup multiplier + "ACTIVE" badge (when current phase is 3); amber border when locked (PRS < threshold)

### Main Area — PRS Panel (left of two-col)

- Large PRS number (e.g., 0.8531)
- Subtitle: threshold status + phase activation message
- **SVG line chart:** inline-rendered; X axis = LoRA versions with dates; Y axis = PRS (0.60–1.00); dashed yellow threshold line at 0.75; area fill under the line; interactive dots with hover tooltip showing: version, PRS, date, training description, loss
- Threshold note below chart

Data source: `GET /api/uc/{uc_id}/prs-history`

Tooltip fields per data point:
```json
{
  "version": "LoRA v3b",
  "date": "Apr 26 13:51",
  "prs": 0.8531,
  "train": "Full corpus · confirmed",
  "loss": "1.967",
  "note": "Confirmed eval. Phase 3 auto-activated."
}
```

### Main Area — Latency + LoRA Panel (right of two-col)

**Inference latency comparison:**
- Horizontal bars, color-coded per phase (blue/teal/rust)
- Speedup badge next to each bar

**LoRA version timeline:**
- Dot-connected timeline of all LoRA checkpoints
- Past versions: blue outlined dot. Current: teal filled dot with glow.
- Each dot: version label, training description, PRS

### Main Area — A/B Comparison Panel

Full-width panel below the two-col section.

**Header row:**
- "Live A/B Query Comparison" title
- Auto-curation pill: animated dot + "Auto-curation · {count} / {threshold} records"

**Query input row:**
- Full-width text input
- "▶ Run both" button

**Two-column response area — Model A (left):**

Settings (collapsible):
- Temperature slider (0.0–1.0, default 0.20)
- Max tokens slider (64–512, default 256)
- Top-p slider (0.50–1.0, default 0.90)
- Confidence threshold slider (0.50–0.95, default 0.75) — minimum confidence to invoke Phase 3
- Phase override: [Phase 1] [Phase 2] [Auto] toggle

Response area: generated text + metadata (latency, phase used, confidence score)

**Two-column response area — Model B (right):**

Settings (collapsible):
- Provider dropdown: Anthropic · OpenAI · Gemini · Custom endpoint
- API key selector: pre-filled from global settings, or "+ Add new key" inline
- Temperature slider
- Max tokens slider (64–1024)
- System prompt textarea (override)

Response area: generated text + metadata (latency, source, estimated cost)

**Verdict row:**
- "Mark acceptable:" label
- [✓ Model A] [✓ Model B] [✓ Both] [✗ Neither] buttons
- Flash notification "📥 Added to training dataset (N/50)" on Model B verdict
- Hint text: changes based on verdict selection

**Auto-curation logic:**
- "✓ Model B" without "✓ Model A" → calls `POST /api/uc/{uc_id}/ab-curate` → appends `{question, answer_b, curated_at}` to `examples/{uc_id}/faqs_curated.json`
- Counter updates in the curation pill and dataset preview
- At `count >= threshold`: retrain suggestion banner appears at top of panel

**Dataset preview panel** (below A/B panel):
- Header: "Auto-curated Fine-tuning Dataset" + record count + filename
- Progress bar: count / threshold
- Scrollable list of last N curated Q&A samples
- At threshold: "🚀 Retrain now" banner with primary CTA → triggers `POST /api/run-step` with step=`train` using `faqs_curated.json`

**A/B query execution:** `POST /api/uc/{uc_id}/ab-query`. Both model requests fired concurrently server-side. Response includes both results with latency measured server-side (more accurate than client-side timing).

### Status Bar

```
● Qdrant online · 6333   ● vLLM 4 workers · 8091–8094   ● GPU 0 1.2/22 GB   ● LoRA v3 active
                                                                               [↑ Re-train] [🎯 PRS eval] [⟳ Refresh]
```

---

## Section 4: Backend Changes

### Modified: `studio/gpu_monitor.py`

Add `get_gpu_realtime() -> dict` function:

**Extended GPU query:**
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader,nounits
```

**Process query:**
```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
```
Cross-reference GPU UUID → index to assign processes to GPUs. Also run `ps aux` filtered to non-vLLM processes for completeness.

Returns:
```json
{
  "gpus": [
    {
      "id": 0, "name": "NVIDIA A10G",
      "used_gb": 1.2, "total_gb": 22.0,
      "util_pct": 8, "temp_c": 36, "power_w": 72,
      "status": "free",
      "processes": [
        {"pid": 1234, "type": "C", "name": "python monitoring_dashboard", "mem_mib": 512}
      ]
    }
  ],
  "has_free_gpu": true,
  "driver_version": "535.104",
  "cuda_version": "12.2"
}
```

### Modified: `studio/routes.py`

```python
@router.get("/wizard", response_class=HTMLResponse)
def wizard_page():
    return FileResponse("templates/studio/wizard.html")

# Change existing /uc/{uc_id} from serving hub.html to uc_detail.html
@router.get("/uc/{uc_id}", response_class=HTMLResponse)
def uc_detail_page(uc_id: str):
    return FileResponse("templates/studio/uc_detail.html")
```

### Modified: `studio/pipeline_runner.py`

Add to `STEP_MODULES`:
```python
"faq-gen-cloud": "pipeline.sleep_faq_generator",
```

`faq-gen-cloud` is NOT in `GPU_REQUIRED_STEPS` (cloud API calls only).

Extend `_build_cmd()` to detect `faq-gen-cloud` and append `--provider` and `--api-key` args read from `settings_manager.get_setting("anthropic_api_key")` (or whichever provider is selected).

### Modified: `studio/api.py`

Ten new endpoints added to `api_router`:

```python
@api_router.get("/gpu/realtime")
def gpu_realtime():
    # calls gpu_monitor.get_gpu_realtime()

@api_router.get("/uc/{uc_id}/prs-history")
def prs_history(uc_id: str):
    # reads version.json prs_history[], enriches with LoRA checkpoint metadata

@api_router.post("/uc/{uc_id}/ab-query")
async def ab_query(uc_id: str, request: Request):
    # calls ab_runner.run_ab_query() — concurrent local + cloud

@api_router.post("/uc/{uc_id}/ab-curate")
async def ab_curate(uc_id: str, request: Request):
    # calls curation_manager.append()

@api_router.get("/uc/{uc_id}/curation-status")
def curation_status(uc_id: str):
    # calls curation_manager.get_status()

@api_router.get("/settings")
def get_settings():
    # calls settings_manager.get_masked()

@api_router.post("/settings")
async def save_settings(request: Request):
    # calls settings_manager.save()

@api_router.post("/wizard/validate-vdb")
async def validate_vdb(request: Request):
    # calls vdb_validator.validate(config)

@api_router.post("/wizard/upload-pdf")
async def upload_pdf(uc_id: str, file: UploadFile):
    # saves to tmp/uploads/{uc_id}/, returns size + chunk estimate

@api_router.post("/wizard/estimate-vram")
async def estimate_vram(request: Request):
    # model_id + lora_rank → vram_required_gb + fits bool
```

### New: `studio/settings_manager.py`

Settings stored at `~/.kvforge/settings.json` (outside the repo — never committed to git).

```python
SETTINGS_FILE = Path.home() / ".kvforge" / "settings.json"

DEFAULTS = {
    "anthropic_api_key": "",
    "openai_api_key": "",
    "gemini_api_key": "",
    "huggingface_token": "",
    "curation_threshold": 50,
    "default_cloud_provider": "anthropic",
    "default_cloud_model": "claude-haiku-4-5-20251001",
}

def get_all() -> dict: ...         # returns raw dict (server-side only)
def get_masked() -> dict: ...      # masks all *_api_key and *_token fields to "••••{last4}"
def get_setting(key: str): ...     # returns single value
def save(updates: dict): ...       # merges updates into DEFAULTS, writes atomically
```

`save()` validates API key format before writing:
- `anthropic_api_key` must start with `sk-ant-`
- `openai_api_key` must start with `sk-`
- `gemini_api_key` must start with `AIza`

### New: `studio/vdb_validator.py`

One `validate(config: dict) -> dict` entry point. Dispatches to per-backend function based on `config["type"]`. Each function does the minimum connectivity check:

| Backend | Test |
|---|---|
| qdrant | `QdrantClient(host, port).get_collections()` |
| chroma | `chromadb.HttpClient(host, port).list_collections()` |
| faiss | `Path(index_path).exists()` |
| pinecone | `Pinecone(api_key).list_indexes()` |
| weaviate | `GET {url}/v1/.well-known/ready` → status 200 |
| milvus | `MilvusClient(uri).list_collections()` |
| generic | `GET {base_url} headers={auth_header}` → status < 500 |

Returns: `{"ok": bool, "error": str | None, "collection_count": int | None}`

All imports are guarded by `try/except ImportError` — missing optional dependencies return `{"ok": false, "error": "pinecone-client not installed — pip install pinecone-client"}`.

### New: `studio/ab_runner.py`

```python
async def run_ab_query(
    uc_id: str,
    query: str,
    model_a_settings: dict,
    model_b_settings: dict,
) -> dict:
    # Fires both concurrently via asyncio.gather
    result_a, result_b = await asyncio.gather(
        _query_local(uc_id, query, model_a_settings),
        _query_cloud(query, model_b_settings),
    )
    return {"response_a": result_a, "response_b": result_b}
```

`_query_local()`: POST to `http://localhost:8090/v1/chat/completions` (existing vLLM round-robin router). Constructs a minimal OpenAI-compatible chat payload. Reads phase and confidence from the response metadata headers that the vLLM workers are assumed to inject.

`_query_cloud()`: Dispatches to the appropriate SDK call based on `model_b_settings["provider"]`:
- `anthropic` → `anthropic.Anthropic(api_key=...).messages.create(...)`
- `openai` → `openai.OpenAI(api_key=...).chat.completions.create(...)`
- `gemini` → `google.generativeai.GenerativeModel(...).generate_content(...)`

API key resolved from `settings_manager.get_setting(f"{provider}_api_key")`, or from `model_b_settings["api_key"]` if provided inline.

Both functions return: `{"text": str, "latency_ms": int, "source": str, ...metadata}`.

### New: `studio/curation_manager.py`

```python
CURATED_FILENAME = "faqs_curated.json"

def _path(uc_id: str) -> Path:
    return ROOT / "examples" / uc_id / CURATED_FILENAME

def append(uc_id: str, question: str, answer: str, source_model: str) -> dict:
    # Loads existing list (or []), appends new entry, writes atomically
    # Returns updated status dict

def get_status(uc_id: str) -> dict:
    # Returns {count, threshold, pct, at_threshold}
    # threshold read from settings_manager

def get_samples(uc_id: str, n: int = 5) -> list:
    # Returns last n entries from faqs_curated.json
```

Writes are atomic (write to `.tmp` then `os.replace`), same pattern as `version.py`.

---

## Out of Scope (not in this implementation)

- Global Settings page (separate HTML page with form for API keys) — the API endpoints are implemented here; the page is a follow-up
- Multimodal ingestion (images, video) — separate spec already exists
- ModelScout integration into the wizard — separate spec already exists
- Dynamic PRS calibration UI — separate spec already exists
- Authentication / multi-user access control

---

## Open Decisions (resolved with defaults)

**API key storage location:** `~/.kvforge/settings.json` (outside the repo root). This prevents accidental git commit of keys. The directory is created on first write.

**PDF upload persistence:** Temp files only. Uploaded PDFs are stored at `tmp/uploads/{session_id}/` and deleted after `kv_indexer` completes (success or failure). If indexing fails, the user re-uploads and re-runs.

---

*Spec written 2026-04-26. Supersedes no prior spec — this is a new feature.*
