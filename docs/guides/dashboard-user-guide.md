# KVForge Dashboard & Studio — User Guide

KVForge has two browser-based UIs that work together: **KVForge Studio** (the control plane, port 8080) and the **Monitoring Dashboard** (per-use-case observability, ports 8081–8084). This guide walks through every feature in the order a new user naturally encounters them.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Starting the UIs](#2-starting-the-uis)
3. [KVForge Portal — the home page](#3-kvforge-portal--the-home-page)
4. [KVForge Studio — control plane](#4-kvforge-studio--control-plane)
   - 4.1 [Sidebar & navigation](#41-sidebar--navigation)
   - 4.2 [Connectivity status bar](#42-connectivity-status-bar)
   - 4.3 [Use-case cards](#43-use-case-cards)
   - 4.4 [Configuring a use case (module panels)](#44-configuring-a-use-case-module-panels)
   - 4.5 [Pipeline Launch Wizard](#45-pipeline-launch-wizard)
   - 4.6 [Real-time pipeline log stream](#46-real-time-pipeline-log-stream)
   - 4.7 [Error assistant toast](#47-error-assistant-toast)
5. [Monitoring Dashboard — per-use-case observability](#5-monitoring-dashboard--per-use-case-observability)
   - 5.1 [Phase Progression stepper](#51-phase-progression-stepper)
   - 5.2 [PRS History chart](#52-prs-history-chart)
   - 5.3 [Flywheel Analytics panel](#53-flywheel-analytics-panel)
   - 5.4 [Tier Distribution & Top Chunks](#54-tier-distribution--top-chunks)
   - 5.5 [FAQ Coverage Heatmap](#55-faq-coverage-heatmap)
   - 5.6 [A/B Query Comparison panel](#56-ab-query-comparison-panel)
   - 5.7 [Connectivity health pills](#57-connectivity-health-pills)
6. [Step-by-step workflow: first use case end to end](#6-step-by-step-workflow-first-use-case-end-to-end)
7. [Reading the dashboard at a glance — quick reference](#7-reading-the-dashboard-at-a-glance--quick-reference)
8. [REST API reference](#8-rest-api-reference)

---

## 1. Architecture overview

```
Browser
  │
  ├── port 8080  ──► KVForge Portal  (landing page + use-case status cards)
  │                    └── /studio  ──► KVForge Studio Hub (config + pipeline)
  │
  ├── port 8081  ──► UC1 Monitoring Dashboard  (observability per use case)
  ├── port 8082  ──► UC2 Monitoring Dashboard
  ├── port 8083  ──► UC3 Monitoring Dashboard
  └── port 8084  ──► UC4 Monitoring Dashboard
```

**KVForge Studio** (port 8080/studio) is where you configure datasets, start pipeline jobs, and watch real-time logs. **Monitoring Dashboards** (8081–8084) show what has *already happened* — charts, chunk access patterns, A/B query results.

---

## 2. Starting the UIs

```bash
# Terminal 1 — Portal + Studio (all use cases)
python kvforge_portal.py --port 8080

# Terminal 2 — UC4 monitoring dashboard (repeat for each UC on a different port)
python -m pipeline.monitoring_dashboard \
    --config examples/usecase4_bedrock_userguide/config.json \
    --port 8084
```

Then open **http://localhost:8080** in your browser.

---

## 3. KVForge Portal — the home page

`http://localhost:8080`

The portal shows one card per configured use case. Each card displays:

| Field | Meaning |
|-------|---------|
| **Use Case title** | Name and dataset description |
| **Phase badge** | Current phase (1 = Text RAG, 2 = KV Injection, 3 = Parametric) |
| **Status dot** | Green = dashboard online, red = offline |
| **Dashboard →** | Link directly to that use case's monitoring dashboard |

The portal pings each dashboard's `/api/health` every 30 seconds. No refresh needed.

**When a dashboard is offline** its card shows a red dot. Start the dashboard process on the correct port to bring it online.

---

## 4. KVForge Studio — control plane

`http://localhost:8080/studio`

Studio is the place where you *do things*: configure datasets, run the pipeline, watch logs, and check GPU availability.

### 4.1 Sidebar & navigation

The collapsible left sidebar has three sections:

- **Workspace** — Studio Hub, KVQ Live Stats, A/B Reports, Settings
- **Use Cases** — color-coded dots (cyan = Phase 3, amber = Phase 2, grey = Phase 1); click any dot to jump to that UC's card
- **+ New Use Case** — opens the inline create form (enter an ID and display name)

Click the `‹` arrow to collapse the sidebar to icon-only mode; click the logo mark to expand it again.

### 4.2 Connectivity status bar

The topbar (top-right) shows three live status pills:

```
[ ● Qdrant ]  [ ● GPU ]  [ ● LLM ]
```

- **Green dot** = service reachable / loaded
- **Red dot** = unreachable / not loaded
- **Grey dot** = not yet checked

These ping the active use case's monitoring dashboard (`/api/connectivity`) every 30 seconds. Click any UC card to make that UC "active" and update the pills.

The existing **GPU pill** (separate from connectivity) shows the count of free GPUs (e.g. `1/1 GPUs free`). Click it to force a refresh.

### 4.3 Use-case cards

Each card on the Studio Hub represents one use case. The card header shows:

- **Phase badge** (Phase 1 / 2 / 3)
- **PRS score** (colour-coded: green ≥ 0.75, amber < 0.75)
- **Journey dots** — 6 dots representing pipeline stages; filled = done, pulsing = active, hollow = locked
- **Dashboard ↗** link — opens the monitoring dashboard in a new tab
- **🚀 Launch** button — opens the Pipeline Launch Wizard

Click the card header to expand/collapse its configuration panels.

### 4.4 Configuring a use case (module panels)

Expanding a card reveals **module chips** along the bottom strip. Click any chip to open its configuration panel:

#### Data Source panel
Configure where KVForge reads its documents:
- **HuggingFace** tab: enter a dataset ID, split, and text column
- **PDF / Local** tab: enter a local file path or directory
- Set **Max rows** to limit dataset size during experiments
- Click **Save Data Config** before moving on

#### Vector DB panel
Configure the embedding and storage backend:
- Choose vector store (Qdrant, ChromaDB, FAISS, Pinecone, etc.)
- Set **embedding model**, **dimensions**, **chunk size**, and **overlap**
- The read-only grid at the bottom shows the authoritative values from `config.json`

#### LLM panel
Configure the local language model:
- Enter the **HuggingFace model ID** (e.g. `meta-llama/Llama-3.2-3B-Instruct`)
- Set quantization (4bit recommended for A10G)
- Optionally set a **vLLM URL** if running an external vLLM server

#### Sleep-time FAQ Generation panel
Configure the cloud LLM used to generate FAQs offline:
- Choose provider (Gemini / Claude / OpenAI)
- Choose model
- Set **FAQ count** (50 is a good starting point)
- Click **Generate FAQs** to launch the job immediately, or configure and use the wizard

#### Training panel
Configure LoRA fine-tuning:
- Adjust learning rate, batch size, and epochs
- View current LoRA version and checkpoint path

#### Evaluation panel
Shows A/B comparison results between KVForge and a third-party LLM (semantic accuracy + latency). Click **Re-run Eval** to launch a fresh evaluation or **Full A/B Report ↗** to open the detailed viewer.

### 4.5 Pipeline Launch Wizard

Click **🚀 Launch** on any use-case card to open the 3-step wizard.

#### Step 1 — Select pipeline step

Choose what to run:

| Step | What it does |
|------|-------------|
| **Index Documents** | Chunks your data → embeds → stores KV tensors in the vector store |
| **Train LoRA** | Fine-tunes the LLM using tier-weighted FAQ replay buffer |
| **Evaluate PRS** | Scores parametric readiness; may trigger a phase advance |
| **Generate FAQs** | Creates FAQs offline via a cloud LLM (sleep-time computation) |
| **Recompute KV Tensors** | Refreshes stale KV caches after a LoRA version bump |

The step progress bar at the top (three coloured segments) shows which wizard step you are on.

#### Step 2 — Configure parameters

Relevant inputs appear depending on which step you chose:

- **Train LoRA / Evaluate PRS** → Epochs (leave blank for config default), Top-K chunks
- **Generate FAQs** → FAQ count (5–500)
- **Index / Recompute** → No extra parameters needed

The **Next** button calls `/studio/api/wizard-validate` before advancing. If a value is out of range (e.g. epochs = 0) an error message appears inline and you cannot proceed until it is fixed.

#### Step 3 — Review & Launch

A summary table shows the step name, use case ID, and any parameters you set. Click **🚀 Launch** to start the job. The wizard closes and the pipeline log stream begins automatically on the UC card.

### 4.6 Real-time pipeline log stream

When a pipeline job is running, a log area appears inside the UC card's active module panel. Every line of stdout/stderr from the subprocess is streamed live via SSE. Log lines are colour-coded:

| Colour | Meaning |
|--------|---------|
| Green | Success / completion message |
| Blue | Informational |
| Amber | Warning |
| Red | Error |

A **Stop** button appears while the job is running. Click it to send SIGTERM to the subprocess. The **Run Next Step** button reappears after the job finishes or is stopped.

### 4.7 Error assistant toast

When an error-pattern line appears in the live log stream (e.g. a Python traceback, `ImportError`, `CUDA out of memory`), a dismissible toast slides up from the bottom-right corner:

```
⚠ Error Detected
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'
💡 CLIP embedder type mismatch — embeddings/clip_embedder.py needs the pooler_output fix
```

The yellow hint line is fetched from the active dashboard's `/api/error-hint` endpoint, which pattern-matches against a table of known errors and their fixes. The toast auto-dismisses after 15 seconds or you can click ✕.

**Known error patterns and their hints:**

| Error pattern | Hint shown |
|---------------|-----------|
| `No module named pypdf` | `pip install pypdf` |
| `No module named pdfplumber` | `pip install pdfplumber pymupdf` |
| `No module named fastembed` | `pip install fastembed` |
| `No module named qdrant_client` | `pip install qdrant-client` |
| `CUDA out of memory` | Reduce batch size or restart to free VRAM |
| `Connection refused` | Check Qdrant is running on port 6333 |
| `401` | Check HF_TOKEN or API key |
| `BaseModelOutputWithPooling` | CLIP embedder pooler_output fix needed |
| `collection already exists` | Delete the collection with curl DELETE |

---

## 5. Monitoring Dashboard — per-use-case observability

`http://localhost:808X` (8081 for UC1, 8082 for UC2, etc.)

The dashboard auto-refreshes every 30 seconds. Each section is described below.

### 5.1 Phase Progression stepper

The top card shows the three KVForge phases as an animated stepper:

```
  [1]──────────[2]──────────[3]
Text RAG    KV Inject    Parametric
```

- **Hollow circle (grey)** = not yet reached
- **Blue filled circle** = completed phase (LoRA trained and advancing)
- **Glowing cyan circle (pulsing)** = current active phase
- **Blue connector line** = fills in as you advance (0% → 100%)

The `?` button next to "Phase Progression" opens the PRS explanation modal, which shows the formula, component weights, and phase thresholds.

**How to advance phases:**
- Phase 1 → 2: PRS ≥ 0.75 in one evaluation round
- Phase 2 → 3: PRS ≥ 0.75 in two *consecutive* evaluation rounds

### 5.2 PRS History chart

Below the stepper, a **Chart.js line chart** plots your Parametric Readiness Score across rounds.

```
PRS
1.0 ┤
0.75┤ ·············· Phase 2 threshold
0.5 ┤        ●
0.25┤    ●
0.0 ┼────────────────
    Rnd 1   Rnd 2  Rnd 3
```

- Hover any point to see the exact PRS value (4 decimal places)
- The chart updates automatically every 30 seconds
- If no PRS data exists yet (no evaluations run), the chart area is empty — run `prs_evaluator.py` or use the wizard to trigger an evaluation

**Interpreting PRS:**

| PRS range | Meaning |
|-----------|---------|
| < 0.60 | Model not ready — more training needed |
| 0.60–0.74 | Coast zone — acceptable but below advancement threshold |
| ≥ 0.75 | Phase advance eligible |
| ≥ 0.75 × 2 consecutive | Phase 3 unlock |

### 5.3 Flywheel Analytics panel

The Flywheel Analytics card shows the cumulative training loop progress:

**Summary row (4 KPI tiles):**

| Tile | Shows |
|------|-------|
| **Rounds** | Total LoRA training rounds completed |
| **Last PRS** | Score from the most recent evaluation |
| **Est. Cost** | Estimated USD cost of all API calls (from analytics DB) |
| **ETA Phase 3** | Projected rounds remaining to reach Phase 3 |

**Bar chart — PRS per round:**
Each training round is one bar. Colour indicates score quality:
- 🟢 Green = PRS ≥ 0.75 (at or above advancement threshold)
- 🟡 Amber = PRS 0.60–0.74 (coast zone)
- 🔴 Red = PRS < 0.60 (regression zone)

**Round snapshot table** (below chart):

| Round | PRS | Phase | Cost ($) |
|-------|-----|-------|---------|
| 1 | 0.653 | 1 | $0.0012 |
| 2 | 0.721 | 2 | $0.0012 |

The flywheel panel refreshes every 60 seconds (less frequent than the main panel because it reads from the analytics SQLite DB which writes are less frequent).

### 5.4 Tier Distribution & Top Chunks

The main `#root` panel shows:

**Tier distribution** — counts of chunks in each access tier:

| Tier | Colour | Definition |
|------|--------|-----------|
| 🔥 Hot | Orange | Top 15% by access, touched < 7 days |
| 🌡 Warm | Yellow | Next 50%, touched < 30 days |
| ❄ Cold | Blue | All other accessed chunks |
| 🧊 Frozen | Grey | Never accessed |

**Top 10 chunks by access count** — a table showing which chunks users query most. Each row shows chunk ID, page, tier, access count, and parametric hit count. Click any **preview text** to open the **Chunk Detail popup**, which shows:
- Full chunk text (scrollable)
- Page number
- Current tier
- Access count
- KV version (the LoRA round whose weights were used to compute this chunk's KV tensor)

A stale KV version (lower than current LoRA version) means that chunk's tensor was computed with an older model — the background daemon will recompute it automatically.

### 5.5 FAQ Coverage Heatmap

Click **Refresh** to load the heatmap. It shows:
- **Rows** = each FAQ question from `faqs.json`
- **Columns** = top-5 matching chunks (by cosine similarity)
- **Cell colour** = similarity score

| Colour | Score |
|--------|-------|
| Red | ≥ 0.85 (strong match) |
| Orange | ≥ 0.75 |
| Yellow | ≥ 0.65 |
| Blue | < 0.65 (weak match) |

**Threshold slider** (0.60–1.00): drag to hide rows where no chunk reaches the threshold. Useful for spotting FAQs with no good coverage — those are knowledge gaps that need more documents or better chunking.

Click any coloured cell to see the full text of that matching chunk.

**How to improve coverage:** FAQs showing only blue cells (all scores < 0.65) indicate the document corpus doesn't contain good answers. Consider adding more source documents or splitting existing chunks more finely.

### 5.6 A/B Query Comparison panel

Type a question in the input box and click **Ask** (or press Enter). KVForge runs the query in parallel against two models:

- **Answer A — KVForge** (local LLM, runs on GPU)
- **Answer B — Cloud LLM** (Gemini / Claude / OpenAI, configurable)

Results show side-by-side with:
- Full answer text
- **Total / Retrieval / Generation latency** breakdown
- **Mode badge** for Answer A:
  - `Parametric (no retrieval)` — Phase 3, model answered from weights
  - `KV Injection` — Phase 2, pre-computed KV tensors injected
  - `Text-in-Context RAG` — Phase 1 (or fallback), standard retrieval
- **Gate info** (Phase 3 only): entropy, hedging, and similarity scores that drove the parametric decision
- **Retrieved chunks** (collapsed by default) for both sides

**Configuring generation parameters** — click the ⚙ Settings gear:
- Top-K (shared, both models)
- Answer A: max new tokens, temperature, top-p, repetition penalty
- Answer B: provider, model, API key, max output tokens, temperature

**Switching Model B provider at runtime** — change the provider dropdown (Gemini / Claude / OpenAI), select the model, and enter your API key. Changes take effect immediately on the next query; no server restart needed.

**Thinking traces** (Claude / Gemini 2.5 models with extended thinking) appear in a collapsible "Thinking" section below Answer B when available.

### 5.7 Connectivity health pills

The monitoring dashboard includes a `/api/connectivity` endpoint that the Studio topbar polls. You can also call it directly:

```bash
curl http://localhost:8084/api/connectivity | jq .
```

```json
{
  "qdrant": {"ok": true, "latency_ms": 4},
  "gpu":    {"ok": true, "name": "NVIDIA A10G", "util_pct": 23, "mem_used_mib": 5120, "mem_total_mib": 24564},
  "llm":    {"ok": true, "loaded": true}
}
```

---

## 6. Step-by-step workflow: first use case end to end

This is the recommended path for bringing a new use case from zero to Phase 3.

### Step 1 — Start the portal

```bash
python kvforge_portal.py --port 8080
```

Open `http://localhost:8080/studio`.

### Step 2 — Create a use case

Click **+ New Use Case** in the sidebar. Enter:
- **ID**: `my-corpus` (alphanumeric, hyphens OK — used as directory name)
- **Display name**: `My Corpus`

Click **Create**. A card appears in the hub.

### Step 3 — Configure the data source

Expand the card → click the **Data** chip. Choose HuggingFace or PDF:
- For PDF: enter the file path in the PDF tab
- Click **Save Data Config**

### Step 4 — Configure the vector DB

Click the **VDB** chip. Choose your vector store and embedding model. The defaults (Qdrant + bge-small, 384 dims) work for most cases. For large PDFs use `mxbai-embed-large-v1` at 1024 dims.

### Step 5 — Configure the LLM

Click the **LLM** chip. Enter the HuggingFace model ID. `meta-llama/Llama-3.2-3B-Instruct` is the recommended starting point.

### Step 6 — Generate FAQs

Click **🚀 Launch** → select **Generate FAQs** → set count (e.g. 50) → **Next** → **Next** → **🚀 Launch**.

Watch the log stream. When it shows `Generated N FAQs → faqs.json`, the FAQ chip turns green.

### Step 7 — Index documents

Click **🚀 Launch** → select **Index Documents** → **Next** → **Next** → **🚀 Launch**.

This chunks your data, embeds each chunk, runs one LLM forward pass per chunk to compute the KV tensor, and upserts everything into the vector store. For a 500-page PDF this takes 10–30 minutes on a single A10G.

Watch the log stream: `Indexed N chunks` confirms completion.

Open the Monitoring Dashboard (`http://localhost:808X`) to verify tier counts — all chunks start as **Frozen** (never queried).

### Step 8 — Evaluate PRS (baseline)

Click **🚀 Launch** → **Evaluate PRS** → **🚀 Launch**.

After a few minutes you get the first PRS score. Check the dashboard:
- The **PRS History chart** shows Round 1
- The **Flywheel Analytics** shows 1 round, last PRS, and ETA to Phase 3
- Phase 1 dot on the stepper is active (glowing)

A typical baseline PRS with no fine-tuning is 0.40–0.55.

### Step 9 — Train LoRA (Round 1)

Click **🚀 Launch** → **Train LoRA** → **Next** → optionally set Epochs → **🚀 Launch**.

Training uses a tier-weighted FAQ replay buffer. Frozen chunks (weight 1) are sampled less than hot chunks (weight 8). Initial training may take 15–45 minutes.

### Step 10 — Evaluate PRS again

Repeat Step 8. If PRS reaches ≥ 0.75 the system automatically advances to **Phase 2** (KV Injection enabled). The stepper on the dashboard animates the connector line filling from Phase 1 → Phase 2.

### Step 11 — Recompute KV tensors

After LoRA version bumps, existing KV tensors are stale (computed with the old model). The background daemon (`kv_background.py`) recomputes them lazily at query time, but you can force a full sweep:

Click **🚀 Launch** → **Recompute KV Tensors** → **🚀 Launch**.

### Step 12 — Repeat until Phase 3

Continue the loop: Train → Evaluate → check dashboard. Two consecutive rounds ≥ 0.75 advances to **Phase 3** (Parametric). At Phase 3 the A/B query panel shows `Parametric (no retrieval)` for high-confidence questions, meaning the LLM answers entirely from its fine-tuned weights — no vector search required.

---

## 7. Reading the dashboard at a glance — quick reference

```
┌──────────────────────────────────────────────────────┐
│  Phase Progression                              [?]   │
│  [1]─────────────[2]────────────────────────[ 3 ●]   │
│  Text RAG     KV Inject                  Parametric   │
│                                                       │
│  PRS History                                          │
│  ▁▃▅▇█  (line chart, last N rounds)                  │
├──────────────────────────────────────────────────────┤
│  Flywheel Analytics                                   │
│  Rounds: 4  │  Last PRS: 0.812  │  ETA: ✓ done       │
│  [bar chart per round]                                │
│  ┌──────────────────────────────────────────┐         │
│  │ Round │ PRS   │ Phase │ Cost ($)         │         │
│  │   1   │ 0.531 │   1   │ $0.0012         │         │
│  │   2   │ 0.712 │   2   │ $0.0012         │         │
│  │   3   │ 0.776 │   2   │ $0.0012         │         │
│  │   4   │ 0.812 │   3   │ $0.0012         │         │
│  └──────────────────────────────────────────┘         │
├──────────────────────────────────────────────────────┤
│  Phase: 3  │  LoRA version: 4  │  Total chunks: 1842 │
│  🔥 Hot: 43  🌡 Warm: 218  ❄ Cold: 821  🧊 Frozen: 760 │
│  Top 10 chunks by access count  [click to expand]    │
├──────────────────────────────────────────────────────┤
│  FAQ Coverage Heatmap              [Refresh]          │
│  Threshold: ──────●──── 0.70                         │
│  FAQ question 1  ██ ██  ░░  ░░  ░░                   │
│  FAQ question 2  ██ ░░  ░░  ░░  ░░                   │
├──────────────────────────────────────────────────────┤
│  Query A/B Comparison                                 │
│  [Ask a question...] [Ask]                            │
│  [⚙ Model & Generation Settings]                     │
│  Answer A (KVForge)  │  Answer B (Gemini)             │
│  Mode: Parametric    │  (cloud RAG)                   │
└──────────────────────────────────────────────────────┘
```

**Signal interpretation summary:**

| What you see | What it means | Action |
|-------------|---------------|--------|
| All chunks Frozen | Not queried yet | Run A/B eval to generate access data |
| PRS stuck < 0.60 | Model not learning | Check FAQ quality; try more epochs |
| PRS 0.60–0.74 for 3+ rounds | Slow convergence | Increase LoRA epochs; add more FAQs |
| PRS ≥ 0.75 once | Phase 2 unlocked | KV injection now active |
| PRS ≥ 0.75 twice | Phase 3 unlocked | Parametric answering now active |
| FAQ heatmap mostly blue | Coverage gaps | Add more source documents |
| Answer A = `Parametric` | Phase 3 confirmed working | Continue monitoring PRS |
| Qdrant pill red in Studio | Qdrant unreachable | `sudo systemctl start qdrant` or check port 6333 |
| GPU pill red in Studio | nvidia-smi failed | Check GPU driver; may be CPU-only mode |

---

## 8. REST API reference

All endpoints are on the monitoring dashboard (e.g. `http://localhost:8084`).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Liveness check → `{"status":"ok"}` |
| `/api/version` | GET | Phase, LoRA version, PRS history |
| `/api/stats` | GET | Tier counts, top-10 chunks, access report |
| `/api/config` | GET | Display-safe config (model names, top_k, etc.) |
| `/api/coverage` | GET | FAQ coverage heatmap data |
| `/api/flywheel` | GET | Flywheel summary: rounds, PRS, cost, ETA |
| `/api/connectivity` | GET | Qdrant / GPU / LLM health check |
| `/api/error-hint` | GET | `?msg=<error text>` → fix suggestion |
| `/api/query` | POST | A/B comparison query (JSON body) |
| `/api/set_model_b_config` | POST | Hot-swap Model B provider/model/key |
| `/api/flywheel/cost-rate` | PATCH | Update cost_per_1k_tokens |

Studio endpoints (under `http://localhost:8080/studio/api`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/registry` | GET | All use cases with phase, PRS, job status |
| `/api/uc/{id}/config` | GET / POST | Read or update use-case config |
| `/api/uc/new` | POST | Create a new use case |
| `/api/gpu-check` | POST | GPU availability and memory |
| `/api/wizard-validate` | POST | Validate wizard parameters before launch |
| `/api/run-step` | POST | Start a pipeline job (returns `job_id`) |
| `/api/job/{job_id}` | DELETE | Stop a running job |
| `/api/stream/{job_id}` | GET (SSE) | Live stdout/stderr stream for a job |
