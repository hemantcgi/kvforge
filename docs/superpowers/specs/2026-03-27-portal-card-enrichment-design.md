# Portal Use-Case Card Enrichment — Design Spec

## Goal

Enrich each of the 4 use-case cards on the KVForge portal (port 8080) with 5 additional information items: vector database (with link), embedding model (HuggingFace link), fine-tuned LLM (HuggingFace link), PRS score (with link to per-UC AB eval report), and a KVQ computation link to a live interactive stats + Claude-generated diagram page.

## Architecture

Three self-contained components:

### 1. Per-UC AB Eval Runner (`pipeline/ab_evaluator.py`)

A new CLI script that generates a per-use-case A/B evaluation report:

**Prerequisite:** `examples/<uc>/faqs.json` must exist. This file is generated on the EC2 host by the pipeline (`index_and_train.py`) during the ingestion phase — it is not committed to git. Running `ab_evaluator.py` on a machine that hasn't run the pipeline will fail with a clear `FileNotFoundError: examples/<uc>/faqs.json not found — run the pipeline first`.

**What it does:**
- Loads the UC's `faqs.json` from `examples/<uc>/faqs.json`
- Samples up to `--max-samples` questions (default 200) from the FAQ list
- For each question, POSTs to the running dashboard's `/api/query` endpoint, which returns raw `answer_a`, `answer_b`, `latency_a_ms`, `latency_b_ms`, `chunks_a`, `chunks_b`, `mode_a`, etc. (no scores — these are computed client-side)
- Computes scores client-side against the FAQ ground-truth answer using `fastembed` (already installed):
  - `sem_sim_a` / `sem_sim_b` — cosine similarity of answer embedding vs ground-truth embedding
  - `rouge_l_a` / `rouge_l_b` — ROUGE-L F1 score (pure-Python, no extra dep)
- Writes `examples/<uc>/ab_eval_results.json` — a JSON array where each element has the shape:
  ```json
  {
    "question": "...",
    "ground_truth": "...",
    "answer_a": "...",
    "answer_b": "...",
    "mode_a": "parametric|rag",
    "latency_a_ms": 123,
    "latency_b_ms": 456,
    "generation_a_ms": 100,
    "generation_b_ms": 300,
    "sem_sim_a": 0.82,
    "sem_sim_b": 0.75,
    "rouge_l_a": 0.41,
    "rouge_l_b": 0.38
  }
  ```
- Generates `examples/<uc>/ab_eval_viewer.html` — a **fully self-contained** HTML file (no server required) where the results array is embedded as an inline JS variable `const AB_DATA = [...];`. This file is modelled on the existing `ab_eval_viewer.html` (UC4) but uses the injected data variable rather than a pre-rendered static table. The generator renders the full HTML string with the data spliced in.

CLI usage:
```
python -m pipeline.ab_evaluator \
  --config examples/usecase1_customer_support/config.json \
  --dashboard-url http://localhost:8081 \
  --gemini-api-key <key> \
  --max-samples 200
```

Requirements: dashboard must be running; `fastembed` and `httpx` (both already present); `GEMINI_API_KEY` env var or `--gemini-api-key` flag.

**New dependencies:** none (fastembed, httpx already in requirements).

### 2. KVQ Interactive Page (`/kvq` on portal, port 8080)

A new FastAPI route on the portal with two panels:

**Panel 1 — Live stats** (auto-refreshes every 10s)

Data fetched from each dashboard's `/api/stats` endpoint. The actual response shape is:
```json
{
  "tier_counts": {"hot": N, "warm": N, "cold": N, "frozen": N},
  "total_chunks": N,
  "version": {
    "phase": 3,
    "prs_history": [{"round": 1, "prs": 0.83}, ...],
    ...
  }
}
```
Relevant fields:
- `tier_counts` — KV cache tier counts (hot/warm/cold/frozen)
- `total_chunks` — total indexed chunks
- `version.phase` — current phase (accessed as `stats_data["version"]["phase"]`)
- `version.prs_history` — accessed as `stats_data["version"]["prs_history"]`

Panel 1 displays per UC:
- KV tier distribution bar: hot / warm / cold / frozen counts as proportional colored segments
- Total chunks indexed
- Current phase badge
- Latest PRS score

Since `/api/stats` does not track per-phase query counters, the phase distribution bar represents **KV cache tier distribution** (not query counts per phase). This is accurate to the data available without modifying any dashboard code.

**Panel 2 — "How KVQ Works" diagram**

- On each page load the page shell (`/kvq`) loads instantly
- A loading spinner is shown immediately
- The browser fetches `/kvq/diagram` (async GET) which calls the Anthropic Claude API (`claude-sonnet-4-6`) with a structured prompt describing the KVQ architecture (Phase 1/2/3 pipeline, KV tensor pre-computation, PRS gate)
- Claude returns an HTML snippet containing an SVG diagram + prose explanation
- The snippet is injected into the page's diagram div
- Anthropic API key sourced from env var `ANTHROPIC_API_KEY`

**New dependency:** `anthropic` Python package (add to `requirements_gpu.txt`).

### 3. Portal Card Enrichment (`kvforge_portal.py`)

**`USE_CASES` dict additions (static per UC):**

| Field | UC1 | UC2 | UC3 | UC4 |
|-------|-----|-----|-----|-----|
| `vectordb` | `"Qdrant"` | `"ChromaDB"` | `"FAISS"` | `"Qdrant"` |
| `vectordb_url` | `"qdrant"` (sentinel) | `"https://www.trychroma.com"` | `"https://faiss.ai"` | `"qdrant"` (sentinel) |

**Note:** The existing portal's `USE_CASES` description strings say "Qdrant + bge-small" for UC2 and UC3. This is incorrect — the config files show `"vector_store": "chroma"` for UC2 and `"vector_store": "faiss"` for UC3. The description strings should be updated to match the actual config when implementing this feature.
| `embed_model` | `"BAAI/bge-small-en-v1.5"` | `"BAAI/bge-small-en-v1.5"` | `"BAAI/bge-small-en-v1.5"` | `"mixedbread-ai/mxbai-embed-large-v1"` |
| `llm_model` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` |
| `ab_eval_dir` | `"examples/usecase1_customer_support"` | `"examples/usecase2_pubmedqa"` | `"examples/usecase3_squad"` | `"examples/usecase4_bedrock_userguide"` |

When `vectordb_url == "qdrant"`, the card JS builds the link at runtime: `` `http://${window.location.hostname}:6333/dashboard` ``. For all other values the string is used as a direct href.

**`/api/status` enhancement:**
- Already fetches `/api/version` per dashboard (returns `prs_history`)
- Extend the returned dict per UC to include `prs`: `prs_history[-1]["prs"]` if `prs_history` is non-empty, else `null`
- No new HTTP calls; the `/api/version` fetch already happens

**New portal routes:**

| Route | Behaviour |
|-------|-----------|
| `GET /ab-eval/{uc_id}` | Looks up `uc_id` in `USE_CASES` by `id` field. If not found → 404 "Unknown use case". If found but `<ab_eval_dir>/ab_eval_viewer.html` doesn't exist → 404 with message "Report not yet generated. Run: `python -m pipeline.ab_evaluator --config <ab_eval_dir>/config.json ...`". Otherwise serves the file as `text/html`. |
| `GET /kvq` | Serves KVQ page shell (instant load, spinner, async diagram fetch) |
| `GET /kvq/diagram` | Calls Claude API, returns `{"html": "<svg>...</svg><p>...</p>"}`. On missing key or API error returns `{"html": "<p>Diagram unavailable — set ANTHROPIC_API_KEY</p>"}` (HTTP 200, no crash). |

**Card UI changes (HTML template):**
- Below `.card-desc`, add a `.card-meta` div with 5 info rows
- Row structure: `<span class="meta-key">Label</span><a class="meta-val" href="..." target="_blank">value ↗</a>` (or plain `<span>` when no link)
- CSS: small monospace font, muted key colour matching existing `.card-desc` style
- PRS value: green (`#22c55e`) if ≥ 0.75, amber (`#f59e0b`) if < 0.75, grey (`#6b7280`) if null — set via inline `style="color:..."` on the value span
- PRS links to `/ab-eval/<uc_id>`; if PRS is null, shows `—` (no link)
- KVQ row always shows "Live stats ↗" linking to `/kvq`
- vectordb_url logic: if `uc.vectordb_url === "qdrant"` set href to `` `http://${window.location.hostname}:6333/dashboard` `` else use value directly

## File Changes

| File | Change |
|------|--------|
| `kvforge_portal.py` | Add metadata to USE_CASES; extend /api/status PRS field; add /ab-eval, /kvq, /kvq/diagram routes; update card HTML/CSS/JS |
| `pipeline/ab_evaluator.py` | New file — AB eval runner CLI |
| `requirements_gpu.txt` | Add `anthropic` |

No changes to any dashboard code (`pipeline/monitoring_dashboard.py` or others).

## Dependencies

| Package | Status |
|---------|--------|
| `httpx` | Already in requirements |
| `fastembed` | Already in requirements |
| `anthropic` | **New** — add to `requirements_gpu.txt` |

## Testing

1. Run AB eval for UC1 with 5 samples and verify output:
   ```
   python -m pipeline.ab_evaluator --config examples/usecase1_customer_support/config.json \
     --dashboard-url http://localhost:8081 --max-samples 5
   ```
   - `examples/usecase1_customer_support/ab_eval_results.json` exists with 5 entries, each containing `sem_sim_a`, `sem_sim_b`, `rouge_l_a`, `rouge_l_b`
   - `examples/usecase1_customer_support/ab_eval_viewer.html` is valid HTML containing `const AB_DATA =`

2. Restart portal. Visit `/ab-eval/uc1` — viewer loads and displays data.

3. Visit `/ab-eval/uc2` (before running ab_evaluator for UC2) — returns 404 with "Report not yet generated" message.

4. Visit `/ab-eval/uc99` — returns 404 with "Unknown use case" message.

5. Visit `/kvq` — page shell loads instantly with spinner; within 5s the Claude diagram appears.

6. Set `ANTHROPIC_API_KEY=""`, visit `/kvq` — page loads, diagram area shows "Diagram unavailable" (no crash, HTTP 200).

7. On `/kvq` Panel 1 — verify all 4 UC rows appear; each shows KV tier bar and total chunks, or `—` if dashboard offline.

8. Visit portal `/` — each of the 4 cards shows 5 info rows with correct values:
   - UC1/UC4: Qdrant link → `:6333/dashboard` (uses current hostname)
   - UC2: ChromaDB link → trychroma.com
   - UC3: FAISS link → faiss.ai
   - Embed and LLM links → correct HuggingFace URLs
   - PRS score coloured green for all three trained UCs (UC1 latest=0.766, UC2 latest=0.887, UC3 latest=0.878 — all ≥ 0.75), `—` for UC4 (empty prs_history → grey dash)

9. Click any card — still navigates to the dashboard (existing behaviour unchanged).
