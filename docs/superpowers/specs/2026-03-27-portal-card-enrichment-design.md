# Portal Use-Case Card Enrichment — Design Spec

## Goal

Enrich each of the 4 use-case cards on the KVForge portal (port 8080) with 5 additional information items: vector database (with link), embedding model (HuggingFace link), fine-tuned LLM (HuggingFace link), PRS score (with link to per-UC AB eval report), and a KVQ computation link to a live interactive stats + Claude-generated diagram page.

## Architecture

Three self-contained components:

### 1. Per-UC AB Eval Runner (`pipeline/ab_evaluator.py`)

A new CLI script that generates a per-use-case A/B evaluation report by:

- Loading the UC's `faqs.json` (already present in `examples/<uc>/faqs.json`)
- Sampling up to a configurable `--max-samples` count (default 200) from the FAQ list
- POSTing each question to the running dashboard's `/api/query` endpoint, which already runs both Model A (KVForge/vLLM parametric) and Model B (Gemini RAG) and returns both answers with scores (`sem_sim_a`, `sem_sim_b`, `rouge_l_a`, `rouge_l_b`, `latency_a_ms`, `latency_b_ms`, etc.)
- Writing results to `examples/<uc>/ab_eval_results.json`
- Generating `examples/<uc>/ab_eval_viewer.html` using the same template as the existing `ab_eval_viewer.html` (UC4)

CLI usage:
```
python -m pipeline.ab_evaluator \
  --config examples/usecase1_customer_support/config.json \
  --dashboard-url http://localhost:8081 \
  --max-samples 200
```

The script requires:
- The target dashboard to be running (queries `/api/query`)
- A Gemini API key (passed via `--gemini-api-key` or env `GEMINI_API_KEY`)

Output files per UC:
- `examples/<uc>/ab_eval_results.json` — list of result dicts
- `examples/<uc>/ab_eval_viewer.html` — self-contained HTML viewer (data embedded inline as a JS variable)

### 2. KVQ Interactive Page (`/kvq` on portal, port 8080)

A new FastAPI route on the portal that renders a two-panel page:

**Panel 1 — Live stats** (auto-refreshes every 10s)
- One row per use case
- Data fetched from each dashboard's `/api/stats` endpoint (returns tier counts, top accessed chunks, etc.)
- Displays: phase distribution bar (Phase 1 / Phase 2 / Phase 3 query counts as proportional colored segments), KV cache registry entry count, median latency per phase
- Pure inline CSS/JS — no external libraries

**Panel 2 — "How KVQ Works" diagram**
- On each page load the portal backend calls the Anthropic Claude API (`claude-sonnet-4-6`) with a prompt describing the KVQ architecture
- Claude generates an HTML/SVG diagram and explanation inline
- The response is streamed into the page
- A loading spinner is shown while the Claude call is in progress (~2–3s)
- Anthropic API key sourced from env var `ANTHROPIC_API_KEY`

The `/kvq` endpoint:
- Renders a full HTML page (not JSON)
- The Claude-generated content is fetched client-side via a `/kvq/diagram` JSON endpoint so the page shell loads instantly and the diagram fills in asynchronously

### 3. Portal Card Enrichment (`kvforge_portal.py`)

**`USE_CASES` dict additions (static per UC):**

| Field | UC1 | UC2 | UC3 | UC4 |
|-------|-----|-----|-----|-----|
| `vectordb` | `"Qdrant"` | `"ChromaDB"` | `"FAISS"` | `"Qdrant"` |
| `vectordb_url` | `"http://<host>:6333/dashboard"` | `"https://www.trychroma.com"` | `"https://faiss.ai"` | `"http://<host>:6333/dashboard"` |
| `embed_model` | `"BAAI/bge-small-en-v1.5"` | `"BAAI/bge-small-en-v1.5"` | `"BAAI/bge-small-en-v1.5"` | `"mixedbread-ai/mxbai-embed-large-v1"` |
| `llm_model` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` | `"meta-llama/Llama-3.2-3B-Instruct"` |

For Qdrant-backed UCs, `vectordb_url` uses `window.location.hostname` client-side (same pattern as existing dashboard links) to work on any host.

**`/api/status` enhancement:**
- Already fetches `/api/version` per dashboard (returns `prs_history`)
- Extend the returned dict per UC to include `prs` (latest `prs_history[-1]["prs"]` or `null`)

**New portal routes:**
- `GET /ab-eval/{uc_id}` — serves `examples/<uc>/ab_eval_viewer.html` as HTML; 404 if not yet generated
- `GET /kvq` — serves the KVQ interactive page shell
- `GET /kvq/diagram` — calls Claude API and returns `{"html": "..."}` JSON

**Card UI changes (HTML template):**
- Below `.card-desc`, add a `.card-meta` div with 5 info rows
- Each row: `<span class="meta-key">label</span> <a class="meta-val" href="..." target="_blank">value</a>`
- PRS value colored via inline style: green (`#22c55e`) if ≥ 0.75, amber (`#f59e0b`) if < 0.75, grey if unavailable
- PRS value links to `/ab-eval/<uc_id>`; shows "no report yet" (non-linked) if file absent
- KVQ row always links to `/kvq`
- Qdrant `vectordb_url` built client-side using `window.location.hostname + ":6333/dashboard"`

## File Changes

| File | Change |
|------|--------|
| `kvforge_portal.py` | Add metadata to USE_CASES; extend /api/status; add /ab-eval, /kvq, /kvq/diagram routes; update card HTML |
| `pipeline/ab_evaluator.py` | New file — AB eval runner CLI |

No changes to any dashboard code. No new dependencies beyond `anthropic` (already installable via pip).

## Error Handling

- If a dashboard is offline, PRS shows as `—` (existing offline state already handled)
- If `/ab-eval/<uc_id>` file missing, route returns 404 with a plain "Report not yet generated. Run: python -m pipeline.ab_evaluator --config ..." message
- If Claude API key is missing or call fails, `/kvq/diagram` returns `{"html": "<p>Diagram unavailable — set ANTHROPIC_API_KEY</p>"}` (no crash)
- If `/api/stats` is unreachable for a UC, that row shows `—` for live stats

## Testing

- Run `python -m pipeline.ab_evaluator --config examples/usecase1_customer_support/config.json --dashboard-url http://localhost:8081 --max-samples 5` and verify `ab_eval_results.json` written with 5 entries and `ab_eval_viewer.html` is valid HTML
- Restart portal, visit `/ab-eval/uc1` — viewer loads
- Visit `/kvq` — page shell loads instantly, spinner appears, diagram fills within 5s
- Visit portal `/` — each card shows 5 info rows; PRS score colored correctly; all links work
- Set `ANTHROPIC_API_KEY=""`, visit `/kvq` — page shows "Diagram unavailable" without crashing
