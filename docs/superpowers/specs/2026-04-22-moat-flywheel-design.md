# KVForge Flywheel Analytics

**Date:** 2026-04-22
**Status:** Approved for implementation

---

## Problem

KVForge's competitive advantage over standard RAG and standalone LLM fine-tuning is real but invisible. A new operator sees a system that costs more to set up (KV compute at index time, LoRA training rounds) without immediately understanding the compounding returns: latency recovery as the model advances phases, cost avoidance as parametric answering replaces retrieval, and learning velocity that increases with query traffic. Without visibility into these returns, the MOAT is not self-evident.

---

## Core Insight

The flywheel makes the MOAT visible. Three signals tell the story:

1. **Learning velocity** — what fraction of real queries is the model now answering from its weights, and how fast is that fraction growing?
2. **Latency recovered** — how many milliseconds per query has the system saved by advancing past Phase 1, and how is that distributed across phases?
3. **Cost savings** — what is the equivalent cloud API spend avoided at the operator's configured rate?

These signals are already computable from data KVForge produces today (PRS state, inference decisions, tier distribution). Flywheel Analytics makes them persistent, queryable, and surfaced in both the per-UC monitoring dashboard and the KVForge Studio cross-UC summary.

---

## Architecture

### Analytics DB (per UC)

A separate SQLite database `<datasource>_analytics.db` per UC. Clean separation from the replay buffer; analytics reads never block training writes.

```sql
-- One row per query (written at inference time, < 1ms, WAL mode)
CREATE TABLE query_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    cluster_id  TEXT,                       -- null pre-clustering (brownfield)
    phase_used  TEXT NOT NULL,              -- 'parametric' | 'kv' | 'retrieval'
    latency_ms  REAL NOT NULL,
    baseline_ms REAL NOT NULL,             -- rolling retrieval avg at time of query
    model_id    TEXT                        -- null for production; set by ModelScout experiments
);

-- One row per LoRA round (written by prs_evaluator after training)
CREATE TABLE round_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                INTEGER NOT NULL,
    lora_version      INTEGER NOT NULL,
    global_prs        REAL NOT NULL,        -- coverage-weighted mean of per-cluster PRSs
    global_phase      INTEGER NOT NULL,     -- min phase across all clusters (conservative)
    parametric_pct    REAL NOT NULL,        -- weighted mean of per-cluster realtime_coverage
    cluster_state     TEXT NOT NULL,        -- JSON: full per-cluster state from version.json
    tier_distribution TEXT NOT NULL,        -- JSON: {hot, warm, cold, frozen} counts
    model_id          TEXT                  -- null for production; set by ModelScout experiments
);

-- Rolling baseline estimator (one row, updated in-place)
CREATE TABLE baseline_stats (
    id              INTEGER PRIMARY KEY,
    rolling_avg_ms  REAL NOT NULL,
    sample_count    INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
```

**`baseline_ms`** is self-calibrating: initialized from the first 50 retrieval queries using simple average, then updated with exponential smoothing (α=0.1) on every subsequent retrieval query. Every parametric or KV query uses the current rolling baseline to compute ms saved — accuracy improves as the UC accumulates retrieval traffic.

**SQLite WAL mode** is enabled on DB init. Analytics reads from the dashboard never block inference writes.

---

### Query-Time Instrumentation

**Hook location:** `pipeline/kv_inference.py` — after `decide_inference_mode` returns, before the response is returned to the caller.

```python
from core.analytics import record_query

result, phase_used, latency_ms, cluster_id = run_inference(query, cfg)
record_query(cfg, cluster_id, phase_used, latency_ms)
```

`record_query` does three things:
1. Reads current `rolling_avg_ms` from `baseline_stats`
2. Inserts a row into `query_events` with `baseline_ms = rolling_avg_ms`
3. If `phase_used == 'retrieval'`, updates `baseline_stats` with exponential smoothing

All three operations are synchronous on the calling thread. SQLite WAL mode keeps each write under 1ms; no background thread or queue is needed.

---

### Training-Round Snapshots

**Hook location:** `pipeline/prs_evaluator.py` — after per-cluster PRS is recomputed and `version.json` is updated by Dynamic PRS.

`record_round` reads directly from the cluster state that Dynamic PRS has already computed — no duplicate calculation:

- `global_prs` = coverage-weighted mean of per-cluster PRSs (per Dynamic PRS spec)
- `global_phase` = min phase across all clusters (conservative view)
- `parametric_pct` = weighted mean of per-cluster `realtime_coverage` values (weights = per-cluster `query_count`)
- `cluster_state` = full cluster dict from `version.json` (serialized to JSON)
- `tier_distribution` = tier counts from `access_tracker`

**PRS projections** use the `prs_history` list already maintained per-cluster by `prs_adapter.py`:

```python
def estimate_days_to_phase3(cluster_state: dict, avg_days_per_round: float) -> str:
    history = cluster_state.get("prs_history", [])
    threshold = cluster_state.get("threshold", 0.72)
    current = cluster_state.get("prs", 0.0)
    if len(history) < 3:
        return "insufficient history"
    slope = compute_slope(history[-3:])
    if slope <= 0:
        return "stalled — check training signal quality"
    rounds = (threshold - current) / slope
    days = rounds * avg_days_per_round
    return f"~{max(1, int(days))} days"
```

---

### ModelScout Integration

`round_snapshots.model_id` is `NULL` for production training rounds. When ModelScout runs a mini-LoRA experiment for a candidate model, it passes the model identifier:

```python
analytics.record_round(cfg, lora_version, cluster_state, tier_dist, model_id="mistral-7b-instruct")
```

This means:
- The Flywheel dashboard can render per-model PRS trajectories in the ModelScout experiments panel
- ModelScout's parameter adjustment rules read from `round_snapshots` (filtered by `model_id`) rather than parsing evaluator stdout
- Post-index ModelScout reads production `round_snapshots` (where `model_id IS NULL`) to establish the baseline before proposing experiments

---

### Per-UC Flywheel Dashboard

A new **"Flywheel"** tab in `pipeline/monitoring_dashboard.py`.

#### Three Metric Panels

**Learning Velocity (headline)**
- Large number: `parametric_pct` from most recent `round_snapshot`
- Sparkline: `parametric_pct` across last 10 `round_snapshots`
- Projection: `estimate_days_to_phase3()` output, or `"stalled"` / `"insufficient history"`

**Latency Recovered**
- `mean(baseline_ms - latency_ms)` over last 1,000 `query_events`
- Stacked bar: parametric / KV / retrieval distribution (fractions from `query_events`)

**Cost Savings**
- `tokens_saved_this_month × (cost_per_1k_tokens / 1000)` where `tokens_saved = latency_ms_saved × tokens_per_ms_baseline`
- Cumulative total since UC creation (sum over all `query_events`)
- Inline "edit" link for `cost_per_1k_tokens` — dashboard-editable, takes effect immediately

#### Cluster Cards

One card per cluster (from `cluster_state` in the latest `round_snapshot`):
- Phase badge (color-coded: green Phase 3, amber Phase 2 advancing, blue Phase 2 early)
- PRS progress bar + numeric value
- Three signal bars: `faq_coverage`, `vdb_coverage`, `realtime_coverage`
- Query count and per-cluster threshold

#### PRS History Chart

Grouped bar chart: one group per LoRA round, one bar per cluster within each group. Bars colored by cluster identity (consistent across rounds). Reads from `round_snapshots.cluster_state` JSON for the last 10 rounds.

#### ModelScout Experiments Panel

Visible only when `round_snapshots` contains rows with non-null `model_id`. Table columns: Model, Rounds, Best PRS, Status (Production / Testing… / Rejected).

---

### Studio Cross-UC Summary Panel

A new **"Flywheel"** section in `studio/routes.py` rendered on the KVForge Studio home page. Loads from each UC's `_analytics.db` at page render.

Summary table — one row per UC:

| UC Name | Phase | Parametric % | Ms Saved/Query | Est. Phase 3 | Cost Saved (mo) |
|---|---|---|---|---|---|
| customer-support | Phase 2→3 ↑ | 71% | 340ms | ~12 days | $142 |
| biomedical | Phase 2 | 43% | 210ms | ~34 days | $67 |
| bedrock-docs | Phase 3 ✓ | 89% | 510ms | Complete ✓ | $218 |

- Clicking any row navigates to that UC's per-UC Flywheel tab
- Flask icon (🧪) next to UC name when active ModelScout experiments exist
- All values read from `_analytics.db`; page render target < 500ms for ≤ 10 UCs

---

## Configuration

New per-UC fields in `DatasourceConfig`:

```json
{
  "cost_per_1k_tokens": 5.00,        // $/1000 tokens for the cloud API being replaced
  "tokens_per_ms_baseline": 0.8      // auto-calibrated from first 50 retrieval queries; editable
}
```

**Dashboard-editable:**

| Field | Dashboard label | Takes effect |
|---|---|---|
| `cost_per_1k_tokens` | "Cloud API rate ($/1k tokens)" | Immediately — cost panel recalculates on next load |

---

## Data Flow

### Query time
```
Query → kv_inference.py → decide_inference_mode → run inference
      → analytics.record_query(cluster_id, phase_used, latency_ms)
      → update baseline_stats if phase_used == 'retrieval'
```

### Training round
```
prs_evaluator.py → compute per-cluster PRS (Dynamic PRS) → update version.json
                 → analytics.record_round(cluster_state, tier_dist, model_id=None)
```

### ModelScout experiment
```
ModelScout → run mini-LoRA → prs_evaluator → analytics.record_round(model_id="<candidate>")
           → read round_snapshots WHERE model_id=<candidate> for parameter adjustment decisions
```

### Dashboard render
```
GET /flywheel → read last round_snapshot → read last 1000 query_events
             → compute metric panels + cluster cards + PRS history chart
             → render Flywheel tab HTML
```

---

## New Modules

| Module | Purpose |
|---|---|
| `core/analytics.py` | DB init, `record_query()`, `record_round()`, rolling baseline estimator, projection calculator |
| `studio/flywheel_routes.py` | FastAPI routes for cross-UC Flywheel summary in Studio |

**Modified modules:**

| Module | Change |
|---|---|
| `pipeline/kv_inference.py` | Call `analytics.record_query()` after each inference decision |
| `pipeline/prs_evaluator.py` | Call `analytics.record_round()` after each LoRA round |
| `pipeline/monitoring_dashboard.py` | Add Flywheel tab (3 metric panels, cluster cards, PRS history, ModelScout panel) |
| `core/config.py` | Add `cost_per_1k_tokens`, `tokens_per_ms_baseline` fields |
| `studio/routes.py` | Add Flywheel summary panel to Studio home |

---

## What Does Not Change

- Dynamic PRS cluster state computation (`core/prs_adapter.py`) — analytics reads from it, does not duplicate it
- LoRA training mechanism — unchanged
- Replay buffer tier system — analytics reads tier distribution, does not modify it
- KV inference routing logic — analytics is a passive observer, not a decision-maker

---

## Success Criteria

1. Cluster cards reflect actual per-cluster phase and three-signal coverage from Dynamic PRS state — no duplicate computation
2. ModelScout experiment rounds tagged with `model_id` appear in the Flywheel experiments panel, with per-model PRS trajectory visible
3. Cost savings number updates immediately when `cost_per_1k_tokens` is edited in the dashboard
4. PRS projection shows `"stalled — check training signal quality"` (not a date) when slope ≤ 0 across last 3 rounds
5. Cross-UC Studio Flywheel panel renders in < 500ms for ≤ 10 UCs
