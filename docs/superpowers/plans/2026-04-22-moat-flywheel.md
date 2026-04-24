# Flywheel Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flywheel analytics layer that instruments every query and LoRA round, persists metrics to per-UC SQLite DBs, and surfaces learning velocity, latency recovered, and cost savings in both the per-UC monitoring dashboard and KVForge Studio cross-UC summary.

**Architecture:** A new `core/analytics.py` module owns DB init, `record_query()` (called from the monitoring dashboard after each inference), and `record_round()` (called from prs_evaluator after each LoRA round). Query helpers power two new dashboard surfaces: a Flywheel tab in the per-UC monitoring dashboard and a cross-UC Flywheel panel in KVForge Studio. The analytics DB is a separate SQLite file (`<collection>_analytics.db`) from the replay buffer so reads never block training.

**Tech Stack:** Python 3.11+, SQLite (stdlib `sqlite3`, WAL mode), FastAPI, Pydantic, inline HTML/CSS/JS

---

## File Structure

**New files:**
- `core/analytics.py` — DB init, `record_query`, `record_round`, projection helpers, query helpers
- `studio/flywheel_routes.py` — FastAPI routes for cross-UC Studio Flywheel summary API
- `tests/test_analytics.py` — unit tests for `core/analytics.py`
- `tests/test_flywheel_routes.py` — tests for `studio/flywheel_routes.py`

**Modified files:**
- `core/config.py` — add `analytics_db`, `cost_per_1k_tokens`, `tokens_per_ms_baseline`
- `pipeline/monitoring_dashboard.py` — hook `record_query` in `run_query`; add `/api/flywheel` endpoint; add `/api/flywheel/cost-rate` PATCH; add Flywheel HTML tab
- `pipeline/prs_evaluator.py` — hook `record_round` in `evaluate()`
- `studio/routes.py` — include `flywheel_routes.router`

---

### Task 1: Analytics DB Schema and Init

**Files:**
- Create: `core/analytics.py`
- Create: `tests/test_analytics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analytics.py
import os
import sqlite3
import tempfile
import pytest


def _cfg(tmp_path, suffix=""):
    return {
        "collection": f"test_collection{suffix}",
        "analytics_db": str(tmp_path / f"test{suffix}_analytics.db"),
        "cost_per_1k_tokens": 5.0,
        "tokens_per_ms_baseline": 0.8,
    }


def test_init_db_creates_tables(tmp_path):
    from core.analytics import init_db, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    path = _db_path(cfg)
    assert os.path.exists(path)
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    db.close()
    assert "query_events" in tables
    assert "round_snapshots" in tables
    assert "baseline_stats" in tables


def test_init_db_enables_wal(tmp_path):
    from core.analytics import init_db, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    db = sqlite3.connect(_db_path(cfg))
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    assert mode == "wal"


def test_db_path_defaults_to_collection(tmp_path):
    from core.analytics import _db_path
    cfg = {"collection": "my_corpus", "analytics_db": ""}
    assert _db_path(cfg) == "my_corpus_analytics.db"


def test_db_path_uses_analytics_db_field(tmp_path):
    from core.analytics import _db_path
    cfg = {"collection": "ignored", "analytics_db": "/some/path/custom.db"}
    assert _db_path(cfg) == "/some/path/custom.db"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_analytics.py -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'core.analytics'`

- [ ] **Step 3: Implement DB init**

```python
# core/analytics.py
"""Per-UC flywheel analytics: SQLite DB for query events and LoRA round snapshots."""
import json
import sqlite3
import time
from typing import Optional


def _db_path(cfg: dict) -> str:
    path = cfg.get("analytics_db", "")
    return path if path else f"{cfg.get('collection', 'default')}_analytics.db"


def _connect(cfg: dict) -> sqlite3.Connection:
    db = sqlite3.connect(_db_path(cfg))
    db.execute("PRAGMA journal_mode=WAL")
    return db


def init_db(cfg: dict) -> None:
    db = _connect(cfg)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS query_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          INTEGER NOT NULL,
            cluster_id  TEXT,
            phase_used  TEXT NOT NULL,
            latency_ms  REAL NOT NULL,
            baseline_ms REAL NOT NULL,
            model_id    TEXT
        );
        CREATE TABLE IF NOT EXISTS round_snapshots (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts                INTEGER NOT NULL,
            lora_version      INTEGER NOT NULL,
            global_prs        REAL NOT NULL,
            global_phase      INTEGER NOT NULL,
            parametric_pct    REAL NOT NULL,
            cluster_state     TEXT NOT NULL,
            tier_distribution TEXT NOT NULL,
            model_id          TEXT
        );
        CREATE TABLE IF NOT EXISTS baseline_stats (
            id              INTEGER PRIMARY KEY,
            rolling_avg_ms  REAL NOT NULL,
            sample_count    INTEGER NOT NULL,
            updated_at      INTEGER NOT NULL
        );
    """)
    db.commit()
    db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_analytics.py::test_init_db_creates_tables tests/test_analytics.py::test_init_db_enables_wal tests/test_analytics.py::test_db_path_defaults_to_collection tests/test_analytics.py::test_db_path_uses_analytics_db_field -v --override-ini="addopts="
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/analytics.py tests/test_analytics.py
git commit -m "feat: add analytics DB schema and init (flywheel task 1)"
```

---

### Task 2: `record_query()` with Rolling Baseline Estimator

**Files:**
- Modify: `core/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_analytics.py

def test_record_query_inserts_row(tmp_path):
    from core.analytics import init_db, record_query, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=800.0)
    db = sqlite3.connect(_db_path(cfg))
    rows = db.execute("SELECT phase_used, latency_ms FROM query_events").fetchall()
    db.close()
    assert len(rows) == 1
    assert rows[0][0] == "retrieval"
    assert rows[0][1] == 800.0


def test_record_query_baseline_bootstraps_from_first_retrieval(tmp_path):
    from core.analytics import init_db, record_query, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=1000.0)
    db = sqlite3.connect(_db_path(cfg))
    row = db.execute("SELECT rolling_avg_ms, sample_count FROM baseline_stats WHERE id=1").fetchone()
    db.close()
    assert row is not None
    assert row[0] == 1000.0
    assert row[1] == 1


def test_record_query_parametric_does_not_update_baseline(tmp_path):
    from core.analytics import init_db, record_query, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=800.0)
    record_query(cfg, cluster_id=None, phase_used="parametric", latency_ms=200.0)
    db = sqlite3.connect(_db_path(cfg))
    row = db.execute("SELECT sample_count FROM baseline_stats WHERE id=1").fetchone()
    db.close()
    assert row[0] == 1  # still 1 — parametric query did not update baseline


def test_record_query_baseline_smooths_after_50(tmp_path):
    from core.analytics import init_db, record_query, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    for _ in range(50):
        record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=1000.0)
    record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=500.0)
    db = sqlite3.connect(_db_path(cfg))
    row = db.execute("SELECT rolling_avg_ms FROM baseline_stats WHERE id=1").fetchone()
    db.close()
    # Expected: 0.9*1000 + 0.1*500 = 950
    assert abs(row[0] - 950.0) < 1.0


def test_record_query_baseline_ms_stored_per_event(tmp_path):
    from core.analytics import init_db, record_query, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=700.0)
    record_query(cfg, cluster_id=None, phase_used="parametric", latency_ms=200.0)
    db = sqlite3.connect(_db_path(cfg))
    rows = db.execute("SELECT baseline_ms FROM query_events ORDER BY id").fetchall()
    db.close()
    # First retrieval: baseline bootstraps from 700 and stores 700 (initial value before update)
    assert rows[0][0] == 700.0
    # Parametric: reads current baseline (700), stores 700
    assert rows[1][0] == 700.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analytics.py::test_record_query_inserts_row -v --override-ini="addopts="
```
Expected: FAIL with `ImportError: cannot import name 'record_query'`

- [ ] **Step 3: Implement `record_query()`**

```python
# append to core/analytics.py

def record_query(cfg: dict, cluster_id: Optional[str], phase_used: str,
                 latency_ms: float, model_id: Optional[str] = None) -> None:
    db = _connect(cfg)
    # Read current baseline
    row = db.execute(
        "SELECT rolling_avg_ms, sample_count FROM baseline_stats WHERE id=1"
    ).fetchone()
    if row:
        baseline_ms, sample_count = row[0], row[1]
    else:
        baseline_ms = latency_ms
        sample_count = 0

    # Insert query event (baseline_ms is the value BEFORE this query updates it)
    db.execute(
        "INSERT INTO query_events (ts, cluster_id, phase_used, latency_ms, baseline_ms, model_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (int(time.time()), cluster_id, phase_used, latency_ms, baseline_ms, model_id),
    )

    # Update rolling baseline only on retrieval queries
    if phase_used == "retrieval":
        if sample_count < 50:
            new_avg = (baseline_ms * sample_count + latency_ms) / (sample_count + 1)
            new_count = sample_count + 1
        else:
            new_avg = 0.9 * baseline_ms + 0.1 * latency_ms
            new_count = sample_count + 1
        db.execute(
            "INSERT OR REPLACE INTO baseline_stats (id, rolling_avg_ms, sample_count, updated_at) "
            "VALUES (1, ?, ?, ?)",
            (new_avg, new_count, int(time.time())),
        )

    db.commit()
    db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_analytics.py::test_record_query_inserts_row tests/test_analytics.py::test_record_query_baseline_bootstraps_from_first_retrieval tests/test_analytics.py::test_record_query_parametric_does_not_update_baseline tests/test_analytics.py::test_record_query_baseline_smooths_after_50 tests/test_analytics.py::test_record_query_baseline_ms_stored_per_event -v --override-ini="addopts="
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/analytics.py tests/test_analytics.py
git commit -m "feat: add record_query with rolling baseline estimator (flywheel task 2)"
```

---

### Task 3: `record_round()`, `compute_slope()`, `estimate_days_to_phase3()`

**Files:**
- Modify: `core/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_analytics.py

def _cluster_state():
    return {
        "0": {
            "label": "auth", "phase": 2, "prs": 0.72,
            "faq_coverage": 0.80, "vdb_coverage": 0.70, "realtime_coverage": 0.65,
            "learned_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
            "threshold": 0.75, "prs_history": [0.55, 0.63, 0.72], "query_count": 100,
        },
        "1": {
            "label": "billing", "phase": 1, "prs": 0.50,
            "faq_coverage": 0.60, "vdb_coverage": 0.45, "realtime_coverage": 0.40,
            "learned_weights": {"faq": 0.4, "vdb": 0.4, "realtime": 0.2},
            "threshold": 0.75, "prs_history": [0.40, 0.45, 0.50], "query_count": 40,
        },
    }


def test_record_round_inserts_snapshot(tmp_path):
    from core.analytics import init_db, record_round, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_round(cfg, lora_version=1, cluster_state=_cluster_state(),
                 tier_distribution={"hot": 5, "warm": 20, "cold": 30, "frozen": 100})
    db = sqlite3.connect(_db_path(cfg))
    rows = db.execute("SELECT lora_version, global_phase FROM round_snapshots").fetchall()
    db.close()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] == 1  # min(phase 2, phase 1) = 1


def test_record_round_global_prs_weighted(tmp_path):
    from core.analytics import init_db, record_round, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_round(cfg, lora_version=1, cluster_state=_cluster_state(),
                 tier_distribution={})
    db = sqlite3.connect(_db_path(cfg))
    row = db.execute("SELECT global_prs FROM round_snapshots").fetchone()
    db.close()
    # weighted mean: (0.72*100 + 0.50*40) / 140 = (72+20)/140 = 92/140 ≈ 0.657
    assert abs(row[0] - (0.72*100 + 0.50*40)/140) < 0.001


def test_record_round_model_id_stored(tmp_path):
    from core.analytics import init_db, record_round, _db_path
    cfg = _cfg(tmp_path)
    init_db(cfg)
    record_round(cfg, lora_version=2, cluster_state=_cluster_state(),
                 tier_distribution={}, model_id="mistral-7b")
    db = sqlite3.connect(_db_path(cfg))
    row = db.execute("SELECT model_id FROM round_snapshots").fetchone()
    db.close()
    assert row[0] == "mistral-7b"


def test_compute_slope_positive():
    from core.analytics import compute_slope
    assert compute_slope([0.5, 0.6, 0.7]) > 0


def test_compute_slope_flat():
    from core.analytics import compute_slope
    assert compute_slope([0.6, 0.6, 0.6]) == 0.0


def test_compute_slope_single_value():
    from core.analytics import compute_slope
    assert compute_slope([0.7]) == 0.0


def test_estimate_days_insufficient_history():
    from core.analytics import estimate_days_to_phase3
    cluster = {"prs": 0.5, "threshold": 0.75, "prs_history": [0.4, 0.5]}
    assert estimate_days_to_phase3(cluster, avg_days_per_round=1.0) == "insufficient history"


def test_estimate_days_stalled():
    from core.analytics import estimate_days_to_phase3
    cluster = {"prs": 0.5, "threshold": 0.75, "prs_history": [0.6, 0.55, 0.5]}
    result = estimate_days_to_phase3(cluster, avg_days_per_round=1.0)
    assert "stalled" in result


def test_estimate_days_returns_days_string():
    from core.analytics import estimate_days_to_phase3
    cluster = {"prs": 0.60, "threshold": 0.75, "prs_history": [0.50, 0.55, 0.60]}
    result = estimate_days_to_phase3(cluster, avg_days_per_round=2.0)
    assert result.startswith("~")
    assert "days" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analytics.py::test_record_round_inserts_snapshot -v --override-ini="addopts="
```
Expected: FAIL with `ImportError: cannot import name 'record_round'`

- [ ] **Step 3: Implement `record_round`, `compute_slope`, `estimate_days_to_phase3`**

```python
# append to core/analytics.py

def compute_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


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


def record_round(cfg: dict, lora_version: int, cluster_state: dict,
                 tier_distribution: dict, model_id: Optional[str] = None) -> None:
    clusters = cluster_state
    total_queries = sum(c.get("query_count", 0) for c in clusters.values())

    if total_queries > 0:
        global_prs = sum(
            c.get("prs", 0.0) * c.get("query_count", 0) for c in clusters.values()
        ) / total_queries
        parametric_pct = sum(
            c.get("realtime_coverage", 0.0) * c.get("query_count", 0)
            for c in clusters.values()
        ) / total_queries
    else:
        n = max(len(clusters), 1)
        global_prs = sum(c.get("prs", 0.0) for c in clusters.values()) / n
        parametric_pct = 0.0

    global_phase = min((c.get("phase", 1) for c in clusters.values()), default=1)

    db = _connect(cfg)
    db.execute(
        "INSERT INTO round_snapshots "
        "(ts, lora_version, global_prs, global_phase, parametric_pct, "
        "cluster_state, tier_distribution, model_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (int(time.time()), lora_version, global_prs, global_phase, parametric_pct,
         json.dumps(cluster_state), json.dumps(tier_distribution), model_id),
    )
    db.commit()
    db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_analytics.py -k "round or slope or estimate" -v --override-ini="addopts="
```
Expected: 9 PASSED

- [ ] **Step 5: Commit**

```bash
git add core/analytics.py tests/test_analytics.py
git commit -m "feat: add record_round, compute_slope, estimate_days_to_phase3 (flywheel task 3)"
```

---

### Task 4: Query Helpers (`get_flywheel_summary`, `get_cluster_cards`, `get_prs_history`, `get_modelscout_experiments`)

**Files:**
- Modify: `core/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_analytics.py

def _seed_db(tmp_path):
    """Seed analytics DB with one round snapshot and several query events."""
    from core.analytics import init_db, record_query, record_round
    cfg = _cfg(tmp_path, suffix="_q4")
    init_db(cfg)
    # 5 retrieval queries to bootstrap baseline
    for _ in range(5):
        record_query(cfg, cluster_id=None, phase_used="retrieval", latency_ms=800.0)
    # parametric and kv queries
    record_query(cfg, cluster_id="0", phase_used="parametric", latency_ms=200.0)
    record_query(cfg, cluster_id="0", phase_used="kv", latency_ms=350.0)
    # round snapshot
    record_round(cfg, lora_version=1, cluster_state=_cluster_state(),
                 tier_distribution={"hot": 5, "warm": 10, "cold": 20, "frozen": 50})
    # modelscout experiment snapshot
    record_round(cfg, lora_version=1, cluster_state=_cluster_state(),
                 tier_distribution={}, model_id="mistral-7b")
    return cfg


def test_get_flywheel_summary_keys(tmp_path):
    from core.analytics import get_flywheel_summary
    cfg = _seed_db(tmp_path)
    summary = get_flywheel_summary(cfg)
    for key in ("global_prs", "global_phase", "parametric_pct", "sparkline",
                "ms_saved_per_query", "phase_dist", "cost_saved_month",
                "cost_saved_total", "cost_per_1k_tokens", "projection"):
        assert key in summary, f"missing key: {key}"


def test_get_flywheel_summary_no_data(tmp_path):
    from core.analytics import init_db, get_flywheel_summary
    cfg = _cfg(tmp_path, suffix="_empty")
    init_db(cfg)
    summary = get_flywheel_summary(cfg)
    assert summary.get("no_data") is True


def test_get_cluster_cards_returns_list(tmp_path):
    from core.analytics import get_cluster_cards
    cfg = _seed_db(tmp_path)
    cards = get_cluster_cards(cfg)
    assert isinstance(cards, list)
    assert len(cards) == 2
    assert all("cluster_id" in c for c in cards)
    assert all("prs" in c for c in cards)
    assert all("faq_coverage" in c for c in cards)


def test_get_prs_history_returns_rounds(tmp_path):
    from core.analytics import get_prs_history
    cfg = _seed_db(tmp_path)
    history = get_prs_history(cfg, n_rounds=10)
    # Only production rounds (model_id IS NULL) — 1 round seeded
    assert len(history) == 1
    assert "lora_version" in history[0]
    assert "per_cluster_prs" in history[0]


def test_get_modelscout_experiments_returns_experiments(tmp_path):
    from core.analytics import get_modelscout_experiments
    cfg = _seed_db(tmp_path)
    exps = get_modelscout_experiments(cfg)
    assert len(exps) == 1
    assert exps[0]["model_id"] == "mistral-7b"
    assert exps[0]["rounds"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_analytics.py::test_get_flywheel_summary_keys -v --override-ini="addopts="
```
Expected: FAIL with `ImportError: cannot import name 'get_flywheel_summary'`

- [ ] **Step 3: Implement query helpers**

```python
# append to core/analytics.py

def get_flywheel_summary(cfg: dict) -> dict:
    db = _connect(cfg)
    snap = db.execute(
        "SELECT global_prs, global_phase, parametric_pct, cluster_state, "
        "tier_distribution, ts FROM round_snapshots WHERE model_id IS NULL "
        "ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    if not snap:
        db.close()
        return {"no_data": True}

    sparkline_rows = db.execute(
        "SELECT parametric_pct FROM round_snapshots WHERE model_id IS NULL "
        "ORDER BY ts DESC LIMIT 10"
    ).fetchall()
    events = db.execute(
        "SELECT phase_used, latency_ms, baseline_ms, ts "
        "FROM query_events ORDER BY ts DESC LIMIT 1000"
    ).fetchall()
    db.close()

    global_prs, global_phase, parametric_pct, cs_json, td_json, snap_ts = snap
    cluster_state = json.loads(cs_json)

    phase_counts: dict[str, int] = {}
    ms_saved_list = []
    for phase_used, latency_ms, baseline_ms, _ in events:
        phase_counts[phase_used] = phase_counts.get(phase_used, 0) + 1
        ms_saved_list.append(baseline_ms - latency_ms)

    total = sum(phase_counts.values())
    phase_dist = {k: round(v / total, 4) for k, v in phase_counts.items()} if total else {}
    ms_saved_per_query = sum(ms_saved_list) / len(ms_saved_list) if ms_saved_list else 0.0

    month_ago = int(time.time()) - 30 * 86400
    monthly_saved_ms = sum(r[2] - r[1] for r in events if r[3] >= month_ago)
    all_saved_ms = sum(r[2] - r[1] for r in events)
    tokens_per_ms = cfg.get("tokens_per_ms_baseline", 0.8)
    cost_per_1k = cfg.get("cost_per_1k_tokens", 5.0)
    cost_saved_month = round(monthly_saved_ms * tokens_per_ms * cost_per_1k / 1000, 2)
    cost_saved_total = round(all_saved_ms * tokens_per_ms * cost_per_1k / 1000, 2)

    projection = "insufficient history"
    for cluster in cluster_state.values():
        proj = estimate_days_to_phase3(cluster, avg_days_per_round=1.0)
        if "stalled" in proj:
            projection = proj
            break
        if proj != "insufficient history":
            projection = proj

    return {
        "global_prs": global_prs,
        "global_phase": global_phase,
        "parametric_pct": round(parametric_pct, 4),
        "sparkline": [r[0] for r in sparkline_rows],
        "ms_saved_per_query": round(ms_saved_per_query, 1),
        "phase_dist": phase_dist,
        "cost_saved_month": cost_saved_month,
        "cost_saved_total": cost_saved_total,
        "cost_per_1k_tokens": cost_per_1k,
        "projection": projection,
        "tier_distribution": json.loads(td_json),
    }


def get_cluster_cards(cfg: dict) -> list[dict]:
    db = _connect(cfg)
    row = db.execute(
        "SELECT cluster_state FROM round_snapshots WHERE model_id IS NULL "
        "ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    db.close()
    if not row:
        return []
    cluster_state = json.loads(row[0])
    cards = []
    for cluster_id, state in cluster_state.items():
        cards.append({
            "cluster_id": cluster_id,
            "label": state.get("label", cluster_id),
            "phase": state.get("phase", 1),
            "prs": state.get("prs", 0.0),
            "faq_coverage": state.get("faq_coverage", 0.0),
            "vdb_coverage": state.get("vdb_coverage", 0.0),
            "realtime_coverage": state.get("realtime_coverage", 0.0),
            "query_count": state.get("query_count", 0),
            "threshold": state.get("threshold", 0.72),
        })
    return sorted(cards, key=lambda c: -c["prs"])


def get_prs_history(cfg: dict, n_rounds: int = 10) -> list[dict]:
    db = _connect(cfg)
    rows = db.execute(
        "SELECT lora_version, cluster_state, ts FROM round_snapshots "
        "WHERE model_id IS NULL ORDER BY ts DESC LIMIT ?",
        (n_rounds,),
    ).fetchall()
    db.close()
    history = []
    for lora_version, cs_json, ts in reversed(rows):
        cluster_state = json.loads(cs_json)
        per_cluster = {cid: state.get("prs", 0.0) for cid, state in cluster_state.items()}
        history.append({"lora_version": lora_version, "ts": ts, "per_cluster_prs": per_cluster})
    return history


def get_modelscout_experiments(cfg: dict) -> list[dict]:
    db = _connect(cfg)
    rows = db.execute(
        "SELECT model_id, COUNT(*) as rounds, MAX(global_prs) as best_prs "
        "FROM round_snapshots WHERE model_id IS NOT NULL "
        "GROUP BY model_id ORDER BY best_prs DESC"
    ).fetchall()
    baseline = db.execute(
        "SELECT MAX(global_prs) FROM round_snapshots WHERE model_id IS NULL"
    ).fetchone()
    db.close()
    baseline_prs = baseline[0] if baseline and baseline[0] else 0.0
    return [
        {
            "model_id": model_id,
            "rounds": rounds,
            "best_prs": best_prs,
            "status": "Candidate" if best_prs >= baseline_prs else "Testing…",
        }
        for model_id, rounds, best_prs in rows
    ]
```

- [ ] **Step 4: Run all analytics tests**

```bash
python -m pytest tests/test_analytics.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add core/analytics.py tests/test_analytics.py
git commit -m "feat: add flywheel query helpers (get_flywheel_summary, cards, history, experiments)"
```

---

### Task 5: Config Fields

**Files:**
- Modify: `core/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_flywheel_config_defaults():
    from core.config import DatasourceConfig
    cfg = DatasourceConfig(
        collection="test", embed_model="BAAI/bge-small-en-v1.5",
        vector_dim=384, llm_model="meta-llama/Llama-3.2-3B-Instruct",
        checkpoint_dir="ckpt", version_file="version.json", replay_db="test_replay.db"
    )
    assert cfg.analytics_db == ""
    assert cfg.cost_per_1k_tokens == 5.0
    assert cfg.tokens_per_ms_baseline == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py::test_flywheel_config_defaults -v --override-ini="addopts="
```
Expected: FAIL with `ValidationError` or `AttributeError`

- [ ] **Step 3: Add config fields**

In `core/config.py`, add these three fields inside `DatasourceConfig`, after the `dashboard_port` field:

```python
    # Flywheel analytics
    analytics_db: str = ""          # defaults to {collection}_analytics.db
    cost_per_1k_tokens: float = 5.0
    tokens_per_ms_baseline: float = 0.8
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py::test_flywheel_config_defaults -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Run all config tests**

```bash
python -m pytest tests/test_config.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add core/config.py tests/test_config.py
git commit -m "feat: add analytics_db, cost_per_1k_tokens, tokens_per_ms_baseline config fields"
```

---

### Task 6: Hook `record_query` into Monitoring Dashboard

**Files:**
- Modify: `pipeline/monitoring_dashboard.py`
- Modify: `tests/test_dashboard.py`

The hook goes in the `run_query` endpoint (line ~857 in monitoring_dashboard.py), right after `result_a` is resolved. The `mode` field in `result_a` uses dashboard-internal values (`"parametric"`, `"kv_injection"`, `"text_in_context"`, `"text_fallback"`); we normalize to spec values (`"parametric"`, `"kv"`, `"retrieval"`) before recording.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard.py
from unittest.mock import patch, MagicMock


def test_run_query_calls_record_query(tmp_path):
    """record_query is called once per run_query invocation."""
    import importlib
    import sys

    # Ensure dashboard is importable without GPU
    with patch.dict("sys.modules", {
        "core.model_loader": MagicMock(),
        "pipeline.kv_background": MagicMock(),
        "pipeline.kv_inference": MagicMock(),
    }):
        from fastapi.testclient import TestClient
        import pipeline.monitoring_dashboard as md

        with patch("core.analytics.record_query") as mock_record, \
             patch("pipeline.monitoring_dashboard._answer_kvforge",
                   return_value={"answer": "ok", "latency_ms": 500, "mode": "parametric"}), \
             patch("pipeline.monitoring_dashboard._answer_gemini",
                   return_value={"answer": "ok", "latency_ms": 600}), \
             patch("pipeline.monitoring_dashboard._load_cfg",
                   return_value={"collection": "test", "analytics_db": str(tmp_path / "t.db"),
                                 "cost_per_1k_tokens": 5.0, "tokens_per_ms_baseline": 0.8}):
            from core.analytics import init_db
            init_db({"collection": "test", "analytics_db": str(tmp_path / "t.db")})
            client = TestClient(md.app)
            resp = client.post("/api/query", json={
                "query": "hello", "a_top_k": 5,
                "a_max_new_tokens": 64, "a_temperature": 0.0,
                "b_max_new_tokens": 64, "b_temperature": 0.0,
            })
            assert mock_record.called
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dashboard.py::test_run_query_calls_record_query -v --override-ini="addopts="
```
Expected: FAIL — `mock_record` not called

- [ ] **Step 3: Add `record_query` hook to `run_query`**

In `pipeline/monitoring_dashboard.py`, add the import at the top (near other core imports):

```python
import core.analytics as _analytics
```

Then in the `run_query` endpoint, after `result_a, result_b = await asyncio.gather(fut_a, fut_b)`, add:

```python
    # Normalize mode and record query event for flywheel analytics
    _mode_map = {"parametric": "parametric", "kv_injection": "kv",
                 "text_in_context": "retrieval", "text_fallback": "retrieval"}
    try:
        _analytics.record_query(
            cfg,
            cluster_id=None,
            phase_used=_mode_map.get(result_a.get("mode", "retrieval"), "retrieval"),
            latency_ms=float(result_a.get("latency_ms", 0)),
        )
    except Exception:
        pass  # analytics must never break inference
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dashboard.py::test_run_query_calls_record_query -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Run full dashboard test suite**

```bash
python -m pytest tests/test_dashboard.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add pipeline/monitoring_dashboard.py tests/test_dashboard.py
git commit -m "feat: hook record_query into monitoring_dashboard run_query endpoint"
```

---

### Task 7: Hook `record_round` into `prs_evaluator.py`

**Files:**
- Modify: `pipeline/prs_evaluator.py`
- Modify: `tests/test_prs_evaluator.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_prs_evaluator.py
from unittest.mock import patch, MagicMock


def test_evaluate_calls_record_round(tmp_path):
    """record_round is called after evaluate() computes PRS."""
    faqs = [{"question": "What is X?", "answer": "X is Y."}]
    cfg = {
        "collection": "test",
        "analytics_db": str(tmp_path / "test_analytics.db"),
        "cost_per_1k_tokens": 5.0,
        "tokens_per_ms_baseline": 0.8,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "faq_question_key": "question",
        "faq_answer_key": "answer",
    }
    from core.analytics import init_db
    init_db(cfg)

    with patch("core.analytics.record_round") as mock_rr, \
         patch("pipeline.prs_evaluator.model_loader") as mock_ml, \
         patch("pipeline.prs_evaluator._generate_parametric", return_value="X is Y."), \
         patch("pipeline.prs_evaluator._extract_confidence", return_value=0.9), \
         patch("pipeline.prs_evaluator._self_consistency", return_value=0.9), \
         patch("pipeline.prs_evaluator.ver") as mock_ver, \
         patch("pipeline.prs_evaluator.TextEmbedding") as mock_emb:
        import numpy as np
        mock_ml.load.return_value = (MagicMock(), MagicMock())
        mock_ver.load.return_value = {"prs_history": [], "phase": 1}
        mock_ver.get_lora_version.return_value = 1
        mock_ver.save = MagicMock()

        fake_emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        mock_emb.return_value.embed.return_value = iter(fake_emb)

        from pipeline.prs_evaluator import evaluate
        evaluate(faqs, cfg, lora_checkpoint=None)

        assert mock_rr.called
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_prs_evaluator.py::test_evaluate_calls_record_round -v --override-ini="addopts="
```
Expected: FAIL — `mock_rr` not called

- [ ] **Step 3: Add `record_round` hook in `evaluate()`**

At the top of `pipeline/prs_evaluator.py`, add:

```python
import core.analytics as _analytics
```

At the end of `evaluate()` in `prs_evaluator.py`, just before `return prs`, add:

```python
    # Record flywheel analytics snapshot
    try:
        version_data = ver.load()
        cluster_state = version_data.get("clusters", {})
        if not cluster_state:
            # Pre-cluster fallback: wrap global PRS as single synthetic cluster
            cluster_state = {"_global": {
                "prs": prs, "phase": version_data.get("phase", 1),
                "faq_coverage": float(np.mean(accuracy_ratios)) if accuracy_ratios else 0.0,
                "vdb_coverage": 0.0, "realtime_coverage": 0.0,
                "query_count": len(faqs), "threshold": cfg.get("prs_threshold", 0.75),
                "prs_history": [h.get("prs", 0.0) for h in version_data.get("prs_history", [])],
            }}
        tier_dist = version_data.get("tier_distribution", {})
        _analytics.record_round(
            cfg,
            lora_version=ver.get_lora_version(),
            cluster_state=cluster_state,
            tier_distribution=tier_dist,
        )
    except Exception:
        pass  # analytics must never break PRS evaluation
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_prs_evaluator.py::test_evaluate_calls_record_round -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Run all PRS evaluator tests**

```bash
python -m pytest tests/test_prs_evaluator.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add pipeline/prs_evaluator.py tests/test_prs_evaluator.py
git commit -m "feat: hook record_round into prs_evaluator.evaluate() (flywheel task 7)"
```

---

### Task 8: `/api/flywheel` JSON Endpoint in Monitoring Dashboard

**Files:**
- Modify: `pipeline/monitoring_dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard.py

def test_api_flywheel_returns_summary(tmp_path):
    from fastapi.testclient import TestClient
    import pipeline.monitoring_dashboard as md
    from core.analytics import init_db, record_round

    cfg = {"collection": "test2",
           "analytics_db": str(tmp_path / "test2_analytics.db"),
           "cost_per_1k_tokens": 5.0,
           "tokens_per_ms_baseline": 0.8}
    init_db(cfg)
    cs = {"0": {"label": "auth", "phase": 2, "prs": 0.72,
                "faq_coverage": 0.8, "vdb_coverage": 0.7, "realtime_coverage": 0.6,
                "learned_weights": {}, "threshold": 0.75,
                "prs_history": [0.55, 0.63, 0.72], "query_count": 50}}
    record_round(cfg, lora_version=1, cluster_state=cs, tier_distribution={})

    with patch("pipeline.monitoring_dashboard._load_cfg", return_value=cfg):
        client = TestClient(md.app)
        resp = client.get("/api/flywheel")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "cluster_cards" in data
        assert "prs_history" in data
        assert "experiments" in data
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dashboard.py::test_api_flywheel_returns_summary -v --override-ini="addopts="
```
Expected: FAIL with 404 (endpoint not found)

- [ ] **Step 3: Add `/api/flywheel` endpoint to `monitoring_dashboard.py`**

```python
@app.get("/api/flywheel")
def get_flywheel():
    cfg = _load_cfg()
    import core.analytics as _an
    _an.init_db(cfg)
    return {
        "summary": _an.get_flywheel_summary(cfg),
        "cluster_cards": _an.get_cluster_cards(cfg),
        "prs_history": _an.get_prs_history(cfg),
        "experiments": _an.get_modelscout_experiments(cfg),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dashboard.py::test_api_flywheel_returns_summary -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitoring_dashboard.py tests/test_dashboard.py
git commit -m "feat: add /api/flywheel JSON endpoint to monitoring dashboard"
```

---

### Task 9: Flywheel HTML Tab in Monitoring Dashboard

**Files:**
- Modify: `pipeline/monitoring_dashboard.py`

This task adds the `/flywheel` HTML endpoint serving the Flywheel tab page. The page fetches `/api/flywheel` on load and renders cluster cards, metric panels, PRS history chart, and ModelScout experiments panel.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard.py

def test_flywheel_tab_returns_html(tmp_path):
    from fastapi.testclient import TestClient
    import pipeline.monitoring_dashboard as md

    cfg = {"collection": "test3", "analytics_db": str(tmp_path / "t3.db"),
           "cost_per_1k_tokens": 5.0, "tokens_per_ms_baseline": 0.8}
    with patch("pipeline.monitoring_dashboard._load_cfg", return_value=cfg):
        client = TestClient(md.app)
        resp = client.get("/flywheel")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Flywheel" in resp.text
        assert "Learning Velocity" in resp.text
        assert "Cluster Progress" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dashboard.py::test_flywheel_tab_returns_html -v --override-ini="addopts="
```
Expected: FAIL with 404

- [ ] **Step 3: Add `/flywheel` HTML endpoint**

```python
@app.get("/flywheel", response_class=HTMLResponse)
def flywheel_tab():
    _load_cfg()
    return HTMLResponse(content=_FLYWHEEL_HTML)
```

And define `_FLYWHEEL_HTML` as a module-level constant in `monitoring_dashboard.py`:

```python
_FLYWHEEL_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>KVForge — Flywheel Analytics</title>
<style>
body{font-family:monospace;background:#0d0d0d;color:#ccc;margin:0;padding:20px}
h1{color:#fff;font-size:18px;margin-bottom:20px}
.nav a{color:#6b9fd4;text-decoration:none;margin-right:16px;font-size:13px}
.nav a:hover{text-decoration:underline}
.panels{display:flex;gap:16px;margin:20px 0}
.panel{flex:1;background:#111;border:1px solid #222;border-radius:6px;padding:16px}
.panel-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#666;margin-bottom:6px}
.panel-value{font-size:32px;font-weight:bold;margin-bottom:4px}
.panel-sub{font-size:12px;color:#888}
.sparkline{display:flex;align-items:flex-end;gap:3px;height:30px;margin-top:8px}
.spark-bar{background:#444;border-radius:2px 2px 0 0;min-width:10px}
.projection{font-size:12px;margin-top:6px}
.phase-bar{display:flex;height:10px;border-radius:4px;overflow:hidden;gap:1px;margin-top:8px}
.phase-legend{display:flex;gap:12px;margin-top:4px;font-size:10px;color:#888}
.section-title{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#666;margin:20px 0 10px}
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.cluster-card{background:#111;border:1px solid #222;border-radius:8px;padding:14px}
.card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.card-name{font-size:13px;font-weight:bold}
.phase-badge{font-size:10px;padding:2px 8px;border-radius:10px}
.phase-3-badge{background:#1a4a1a;color:#6fcf6f}
.phase-2-badge{background:#2a2a00;color:#f0c040}
.phase-1-badge{background:#1a1a3a;color:#6b9fd4}
.prs-bar-wrap{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.prs-bar-track{flex:1;background:#1a1a1a;border-radius:3px;height:8px;overflow:hidden}
.prs-bar-fill{height:100%;border-radius:3px}
.prs-value{font-size:13px;font-weight:bold;min-width:36px}
.signal-row{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.signal-label{font-size:10px;color:#888;width:56px}
.signal-track{flex:1;background:#1a1a1a;border-radius:2px;height:5px;overflow:hidden}
.signal-fill{height:100%}
.signal-val{font-size:10px;min-width:28px}
.card-footer{font-size:10px;color:#555;margin-top:8px}
.history-chart{background:#111;border:1px solid #222;border-radius:6px;padding:14px;margin-top:0}
.chart-bars{display:flex;gap:16px;align-items:flex-end;height:60px;padding:0 8px;margin-top:8px}
.chart-group{display:flex;gap:3px;align-items:flex-end}
.chart-bar{width:10px;border-radius:2px 2px 0 0}
.chart-labels{display:flex;justify-content:space-around;font-size:10px;color:#555;margin-top:4px}
.chart-legend{display:flex;gap:12px;margin-top:6px;font-size:10px;color:#888}
.legend-dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:3px}
.experiments-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.experiments-table th{color:#555;text-align:left;padding:4px;border-bottom:1px solid #222;font-size:10px}
.experiments-table td{padding:6px 4px}
.no-data{color:#555;font-style:italic;padding:20px 0}
</style>
</head><body>
<h1>Flywheel Analytics</h1>
<div class="nav">
  <a href="/">Dashboard</a>
  <a href="/flywheel">Flywheel</a>
</div>
<div id="root"><p style="color:#555">Loading…</p></div>
<script>
const CLUSTER_COLORS = ['#6fcf6f','#f0c040','#6b9fd4','#e07070','#c084fc','#38bdf8'];

async function load() {
  const r = await fetch('/api/flywheel');
  const d = await r.json();
  render(d);
}

function pct(v){ return Math.round((v||0)*100); }
function bar(pct_val, color, cls=''){
  const w = Math.max(0, Math.min(100, pct_val));
  return `<div class="signal-fill ${cls}" style="background:${color};width:${w}%"></div>`;
}

function phaseColor(p){ return p>=3?'#6fcf6f':p===2?'#f0c040':'#6b9fd4'; }
function phaseBadge(p){ const c=p>=3?'phase-3-badge':p===2?'phase-2-badge':'phase-1-badge';
  return `<span class="phase-badge ${c}">Phase ${p}${p>=3?' ✓':''}</span>`; }

function renderSummary(s) {
  if (s.no_data) return '<div class="no-data">No analytics data yet — run some queries and a LoRA round first.</div>';
  const pct_v = pct(s.parametric_pct);
  const sparkMax = Math.max(...(s.sparkline||[0.01]));
  const sparks = (s.sparkline||[]).reverse().map(v=>{
    const h = Math.max(10, Math.round(v/sparkMax*30));
    return `<div class="spark-bar" style="height:${h}px;background:#6fcf6f"></div>`;
  }).join('');
  const pd = s.phase_dist||{};
  const pPct = pct(pd.parametric||0), kPct=pct(pd.kv||0), rPct=pct(pd.retrieval||0);
  return `
  <div class="panels">
    <div class="panel" style="border-color:#2a4a2a">
      <div class="panel-label">Learning Velocity</div>
      <div class="panel-value" style="color:#6fcf6f">${pct_v}%</div>
      <div class="panel-sub">of queries answered parametrically</div>
      <div class="sparkline">${sparks}</div>
      <div class="projection" style="color:${s.projection&&s.projection.startsWith('~')?'#f0c040':'#888'}">
        ⏱ ${s.projection||'—'}
      </div>
    </div>
    <div class="panel" style="border-color:#2a2a4a">
      <div class="panel-label">Latency Recovered</div>
      <div class="panel-value" style="color:#6b9fd4">${Math.round(s.ms_saved_per_query||0)}ms</div>
      <div class="panel-sub">saved per query on average</div>
      <div class="phase-bar" style="margin-top:12px">
        <div style="background:#6fcf6f;flex:${pPct}" title="Parametric"></div>
        <div style="background:#6b9fd4;flex:${kPct}" title="KV"></div>
        <div style="background:#444;flex:${rPct}" title="Retrieval"></div>
      </div>
      <div class="phase-legend">
        <span><span style="color:#6fcf6f">■</span> Parametric ${pPct}%</span>
        <span><span style="color:#6b9fd4">■</span> KV ${kPct}%</span>
        <span><span style="color:#444">■</span> Retrieval ${rPct}%</span>
      </div>
    </div>
    <div class="panel" style="border-color:#4a2a2a">
      <div class="panel-label">Cost Savings</div>
      <div class="panel-value" style="color:#e07070">$${s.cost_saved_month||0}</div>
      <div class="panel-sub">saved this month</div>
      <div style="font-size:11px;color:#888;margin-top:8px">
        vs. cloud API at
        <strong style="color:#aaa">$${s.cost_per_1k_tokens||5}/1k tokens</strong>
        <span id="edit-cost-btn" onclick="showCostEdit()"
              style="margin-left:6px;color:#6b9fd4;cursor:pointer;text-decoration:underline">edit</span>
      </div>
      <div style="color:#aaa;font-size:11px;margin-top:4px">
        Cumulative: <strong style="color:#e07070">$${s.cost_saved_total||0}</strong>
      </div>
      <div id="cost-edit-form" style="display:none;margin-top:8px">
        <input id="cost-input" type="number" step="0.01" value="${s.cost_per_1k_tokens||5}"
               style="width:70px;background:#1a1a1a;border:1px solid #444;color:#fff;padding:3px">
        <button onclick="saveCostRate()"
                style="margin-left:4px;background:#2a4a2a;border:none;color:#6fcf6f;padding:3px 8px;cursor:pointer">
          Save
        </button>
      </div>
    </div>
  </div>`;
}

function renderClusterCards(cards) {
  if (!cards || !cards.length) return '<div class="no-data">No cluster data yet.</div>';
  return cards.map((c, i) => {
    const color = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
    const prsW = Math.round((c.prs||0)*100);
    return `
    <div class="cluster-card" style="border-color:${color}33">
      <div class="card-header">
        <span class="card-name">${c.label||c.cluster_id}</span>
        ${phaseBadge(c.phase||1)}
      </div>
      <div class="prs-bar-wrap">
        <div class="prs-bar-track">
          <div class="prs-bar-fill" style="background:${color};width:${prsW}%"></div>
        </div>
        <span class="prs-value" style="color:${color}">${(c.prs||0).toFixed(2)}</span>
      </div>
      <div style="font-size:10px;color:#666;margin-bottom:4px">Three-signal coverage</div>
      ${['faq_coverage','vdb_coverage','realtime_coverage'].map(k=>{
        const label = k==='faq_coverage'?'FAQ':k==='vdb_coverage'?'VDB':'Realtime';
        const w = Math.round((c[k]||0)*100);
        return `<div class="signal-row">
          <span class="signal-label">${label}</span>
          <div class="signal-track"><div class="signal-fill" style="background:${color};width:${w}%"></div></div>
          <span class="signal-val" style="color:${color}">${w}%</span>
        </div>`;
      }).join('')}
      <div class="card-footer">${c.query_count||0} queries · threshold ${(c.threshold||0.72).toFixed(2)}</div>
    </div>`;
  }).join('');
}

function renderHistory(history) {
  if (!history || !history.length) return '<div class="no-data">No round history yet.</div>';
  const allClusters = [...new Set(history.flatMap(h=>Object.keys(h.per_cluster_prs||{})))];
  const maxPrs = Math.max(...history.flatMap(h=>Object.values(h.per_cluster_prs||{0:0.01})), 0.01);
  const groups = history.map((h, ri) => {
    const bars = allClusters.map((cid, ci) => {
      const prs = h.per_cluster_prs[cid]||0;
      const height = Math.max(5, Math.round(prs/maxPrs*60));
      const color = CLUSTER_COLORS[ci%CLUSTER_COLORS.length];
      return `<div class="chart-bar" style="height:${height}px;background:${color}" title="${cid}: ${prs.toFixed(2)}"></div>`;
    }).join('');
    return `<div class="chart-group">${bars}</div>`;
  });
  const labels = history.map(h=>`v${h.lora_version}`);
  const legend = allClusters.map((cid,ci)=>{
    const color = CLUSTER_COLORS[ci%CLUSTER_COLORS.length];
    return `<span><span class="legend-dot" style="background:${color}"></span>${cid}</span>`;
  }).join('');
  return `
  <div class="history-chart">
    <div class="chart-bars">${groups.join('')}</div>
    <div class="chart-labels">${labels.map(l=>`<span>${l}</span>`).join('')}</div>
    <div class="chart-legend">${legend}</div>
  </div>`;
}

function renderExperiments(exps) {
  if (!exps || !exps.length) return '<div class="no-data">No ModelScout experiments recorded.</div>';
  const rows = exps.map(e=>`
    <tr>
      <td style="color:${e.status==='Candidate'?'#6fcf6f':'#aaa'}">${e.model_id}</td>
      <td style="color:#666">${e.rounds}</td>
      <td style="color:${e.status==='Candidate'?'#6fcf6f':'#aaa'}">${(e.best_prs||0).toFixed(3)}</td>
      <td style="color:${e.status==='Candidate'?'#6fcf6f':'#f0c040'}">${e.status}</td>
    </tr>`).join('');
  return `<table class="experiments-table">
    <tr><th>Model</th><th>Rounds</th><th>Best PRS</th><th>Status</th></tr>
    ${rows}
  </table>`;
}

function render(d) {
  document.getElementById('root').innerHTML = `
    ${renderSummary(d.summary||{})}
    <div class="section-title">Cluster Progress</div>
    <div class="cluster-grid">${renderClusterCards(d.cluster_cards||[])}</div>
    <div class="section-title">PRS History — Per Cluster</div>
    ${renderHistory(d.prs_history||[])}
    <div class="section-title">🧪 ModelScout Experiments</div>
    ${renderExperiments(d.experiments||[])}
  `;
}

function showCostEdit() {
  document.getElementById('cost-edit-form').style.display = 'block';
}

async function saveCostRate() {
  const val = parseFloat(document.getElementById('cost-input').value);
  if (isNaN(val) || val <= 0) return;
  await fetch('/api/flywheel/cost-rate', {
    method: 'PATCH',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({cost_per_1k_tokens: val})
  });
  document.getElementById('cost-edit-form').style.display = 'none';
  load();
}

load();
</script>
</body></html>"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dashboard.py::test_flywheel_tab_returns_html -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/monitoring_dashboard.py tests/test_dashboard.py
git commit -m "feat: add Flywheel HTML tab to monitoring dashboard (cluster cards, PRS history, cost)"
```

---

### Task 10: Cost Rate Edit Endpoint (`PATCH /api/flywheel/cost-rate`)

**Files:**
- Modify: `pipeline/monitoring_dashboard.py`
- Modify: `tests/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_dashboard.py

def test_patch_cost_rate_updates_config(tmp_path):
    from fastapi.testclient import TestClient
    import pipeline.monitoring_dashboard as md
    import json

    config_file = tmp_path / "cfg.json"
    cfg_data = {
        "collection": "test4", "embed_model": "BAAI/bge-small-en-v1.5",
        "vector_dim": 384, "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "checkpoint_dir": "ckpt", "version_file": "v.json", "replay_db": "r.db",
        "cost_per_1k_tokens": 5.0, "tokens_per_ms_baseline": 0.8,
        "analytics_db": str(tmp_path / "t4.db"),
    }
    config_file.write_text(json.dumps(cfg_data))

    with patch("pipeline.monitoring_dashboard._config_path", str(config_file)), \
         patch("pipeline.monitoring_dashboard._cfg", {}):
        client = TestClient(md.app)
        resp = client.patch("/api/flywheel/cost-rate",
                            json={"cost_per_1k_tokens": 12.5})
        assert resp.status_code == 200
        updated = json.loads(config_file.read_text())
        assert updated["cost_per_1k_tokens"] == 12.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_dashboard.py::test_patch_cost_rate_updates_config -v --override-ini="addopts="
```
Expected: FAIL with 404 or 405

- [ ] **Step 3: Add `PATCH /api/flywheel/cost-rate` endpoint**

```python
class CostRateUpdate(BaseModel):
    cost_per_1k_tokens: float


@app.patch("/api/flywheel/cost-rate")
def update_cost_rate(body: CostRateUpdate):
    global _cfg
    cfg = _load_cfg()
    cfg["cost_per_1k_tokens"] = body.cost_per_1k_tokens
    _cfg = cfg
    # Persist to config file
    with open(_config_path) as f:
        data = json.load(f)
    data["cost_per_1k_tokens"] = body.cost_per_1k_tokens
    with open(_config_path, "w") as f:
        json.dump(data, f, indent=2)
    return {"cost_per_1k_tokens": body.cost_per_1k_tokens}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_dashboard.py::test_patch_cost_rate_updates_config -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Run full dashboard test suite**

```bash
python -m pytest tests/test_dashboard.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add pipeline/monitoring_dashboard.py tests/test_dashboard.py
git commit -m "feat: add PATCH /api/flywheel/cost-rate endpoint (dashboard-editable cost setting)"
```

---

### Task 11: Studio Cross-UC Flywheel Summary (`studio/flywheel_routes.py`)

**Files:**
- Create: `studio/flywheel_routes.py`
- Create: `tests/test_flywheel_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_flywheel_routes.py
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_app():
    from studio.flywheel_routes import flywheel_router
    app = FastAPI()
    app.include_router(flywheel_router)
    return app


def test_flywheel_all_ucs_returns_list(tmp_path):
    import core.analytics as an

    cfg1 = {"collection": "uc1",
            "analytics_db": str(tmp_path / "uc1_analytics.db"),
            "cost_per_1k_tokens": 5.0, "tokens_per_ms_baseline": 0.8}
    an.init_db(cfg1)
    cs = {"0": {"label": "auth", "phase": 2, "prs": 0.72,
                "faq_coverage": 0.8, "vdb_coverage": 0.7, "realtime_coverage": 0.6,
                "learned_weights": {}, "threshold": 0.75,
                "prs_history": [0.55, 0.63, 0.72], "query_count": 50}}
    an.record_round(cfg1, lora_version=1, cluster_state=cs, tier_distribution={})

    mock_uc = MagicMock()
    mock_uc.id = "uc1"
    mock_uc.config = cfg1

    with patch("studio.flywheel_routes._get_all_uc_configs", return_value=[mock_uc]):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/flywheel/all")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["uc_id"] == "uc1"


def test_flywheel_all_ucs_empty(tmp_path):
    with patch("studio.flywheel_routes._get_all_uc_configs", return_value=[]):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/flywheel/all")
        assert resp.status_code == 200
        assert resp.json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_flywheel_routes.py -v --override-ini="addopts="
```
Expected: FAIL with `ModuleNotFoundError: No module named 'studio.flywheel_routes'`

- [ ] **Step 3: Implement `studio/flywheel_routes.py`**

```python
# studio/flywheel_routes.py
"""FastAPI router for cross-UC Flywheel summary in KVForge Studio."""
from fastapi import APIRouter
from pathlib import Path
import json

import core.analytics as _analytics

flywheel_router = APIRouter()

ROOT = Path(__file__).resolve().parent.parent


def _get_all_uc_configs():
    """Return list of objects with .id and .config (dict) for all known UCs.

    Reads datasource_*.json config files from the project root.
    """
    class _UC:
        def __init__(self, uc_id, config):
            self.id = uc_id
            self.config = config

    ucs = []
    for cfg_file in sorted(ROOT.glob("datasource_*.json")):
        try:
            with open(cfg_file) as f:
                data = json.load(f)
            uc_id = cfg_file.stem.replace("datasource_", "")
            ucs.append(_UC(uc_id, data))
        except Exception:
            continue
    return ucs


@flywheel_router.get("/api/flywheel/all")
def get_all_flywheel_summaries():
    """Return flywheel summary for every known UC — powers Studio cross-UC panel."""
    ucs = _get_all_uc_configs()
    results = []
    for uc in ucs:
        try:
            _analytics.init_db(uc.config)
            summary = _analytics.get_flywheel_summary(uc.config)
            experiments = _analytics.get_modelscout_experiments(uc.config)
            results.append({
                "uc_id": uc.id,
                "summary": summary,
                "has_experiments": len(experiments) > 0,
            })
        except Exception:
            results.append({"uc_id": uc.id, "summary": {"no_data": True},
                            "has_experiments": False})
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_flywheel_routes.py -v --override-ini="addopts="
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add studio/flywheel_routes.py tests/test_flywheel_routes.py
git commit -m "feat: add Studio cross-UC flywheel summary route (flywheel task 11)"
```

---

### Task 12: Wire Studio Flywheel Routes into `studio/routes.py`

**Files:**
- Modify: `studio/routes.py`
- Modify: `tests/test_studio_routes.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_studio_routes.py
def test_studio_includes_flywheel_router():
    from studio.routes import router
    routes = [r.path for r in router.routes]
    assert any("flywheel" in p for p in routes)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_studio_routes.py::test_studio_includes_flywheel_router -v --override-ini="addopts="
```
Expected: FAIL — no flywheel route found

- [ ] **Step 3: Include flywheel router in `studio/routes.py`**

In `studio/routes.py`, add this import at the top with the other imports:

```python
from studio.flywheel_routes import flywheel_router
```

Then add this line immediately after `router.include_router(api_router)`:

```python
router.include_router(flywheel_router)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_studio_routes.py::test_studio_includes_flywheel_router -v --override-ini="addopts="
```
Expected: PASS

- [ ] **Step 5: Run full studio routes test suite**

```bash
python -m pytest tests/test_studio_routes.py -v --override-ini="addopts="
```
Expected: All PASSED

- [ ] **Step 6: Run all tests**

```bash
python -m pytest tests/ -v --override-ini="addopts="
```
Expected: All PASSED (skip any that require GPU)

- [ ] **Step 7: Commit**

```bash
git add studio/routes.py tests/test_studio_routes.py
git commit -m "feat: wire flywheel_routes into studio/routes.py (flywheel analytics complete)"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| SQLite analytics DB per UC, WAL mode, 3 tables | Task 1 |
| `record_query` with rolling baseline estimator (α=0.1, 50-query init) | Task 2 |
| `record_round` reading from Dynamic PRS cluster_state | Task 3, 7 |
| `compute_slope`, `estimate_days_to_phase3` with stalled/insufficient edge cases | Task 3 |
| Query helpers: `get_flywheel_summary`, `get_cluster_cards`, `get_prs_history`, `get_modelscout_experiments` | Task 4 |
| `analytics_db`, `cost_per_1k_tokens`, `tokens_per_ms_baseline` config fields | Task 5 |
| `record_query` hooked into monitoring dashboard's `run_query` | Task 6 |
| `record_round` hooked into `prs_evaluator.evaluate()` | Task 7 |
| `GET /api/flywheel` JSON endpoint | Task 8 |
| Flywheel HTML tab: 3 metric panels, cluster cards, PRS history chart, ModelScout panel | Task 9 |
| `PATCH /api/flywheel/cost-rate` — dashboard-editable cost rate | Task 10 |
| Studio cross-UC Flywheel summary API | Task 11 |
| Wire flywheel routes into Studio | Task 12 |
| `model_id` in `round_snapshots` for ModelScout experiment tagging | Tasks 3, 4, 11 |
| Analytics never breaks inference (try/except guards) | Tasks 6, 7 |

No gaps found.

**Type consistency:** `record_query(cfg, cluster_id, phase_used, latency_ms, model_id=None)` is called consistently in Tasks 2, 6. `record_round(cfg, lora_version, cluster_state, tier_distribution, model_id=None)` is called consistently in Tasks 3, 7, 11. `_db_path(cfg)` used throughout Tasks 1–4.
