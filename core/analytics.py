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


def compute_slope(values: list) -> float:
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

    phase_counts: dict = {}
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


def get_cluster_cards(cfg: dict) -> list:
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


def get_prs_history(cfg: dict, n_rounds: int = 10) -> list:
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


def get_modelscout_experiments(cfg: dict) -> list:
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
