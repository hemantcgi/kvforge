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


# ── record_query tests ────────────────────────────────────────────────────────

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


# ── record_round / compute_slope / estimate_days tests ────────────────────────

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


# ── query helper tests ────────────────────────────────────────────────────────

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
