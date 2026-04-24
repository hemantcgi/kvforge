"""Tests for pipeline/query_logger.py — SQLite query logging."""

import time
import pytest
from pipeline.query_logger import init_db, log_query, mark_requeried, get_cluster_stats, get_training_pairs


def test_init_creates_table(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("query_log",) in tables


def test_log_and_retrieve_retrieval_query(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    row_id = log_query(db, "What is RAG?", "RAG stands for...", "retrieval", cluster_id="0")
    assert row_id == 1
    pairs = get_training_pairs(db)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What is RAG?"
    assert pairs[0]["cluster_id"] == "0"


def test_parametric_queries_not_in_training_pairs(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Who is CEO?", "Tim Cook", "parametric", cluster_id="1")
    pairs = get_training_pairs(db)
    assert len(pairs) == 0


def test_get_cluster_stats_no_requery(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "parametric", cluster_id="0")
    log_query(db, "Q2", "A2", "parametric", cluster_id="0")
    stats = get_cluster_stats(db, "0", window_minutes=10)
    assert stats["realtime_coverage"] == 1.0
    assert stats["query_count"] == 2


def test_mark_requeried_lowers_coverage(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "parametric", cluster_id="0")
    mark_requeried(db, "Q1", window_minutes=10)
    stats = get_cluster_stats(db, "0", window_minutes=10)
    assert stats["realtime_coverage"] == 0.0


def test_get_cluster_stats_empty(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    stats = get_cluster_stats(db, "99", window_minutes=10)
    assert stats["realtime_coverage"] == 0.0
    assert stats["query_count"] == 0


def test_get_training_pairs_filtered_by_cluster(tmp_path):
    db = str(tmp_path / "q.db")
    init_db(db)
    log_query(db, "Q1", "A1", "retrieval", cluster_id="0")
    log_query(db, "Q2", "A2", "retrieval", cluster_id="1")
    pairs = get_training_pairs(db, cluster_id="0")
    assert len(pairs) == 1
    assert pairs[0]["question"] == "Q1"
