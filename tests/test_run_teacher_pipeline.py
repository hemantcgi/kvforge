"""Tests for tools/run_teacher_pipeline.py — Sprint 2 teacher pipeline."""

import json
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_run_teacher_pipeline_quality_filters(monkeypatch):
    from tools import run_teacher_pipeline

    query_pool = [
        {"question": "Q1", "expected_answer": "A1", "source": "faq"},
        {"question": "Q2", "expected_answer": "A2", "source": "faq"},
        {"question": "Q3", "expected_answer": "", "source": "chunk"},
    ]

    call_count = {"n": 0}

    def fake_answer_with_mode(query, cfg, force_mode=None):
        call_count["n"] += 1
        return f"answer to {query}", "text_rag"

    def fake_retrieve_chunk_ids(query, cfg, embedder, store):
        return ["1", "2"]

    monkeypatch.setattr(run_teacher_pipeline, "answer_with_mode", fake_answer_with_mode)
    monkeypatch.setattr(run_teacher_pipeline, "_retrieve_chunk_ids", fake_retrieve_chunk_ids)

    def fake_quality_filter(pairs, threshold, judge_client=None, judge_model="gpt-4o-mini"):
        return [
            {**p, "factual_accuracy": 0.9}
            for p in pairs
        ], {"kept": 2, "dropped": 0, "drop_rate": 0.0}

    monkeypatch.setattr(run_teacher_pipeline, "quality_filter", fake_quality_filter)

    with patch("tools.run_teacher_pipeline.model_loader.load",
               return_value=(MagicMock(), MagicMock())), \
         patch("tools.run_teacher_pipeline.TextEmbedding"), \
         patch("tools.run_teacher_pipeline.get_store", return_value=MagicMock()):
        result = run_teacher_pipeline.run_teacher_pipeline(
            {"query_log_db": None}, query_pool, quality_threshold=0.7
        )

    assert call_count["n"] == 3
    assert len(result["teacher_pairs"]) == 3
    assert result["stats"]["kept"] == 2
    assert result["stats"]["non_evaluable"] == 1


def test_main_cli(tmp_path):
    from tools.run_teacher_pipeline import main

    cfg_path = tmp_path / "cfg.json"
    pool_path = tmp_path / "pool.json"
    out_path = tmp_path / "out.json"

    cfg_path.write_text(json.dumps({"query_log_db": None}))
    pool_path.write_text(json.dumps([
        {"question": "Q1", "expected_answer": "A1", "source": "faq"},
    ]))

    def fake_run(*args, **kwargs):
        return {
            "teacher_pairs": [
                {"question": "Q1", "teacher_answer": "A1", "expected_answer": "A1"},
            ],
            "stats": {"kept": 1, "dropped": 0},
        }

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "run_teacher_pipeline.py",
            "--config", str(cfg_path),
            "--query-pool", str(pool_path),
            "--output", str(out_path),
        ]
        with patch("tools.run_teacher_pipeline.run_teacher_pipeline", fake_run):
            main()
    finally:
        sys.argv = old_argv

    output = json.loads(out_path.read_text())
    assert len(output["teacher_pairs"]) == 1
    assert output["teacher_pairs"][0]["question"] == "Q1"
