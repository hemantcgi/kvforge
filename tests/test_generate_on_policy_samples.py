"""Tests for tools/generate_on_policy_samples.py — Sprint 2.5 on-policy labeler."""

import json
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_main_cli(tmp_path):
    from tools.generate_on_policy_samples import main

    cfg_path = tmp_path / "cfg.json"
    teacher_cfg_path = tmp_path / "teacher_cfg.json"
    pool_path = tmp_path / "pool.json"
    out_path = tmp_path / "out.json"

    cfg_path.write_text(json.dumps({"query_log_db": None}))
    teacher_cfg_path.write_text(json.dumps({"query_log_db": None}))
    pool_path.write_text(json.dumps([
        {"question": "Q1", "expected_answer": "A1", "source": "faq"},
    ]))

    def fake_generate(query_pool, student_model, student_tokenizer, teacher_model,
                     teacher_tokenizer, cfg, sft_format="chat", judge_client=None,
                     judge_model="gpt-4o-mini"):
        return [
            {
                "question": "Q1",
                "student_answer": "S1",
                "teacher_answer": "A1",
                "confidence_label": True,
                "factual_accuracy": 0.85,
            }
        ]

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "generate_on_policy_samples.py",
            "--config", str(cfg_path),
            "--teacher-config", str(teacher_cfg_path),
            "--query-pool", str(pool_path),
            "--output", str(out_path),
        ]
        with patch("tools.generate_on_policy_samples.generate_on_policy_samples", fake_generate), \
             patch("tools.generate_on_policy_samples.model_loader.load",
                   return_value=(MagicMock(), MagicMock())):
            main()
    finally:
        sys.argv = old_argv

    output = json.loads(out_path.read_text())
    assert len(output["on_policy_pairs"]) == 1
    assert output["on_policy_pairs"][0]["confidence_label"] is True
