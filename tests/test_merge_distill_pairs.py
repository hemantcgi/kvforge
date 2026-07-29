"""Tests for tools/merge_distill_pairs.py."""

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.merge_distill_pairs import merge


def test_merge_combines_both_sources():
    teacher = {"teacher_pairs": [{"question": "Q?", "teacher_answer": "A"}]}
    on_policy = {"on_policy_pairs": [{"question": "Q2?", "answer": "A2", "confidence_label": True}]}

    tp = NamedTemporaryFile(suffix=".json", delete=False)
    op = NamedTemporaryFile(suffix=".json", delete=False)
    try:
        Path(tp.name).write_text(json.dumps(teacher))
        Path(op.name).write_text(json.dumps(on_policy))
        result = merge(tp.name, op.name)
        assert len(result["teacher_pairs"]) == 1
        assert len(result["on_policy_pairs"]) == 1
        assert result["teacher_pairs"][0]["question"] == "Q?"
        assert result["on_policy_pairs"][0]["confidence_label"] is True
    finally:
        Path(tp.name).unlink(missing_ok=True)
        Path(op.name).unlink(missing_ok=True)


def test_merge_handles_empty_inputs():
    teacher = {"teacher_pairs": []}
    on_policy = {"on_policy_pairs": []}

    tp = NamedTemporaryFile(suffix=".json", delete=False)
    op = NamedTemporaryFile(suffix=".json", delete=False)
    try:
        Path(tp.name).write_text(json.dumps(teacher))
        Path(op.name).write_text(json.dumps(on_policy))
        result = merge(tp.name, op.name)
        assert result["teacher_pairs"] == []
        assert result["on_policy_pairs"] == []
    finally:
        Path(tp.name).unlink(missing_ok=True)
        Path(op.name).unlink(missing_ok=True)


def test_merge_missing_keys_defaults_to_empty():
    teacher = {}
    on_policy = {}

    tp = NamedTemporaryFile(suffix=".json", delete=False)
    op = NamedTemporaryFile(suffix=".json", delete=False)
    try:
        Path(tp.name).write_text(json.dumps(teacher))
        Path(op.name).write_text(json.dumps(on_policy))
        result = merge(tp.name, op.name)
        assert result["teacher_pairs"] == []
        assert result["on_policy_pairs"] == []
    finally:
        Path(tp.name).unlink(missing_ok=True)
        Path(op.name).unlink(missing_ok=True)


def test_cli_produces_valid_output():
    teacher = {"teacher_pairs": [{"question": "Q?", "teacher_answer": "A"}]}
    on_policy = {"on_policy_pairs": [{"question": "Q2?", "answer": "A2", "confidence_label": True}]}

    tp = NamedTemporaryFile(suffix=".json", delete=False)
    op = NamedTemporaryFile(suffix=".json", delete=False)
    out = NamedTemporaryFile(suffix=".json", delete=False)
    try:
        Path(tp.name).write_text(json.dumps(teacher))
        Path(op.name).write_text(json.dumps(on_policy))

        old_argv = sys.argv
        sys.argv = [
            "merge",
            "--teacher-pairs", tp.name,
            "--on-policy", op.name,
            "--output", out.name,
        ]
        try:
            from tools.merge_distill_pairs import main
            main()
        finally:
            sys.argv = old_argv

        result = json.loads(Path(out.name).read_text())
        assert "teacher_pairs" in result
        assert "on_policy_pairs" in result
    finally:
        Path(tp.name).unlink(missing_ok=True)
        Path(op.name).unlink(missing_ok=True)
        Path(out.name).unlink(missing_ok=True)
