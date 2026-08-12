import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_heldout_eval import (
    compute_version_hash,
    is_overlap,
    normalize_text,
)


def test_normalize_text():
    assert normalize_text("The AWS service!") == "aws service"


def test_is_overlap_exact():
    train = ["What is Amazon Bedrock?"]
    assert is_overlap("What is Amazon Bedrock?", train, threshold=0.85)


def test_is_overlap_different():
    train = ["What is Amazon Bedrock?"]
    assert not is_overlap("How do I configure IAM roles?", train, threshold=0.85)


def test_is_overlap_paraphrase():
    train = ["What is Amazon Bedrock?"]
    # A close paraphrase should be flagged at a lower threshold.
    assert is_overlap("What exactly is Amazon Bedrock used for?", train, threshold=0.5)


def test_compute_version_hash_is_deterministic():
    items = [
        {"question": "q1", "answer": "a1", "type": "novel"},
        {"question": "q2", "answer": "a2", "type": "paraphrase"},
    ]
    h1 = compute_version_hash(items)
    h2 = compute_version_hash(items)
    assert h1 == h2
    assert len(h1) == 16


def test_compute_version_hash_changes_with_content():
    items1 = [{"question": "q1", "answer": "a1", "type": "novel"}]
    items2 = [{"question": "q2", "answer": "a2", "type": "novel"}]
    assert compute_version_hash(items1) != compute_version_hash(items2)

