"""Tests for the Knowledge Differentiation Score (KDS) in prs_evaluator."""

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import numpy as np


class _FakeStore:
    """In-memory VectorStore implementation sufficient for KDS tests."""

    def __init__(self, points):
        self.points = points
        self.set_payload_calls = []

    def scroll(self, collection, limit=1000, with_payload=True, offset=None):
        if offset is None:
            return (self.points, None)
        return ([], None)

    def set_payload(self, collection, point_id, payload):
        self.set_payload_calls.append((collection, point_id, payload))


def _make_self_consistency():
    """Return a deterministic _self_consistency replacement.

    Embeddings are clustered around [1,0,0] for chunk1-style questions and
    around [0,1,0] for chunk2-style questions, with a small controlled
    within-chunk variance.
    """
    def _self_consistency(question, pipe_sample, embedder, tokenizer=None,
                          sft_format="chat", n=3, return_embeddings=False):
        if "chunk1" in question.lower() or question.lower().startswith("what is"):
            base = np.array([1.0, 0.0, 0.0])
        else:
            base = np.array([0.0, 1.0, 0.0])
        embs = np.array([
            base + np.array([0.05, 0.0, 0.0]),
            base + np.array([-0.05, 0.0, 0.0]),
            base + np.array([0.0, 0.0, 0.03]),
        ])
        if return_embeddings:
            return 0.99, embs
        return 0.99
    return _self_consistency


@contextmanager
def _kds_patches(store, sc_side_effect):
    """Common patches for compute_kds heavy dependencies."""
    fake_transformers = MagicMock()
    fake_transformers.pipeline = MagicMock(return_value=MagicMock())
    mocks = {}
    with patch.dict("sys.modules", {"transformers": fake_transformers}), ExitStack() as stack:
        mocks["model_loader"] = stack.enter_context(
            patch(
                "pipeline.prs_evaluator.model_loader",
                MagicMock(load=MagicMock(return_value=(MagicMock(), MagicMock()))),
            )
        )
        mocks["ver"] = stack.enter_context(
            patch(
                "pipeline.prs_evaluator.ver",
                MagicMock(
                    init=MagicMock(),
                    get_lora_version=MagicMock(return_value=2),
                    append_kds=MagicMock(),
                ),
            )
        )
        mocks["TextEmbedding"] = stack.enter_context(
            patch("pipeline.prs_evaluator.TextEmbedding", MagicMock())
        )
        mocks["get_store"] = stack.enter_context(
            patch("pipeline.prs_evaluator.get_store", MagicMock(return_value=store))
        )
        mocks["_self_consistency"] = stack.enter_context(
            patch(
                "pipeline.prs_evaluator._self_consistency",
                MagicMock(side_effect=sc_side_effect),
            )
        )
        yield mocks


def test_compute_kds_returns_values_in_zero_one():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one"}),
        MagicMock(id="chunk2", payload={"text": "text two"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {"question": "What is chunk1?", "answer": "A", "source_chunk_ids": ["chunk1"]},
        {"question": "What is chunk2?", "answer": "B", "source_chunk_ids": ["chunk2"]},
    ]
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()):
        mean_kds, kds_by_chunk = compute_kds(faqs, cfg, sample_cap=300, n=3)

    assert isinstance(mean_kds, float)
    assert 0.0 <= mean_kds <= 1.0
    assert set(kds_by_chunk.keys()) == {"chunk1", "chunk2"}
    for kds in kds_by_chunk.values():
        assert 0.0 <= kds <= 1.0


def test_kds_and_last_kds_round_persisted_to_vector_store():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one"}),
        MagicMock(id="chunk2", payload={"text": "text two"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {"question": "What is chunk1?", "answer": "A", "source_chunk_ids": ["chunk1"]},
        {"question": "What is chunk2?", "answer": "B", "source_chunk_ids": ["chunk2"]},
    ]
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()):
        compute_kds(faqs, cfg, sample_cap=300, n=3)

    assert len(store.set_payload_calls) == 2
    for collection, point_id, payload in store.set_payload_calls:
        assert collection == "test"
        assert point_id in ("chunk1", "chunk2")
        assert "kds" in payload
        assert "last_kds_round" in payload
        assert 0.0 <= payload["kds"] <= 1.0
        assert payload["last_kds_round"] == 2


def test_kds_history_appended_to_version_json():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {"question": "What is chunk1?", "answer": "A", "source_chunk_ids": ["chunk1"]},
    ]
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()) as mocks:
        mean_kds, kds_by_chunk = compute_kds(faqs, cfg, sample_cap=300, n=3)

    mocks["ver"].append_kds.assert_called_once()
    args = mocks["ver"].append_kds.call_args[0]
    assert args[0] == 2
    assert 0.0 <= args[1] <= 1.0
    assert args[2] == 1


def test_chunks_without_faqs_excluded():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {"question": "What is chunk1?", "answer": "A", "source_chunk_ids": ["other"]},
    ]
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()) as mocks:
        mean_kds, kds_by_chunk = compute_kds(faqs, cfg, sample_cap=300, n=3)

    assert kds_by_chunk == {}
    assert mean_kds == 0.0
    assert len(store.set_payload_calls) == 0
    mocks["ver"].append_kds.assert_called_once()
    args = mocks["ver"].append_kds.call_args[0]
    assert args[1] == 0.0
    assert args[2] == 0


def test_compute_kds_strips_variant_suffix_in_chat_mode():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {
            "question": "What is chunk1? (variant 1)",
            "answer": "A",
            "source_chunk_ids": ["chunk1"],
        },
    ]
    cfg = {"collection": "test", "sft_format": "chat"}

    sc = _make_self_consistency()
    with _kds_patches(store, sc) as mocks:
        compute_kds(faqs, cfg, sample_cap=300, n=3)

    called_questions = [call.args[0] for call in mocks["_self_consistency"].call_args_list]
    assert any("What is chunk1?" in q and "(variant" not in q for q in called_questions)


def test_compute_kds_handles_empty_input():
    from pipeline.prs_evaluator import compute_kds

    store = _FakeStore([])
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()) as mocks:
        mean_kds, kds_by_chunk = compute_kds([], cfg, sample_cap=300, n=3)

    assert mean_kds == 0.0
    assert kds_by_chunk == {}
    mocks["ver"].append_kds.assert_called_once()
    args = mocks["ver"].append_kds.call_args[0]
    assert args[1] == 0.0
    assert args[2] == 0


def test_compute_kds_respects_sample_cap():
    from pipeline.prs_evaluator import compute_kds

    points = [
        MagicMock(id="chunk1", payload={"text": "text one", "last_kds_round": 1}),
        MagicMock(id="chunk2", payload={"text": "text two"}),
        MagicMock(id="chunk3", payload={"text": "text three"}),
    ]
    store = _FakeStore(points)
    faqs = [
        {"question": "What is chunk1?", "answer": "A", "source_chunk_ids": ["chunk1"]},
        {"question": "What is chunk2?", "answer": "B", "source_chunk_ids": ["chunk2"]},
        {"question": "What is chunk3?", "answer": "C", "source_chunk_ids": ["chunk3"]},
    ]
    cfg = {"collection": "test", "sft_format": "bare"}

    with _kds_patches(store, _make_self_consistency()):
        mean_kds, kds_by_chunk = compute_kds(faqs, cfg, sample_cap=2, n=3)

    assert len(kds_by_chunk) == 2
    assert "chunk1" not in kds_by_chunk  # chunk1 was measured in a previous round
