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


@contextmanager
def _fkds_patches(store, embeddings_by_question, factual_scores):
    """Common patches for compute_fkds heavy dependencies."""
    def _fake_pipe(prompt, *args, **kwargs):
        # In bare mode the prompt is the question text; return generated text
        # that, after the prompt is stripped, matches the keys in our fixtures.
        if "chunk1" in prompt:
            return [{"generated_text": "What is chunk1? answer chunk1 a"}]
        if "chunk2" in prompt:
            return [{"generated_text": "What is chunk2? answer chunk2 a"}]
        return [{"generated_text": f"{prompt} default answer"}]

    fake_transformers = MagicMock()
    fake_transformers.pipeline = MagicMock(return_value=_fake_pipe)

    class FakeEmbedder:
        def embed(self, texts):
            return iter([embeddings_by_question.get(t, np.array([0.0, 0.0, 1.0])) for t in texts])

    fake_metrics = MagicMock()
    fake_metrics.token_f1 = MagicMock(side_effect=lambda pred, ref: factual_scores.get((pred, ref), 0.5))
    fake_metrics.llm_judge = MagicMock(side_effect=lambda q, a, gt, **kw: {
        "factually_correct": factual_scores.get((q, a), 0.5)
    })

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
                    append_fkds=MagicMock(),
                ),
            )
        )
        mocks["TextEmbedding"] = stack.enter_context(
            patch("pipeline.prs_evaluator.TextEmbedding", MagicMock(return_value=FakeEmbedder()))
        )
        mocks["get_store"] = stack.enter_context(
            patch("pipeline.prs_evaluator.get_store", MagicMock(return_value=store))
        )
        # compute_fkds imports eval.metrics inside the function; patch its attributes.
        mocks["eval_metrics"] = {}
        mocks["eval_metrics"]["token_f1"] = stack.enter_context(
            patch("eval.metrics.token_f1", fake_metrics.token_f1)
        )
        mocks["eval_metrics"]["llm_judge"] = stack.enter_context(
            patch("eval.metrics.llm_judge", fake_metrics.llm_judge)
        )
        yield mocks


def test_compute_fkds_returns_kds_fkds_and_persists():
    from pipeline.prs_evaluator import compute_fkds

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

    embeddings_by_question = {
        "answer chunk1 a": np.array([1.0, 0.0, 0.0]),
        "answer chunk2 a": np.array([0.0, 1.0, 0.0]),
    }
    factual_scores = {
        ("answer chunk1 a", "A"): 0.8,
        ("What is chunk1?", "answer chunk1 a"): 0.9,
        ("answer chunk2 a", "B"): 0.4,
        ("What is chunk2?", "answer chunk2 a"): 0.5,
    }

    with _fkds_patches(store, embeddings_by_question, factual_scores) as mocks:
        mean_kds, mean_fkds, fkds_by_chunk = compute_fkds(
            faqs, cfg, sample_cap=300, n=3, factual_weight=0.1
        )

    assert isinstance(mean_kds, float)
    assert isinstance(mean_fkds, float)
    assert 0.0 <= mean_kds <= 1.0
    assert 0.0 <= mean_fkds <= 1.0
    assert set(fkds_by_chunk.keys()) == {"chunk1", "chunk2"}
    for entry in fkds_by_chunk.values():
        assert "kds" in entry
        assert "factual_accuracy" in entry
        assert "fkds" in entry
        assert 0.0 <= entry["fkds"] <= 1.0

    # fKDS should blend consistency and factual accuracy; chunk1 is more factual.
    assert fkds_by_chunk["chunk1"]["fkds"] > fkds_by_chunk["chunk2"]["fkds"]

    assert len(store.set_payload_calls) == 2
    for collection, point_id, payload in store.set_payload_calls:
        assert collection == "test"
        assert point_id in ("chunk1", "chunk2")
        assert "kds" in payload
        assert "fkds" in payload
        assert "factual_accuracy" in payload
        assert "last_kds_round" in payload

    mocks["ver"].append_kds.assert_called_once()
    mocks["ver"].append_fkds.assert_called_once()
