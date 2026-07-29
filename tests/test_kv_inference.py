# tests/test_kv_inference.py
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_chunk(kv_version, chunk_id=1, kds=1.0):
    import core.kv_utils as kv_utils
    fake_kv = np.zeros((28, 2, 8, 128), dtype=np.float16)
    return {
        "chunk_id": chunk_id,
        "text": "Amazon Bedrock is a managed service.",
        "page": 7,
        "score": 0.9,
        "kv_cache": kv_utils.serialize_kv(fake_kv),
        "kv_version": kv_version,
        "kds": kds,
    }


def test_all_fresh_uses_kv_path():
    """When all chunks have current kv_version, should call generate_with_kv."""
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(kv_version=5, chunk_id=i) for i in range(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=0.0)
    assert mode == "kv_injection"


def test_any_stale_uses_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5), _fake_chunk(None), _fake_chunk(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=0.0)
    assert mode == "text_fallback"


def test_stale_chunks_are_queued():
    from pipeline.kv_inference import decide_inference_mode, get_stale_chunk_ids
    chunks = [_fake_chunk(5), _fake_chunk(None, 2), _fake_chunk(3, 3)]
    stale = get_stale_chunk_ids(chunks, current_lora_version=5)
    assert set(stale) == {2, 3}


def test_kds_above_threshold_keeps_kv_injection():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.8), _fake_chunk(5, chunk_id=2, kds=0.9)]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=0.5)
    assert mode == "kv_injection"


def test_kds_below_threshold_forces_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.8), _fake_chunk(5, chunk_id=2, kds=0.3)]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=0.5)
    assert mode == "text_fallback"


def test_missing_kds_forces_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    missing_kds_chunk = {
        "chunk_id": 2,
        "kv_version": 5,
        "kv_cache": _fake_chunk(5)["kv_cache"],
    }
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.8), missing_kds_chunk]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=0.5)
    assert mode == "text_fallback"


def test_kds_threshold_none_disables_kv_injection():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.9)]
    mode = decide_inference_mode(chunks, current_lora_version=5, kds_threshold=None)
    assert mode == "text_fallback"


def test_phase_below_two_disables_kv_injection():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.9)]
    mode = decide_inference_mode(chunks, current_lora_version=5, phase=1, kds_threshold=0.0)
    assert mode == "text_fallback"


def _fake_chunk_with_fkds(kv_version, chunk_id=1, kds=1.0, fkds=None):
    chunk = _fake_chunk(kv_version, chunk_id=chunk_id, kds=kds)
    if fkds is not None:
        chunk["fkds"] = fkds
    return chunk


def test_fkds_above_threshold_keeps_kv_injection():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [
        _fake_chunk_with_fkds(5, chunk_id=1, kds=0.2, fkds=0.8),
        _fake_chunk_with_fkds(5, chunk_id=2, kds=0.2, fkds=0.9),
    ]
    mode = decide_inference_mode(chunks, current_lora_version=5, fkds_threshold=0.5)
    assert mode == "kv_injection"


def test_fkds_below_threshold_forces_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [
        _fake_chunk_with_fkds(5, chunk_id=1, kds=0.8, fkds=0.8),
        _fake_chunk_with_fkds(5, chunk_id=2, kds=0.8, fkds=0.3),
    ]
    mode = decide_inference_mode(chunks, current_lora_version=5, fkds_threshold=0.5)
    assert mode == "text_fallback"


def test_fkds_missing_forces_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5, chunk_id=1, kds=0.8)]
    mode = decide_inference_mode(chunks, current_lora_version=5, fkds_threshold=0.5)
    assert mode == "text_fallback"


def test_kv_stacking_produces_correct_past_key_values_shape():
    """stack_past_key_values must produce HuggingFace-compatible past_key_values."""
    import core.kv_utils as kv_utils
    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, N_CHUNKS = 28, 8, 128, 5
    # Simulate 5 fresh chunks
    chunks = [_fake_chunk(kv_version=3, chunk_id=i) for i in range(N_CHUNKS)]
    chunk_arrs = [
        kv_utils.deserialize_kv(c["kv_cache"], shape=(NUM_LAYERS, 2, NUM_KV_HEADS, HEAD_DIM))
        for c in chunks
    ]
    pkv = kv_utils.stack_past_key_values(chunk_arrs, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)
    layers = list(kv_utils._iter_kv_layers(pkv))
    assert len(layers) == NUM_LAYERS
    k, v = layers[0]
    # [batch=1, num_kv_heads, N_chunks, head_dim]
    assert k.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)
    assert v.shape == (1, NUM_KV_HEADS, N_CHUNKS, HEAD_DIM)


def test_answer_with_retrieval_calls_log_query(tmp_path):
    """answer_with_retrieval must call query_logger.log_query after generating an answer."""
    import sys
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "collection": "test",
        "top_k": 3,
        "query_log_db": str(tmp_path / "q.db"),
        "checkpoint_dir": str(tmp_path),
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    }

    fake_hit = MagicMock()
    fake_hit.id = 42
    fake_hit.score = 0.9
    fake_hit.payload = {
        "text": "Some context text.",
        "page": 1,
        "kv_cache": None,
        "kv_version": None,
    }

    logged = []

    with patch("pipeline.kv_inference._run_search", return_value=[fake_hit]), \
         patch("pipeline.kv_inference.get_store", return_value=MagicMock()), \
         patch("pipeline.kv_inference.model_loader.load",
               return_value=(MagicMock(), MagicMock())), \
         patch("pipeline.kv_inference.ver.get_lora_version", return_value=1), \
         patch("pipeline.kv_inference.ver.load",
               return_value={"checkpoint_path": None, "phase": 2}), \
         patch("pipeline.kv_inference.kv_background.record_access"), \
         patch("pipeline.kv_inference.kv_background.enqueue_kv_recompute"), \
         patch("pipeline.kv_inference.generate_text_in_context",
               return_value="mocked answer"), \
         patch("pipeline.query_logger.log_query",
               side_effect=lambda **kw: logged.append(kw)):
        from pipeline.kv_inference import answer_with_retrieval
        result = answer_with_retrieval("What is Bedrock?", cfg)

    assert result == "mocked answer"
    assert len(logged) == 1
    assert isinstance(logged[0]["db_path"], str)
    assert logged[0]["query_text"] == "What is Bedrock?"
    assert logged[0]["answer_text"] == "mocked answer"
    assert logged[0]["routed_to"] == "retrieval"


def test_answer_with_retrieval_survives_log_query_exception(tmp_path):
    """A log_query crash must not propagate — answer must still be returned."""
    import sys
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "collection": "test",
        "top_k": 3,
        "query_log_db": str(tmp_path / "q.db"),
        "checkpoint_dir": str(tmp_path),
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
    }

    fake_hit = MagicMock()
    fake_hit.id = 1
    fake_hit.score = 0.9
    fake_hit.payload = {"text": "ctx", "page": 1, "kv_cache": None, "kv_version": None}

    with patch("pipeline.kv_inference._run_search", return_value=[fake_hit]), \
         patch("pipeline.kv_inference.get_store", return_value=MagicMock()), \
         patch("pipeline.kv_inference.model_loader.load",
               return_value=(MagicMock(), MagicMock())), \
         patch("pipeline.kv_inference.ver.get_lora_version", return_value=0), \
         patch("pipeline.kv_inference.ver.load",
               return_value={"checkpoint_path": None, "phase": 2}), \
         patch("pipeline.kv_inference.kv_background.record_access"), \
         patch("pipeline.kv_inference.kv_background.enqueue_kv_recompute"), \
         patch("pipeline.kv_inference.generate_text_in_context",
               return_value="safe answer"), \
         patch("pipeline.query_logger.log_query",
               side_effect=RuntimeError("disk full")):
        from pipeline.kv_inference import answer_with_retrieval
        result = answer_with_retrieval("Q?", cfg)

    assert result == "safe answer"


def test_select_recompute_tokens_zero_ratio():
    from pipeline.kv_inference import _select_recompute_tokens
    deviation = np.array([0.1, 0.5, 0.2, 0.9])
    mask = _select_recompute_tokens(deviation, 0.0)
    assert mask.sum() == 0


def test_select_recompute_tokens_full_ratio():
    from pipeline.kv_inference import _select_recompute_tokens
    deviation = np.array([0.1, 0.5, 0.2, 0.9])
    mask = _select_recompute_tokens(deviation, 1.0)
    assert mask.sum() == 4


def test_select_recompute_tokens_partial_ratio():
    from pipeline.kv_inference import _select_recompute_tokens
    deviation = np.array([0.1, 0.5, 0.2, 0.9])
    mask = _select_recompute_tokens(deviation, 0.5)
    # 50% of 4 tokens = 2, including the attention sink (index 0).
    assert mask[0]  # attention sink
    assert mask.sum() == 2
    assert mask[3]  # highest deviation
