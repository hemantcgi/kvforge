# tests/test_kv_inference.py
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_chunk(kv_version, chunk_id=1):
    import core.kv_utils as kv_utils
    fake_kv = np.zeros((28, 2, 8, 128), dtype=np.float16)
    return {
        "chunk_id": chunk_id,
        "text": "Amazon Bedrock is a managed service.",
        "page": 7,
        "score": 0.9,
        "kv_cache": kv_utils.serialize_kv(fake_kv),
        "kv_version": kv_version,
    }


def test_all_fresh_uses_kv_path():
    """When all chunks have current kv_version, should call generate_with_kv."""
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(kv_version=5, chunk_id=i) for i in range(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "kv_injection"


def test_any_stale_uses_text_fallback():
    from pipeline.kv_inference import decide_inference_mode
    chunks = [_fake_chunk(5), _fake_chunk(None), _fake_chunk(5)]
    mode = decide_inference_mode(chunks, current_lora_version=5)
    assert mode == "text_fallback"


def test_stale_chunks_are_queued():
    from pipeline.kv_inference import decide_inference_mode, get_stale_chunk_ids
    chunks = [_fake_chunk(5), _fake_chunk(None, 2), _fake_chunk(3, 3)]
    stale = get_stale_chunk_ids(chunks, current_lora_version=5)
    assert set(stale) == {2, 3}


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
               return_value={"checkpoint_path": None}), \
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
               return_value={"checkpoint_path": None}), \
         patch("pipeline.kv_inference.kv_background.record_access"), \
         patch("pipeline.kv_inference.kv_background.enqueue_kv_recompute"), \
         patch("pipeline.kv_inference.generate_text_in_context",
               return_value="safe answer"), \
         patch("pipeline.query_logger.log_query",
               side_effect=RuntimeError("disk full")):
        from pipeline.kv_inference import answer_with_retrieval
        result = answer_with_retrieval("Q?", cfg)

    assert result == "safe answer"
