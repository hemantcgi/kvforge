import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _mock_cfg(tmp_path):
    return {
        "collection": "test_col",
        "version_file": str(tmp_path / "version.json"),
        "per_token_kv_dir": str(tmp_path / "per_token_kv"),
        "kv_num_layers": 4,
        "kv_num_heads": 2,
        "kv_head_dim": 64,
        "chunk_size": 32,
    }


def test_promotion_writes_file(tmp_path):
    from pipeline.kv_background import promote_chunk_to_enhanced_tier
    import torch

    cfg = _mock_cfg(tmp_path)
    chunk_id = "abc123"
    chunk_text = "This is a test chunk about machine learning."

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_vs = MagicMock()

    with patch("pipeline.kv_background.compute_per_token_kv") as mock_cptk:
        mock_cptk.return_value = np.random.randn(4, 2, 2, 8, 64).astype(np.float16)
        promote_chunk_to_enhanced_tier(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            cfg=cfg,
            model=mock_model,
            tokenizer=mock_tokenizer,
            vector_store=mock_vs,
            tq_config=None,
        )

    kv_dir = Path(cfg["per_token_kv_dir"])
    files = list(kv_dir.glob(f"{chunk_id}*.npz"))
    assert len(files) == 1, "Expected one .npz file written"
    mock_vs.set_payload.assert_called_once()
    positional_args, _ = mock_vs.set_payload.call_args
    assert "kv_token_path" in positional_args[2]


def test_promotion_skips_already_enhanced(tmp_path):
    from pipeline.kv_background import promote_chunk_to_enhanced_tier

    cfg = _mock_cfg(tmp_path)
    mock_vs = MagicMock()

    with patch("pipeline.kv_background.compute_per_token_kv") as mock_cptk:
        promote_chunk_to_enhanced_tier(
            chunk_id="already_done",
            chunk_text="text",
            cfg=cfg,
            model=MagicMock(),
            tokenizer=MagicMock(),
            vector_store=mock_vs,
            tq_config=None,
            existing_kv_token_path="/some/existing/path.npz",
        )
    mock_cptk.assert_not_called()


def _make_point(point_id, payload):
    return SimpleNamespace(id=point_id, payload=payload)


def _make_scrollable_store(access_counts):
    """Return a mock vector store whose scroll returns all access_counts."""
    store = MagicMock()
    points = [_make_point(i, {"access_count": c}) for i, c in enumerate(access_counts)]
    store.scroll.return_value = (points, None)
    return store


def test_maybe_promote_trigger_when_enabled_and_hot(tmp_path):
    from pipeline.kv_background import _maybe_promote_chunk_to_enhanced_tier

    cfg = _mock_cfg(tmp_path)
    cfg["enable_enhanced_tier"] = True

    # 100 chunks with counts 1..100; target chunk id=99 has count 100 -> top 15%.
    store = _make_scrollable_store(list(range(1, 101)))
    payload = {"text": "hot chunk text", "access_count": 100}

    with patch("pipeline.kv_background.promote_chunk_to_enhanced_tier") as mock_promote:
        _maybe_promote_chunk_to_enhanced_tier(
            cfg=cfg,
            chunk_id=99,
            chunk_payload=payload,
            model=MagicMock(),
            tokenizer=MagicMock(),
            vector_store=store,
        )

    mock_promote.assert_called_once()
    call_kwargs = mock_promote.call_args[1]
    assert call_kwargs["chunk_id"] == "99"
    assert call_kwargs["chunk_text"] == "hot chunk text"
    assert call_kwargs["cfg"] is cfg


def test_maybe_promote_noop_when_disabled(tmp_path):
    from pipeline.kv_background import _maybe_promote_chunk_to_enhanced_tier

    cfg = _mock_cfg(tmp_path)
    cfg["enable_enhanced_tier"] = False

    store = _make_scrollable_store(list(range(1, 101)))
    payload = {"text": "hot chunk text", "access_count": 100}

    with patch("pipeline.kv_background.promote_chunk_to_enhanced_tier") as mock_promote:
        _maybe_promote_chunk_to_enhanced_tier(
            cfg=cfg,
            chunk_id=99,
            chunk_payload=payload,
            model=MagicMock(),
            tokenizer=MagicMock(),
            vector_store=store,
        )

    mock_promote.assert_not_called()
    store.scroll.assert_not_called()


def test_maybe_promote_noop_when_already_enhanced(tmp_path):
    from pipeline.kv_background import _maybe_promote_chunk_to_enhanced_tier

    cfg = _mock_cfg(tmp_path)
    cfg["enable_enhanced_tier"] = True

    store = _make_scrollable_store(list(range(1, 101)))
    payload = {"text": "already enhanced", "access_count": 100, "kv_token_path": "/x.npz"}

    with patch("pipeline.kv_background.promote_chunk_to_enhanced_tier") as mock_promote:
        _maybe_promote_chunk_to_enhanced_tier(
            cfg=cfg,
            chunk_id=99,
            chunk_payload=payload,
            model=MagicMock(),
            tokenizer=MagicMock(),
            vector_store=store,
        )

    mock_promote.assert_not_called()
    store.scroll.assert_not_called()


def test_maybe_promote_noop_when_not_hot(tmp_path):
    from pipeline.kv_background import _maybe_promote_chunk_to_enhanced_tier

    cfg = _mock_cfg(tmp_path)
    cfg["enable_enhanced_tier"] = True

    # 100 chunks; target chunk id=0 has count 1 -> not in top 15%.
    store = _make_scrollable_store(list(range(1, 101)))
    payload = {"text": "cold chunk", "access_count": 1}

    with patch("pipeline.kv_background.promote_chunk_to_enhanced_tier") as mock_promote:
        _maybe_promote_chunk_to_enhanced_tier(
            cfg=cfg,
            chunk_id=0,
            chunk_payload=payload,
            model=MagicMock(),
            tokenizer=MagicMock(),
            vector_store=store,
        )

    mock_promote.assert_not_called()
