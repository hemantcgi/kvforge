import pytest, numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path


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
    mock_vs.update_payload.assert_called_once()
    call_kwargs = mock_vs.update_payload.call_args[1]
    assert "kv_token_path" in call_kwargs["payload"]


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
