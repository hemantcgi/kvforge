import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_select_noise_targets_deterministic():
    from tools.noise_injector import _select_noise_targets
    chunks = [{"chunk_id": i, "text": f"Chunk {i}"} for i in range(100)]
    targets1 = _select_noise_targets(chunks, 0.10, seed=42)
    targets2 = _select_noise_targets(chunks, 0.10, seed=42)
    assert targets1 == targets2
    assert len(targets1) == 10
    assert all(0 <= i < 100 for i in targets1)


def test_select_noise_targets_zero_rate():
    from tools.noise_injector import _select_noise_targets
    chunks = [{"chunk_id": i} for i in range(50)]
    targets = _select_noise_targets(chunks, 0.0)
    assert targets == []


def test_select_noise_targets_full_rate():
    from tools.noise_injector import _select_noise_targets
    chunks = [{"chunk_id": i} for i in range(20)]
    targets = _select_noise_targets(chunks, 1.0)
    assert len(targets) == 20


def test_build_noise_prompt_contains_original_text():
    from tools.noise_injector import _build_noise_prompt
    prompt = _build_noise_prompt("Amazon Bedrock is a managed service for LLMs.")
    assert "Amazon Bedrock is a managed service for LLMs." in prompt
    assert "incorrect" in prompt.lower() or "wrong" in prompt.lower()


def test_inject_noise_with_mock_llm():
    from tools.noise_injector import inject_noise
    chunks = [{"chunk_id": i, "text": f"Fact {i}."} for i in range(20)]
    with patch("tools.noise_injector._call_llm", return_value="Incorrect fact."):
        result = inject_noise(chunks, noise_rate=0.5, api_key="fake")
    assert len(result) == 20
    corrupted = [c for c in result if c["text"] != f"Fact {c['chunk_id']}."]
    assert len(corrupted) == 10
    assert all(c["text"] == "Incorrect fact." for c in corrupted)
    clean = [c for c in result if c["text"] == f"Fact {c['chunk_id']}."]
    assert len(clean) == 10


def test_inject_noise_zero_rate_returns_copy():
    from tools.noise_injector import inject_noise
    chunks = [{"chunk_id": 0, "text": "Original."}]
    result = inject_noise(chunks, noise_rate=0.0, api_key="fake")
    assert result[0]["text"] == "Original."
    assert result is not chunks
