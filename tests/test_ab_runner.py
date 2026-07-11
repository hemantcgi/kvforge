# tests/test_ab_runner.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from studio.ab_runner import _query_local, _query_cloud


@pytest.mark.asyncio
async def test_query_local_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42."}}],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("studio.ab_runner.httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value = ctx

        result = await _query_local("What is 6×7?", {})

    assert result["text"] == "The answer is 42."
    assert result["source"] == "local-vllm"
    assert "latency_ms" in result


@pytest.mark.asyncio
async def test_query_local_connection_error():
    with patch("studio.ab_runner.httpx.AsyncClient") as MockClient:
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(side_effect=Exception("Connection refused"))
        MockClient.return_value = ctx

        result = await _query_local("Q?", {})

    assert result["text"] == ""
    assert "error" in result


@pytest.mark.asyncio
async def test_query_cloud_uses_stored_key():
    with patch("studio.settings_manager.get_setting", return_value="sk-ant-api03-test"):
        with patch("studio.ab_runner._call_anthropic", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ("Cloud answer.", 0.0003)
            result = await _query_cloud("Q?", {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})

    assert result["text"] == "Cloud answer."
    assert result["source"] == "anthropic"
    assert result["cost_est_usd"] == 0.0003


@pytest.mark.asyncio
async def test_query_cloud_unknown_provider():
    result = await _query_cloud("Q?", {"provider": "cohere"})
    assert result["text"] == ""
    assert "error" in result


@pytest.mark.asyncio
async def test_run_ab_query_returns_both():
    from studio.ab_runner import run_ab_query
    with patch("studio.ab_runner._query_local", new_callable=AsyncMock) as ml, \
         patch("studio.ab_runner._query_cloud", new_callable=AsyncMock) as mc:
        ml.return_value = {"text": "Local.", "source": "local-vllm", "latency_ms": 800}
        mc.return_value = {"text": "Cloud.", "source": "anthropic", "latency_ms": 1200}
        result = await run_ab_query("uc-test", "Q?", {}, {})

    assert result["response_a"]["text"] == "Local."
    assert result["response_b"]["text"] == "Cloud."
