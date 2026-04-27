# studio/ab_runner.py
import asyncio
import re
import time
import httpx

_VLLM_URL = "http://localhost:8090/v1/chat/completions"
_AB_TIMEOUT = 45.0


def _sanitize_error(msg: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9\-]{6,}", "sk-***", msg)


async def run_ab_query(
    uc_id: str,
    query: str,
    model_a_settings: dict,
    model_b_settings: dict,
) -> dict:
    # uc_id is reserved for future per-UC vLLM port routing
    try:
        result_a, result_b = await asyncio.wait_for(
            asyncio.gather(
                _query_local(query, model_a_settings),
                _query_cloud(query, model_b_settings),
            ),
            timeout=_AB_TIMEOUT,
        )
    except asyncio.TimeoutError:
        timeout_err = {"text": "", "latency_ms": int(_AB_TIMEOUT * 1000), "error": "Query timed out"}
        return {"response_a": timeout_err, "response_b": timeout_err}
    return {"response_a": result_a, "response_b": result_b}


async def _query_local(query: str, settings: dict) -> dict:
    payload = {
        "model": "kvforge-local",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        "temperature": float(settings.get("temperature", 0.2)),
        "max_tokens": int(settings.get("max_tokens", 256)),
        "top_p": float(settings.get("top_p", 0.9)),
    }
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(_VLLM_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data["choices"][0]["message"]["content"],
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "source": "local-vllm",
            "phase_used": data.get("phase", "unknown"),
            "confidence": data.get("confidence"),
        }
    except Exception as e:
        return {"text": "", "latency_ms": 0, "source": "local-vllm", "error": _sanitize_error(str(e))}


async def _query_cloud(query: str, settings: dict) -> dict:
    from studio.settings_manager import get_setting
    provider = settings.get("provider", "anthropic")
    api_key = settings.get("api_key") or get_setting(f"{provider}_api_key") or ""
    model = settings.get("model", "claude-haiku-4-5-20251001")
    temperature = float(settings.get("temperature", 0.3))
    max_tokens = int(settings.get("max_tokens", 512))
    system_prompt = settings.get("system_prompt", "You are a helpful assistant.")

    t0 = time.monotonic()
    try:
        if provider == "anthropic":
            text, cost = await _call_anthropic(api_key, model, query, system_prompt, temperature, max_tokens)
        elif provider == "openai":
            text, cost = await _call_openai(api_key, model, query, system_prompt, temperature, max_tokens)
        elif provider == "gemini":
            text, cost = await _call_gemini(api_key, model, query, temperature, max_tokens)
        else:
            return {"text": "", "latency_ms": 0, "source": provider, "error": f"Unknown provider: {provider}"}
        return {
            "text": text,
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "source": provider,
            "cost_est_usd": cost,
        }
    except Exception as e:
        return {"text": "", "latency_ms": 0, "source": provider, "error": _sanitize_error(str(e))}


async def _call_anthropic(api_key, model, query, system_prompt, temperature, max_tokens) -> tuple[str, float]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
        temperature=temperature,
    )
    text = msg.content[0].text
    cost = round((msg.usage.input_tokens * 0.00000025) + (msg.usage.output_tokens * 0.00000125), 6)
    return text, cost


async def _call_openai(api_key, model, query, system_prompt, temperature, max_tokens) -> tuple[str, float]:
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
    )
    text = resp.choices[0].message.content
    cost = round((resp.usage.prompt_tokens * 0.00000015) + (resp.usage.completion_tokens * 0.0000006), 6)
    return text, cost


async def _call_gemini(api_key, model, query, temperature, max_tokens) -> tuple[str, float]:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    resp = await asyncio.to_thread(
        gen_model.generate_content, query,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return resp.text, 0.0
