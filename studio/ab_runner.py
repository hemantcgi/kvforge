# studio/ab_runner.py
"""
A/B query runner.

Both models share the same top-K Qdrant retrieval:
  Model A — KVForge RAG + GPU/vLLM:  retrieved chunks → configured LLM endpoint → answer
  Model B — Cloud LLM:               retrieved chunks → Anthropic/OpenAI/Gemini → answer
"""
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
_AB_TIMEOUT = 90.0
_AB_TOP_K = 10
_AB_SCORE_THRESHOLD = 0.55

_NO_CORPUS_MSG = (
    "No corpus chunks scored above the relevance threshold ({threshold}).\n\n"
    "The indexed corpus does not contain content closely related to this query. "
    "Try a query about topics covered by your data source."
)

_RAG_SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the provided context passages. "
    "Cite relevant passages by their number, e.g. [1]. "
    "If the context does not contain enough information, say so explicitly."
)


def _sanitize_error(msg: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9\-]{6,}", "sk-***", msg)


def _load_uc_cfg(uc_id: str) -> dict:
    config_path = ROOT / "examples" / uc_id / "config.json"
    if not config_path.exists():
        from studio.pipeline_runner import _ensure_config_json
        _ensure_config_json(uc_id)
    if config_path.exists():
        raw = json.loads(config_path.read_text())
        if "addon_config" in raw:
            from core.config import KVForgeConfig
            dc = KVForgeConfig(**raw)
            cfg = dc.get_merged_config("indexing", "inference", "training")
            cfg.setdefault("version_file", raw.get("version_file", f"examples/{uc_id}/version.json"))
            cfg.setdefault("collection", raw.get("collection", uc_id))
            return cfg
        return raw
    return {}


def _do_rag_search(query: str, cfg: dict) -> list:
    from fastembed import TextEmbedding
    from vectorstore.registry import get_store
    from pipeline.bedrock_rag import _run_search, Config

    search_cfg = dict(cfg)
    search_cfg["top_k"] = _AB_TOP_K
    search_cfg["score_threshold"] = _AB_SCORE_THRESHOLD

    embedder = TextEmbedding(
        model_name=cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
        show_download_progress=False,
    )
    store = get_store(search_cfg)
    rag_cfg = Config(**{k: search_cfg[k] for k in Config.__dataclass_fields__ if k in search_cfg})
    return _run_search(query, embedder, store, rag_cfg)


def _format_context(hits: list) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        payload = h.payload if hasattr(h, "payload") else {}
        text = payload.get("text", "").strip()
        page = payload.get("page", "?")
        parts.append(f"[{i}] (page {page})\n{text}")
    return "\n\n".join(parts)


async def run_ab_query(
    uc_id: str,
    query: str,
    model_a_settings: dict,
    model_b_settings: dict,
) -> dict:
    # Load UC config for defaults (vllm_url, llm_model, etc.)
    try:
        cfg = await asyncio.to_thread(_load_uc_cfg, uc_id)
    except Exception as e:
        err = {"text": "", "latency_ms": 0, "error": _sanitize_error(str(e))}
        return {"response_a": err, "response_b": err}

    if not cfg:
        err = {"text": "", "latency_ms": 0, "error": f"Config not found for UC '{uc_id}'"}
        return {"response_a": err, "response_b": err}

    # ── Shared retrieval ──────────────────────────────────────────────────────
    t_search = time.monotonic()
    try:
        hits = await asyncio.to_thread(_do_rag_search, query, cfg)
    except Exception as e:
        err = {"text": "", "latency_ms": 0,
               "error": f"Retrieval failed: {_sanitize_error(str(e))}"}
        return {"response_a": err, "response_b": err}
    search_ms = int((time.monotonic() - t_search) * 1000)

    if not hits:
        msg = _NO_CORPUS_MSG.format(threshold=_AB_SCORE_THRESHOLD)
        no_hits = {"text": msg, "latency_ms": search_ms, "retrieval_ms": search_ms,
                   "inference_ms": 0, "chunks_retrieved": 0, "chunks": []}
        return {"response_a": {**no_hits, "source": "kvforge-rag"},
                "response_b": {**no_hits, "source": "cloud-llm"}}

    context = _format_context(hits)
    chunks_data = [
        {
            "text": (h.payload.get("text", "") if hasattr(h, "payload") else ""),
            "score": round(h.score, 3) if hasattr(h, "score") else None,
            "source": (h.payload.get("source", h.payload.get("file_path", h.payload.get("filename", "")))
                       if hasattr(h, "payload") else ""),
        }
        for h in hits
    ]

    # ── Generate answers in parallel ──────────────────────────────────────────
    try:
        result_a, result_b = await asyncio.wait_for(
            asyncio.gather(
                _model_a_generate(query, context, model_a_settings, cfg, search_ms, len(hits), chunks_data),
                _model_b_generate(query, context, model_b_settings, search_ms, len(hits), chunks_data),
            ),
            timeout=_AB_TIMEOUT,
        )
    except asyncio.TimeoutError:
        timeout_err = {"text": "", "latency_ms": int(_AB_TIMEOUT * 1000), "error": "Query timed out"}
        return {"response_a": timeout_err, "response_b": timeout_err}

    return {"response_a": result_a, "response_b": result_b}


async def _model_a_generate(
    query: str,
    context: str,
    settings: dict,
    cfg: dict,
    search_ms: int,
    n_chunks: int,
    chunks_data: list,
) -> dict:
    """Call the GPU/vLLM endpoint with retrieved context to produce an answer."""
    # Resolve endpoint: UI setting > UC config > nothing
    endpoint = (settings.get("endpoint_url") or cfg.get("vllm_url") or "").rstrip("/")
    model_name = settings.get("model_name") or cfg.get("vllm_model") or cfg.get("llm_model") or "kvforge-local"
    temperature = float(settings.get("temperature", 0.2))
    max_tokens = int(settings.get("max_tokens", 256))
    system_prompt = settings.get("system_prompt") or _RAG_SYSTEM_PROMPT

    if not endpoint:
        return {
            "text": "",
            "latency_ms": search_ms,
            "retrieval_ms": search_ms,
            "inference_ms": 0,
            "source": "kvforge-rag",
            "chunks_retrieved": n_chunks,
            "chunks": chunks_data,
            "error": (
                "No GPU endpoint configured for Model A. "
                "Set the vLLM endpoint URL in Model A settings, "
                "or configure vllm_url in your datasource."
            ),
        }

    # Normalise: strip any trailing /v1 so we don't double it
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/v1/chat/completions"
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        inference_ms = int((time.monotonic() - t0) * 1000)
        return {
            "text": text,
            "retrieval_ms": search_ms,
            "inference_ms": inference_ms,
            "latency_ms": search_ms + inference_ms,
            "source": f"vllm/{model_name}",
            "chunks_retrieved": n_chunks,
            "chunks": chunks_data,
            "phase_used": data.get("phase", "phase1-rag"),
        }
    except Exception as e:
        inference_ms = int((time.monotonic() - t0) * 1000)
        return {
            "text": "",
            "retrieval_ms": search_ms,
            "inference_ms": inference_ms,
            "latency_ms": search_ms + inference_ms,
            "source": f"vllm/{model_name}",
            "chunks_retrieved": n_chunks,
            "chunks": chunks_data,
            "error": _sanitize_error(str(e)),
        }


async def _model_b_generate(
    query: str,
    context: str,
    settings: dict,
    search_ms: int,
    n_chunks: int,
    chunks_data: list,
) -> dict:
    """Call cloud LLM with retrieved context to produce an answer."""
    from studio.settings_manager import get_setting
    provider    = settings.get("provider", "anthropic")
    api_key     = settings.get("api_key") or get_setting(f"{provider}_api_key") or ""
    model       = settings.get("model", "claude-haiku-4-5-20251001")
    temperature = float(settings.get("temperature", 0.3))
    max_tokens  = int(settings.get("max_tokens", 1024))
    system_prompt = settings.get("system_prompt") or _RAG_SYSTEM_PROMPT
    user_message  = f"Context:\n{context}\n\nQuestion: {query}"

    t0 = time.monotonic()
    try:
        if provider == "anthropic":
            text, cost = await _call_anthropic(api_key, model, user_message, system_prompt, temperature, max_tokens)
        elif provider == "openai":
            text, cost = await _call_openai(api_key, model, user_message, system_prompt, temperature, max_tokens)
        elif provider == "gemini":
            text, cost = await _call_gemini(api_key, model, user_message, temperature, max_tokens)
        else:
            return {"text": "", "latency_ms": 0, "retrieval_ms": search_ms,
                    "inference_ms": 0, "chunks_retrieved": n_chunks, "chunks": chunks_data,
                    "source": provider, "error": f"Unknown provider: {provider}"}
        inference_ms = int((time.monotonic() - t0) * 1000)
        return {
            "text": text,
            "retrieval_ms": search_ms,
            "inference_ms": inference_ms,
            "latency_ms": search_ms + inference_ms,
            "source": f"{provider}/{model}",
            "chunks_retrieved": n_chunks,
            "chunks": chunks_data,
            "cost_est_usd": cost,
        }
    except Exception as e:
        inference_ms = int((time.monotonic() - t0) * 1000)
        return {"text": "", "retrieval_ms": search_ms, "inference_ms": inference_ms,
                "latency_ms": search_ms + inference_ms, "chunks_retrieved": n_chunks,
                "chunks": chunks_data, "source": provider, "error": _sanitize_error(str(e))}


async def _call_anthropic(api_key, model, user_message, system_prompt, temperature, max_tokens):
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    msg = await client.messages.create(
        model=model, max_tokens=max_tokens, system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=temperature,
    )
    text = msg.content[0].text
    cost = round((msg.usage.input_tokens * 0.00000025) + (msg.usage.output_tokens * 0.00000125), 6)
    return text, cost


async def _call_openai(api_key, model, user_message, system_prompt, temperature, max_tokens):
    import openai
    client = openai.AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
    )
    text = resp.choices[0].message.content
    cost = round((resp.usage.prompt_tokens * 0.00000015) + (resp.usage.completion_tokens * 0.0000006), 6)
    return text, cost


async def _call_gemini(api_key, model, user_message, temperature, max_tokens):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    resp = await asyncio.to_thread(
        gen_model.generate_content, user_message,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    return resp.text, 0.0
