"""FastAPI monitoring dashboard for KVForge.

Provides a REST API consumed by the bundled HTML dashboard UI.  Key endpoints:

* ``GET /api/health``        — liveness check.
* ``GET /api/version``       — current LoRA version and phase.
* ``GET /api/stats``         — tier counts, top accessed chunks, access report.
* ``GET /api/config``        — display-safe config fields for the UI settings panel.
* ``POST /api/query``        — A/B query: runs KVForge (Model A) and Gemini
  (Model B) in parallel and returns both answers with latency metrics.

GPU modules (torch/transformers) are imported and the base model is pre-warmed
in the startup event so that worker threads never race on the lazy-import lock.

Start the server::

    python3 -m pipeline.monitoring_dashboard
    python3 -m pipeline.monitoring_dashboard --config examples/usecase4_bedrock_userguide/config.json
    uvicorn pipeline.monitoring_dashboard:app --port 8084 --reload
"""

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

# Ensure project root is on sys.path before any local imports (do once, not per-thread)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import core.version as ver
from vectorstore.registry import get_store

# Heavy modules (torch/transformers) loaded once at startup in main thread
# so worker threads never race on the lazy-import lock.
_model_loader = None
_kv_background = None
_kv_inference = None

_model_b_config: dict = {
    "provider": "gemini",
    "model": "gemini-2.0-flash",  # must match first item in JS MODELS_B["gemini"]
    "api_key": "",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Import GPU modules and pre-warm the model+LoRA in the main thread at startup."""
    global _model_loader, _kv_background, _kv_inference
    try:
        import core.model_loader as _ml
        import pipeline.kv_background as _kb
        import pipeline.kv_inference as _ki
        _model_loader = _ml
        _kv_background = _kb
        _kv_inference = _ki
        # Pre-warm: load the SAME checkpoint that queries will request so the
        # first query hits the cache.  Loading only the base model (None) here
        # causes a cache miss on every query because _answer_kvforge always
        # calls load(lora_ckpt) where lora_ckpt != None.
        cfg = _load_cfg()
        _ml.init(cfg)
        ver.init(cfg)
        lora_ckpt = ver.load().get("checkpoint_path")
        print(f"[dashboard] pre-warming model (lora_ckpt={lora_ckpt})…", flush=True)
        _ml.load(lora_ckpt)
        print("[dashboard] model ready", flush=True)
    except Exception as e:
        print(f"[dashboard] inference modules unavailable: {e}", flush=True)
    yield


app = FastAPI(title="KVForge Dashboard", lifespan=_lifespan)
_cfg: dict = {}
_vector_store = None
_query_executor = ThreadPoolExecutor(max_workers=2)

# Config file path — overridden by --config CLI arg at startup (no annotation so global works)
_config_path = "my_config.json"


def _load_cfg() -> dict:
    global _cfg
    if not _cfg:
        with open(_config_path) as f:
            _cfg = json.load(f)
    return _cfg


def _get_store():
    """Return the module-level VectorStore singleton, creating it on first call.

    Returns:
        A ``VectorStore``-protocol-compatible instance configured from the
        loaded datasource config.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = get_store(_load_cfg())
    return _vector_store


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/api/version")
def get_version():
    return ver.load()


@app.get("/api/stats")
def get_stats():
    """Return collection statistics for the dashboard overview panel.

    Paginates through the entire collection to compute tier distribution counts
    and the top-10 most-accessed chunks.  Also reads ``access_report.json``
    if it exists.

    Returns:
        JSON object with keys ``version``, ``tier_counts``, ``top_chunks``,
        ``access_report``, and ``total_chunks``.
    """
    cfg = _load_cfg()
    v = ver.load()

    # Vector store tier counts — paginate to handle >5000 chunks
    store = _get_store()
    tier_counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}
    top_chunks = []
    try:
        all_results = []
        offset = None
        while True:
            batch, offset = store.scroll(
                cfg["collection"],
                limit=500,
                with_payload=True,
                offset=offset,
            )
            all_results.extend(batch)
            if offset is None:
                break

        for r in all_results:
            t = r.payload.get("tier", "frozen")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        top_chunks = sorted(
            [{"chunk_id": r.id,
              "page": r.payload.get("page", 0),
              "access_count": r.payload.get("access_count", 0),
              "parametric_hit_count": r.payload.get("parametric_hit_count", 0),
              "tier": r.payload.get("tier", "frozen"),
              "text_preview": (r.payload.get("text", "")[:80] + "…")}
             for r in all_results],
            key=lambda x: x["access_count"],
            reverse=True,
        )[:10]
    except Exception as e:
        tier_counts["error"] = str(e)

    # Access report
    access_report = {}
    rp = Path("access_report.json")
    if rp.exists():
        with open(rp) as f:
            access_report = json.load(f)

    return {
        "version": v,
        "tier_counts": tier_counts,
        "top_chunks": top_chunks,
        "access_report": access_report,
        "total_chunks": sum(v for k, v in tier_counts.items() if k != "error"),
    }


@app.get("/api/config")
def get_config():
    """Return display-safe config fields for the dashboard settings panel."""
    cfg = _load_cfg()
    llm_model = cfg.get("llm_model", "meta-llama/Llama-3.2-3B-Instruct")
    embed_model = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    hf_base = "https://huggingface.co/"
    return {
        "llm_model": llm_model,
        "llm_model_url": hf_base + llm_model,
        "embed_model": embed_model,
        "embed_model_url": hf_base + embed_model,
        "model_b_provider": _model_b_config.get("provider", "gemini"),
        "model_b_model": _model_b_config.get("model", "gemini-2.0-flash"),
        "top_k": cfg.get("top_k", 5),
        "collection": cfg.get("collection", ""),
    }


@app.get("/api/access-report")
def get_access_report():
    rp = Path("access_report.json")
    if not rp.exists():
        return JSONResponse({"error": "No report yet"}, status_code=404)
    with open(rp) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# A/B Query Comparison
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    # Answer A (local LLM RAG) generation params
    a_top_k: int = 5
    a_max_new_tokens: int = 64
    a_temperature: float = 0.7
    a_top_p: float = 0.9
    a_repetition_penalty: float = 1.2
    # Answer B (Gemini) params
    b_top_k: int = 5
    b_max_output_tokens: int = 4096
    b_temperature: float = 1.0


class ModelBConfigRequest(BaseModel):
    provider: Literal["gemini", "openai"]
    model: str
    api_key: str

@app.post("/api/set_model_b_config")
def set_model_b_config(req: ModelBConfigRequest):
    global _model_b_config
    _model_b_config = {"provider": req.provider, "model": req.model, "api_key": req.api_key}
    return {"ok": True}


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _answer_kvforge(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "A:KVForge"
    _log(tag, f"START query={query!r} top_k={params.a_top_k} max_new_tokens={params.a_max_new_tokens} temp={params.a_temperature}")
    if _model_loader is None or _kv_inference is None:
        _log(tag, "SKIP — inference modules not loaded (GPU required)")
        return {"answer": "KVForge inference modules not available (GPU required)", "latency_ms": 0}
    retrieval_ms = 0
    generation_ms = 0
    chunks = []
    mode = "text_in_context"
    gate_info = {}
    try:
        import torch
        ver.init(cfg)
        _kv_background.start(cfg)

        phase = ver.get_phase()
        vllm_url = cfg.get("vllm_url")

        # ── vLLM path: fast generation via dedicated inference server ──────
        # When vllm_url is set, all generation (Phase 3 parametric and
        # Phase 1/2 text-in-context) is routed to the vLLM server.
        # KV injection (Phase 2) falls back to the local model if vLLM is up;
        # if the vLLM server is not reachable the local model is used instead.
        if vllm_url:
            import core.vllm_client as _vllm
            import numpy as _np
            vllm_model = cfg.get("vllm_model", cfg.get("llm_model", "default"))

            # ── Per-query PRS gate ────────────────────────────────────────────
            # Embed the query and find max cosine similarity to known-good
            # query embeddings stored by prs_evaluator.  If ≥ 0.75 the model
            # has "mastered" this question → bypass retrieval entirely.
            per_query_prs = 0.0
            use_parametric = (phase >= 3)

            if not use_parametric and phase >= 2:
                version_data = ver.load()
                known_good = version_data.get("known_good_queries", [])
                if known_good:
                    from fastembed import TextEmbedding as _TEprs
                    _q_emb = list(_TEprs(model_name=cfg["embed_model"],
                                         show_download_progress=False).embed([query]))[0]
                    _nq = float(_np.linalg.norm(_q_emb))
                    if _nq > 1e-9:
                        sims = [
                            float(_np.dot(_q_emb, _np.array(kg))
                                  / (_nq * (float(_np.linalg.norm(_np.array(kg))) + 1e-9)))
                            for kg in known_good
                        ]
                        per_query_prs = max(sims)
                        if per_query_prs >= 0.75:
                            use_parametric = True
                            _log(tag, f"per-query PRS={per_query_prs:.3f} ≥ 0.75 → parametric override (no retrieval)")

            if use_parametric:
                mode = "parametric"
                if per_query_prs == 0.0:
                    _log(tag, f"Phase 3 via vLLM ({vllm_url}, model={vllm_model})…")
                # Apply Llama chat template client-side so vLLM sees the same
                # prompt as the local model path.
                from transformers import AutoTokenizer as _AT
                _tok = _AT.from_pretrained(cfg["llm_model"])
                messages = [{"role": "user", "content": query}]
                prompt = _tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                t_gen = time.time()
                answer = _vllm.generate(
                    prompt,
                    url=vllm_url,
                    model=vllm_model,
                    max_tokens=params.a_max_new_tokens,
                    temperature=params.a_temperature if params.a_temperature > 0 else 0.0,
                    stop=["<|eot_id|>", "<|end_of_text|>"],
                )
                generation_ms = int((time.time() - t_gen) * 1000)
                _log(tag, f"generation done mode=parametric via vLLM ({len(answer)} chars, {generation_ms}ms)")
                elapsed = int((time.time() - t0) * 1000)
                _log(tag, f"DONE {elapsed}ms")
                return {"answer": answer, "latency_ms": elapsed,
                        "retrieval_ms": 0, "generation_ms": generation_ms,
                        "chunks": [], "mode": mode, "gate": {},
                        "prs_score": round(per_query_prs, 4)}

            # Phase 1/2 with vLLM: retrieve then generate
            from fastembed import TextEmbedding
            from vectorstore.registry import get_store as _gs
            cfg_a = dict(cfg, top_k=params.a_top_k)
            embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
            store = _gs(cfg)
            _log(tag, "embedding query…")
            q_vec = list(embedder.embed([query]))[0].tolist()
            t_ret = time.time()
            hits = store.query(cfg["collection"], q_vec, params.a_top_k)
            retrieval_ms = int((time.time() - t_ret) * 1000)
            _log(tag, f"search done — {len(hits)} hits, retrieval={retrieval_ms}ms")
            if not hits:
                answer = "No relevant chunks found."
            else:
                context = "\n\n---\n\n".join(
                    f"[page {h.payload.get('page', '?')}, score {h.score:.4f}]\n{h.payload.get('text', '')}"
                    for h in hits
                )
                prompt = (
                    f"Using only the context below, answer the question in 2-4 sentences. "
                    f"Cite page numbers.\n\nContext:\n{context}\n\n"
                    f"Question: {query}\n\nAnswer:"
                )
                t_gen = time.time()
                answer = _vllm.generate(
                    prompt,
                    url=vllm_url,
                    model=vllm_model,
                    max_tokens=params.a_max_new_tokens,
                    temperature=params.a_temperature,
                )
                generation_ms = int((time.time() - t_gen) * 1000)
                _log(tag, f"generation done mode=text_in_context via vLLM ({len(answer)} chars, {generation_ms}ms)")
        else:
            # ── Local HF transformers path (fallback when vLLM not configured) ─
            _model_loader.init(cfg)
            lora_ckpt = ver.load().get("checkpoint_path")
            model, tokenizer = _model_loader.load(lora_ckpt)
            # model is already in the correct dtype (fp16 or quantized)

            # ── Phase 3: answer directly from fine-tuned weights, no retrieval ─
            if phase >= 3:
                mode = "parametric"
                _log(tag, "Phase 3 — answering from fine-tuned weights (no retrieval)…")
                t_gen = time.time()
                messages = [{"role": "user", "content": query}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=params.a_max_new_tokens,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                answer = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                generation_ms = int((time.time() - t_gen) * 1000)
                _log(tag, f"generation done mode=parametric ({len(answer)} chars, {generation_ms}ms)")
                elapsed = int((time.time() - t0) * 1000)
                _log(tag, f"DONE {elapsed}ms")
                return {"answer": answer, "latency_ms": elapsed,
                        "retrieval_ms": 0, "generation_ms": generation_ms,
                        "chunks": [], "mode": mode, "gate": {}}

            # ── Phase 1/2: retrieval pipeline ────────────────────────────────
            from fastembed import TextEmbedding
            from pipeline.bedrock_rag import _run_search, Config
            cfg_a = dict(cfg, top_k=params.a_top_k)

            _log(tag, "embedding query…")
            embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
            store_b = _gs(cfg_a)
            rag_cfg = Config(**{k: cfg_a[k] for k in Config.__dataclass_fields__ if k in cfg_a})

            _log(tag, "searching Qdrant…")
            t_ret = time.time()
            hits = _run_search(query, embedder, store_b, rag_cfg)
            retrieval_ms = int((time.time() - t_ret) * 1000)
            top_score_a = f"{hits[0].score:.4f}" if hits else "n/a"
            _log(tag, f"search done — {len(hits)} hits, top_score={top_score_a}, retrieval={retrieval_ms}ms")

            if not hits:
                answer = "No relevant chunks found."
            else:
                chunks = [{"chunk_id": h.id,
                           "text": h.payload["text"],
                           "page": h.payload.get("page", 0), "score": round(h.score, 4),
                           "kv_cache": h.payload.get("kv_cache"),
                           "kv_version": h.payload.get("kv_version")} for h in hits]
                current_ver = ver.get_lora_version()

                stale = _kv_inference.get_stale_chunk_ids(chunks, current_ver)
                if stale:
                    _kv_background.enqueue_kv_recompute(stale)

                mode = _kv_inference.decide_inference_mode(chunks, current_ver)
                _log(tag, f"mode={mode}, recording access + generating…")
                t_gen = time.time()
                for rank, chunk in enumerate(chunks, start=1):
                    _kv_background.record_access(chunk["chunk_id"], rank)

                if mode == "kv_injection":
                    answer = _kv_inference.generate_with_kv(query, chunks, model, tokenizer, cfg)
                else:
                    answer = _kv_inference.generate_text_in_context(
                        query, chunks, model, tokenizer,
                        max_new_tokens=params.a_max_new_tokens,
                        temperature=params.a_temperature,
                        top_p=params.a_top_p,
                        repetition_penalty=params.a_repetition_penalty,
                    )
                generation_ms = int((time.time() - t_gen) * 1000)
            _log(tag, f"generation done mode={mode} ({len(answer)} chars, {generation_ms}ms)")
    except Exception as e:
        import traceback
        _log(tag, f"ERROR: {e}\n{traceback.format_exc()}")
        answer = f"Error: {e}"
        retrieval_ms = 0
        generation_ms = 0
    elapsed = int((time.time() - t0) * 1000)
    _log(tag, f"DONE {elapsed}ms")
    display_chunks = [dict(c, text=c["text"][:500]) for c in chunks]
    return {"answer": answer, "latency_ms": elapsed, "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms, "chunks": display_chunks,
            "mode": mode, "gate": gate_info, "prs_score": 0.0}


def _answer_gemini(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "B:Gemini"
    _log(tag, f"START query={query!r} top_k={params.b_top_k} max_tokens={params.b_max_output_tokens} temp={params.b_temperature}")
    chunks = []
    retrieval_ms = 0
    generation_ms = 0
    thinking = ""
    try:
        from fastembed import TextEmbedding

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        q_vec = list(embedder.embed([query]))[0].tolist()

        _log(tag, "searching vector store…")
        store = _get_store()
        t_ret = time.time()
        hits = store.query(cfg["collection"], q_vec, top_k=params.b_top_k)
        retrieval_ms = int((time.time() - t_ret) * 1000)
        top_score_b = f"{hits[0].score:.4f}" if hits else "n/a"
        _log(tag, f"search done — {len(hits)} hits, top_score={top_score_b}, retrieval={retrieval_ms}ms")
        chunks = [
            {
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:2000],
            }
            for h in hits
        ]

        # 2. build prompt
        context = "\n\n".join(
            f"[page {c['page']}, score {c['score']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"Cite sources as [page N].\n\nContext:\n{context}\n\nQuestion: {query}"
        )

        # 3. call Gemini REST API
        # Use runtime config if set; fall back to config.json values
        api_key = _model_b_config.get("api_key") or cfg.get("gemini_api_key", "")
        model = _model_b_config.get("model") or cfg.get("gemini_model", "gemini-3.1-pro-preview")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        _log(tag, f"calling Gemini API ({model}, timeout=90s)…")
        t_gen = time.time()
        resp = httpx.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": params.b_temperature,
                    "maxOutputTokens": params.b_max_output_tokens,
                },
            },
            timeout=90,
        )
        _log(tag, f"Gemini HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            # Gemini returns no candidates on safety blocks or quota issues
            block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
            _log(tag, f"Gemini returned 0 candidates — blockReason={block_reason}, raw={str(data)[:300]}")
            answer = f"Gemini returned no answer (blockReason: {block_reason})"
        else:
            finish = candidates[0].get("finishReason", "unknown")
            # Separate thinking parts (thought=True) from response text
            all_parts = candidates[0].get("content", {}).get("parts", []) or []
            text_parts = [p.get("text", "") for p in all_parts if not p.get("thought")]
            thinking_parts = [p.get("text", "") for p in all_parts if p.get("thought")]
            thinking = "".join(thinking_parts).strip()
            answer = "".join(text_parts).strip()
            if not answer:
                _log(tag, f"Gemini candidate has no text parts — finishReason={finish}, raw={str(candidates[0])[:300]}")
                answer = f"Gemini returned empty content (finishReason: {finish})"
            else:
                generation_ms = int((time.time() - t_gen) * 1000)
                if finish == "MAX_TOKENS":
                    _log(tag, f"Gemini hit MAX_TOKENS — partial response ({len(answer)} chars, {generation_ms}ms)")
                    answer += "\n[response truncated — increase Max output tokens]"
                else:
                    _log(tag, f"generation done ({len(answer)} chars, {generation_ms}ms)")
    except Exception as e:
        import traceback
        _log(tag, f"ERROR: {e}\n{traceback.format_exc()}")
        answer = f"Error: {e}"
        thinking = ""
        retrieval_ms = 0
        generation_ms = 0
    elapsed = int((time.time() - t0) * 1000)
    _log(tag, f"DONE {elapsed}ms")
    return {"answer": answer, "latency_ms": elapsed, "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms, "chunks": chunks, "thinking": thinking}


def _answer_openai(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "B:OpenAI"
    _log(tag, f"START query={query!r} top_k={params.b_top_k} max_tokens={params.b_max_output_tokens} temp={params.b_temperature}")
    chunks = []
    retrieval_ms = 0
    generation_ms = 0
    answer = ""
    try:
        from fastembed import TextEmbedding

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        q_vec = list(embedder.embed([query]))[0].tolist()

        _log(tag, "searching vector store…")
        store = _get_store()
        t_ret = time.time()
        hits = store.query(cfg["collection"], q_vec, top_k=params.b_top_k)
        retrieval_ms = int((time.time() - t_ret) * 1000)
        top_score_b = f"{hits[0].score:.4f}" if hits else "n/a"
        _log(tag, f"search done — {len(hits)} hits, top_score={top_score_b}, retrieval={retrieval_ms}ms")
        chunks = [
            {
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:2000],
            }
            for h in hits
        ]

        # build prompt
        context = "\n\n".join(
            f"[page {c['page']}, score {c['score']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"Cite sources as [page N].\n\nContext:\n{context}\n\nQuestion: {query}"
        )

        # call OpenAI REST API
        api_key = _model_b_config.get("api_key") or ""
        model = _model_b_config.get("model", "gpt-4o")
        url = "https://api.openai.com/v1/chat/completions"
        _log(tag, f"calling OpenAI API ({model}, timeout=90s)…")
        t_gen = time.time()
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": params.b_max_output_tokens,
                "temperature": params.b_temperature,
            },
            timeout=90,
        )
        _log(tag, f"OpenAI HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            answer = "OpenAI returned no choices"
        else:
            finish = choices[0].get("finish_reason", "unknown")
            answer = choices[0]["message"]["content"].strip()
            if not answer:
                answer = f"OpenAI returned empty content (finish_reason: {finish})"
            else:
                generation_ms = int((time.time() - t_gen) * 1000)
                if finish == "length":
                    answer += "\n[response truncated — increase Max output tokens]"
    except Exception as e:
        import traceback
        _log(tag, f"ERROR: {e}\n{traceback.format_exc()}")
        answer = f"Error: {e}"
        retrieval_ms = 0
        generation_ms = 0
    elapsed = int((time.time() - t0) * 1000)
    _log(tag, f"DONE {elapsed}ms")
    return {"answer": answer, "latency_ms": elapsed, "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms, "chunks": chunks, "thinking": ""}


@app.post("/api/query")
async def run_query(req: QueryRequest):
    _log("query", f"received query={req.query!r}")
    cfg = _load_cfg()
    loop = asyncio.get_event_loop()
    fut_a = loop.run_in_executor(_query_executor, _answer_kvforge, req.query, cfg, req)
    provider = _model_b_config.get("provider", "gemini")
    fn_b = _answer_openai if provider == "openai" else _answer_gemini
    fut_b = loop.run_in_executor(_query_executor, fn_b, req.query, cfg, req)
    result_a, result_b = await asyncio.gather(fut_a, fut_b)
    return {
        "answer_a": result_a["answer"],
        "latency_a_ms": result_a["latency_ms"],
        "retrieval_a_ms": result_a.get("retrieval_ms", 0),
        "generation_a_ms": result_a.get("generation_ms", 0),
        "chunks_a": result_a.get("chunks", []),
        "mode_a": result_a.get("mode", "text_in_context"),
        "prs_score_a": result_a.get("prs_score", 0.0),
        "gate_a": result_a.get("gate", {}),
        "answer_b": result_b["answer"],
        "latency_b_ms": result_b["latency_ms"],
        "retrieval_b_ms": result_b.get("retrieval_ms", 0),
        "generation_b_ms": result_b.get("generation_ms", 0),
        "chunks_b": result_b.get("chunks", []),
        "thinking_b": result_b.get("thinking", ""),
    }


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KVForge Dashboard</title>
<style>
  body { font-family: monospace; background:#111; color:#eee; padding:20px; }
  h1 { color:#7af; }
  .card { border:1px solid #333; padding:12px; margin:10px 0; }
  table { border-collapse:collapse; width:100%; }
  td,th { border:1px solid #333; padding:6px 10px; text-align:left; }
  .hot{color:#f90} .warm{color:#ff0} .cold{color:#0af} .frozen{color:#aaa}
  .param-grid {
    display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
    gap:8px; margin-top:8px;
  }
  .param-group { display:flex; flex-direction:column; gap:3px; }
  .param-group label { color:#888; font-size:0.8em; }
  .param-group input {
    background:#1a1a1a; color:#eee; border:1px solid #444;
    padding:4px 6px; font-family:monospace; font-size:0.9em; width:100%; box-sizing:border-box;
  }
  .param-group input:focus { border-color:#27a; outline:none; }
  .section-label { color:#7af; font-size:0.85em; margin:10px 0 4px; font-weight:bold; }
  .model-info { font-size:0.85em; color:#888; margin:4px 0; }
  .model-info a { color:#7af; text-decoration:none; }
  .model-info a:hover { text-decoration:underline; }
  /* PRS modal */
  .help-btn {
    display:inline-block; width:16px; height:16px; line-height:16px;
    background:#27a; color:#fff; border-radius:50%; font-size:0.75em;
    text-align:center; cursor:pointer; margin-left:6px; vertical-align:middle;
    user-select:none;
  }
  .help-btn:hover { background:#39b; }
  .modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,0.7); z-index:1000;
    align-items:center; justify-content:center;
  }
  .modal-overlay.open { display:flex; }
  .modal-box {
    background:#1a1a1a; border:1px solid #444; border-radius:6px;
    padding:24px; max-width:560px; width:90%; position:relative;
    color:#eee; font-family:monospace; line-height:1.6;
  }
  .modal-box h2 { color:#7af; margin:0 0 12px; font-size:1.1em; }
  .modal-box table { width:100%; margin:10px 0; font-size:0.88em; }
  .modal-box td, .modal-box th { border:1px solid #333; padding:5px 8px; }
  .modal-box th { color:#7af; }
  .modal-close {
    position:absolute; top:10px; right:14px; cursor:pointer;
    color:#888; font-size:1.2em;
  }
  .modal-close:hover { color:#eee; }
  .prs-bar-wrap { background:#222; border-radius:4px; height:12px; margin:3px 0 8px; }
  .prs-bar { height:12px; border-radius:4px; background:#27a; transition:width 0.4s; }
</style>
</head>
<body>

<!-- PRS explanation modal -->
<div class="modal-overlay" id="prs-modal" onclick="if(event.target===this)closePrsModal()">
  <div class="modal-box">
    <span class="modal-close" onclick="closePrsModal()">&#x2715;</span>
    <h2>Parametric Readiness Score (PRS)</h2>
    <p>PRS measures how well the fine-tuned LLM has <b>memorised</b> the document knowledge
    and can answer questions <i>without</i> retrieving chunks. Higher = more reliable
    parametric memory.</p>

    <p><b>Formula:</b></p>
    <pre style="background:#111;padding:8px;border-radius:4px;font-size:0.85em">
PRS = 0.5 × Accuracy
    + 0.3 × Calibration
    + 0.2 × Self-Consistency</pre>

    <table>
      <tr><th>Component</th><th>What it measures</th><th>Weight</th></tr>
      <tr><td><b>Accuracy</b></td>
          <td>Fraction of FAQ answers that match the LLM's direct answer
              (no retrieval, cosine similarity ≥ threshold)</td>
          <td>50%</td></tr>
      <tr><td><b>Calibration</b></td>
          <td>Whether the model's confidence token probabilities
              match its actual accuracy (low entropy on correct answers)</td>
          <td>30%</td></tr>
      <tr><td><b>Self-Consistency</b></td>
          <td>Mean pairwise cosine similarity of 3 answers sampled
              at temperature 0.7 — measures how stable the knowledge is</td>
          <td>20%</td></tr>
    </table>

    <p><b>Phase thresholds:</b></p>
    <table>
      <tr><th>Phase</th><th>Condition</th><th>Behaviour</th></tr>
      <tr><td>1</td><td>—</td><td>Standard RAG — text-in-context only</td></tr>
      <tr><td>2</td><td>PRS ≥ 0.75 (one round)</td><td>KV injection enabled</td></tr>
      <tr><td>3</td><td>PRS ≥ 0.80 (two consecutive rounds)</td>
          <td>Confidence gate — high-confidence queries answered from weights directly</td></tr>
    </table>

    <div id="prs-live"></div>
  </div>
</div>

<h1>KVForge Dashboard</h1>
<div id="root">Loading…</div>

<div class="card" id="query-section">
  <b>Query A/B Comparison</b>
  <div style="margin:10px 0">
    <input id="qinput" type="text" placeholder="Ask a question..."
      style="width:70%;padding:8px;background:#222;color:#eee;border:1px solid #555;font-family:monospace"/>
    <button onclick="runQuery()"
      style="padding:8px 16px;background:#27a;color:#fff;border:none;cursor:pointer;font-family:monospace;margin-left:8px">
      Ask
    </button>
  </div>

  <details id="settings-panel">
    <summary style="cursor:pointer;color:#888;margin-bottom:6px">&#9881; Model &amp; Generation Settings</summary>

    <div id="model-info-a" class="model-info">Loading model info…</div>
    <div id="model-info-b" class="model-info"></div>

    <div class="section-label">Shared Retrieval Settings</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Context chunks (top_k) — both models</label>
        <input id="top_k" type="number" min="1" max="10" value="5"/>
      </div>
    </div>

    <div class="section-label" id="label-a">Answer A — KVForge</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Max new tokens</label>
        <input id="a_max_new_tokens" type="number" min="16" max="2048" value="64"/>
      </div>
      <div class="param-group">
        <label>Temperature</label>
        <input id="a_temperature" type="number" min="0.1" max="2.0" step="0.05" value="0.7"/>
      </div>
      <div class="param-group">
        <label>Top-p (nucleus sampling)</label>
        <input id="a_top_p" type="number" min="0.1" max="1.0" step="0.05" value="0.9"/>
      </div>
      <div class="param-group">
        <label>Repetition penalty</label>
        <input id="a_repetition_penalty" type="number" min="1.0" max="2.0" step="0.05" value="1.2"/>
      </div>
    </div>

    <div class="section-label" id="label-b">Answer B — Gemini RAG</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Provider</label>
        <select id="b_provider" style="background:#1a1a1a;color:#eee;border:1px solid #444;padding:4px 6px;font-family:monospace;font-size:0.9em;width:100%;box-sizing:border-box;">
          <option value="gemini">Gemini</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>
      <div class="param-group">
        <label>Model</label>
        <select id="b_model" style="background:#1a1a1a;color:#eee;border:1px solid #444;padding:4px 6px;font-family:monospace;font-size:0.9em;width:100%;box-sizing:border-box;"></select>
      </div>
      <div class="param-group">
        <label>API Key</label>
        <input id="b_api_key" type="password" placeholder="Paste API key…"/>
      </div>
      <div class="param-group">
        <label>Max output tokens</label>
        <input id="b_max_output_tokens" type="number" min="128" max="65536" value="4096"/>
      </div>
      <div class="param-group">
        <label>Temperature</label>
        <input id="b_temperature" type="number" min="0.0" max="2.0" step="0.05" value="1.0"/>
      </div>
    </div>
  </details>

  <div id="ab-result" style="display:none">
    <div style="display:flex;gap:16px;margin-top:12px">
      <div style="flex:1;border:1px solid #444;padding:12px">
        <b style="color:#7af" id="header-a">Answer A — KVForge</b>
        <div id="latency-a" style="color:#888;font-size:0.85em;margin:4px 0;font-family:monospace"></div>
        <div id="mode-a" style="font-size:0.85em;margin:4px 0"></div>
        <pre id="answer-a" style="white-space:pre-wrap;margin:8px 0;color:#eee;font-size:0.9em"></pre>
        <details style="margin-top:8px">
          <summary style="cursor:pointer;color:#888">Retrieved context (reasoning)</summary>
          <div id="chunks-a" style="font-size:0.8em;margin-top:6px"></div>
        </details>
      </div>
      <div style="flex:1;border:1px solid #444;padding:12px">
        <b style="color:#fa7" id="header-b">Answer B — Gemini RAG</b>
        <div id="latency-b" style="color:#888;font-size:0.85em;margin:4px 0;font-family:monospace"></div>
        <pre id="answer-b" style="white-space:pre-wrap;margin:8px 0;color:#eee;font-size:0.9em"></pre>
        <details id="thinking-b-details" style="margin-top:8px;display:none">
          <summary style="cursor:pointer;color:#adf">Reasoning (thinking tokens)</summary>
          <pre id="thinking-b" style="white-space:pre-wrap;font-size:0.8em;margin-top:6px;color:#9c9;max-height:400px;overflow-y:auto"></pre>
        </details>
        <details style="margin-top:8px">
          <summary style="cursor:pointer;color:#888">Retrieved chunks</summary>
          <div id="chunks-b" style="font-size:0.8em;margin-top:6px"></div>
        </details>
      </div>
    </div>
  </div>
  <div id="query-loading" style="display:none;color:#888;margin-top:8px">Running queries (A/B in parallel)…</div>
</div>

<script>
const MODELS_B = {
  gemini: ['gemini-2.0-flash','gemini-2.5-pro','gemini-1.5-pro','gemini-1.5-flash'],
  openai: ['gpt-4.1','gpt-4.1-mini','gpt-4o','gpt-4o-mini'],
};

function populateModelDropdown(provider) {
  const sel = document.getElementById('b_model');
  sel.innerHTML = MODELS_B[provider].map(m => `<option value="${m}">${m}</option>`).join('');
  const saved = localStorage.getItem(`modelb_${provider}_model`);
  if (saved && MODELS_B[provider].includes(saved)) sel.value = saved;
}

function saveAndSyncModelB() {
  const provider = document.getElementById('b_provider').value;
  const model = document.getElementById('b_model').value;
  const apiKey = document.getElementById('b_api_key').value;
  localStorage.setItem('modelb_provider', provider);
  localStorage.setItem(`modelb_${provider}_model`, model);
  localStorage.setItem(`modelb_${provider}_key`, apiKey);
  fetch('/api/set_model_b_config', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({provider, model, api_key: apiKey}),
  }).catch(e => console.error('[ModelB config sync failed]', e));
  const label = provider === 'openai' ? 'OpenAI' : 'Gemini';
  document.getElementById('label-b').textContent = `Answer B — ${label} RAG`;
  document.getElementById('header-b').textContent = `Answer B — ${label} (${model})`;
  document.getElementById('model-info-b').innerHTML =
    `${label} model: <b style="color:#fa7">${model}</b>`;
}

document.getElementById('b_provider').addEventListener('change', function() {
  const provider = this.value;
  populateModelDropdown(provider);
  document.getElementById('b_api_key').value = localStorage.getItem(`modelb_${provider}_key`) || '';
  saveAndSyncModelB();
});
document.getElementById('b_model').addEventListener('change', saveAndSyncModelB);
document.getElementById('b_api_key').addEventListener('change', saveAndSyncModelB);

async function loadConfig() {
  // Restore Model B config from localStorage (must run before fetch to pre-populate)
  const savedProvider = localStorage.getItem('modelb_provider') || 'gemini';
  document.getElementById('b_provider').value = savedProvider;
  populateModelDropdown(savedProvider);
  document.getElementById('b_api_key').value = localStorage.getItem(`modelb_${savedProvider}_key`) || '';
  saveAndSyncModelB();
  try {
    const cfg = await fetch('/api/config').then(r => r.json());
    document.getElementById('model-info-a').innerHTML =
      `LLM: <a href="${cfg.llm_model_url}" target="_blank" rel="noopener">${cfg.llm_model}</a>` +
      ` &nbsp;|&nbsp; Embedder: <a href="${cfg.embed_model_url}" target="_blank" rel="noopener">${cfg.embed_model}</a>` +
      ` &nbsp;|&nbsp; Collection: ${cfg.collection}`;
    // set default top_k from server config (shared for both models)
    document.getElementById('top_k').value = cfg.top_k;
    // update section labels and answer headers with actual model name
    const shortName = cfg.llm_model.split('/').pop();
    document.getElementById('label-a').textContent = `Answer A — KVForge (${shortName})`;
    document.getElementById('header-a').textContent = `Answer A — KVForge (${shortName})`;
  } catch(e) {
    document.getElementById('model-info-a').textContent = 'Could not load config';
  }
}

let _prsHistory = [];

async function load(){
  const [stats, ver] = await Promise.all(
    [fetch('/api/stats').then(r=>r.json()), fetch('/api/version').then(r=>r.json())]
  );
  _prsHistory = ver.prs_history || [];
  const tc = stats.tier_counts;
  document.getElementById('root').innerHTML = `
    <div class="card"><b>Phase:</b> ${ver.phase} &nbsp;|&nbsp;
      <b>LoRA version:</b> ${ver.current_lora_version} &nbsp;|&nbsp;
      <b>Total chunks:</b> ${stats.total_chunks}
    </div>
    <div class="card"><b>Tier distribution:</b>
      <span class="hot">Hot: ${tc.hot||0}</span> &nbsp;
      <span class="warm">Warm: ${tc.warm||0}</span> &nbsp;
      <span class="cold">Cold: ${tc.cold||0}</span> &nbsp;
      <span class="frozen">Frozen: ${tc.frozen||0}</span>
    </div>
    <div class="card"><b>Top 10 chunks by access count:</b>
      <table><tr><th>ID</th><th>Page</th><th>Tier</th><th>Access</th><th>Parametric</th><th>Preview</th></tr>
      ${stats.top_chunks.map(c=>`<tr><td>${c.chunk_id}</td><td>${c.page}</td>
        <td class="${c.tier}">${c.tier}</td><td>${c.access_count}</td>
        <td>${c.parametric_hit_count}</td><td>${c.text_preview}</td></tr>`).join('')}
      </table>
    </div>
    <div class="card">
      <b>PRS history</b>
      <span class="help-btn" onclick="openPrsModal()">?</span>
      <div style="margin-top:8px">
      ${(ver.prs_history||[]).length === 0 ? '<span style="color:#666">No data yet — run prs_evaluator.py to generate scores</span>' :
        (ver.prs_history||[]).map(r => {
          const pct = Math.round((r.prs||0)*100);
          const col = pct >= 80 ? '#4d4' : pct >= 75 ? '#fa7' : '#f77';
          return `<div style="margin:4px 0">
            <span style="color:#888;font-size:0.85em">Round ${r.round}</span>
            <span style="color:${col};font-weight:bold;margin:0 8px">${(r.prs||0).toFixed(4)}</span>
            <div class="prs-bar-wrap" style="display:inline-block;width:160px;vertical-align:middle">
              <div class="prs-bar" style="width:${pct}%;background:${col}"></div>
            </div>
            <span style="color:#666;font-size:0.8em;margin-left:6px">${pct}%</span>
          </div>`;
        }).join('')
      }
      </div>
    </div>`;
}

function getNum(id) { return parseFloat(document.getElementById(id).value) || 0; }
function getInt(id) { return parseInt(document.getElementById(id).value) || 0; }

async function runQuery() {
  const q = document.getElementById('qinput').value.trim();
  if (!q) return;
  document.getElementById('query-loading').style.display = 'block';
  document.getElementById('ab-result').style.display = 'none';
  const topK = getInt('top_k');
  const payload = {
    query: q,
    a_top_k:               topK,
    a_max_new_tokens:      getInt('a_max_new_tokens'),
    a_temperature:         getNum('a_temperature'),
    a_top_p:               getNum('a_top_p'),
    a_repetition_penalty:  getNum('a_repetition_penalty'),
    b_top_k:               topK,
    b_max_output_tokens:   getInt('b_max_output_tokens'),
    b_temperature:         getNum('b_temperature'),
  };
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  }).then(r => r.json());
  document.getElementById('query-loading').style.display = 'none';
  document.getElementById('answer-a').textContent = res.answer_a;
  document.getElementById('latency-a').innerHTML =
    `Total: <b>${res.latency_a_ms}ms</b> &nbsp;|&nbsp; ` +
    `<span style="color:#4af">Retrieval: ${res.retrieval_a_ms}ms</span> &nbsp;|&nbsp; ` +
    `<span style="color:#fa4">Generation: ${res.generation_a_ms}ms</span>`;
  // Mode badge + gate info
  const modeA = res.mode_a || 'text_in_context';
  const modeColors = {parametric: '#4f4', kv_injection: '#4af', text_in_context: '#fa4'};
  const modeLabels = {parametric: 'Parametric (no retrieval)', kv_injection: 'KV Injection', text_in_context: 'Text-in-Context RAG'};
  let modeHtml = `<span style="color:${modeColors[modeA]||'#aaa'};font-weight:bold">\u25cf ${modeLabels[modeA]||modeA}</span>`;
  const gate = res.gate_a || {};
  if (gate.decision) {
    modeHtml += ` &nbsp;<span style="color:#888;font-size:0.85em">[gate: entropy=${gate.entropy} hedging=${gate.hedging} sim=${gate.similarity} \u2192 <b>${gate.decision}</b>]</span>`;
  }
  document.getElementById('mode-a').innerHTML = modeHtml;
  document.getElementById('chunks-a').innerHTML = (res.chunks_a||[]).map(c =>
    `<div style="margin:4px 0;border-top:1px solid #333;padding-top:4px">
      <span style="color:#888">page ${c.page} \u00b7 score ${c.score}</span><br>${c.text}\u2026</div>`
  ).join('');
  document.getElementById('answer-b').textContent = res.answer_b;
  document.getElementById('latency-b').innerHTML =
    `Total: <b>${res.latency_b_ms}ms</b> &nbsp;|&nbsp; ` +
    `<span style="color:#4af">Retrieval: ${res.retrieval_b_ms}ms</span> &nbsp;|&nbsp; ` +
    `<span style="color:#fa4">Generation: ${res.generation_b_ms}ms</span>`;
  const thinkingB = res.thinking_b || '';
  const thinkingDetails = document.getElementById('thinking-b-details');
  if (thinkingB) {
    document.getElementById('thinking-b').textContent = thinkingB;
    thinkingDetails.style.display = 'block';
  } else {
    thinkingDetails.style.display = 'none';
  }
  document.getElementById('chunks-b').innerHTML = (res.chunks_b||[]).map(c =>
    `<div style="margin:4px 0;border-top:1px solid #333;padding-top:4px">
      <span style="color:#888">page ${c.page} \u00b7 score ${c.score}</span><br>${c.text}\u2026</div>`
  ).join('');
  document.getElementById('ab-result').style.display = 'block';
}

function openPrsModal() {
  const history = _prsHistory;
  // Populate live scores inside the modal
  const live = document.getElementById('prs-live');
  if (history && history.length > 0) {
    const latest = history[history.length - 1];
    const pct = Math.round((latest.prs||0)*100);
    const col = pct >= 80 ? '#4d4' : pct >= 75 ? '#fa7' : '#f77';
    const next = pct >= 80 ? 'Phase 3 threshold met \u2714' :
                 pct >= 75 ? 'Phase 2 threshold met \u2714 — need \u226580% \u00d72 for Phase 3' :
                 'Below Phase 2 threshold (need \u226575%)';
    live.innerHTML = `<p style="margin-top:12px"><b>Latest score:</b>
      <span style="color:${col};font-size:1.1em;font-weight:bold"> ${(latest.prs||0).toFixed(4)}</span>
      &nbsp;&mdash;&nbsp;<span style="color:${col}">${next}</span></p>`;
  } else {
    live.innerHTML = '<p style="color:#666;margin-top:12px">No PRS data yet.</p>';
  }
  document.getElementById('prs-modal').classList.add('open');
}

function closePrsModal() {
  document.getElementById('prs-modal').classList.remove('open');
}

// Close modal on Escape key
document.addEventListener('keydown', e => { if (e.key === 'Escape') closePrsModal(); });

loadConfig();
load();
setInterval(load, 30000);

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('qinput').addEventListener('keydown', e => {
    if (e.key === 'Enter') runQuery();
  });
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the self-contained monitoring dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


def _main():
    global _config_path
    parser = argparse.ArgumentParser(description="KVForge Dashboard")
    parser.add_argument(
        "--config",
        default="my_config.json",
        help="Path to JSON config file (default: my_config.json)",
    )
    parser.add_argument("--port", type=int, default=None,
                        help="Dashboard port (overrides config dashboard_port)")
    args = parser.parse_args()
    _config_path = args.config
    cfg = _load_cfg()
    port = args.port if args.port is not None else cfg.get("dashboard_port", 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    _main()
