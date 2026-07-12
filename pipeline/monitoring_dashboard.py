"""FastAPI monitoring dashboard for KVForge.

Provides a self-contained REST API + HTML UI for observing one use-case
collection.  One dashboard process per use-case (ports 8081-8084 on the
reference EC2 deployment).

REST endpoints
--------------
* ``GET  /api/health``          — liveness check.
* ``GET  /api/version``         — current LoRA version and phase.
* ``GET  /api/stats``           — tier counts, top-10 most-accessed chunks
                                  (with full text), access report.
* ``GET  /api/config``          — display-safe config for the settings panel.
* ``GET  /api/access-report``   — raw access_report.json if present.
* ``GET  /api/coverage``        — FAQ coverage heatmap data: for each FAQ in
                                  faqs.json, returns the top-K closest chunks
                                  (cosine similarity) with tier, text, page,
                                  access_count, kv_version, and score.
* ``POST /api/query``           — A/B comparison query.  Model A = KVForge
                                  (local vLLM or HF transformers).  Model B =
                                  Gemini / Claude / OpenAI (configurable at
                                  runtime).  Both sides record chunk access
                                  counts so tier data stays accurate.
* ``POST /api/set_model_b_config`` — hot-swap Model B provider/model/api_key.

Dashboard UI features
---------------------
* Phase, LoRA version, total chunks at a glance.
* Tier distribution bar (hot / warm / cold / frozen) with dynamic counts.
* Top-10 most-accessed chunks table — click any preview to open a full-text
  popup (chunk text, page, access count, kv_version, tier).
* PRS history with progress bars and a help modal explaining the formula.
* FAQ Coverage Heatmap — FAQs as rows, top-K matching chunks as columns,
  cells coloured by cosine similarity score (≥0.85 red, ≥0.75 orange,
  ≥0.65 yellow, <0.65 blue).  A threshold slider (0.60–1.00) hides rows
  with no match above the cutoff.  Click any cell for the full chunk popup.
* A/B query panel with configurable generation params for both models.

GPU model loading
-----------------
Heavy modules (torch, transformers) are imported and the LoRA checkpoint is
pre-warmed in the FastAPI lifespan startup hook so worker threads never race
on the lazy-import lock.

Start the server::

    python -m pipeline.monitoring_dashboard --config examples/usecase1_customer_support/config.json
    python -m pipeline.monitoring_dashboard --config examples/usecase4_bedrock_userguide/config.json --port 8084
    uvicorn pipeline.monitoring_dashboard:app --port 8081 --reload
"""

import argparse
import asyncio
import json
import subprocess as _sp
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

# Fastembed singleton — constructed once at startup so the ONNX session is
# reused across queries instead of being rebuilt every call.
_embedder = None

_model_b_config: dict = {
    "provider": "gemini",
    "model": "gemini-2.5-flash",  # must match first item in JS MODELS_B["gemini"]
    "api_key": "",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Import GPU modules and pre-warm the model+LoRA in the main thread at startup."""
    global _model_loader, _kv_background, _kv_inference, _embedder
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
        _kb.start(cfg)
        lora_ckpt = ver.load().get("checkpoint_path")
        print(f"[dashboard] pre-warming model (lora_ckpt={lora_ckpt})…", flush=True)
        _ml.load(lora_ckpt)
        print("[dashboard] model ready", flush=True)
    except Exception as e:
        print(f"[dashboard] inference modules unavailable: {e}", flush=True)
    # Pre-warm the fastembed embedder singleton so the first query doesn't pay
    # the ONNX session construction cost.
    try:
        from fastembed import TextEmbedding
        cfg = _load_cfg()
        _embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        print("[dashboard] embedder ready", flush=True)
    except Exception as e:
        print(f"[dashboard] embedder unavailable: {e}", flush=True)
    yield


def _get_embedder(cfg: dict):
    """Return the module-level fastembed singleton, creating it on first call."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
    return _embedder


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
    v = dict(ver.load())
    v.pop("known_good_queries", None)
    return v


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
              "kv_version": r.payload.get("kv_version"),
              "text": r.payload.get("text", ""),
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

    v_slim = dict(v)
    v_slim.pop("known_good_queries", None)
    return {
        "version": v_slim,
        "tier_counts": tier_counts,
        "top_chunks": top_chunks,
        "access_report": access_report,
        "total_chunks": sum(tc_v for k, tc_v in tier_counts.items() if k != "error"),
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


def _check_qdrant(cfg: dict) -> dict:
    url = cfg.get("qdrant_url", "http://localhost:6333")
    try:
        t0 = time.time()
        r = httpx.get(f"{url}/healthz", timeout=2)
        ms = int((time.time() - t0) * 1000)
        return {"ok": r.status_code == 200, "latency_ms": ms}
    except Exception as exc:
        return {"ok": False, "latency_ms": -1, "error": str(exc)}


def _check_gpu() -> dict:
    try:
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=_sp.DEVNULL, timeout=3, text=True,
        ).strip().splitlines()[0]
        name, util, used, total = [x.strip() for x in out.split(",")]
        return {"ok": True, "name": name, "util_pct": int(util),
                "mem_used_mib": int(used), "mem_total_mib": int(total)}
    except Exception:
        return {"ok": False}


def _check_llm() -> dict:
    loaded = _model_loader is not None
    return {"ok": True, "loaded": loaded}


_ERROR_HINTS: list[tuple[str, str]] = [
    ("No module named pypdf",          "Install missing package: pip install pypdf"),
    ("No module named pdfplumber",     "Install missing package: pip install pdfplumber pymupdf"),
    ("No module named fastembed",      "Install missing package: pip install fastembed"),
    ("No module named qdrant_client",  "Install missing package: pip install qdrant-client"),
    ("CUDA out of memory",             "GPU out of memory — reduce batch size or chunk count, or restart to free VRAM"),
    ("Connection refused",             "Service unreachable — check Qdrant is running (port 6333) and config URL is correct"),
    ("401",                            "Authentication error — check your HF_TOKEN or API key is set correctly"),
    ("BaseModelOutputWithPooling",     "CLIP embedder type mismatch — embeddings/clip_embedder.py needs the pooler_output fix"),
    ("AutoModelForCausalLM",           "Wrong model class for LLaVA — use LlavaForConditionalGeneration in core/multimodal_loader.py"),
    ("collection already exists",      "Collection 409 conflict — delete the existing empty collection: curl -X DELETE http://localhost:6333/collections/<name>"),
]


@app.get("/api/connectivity")
def get_connectivity():
    cfg = _load_cfg()
    return {
        "qdrant": _check_qdrant(cfg),
        "gpu": _check_gpu(),
        "llm": _check_llm(),
    }


@app.get("/api/error-hint")
def get_error_hint(msg: str = ""):
    for pattern, hint in _ERROR_HINTS:
        if pattern.lower() in msg.lower():
            return {"hint": hint, "severity": "error"}
    return {"hint": None, "severity": None}


@app.get("/api/access-report")
def get_access_report():
    rp = Path("access_report.json")
    if not rp.exists():
        return JSONResponse({"error": "No report yet"}, status_code=404)
    with open(rp) as f:
        return json.load(f)


@app.get("/api/coverage")
def get_coverage(top_k: int = 5):
    """Return FAQ coverage heatmap data.

    For each FAQ in faqs.json, finds the top_k closest chunks by cosine
    similarity using the configured embedding model and vector store.

    Returns:
        JSON with ``faqs`` list (question, answer) and ``matches`` dict
        keyed by FAQ index containing chunk matches with id, tier, text,
        page, access_count, kv_version, and similarity score.
    """
    cfg = _load_cfg()
    faqs_path = Path(_config_path).parent / "faqs.json"
    if not faqs_path.exists():
        return JSONResponse({"error": "faqs.json not found"}, status_code=404)
    with open(faqs_path) as f:
        faqs = json.load(f)
    if not faqs:
        return JSONResponse({"error": "faqs.json is empty"}, status_code=404)

    try:
        embedder = _get_embedder(cfg)
        store = _get_store()
        collection = cfg["collection"]
        matches = {}
        questions = [f.get("question", "") for f in faqs]
        # Batch-embed all questions for efficiency
        vecs = list(embedder.embed(questions))
        for i, (faq, qvec) in enumerate(zip(faqs, vecs)):
            hits = store.query(collection, qvec.tolist(), top_k)
            matches[i] = [
                {
                    "chunk_id": h.id,
                    "score": round(float(h.score), 4),
                    "tier": h.payload.get("tier", "frozen"),
                    "text": h.payload.get("text", ""),
                    "page": h.payload.get("page", 0),
                    "access_count": h.payload.get("access_count", 0),
                    "kv_version": h.payload.get("kv_version"),
                    "text_preview": h.payload.get("text", "")[:80] + "…",
                }
                for h in hits
            ]
        return {"faqs": faqs, "matches": matches}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
    provider: Literal["gemini", "openai", "claude"]
    model: str
    api_key: str
    base_url: str = ""   # optional override for OpenAI-compatible endpoints

@app.post("/api/set_model_b_config")
def set_model_b_config(req: ModelBConfigRequest):
    global _model_b_config
    _model_b_config = {
        "provider": req.provider, "model": req.model,
        "api_key": req.api_key, "base_url": req.base_url,
    }
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
        # ver.init() and _kv_background.start() are already called at startup
        # in _lifespan; calling them per-query acquires locks for no benefit.
        version_data = ver.load()
        phase = version_data.get("phase", 1)
        vllm_url = cfg.get("vllm_url")

        # ── vLLM path: fast generation via dedicated inference server ──────
        # Probe vLLM before committing — if the server is down we fall through
        # to the local HF transformers path below.
        if vllm_url:
            try:
                httpx.get(f"{vllm_url}/health", timeout=2)
            except Exception as _ve:
                _log(tag, f"vLLM unreachable ({vllm_url}): {_ve} — falling back to local HF path")
                vllm_url = None

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

            # For Phase 3 seed per_query_prs with the latest global PRS so the
            # AB eval report shows a meaningful score instead of 0.0.
            if use_parametric:
                _hist = version_data.get("prs_history", [])
                if _hist:
                    per_query_prs = float(_hist[-1].get("prs", 0.0))

            if not use_parametric and phase >= 2:
                known_good = version_data.get("known_good_queries", [])
                if known_good:
                    _q_emb = list(_get_embedder(cfg).embed([query]))[0]
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
            cfg_a = dict(cfg, top_k=params.a_top_k)
            embedder = _get_embedder(cfg)
            store = _get_store()
            _log(tag, "embedding query…")
            q_vec = list(embedder.embed([query]))[0].tolist()
            t_ret = time.time()
            hits = store.query(cfg["collection"], q_vec, params.a_top_k)
            retrieval_ms = int((time.time() - t_ret) * 1000)
            _log(tag, f"search done — {len(hits)} hits, retrieval={retrieval_ms}ms")
            if hits and _kv_background:
                for rank, h in enumerate(hits, start=1):
                    _kv_background.record_access(h.id, rank)
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
            lora_ckpt = version_data.get("checkpoint_path")
            model, tokenizer = _model_loader.load(lora_ckpt)
            # model is already in the correct dtype (fp16 or quantized)

            # ── Phase 3: corpus-wide parametric answering — answer directly
            # from fine-tuned weights, no retrieval ─
            if phase >= 3:
                mode = "parametric"
                _log(tag, "Phase 3 — corpus-wide parametric answering from fine-tuned weights (no retrieval)…")
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
            from pipeline.bedrock_rag import _run_search, Config
            cfg_a = dict(cfg, top_k=params.a_top_k)

            _log(tag, "embedding query…")
            embedder = _get_embedder(cfg)
            store_b = _get_store()
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
        _log(tag, "embedding query…")
        embedder = _get_embedder(cfg)
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
                "chunk_id": h.id,
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:2000],
            }
            for h in hits
        ]
        if _kv_background and hits:
            for rank, chunk in enumerate(chunks, start=1):
                _kv_background.record_access(chunk["chunk_id"], rank)

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


def _answer_claude(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "B:Claude"
    _log(tag, f"START query={query!r} top_k={params.b_top_k} max_tokens={params.b_max_output_tokens} temp={params.b_temperature}")
    chunks = []
    retrieval_ms = 0
    generation_ms = 0
    answer = ""
    try:
        _log(tag, "embedding query…")
        embedder = _get_embedder(cfg)
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
                "chunk_id": h.id,
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:2000],
            }
            for h in hits
        ]
        if _kv_background and hits:
            for rank, chunk in enumerate(chunks, start=1):
                _kv_background.record_access(chunk["chunk_id"], rank)

        context = "\n\n".join(
            f"[page {c['page']}, score {c['score']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"Cite sources as [page N].\n\nContext:\n{context}\n\nQuestion: {query}"
        )

        api_key = _model_b_config.get("api_key") or ""
        model = _model_b_config.get("model", "claude-sonnet-4-6")
        _log(tag, f"calling Anthropic API ({model}, timeout=90s)…")
        t_gen = time.time()
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": params.b_max_output_tokens,
                "temperature": params.b_temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        _log(tag, f"Anthropic HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            answer = "Claude returned no content"
        else:
            answer = "".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
            stop_reason = data.get("stop_reason", "unknown")
            if not answer:
                answer = f"Claude returned empty content (stop_reason: {stop_reason})"
            else:
                generation_ms = int((time.time() - t_gen) * 1000)
                if stop_reason == "max_tokens":
                    answer += "\n[response truncated — increase Max output tokens]"
                else:
                    _log(tag, f"generation done ({len(answer)} chars, {generation_ms}ms)")
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


def _answer_openai(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "B:OpenAI"
    _log(tag, f"START query={query!r} top_k={params.b_top_k} max_tokens={params.b_max_output_tokens} temp={params.b_temperature}")
    chunks = []
    retrieval_ms = 0
    generation_ms = 0
    answer = ""
    try:
        _log(tag, "embedding query…")
        embedder = _get_embedder(cfg)
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
                "chunk_id": h.id,
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:2000],
            }
            for h in hits
        ]
        if _kv_background and hits:
            for rank, chunk in enumerate(chunks, start=1):
                _kv_background.record_access(chunk["chunk_id"], rank)

        # build prompt
        context = "\n\n".join(
            f"[page {c['page']}, score {c['score']}]\n{c['text']}" for c in chunks
        )
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"Cite sources as [page N].\n\nContext:\n{context}\n\nQuestion: {query}"
        )

        # call OpenAI-compatible REST API (base_url can be a full chat/completions URL
        # or a base like https://api.openai.com/v1 — we append /chat/completions if needed)
        api_key = _model_b_config.get("api_key") or ""
        model = _model_b_config.get("model", "gpt-4o")
        _base = (_model_b_config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        url = _base if _base.endswith("/chat/completions") else f"{_base}/chat/completions"
        _log(tag, f"calling OpenAI-compat API ({model} @ {url}, timeout=90s)…")
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
    fn_b = {"openai": _answer_openai, "claude": _answer_claude}.get(provider, _answer_gemini)
    fut_b = loop.run_in_executor(_query_executor, fn_b, req.query, cfg, req)
    result_a, result_b = await asyncio.gather(fut_a, fut_b)
    response = {
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
    # Flywheel: record query event for analytics
    try:
        from core.analytics import record_query, init_db
        init_db(cfg)
        cluster_id = (result_a.get("chunks", [{}]) or [{}])[0].get("cluster_id")
        record_query(cfg, cluster_id=cluster_id,
                     phase_used=result_a.get("mode", "text_in_context"),
                     latency_ms=result_a["latency_ms"])
    except Exception:
        pass
    return response


@app.get("/api/flywheel")
async def get_flywheel():
    """Return Flywheel summary: PRS history, cost estimate, ETA to Phase 3."""
    cfg = _load_cfg()
    try:
        from core.analytics import get_flywheel_summary, init_db
        init_db(cfg)
        return get_flywheel_summary(cfg)
    except Exception as exc:
        return {"error": str(exc)}


@app.patch("/api/flywheel/cost-rate")
async def update_cost_rate(body: dict):
    """Update cost_per_1k_tokens in the datasource config."""
    new_rate = float(body.get("cost_per_1k_tokens", 5.0))
    try:
        import json as _json
        from pathlib import Path as _Path
        data = _json.loads(_Path(_config_path).read_text())
        data["cost_per_1k_tokens"] = new_rate
        _Path(_config_path).write_text(_json.dumps(data, indent=2))
        return {"cost_per_1k_tokens": new_rate}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KVForge Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
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
  /* FAQ Coverage Heatmap */
  .heatmap-table { border-collapse:collapse; font-size:0.82em; }
  .heatmap-table th { background:#1a1a1a; color:#888; padding:5px 8px; position:sticky; top:0; z-index:2; white-space:nowrap; }
  .heatmap-table td.faq-label { color:#adf; max-width:260px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding:5px 10px 5px 4px; }
  .hm-cell {
    width:52px; height:42px; cursor:pointer; border:1px solid #1a1a1a;
    transition: opacity 0.15s; text-align:center; vertical-align:middle;
    font-size:0.78em; color:#111; font-weight:bold; padding:2px;
  }
  .hm-cell:hover { opacity:0.75; outline:1px solid #fff; }
  .hm-hot    { background:#e74c3c; color:#fff; }
  .hm-warm   { background:#e67e22; color:#fff; }
  .hm-cold   { background:#f1c40f; color:#111; }
  .hm-frozen { background:#3498db; color:#fff; }
  /* Chunk detail modal */
  .chunk-modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,0.75); z-index:2000;
    align-items:center; justify-content:center;
  }
  .chunk-modal-overlay.open { display:flex; }
  .chunk-modal-box {
    background:#1a1a1a; border:1px solid #444; border-radius:6px;
    padding:24px; max-width:680px; width:92%; position:relative;
    color:#eee; font-family:monospace; line-height:1.6; max-height:80vh; overflow-y:auto;
  }
  .chunk-modal-box h2 { color:#7af; margin:0 0 10px; font-size:1em; }
  .chunk-meta { font-size:0.82em; color:#888; margin-bottom:10px; }
  .chunk-meta span { margin-right:14px; }
  .chunk-text-body { background:#111; padding:12px; border-radius:4px; font-size:0.88em; white-space:pre-wrap; word-break:break-word; color:#dde; line-height:1.5; }
  .heatmap-legend { display:flex; gap:14px; margin:8px 0 12px; font-size:0.82em; align-items:center; }
  .legend-dot { display:inline-block; width:14px; height:14px; border-radius:2px; vertical-align:middle; margin-right:4px; }
  /* Phase stepper */
  .phase-stepper { display:flex; align-items:center; margin:12px 0 18px; }
  .ps-node { display:flex; flex-direction:column; align-items:center; gap:4px; }
  .ps-circle {
    width:40px; height:40px; border-radius:50%; border:2px solid #333;
    display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:13px; color:#555; background:#1a1a1a;
    transition:all 0.5s ease;
  }
  .ps-circle.done { border-color:#27a; color:#27a; background:#0d1f2d; }
  .ps-circle.active {
    border-color:#7af; color:#7af; background:#0d1a2a;
    box-shadow:0 0 14px #27a888; animation:ps-pulse 2s infinite;
  }
  @keyframes ps-pulse { 0%,100%{box-shadow:0 0 8px #27a8} 50%{box-shadow:0 0 20px #27af} }
  .ps-label { font-size:9px; color:#555; text-transform:uppercase; letter-spacing:.08em; }
  .ps-label.done { color:#27a; }
  .ps-label.active { color:#7af; }
  .ps-line { flex:1; height:2px; background:#222; margin:0 6px; margin-bottom:14px;
             position:relative; overflow:hidden; min-width:30px; }
  .ps-line-fill { height:100%; background:#27a; width:0%; transition:width 1s ease; }
  /* PRS chart container */
  .prs-chart-wrap { position:relative; height:160px; margin:10px 0; }
  /* Flywheel */
  .fw-summary { display:flex; gap:20px; flex-wrap:wrap; margin:8px 0 12px; }
  .fw-kv { display:flex; flex-direction:column; gap:2px; }
  .fw-kv-label { font-size:9px; color:#666; text-transform:uppercase; letter-spacing:.07em; }
  .fw-kv-val { font-size:1.3em; font-weight:700; color:#7af; }
  .fw-chart-wrap { position:relative; height:140px; margin:8px 0; }
  .fw-table { width:100%; border-collapse:collapse; font-size:0.82em; margin-top:10px; }
  .fw-table th { color:#7af; border-bottom:1px solid #333; padding:4px 8px; text-align:left; }
  .fw-table td { border-bottom:1px solid #1a1a1a; padding:4px 8px; color:#ccc; }
  .fw-table tr:hover td { background:#1a1a1a; }
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
PRS = 0.7 × Accuracy
    + 0.15 × Calibration
    + 0.15 × Self-Consistency</pre>

    <table>
      <tr><th>Component</th><th>What it measures</th><th>Weight</th></tr>
      <tr><td><b>Accuracy</b></td>
          <td>Fraction of FAQ answers that match the LLM's direct answer
              (no retrieval, cosine similarity ≥ threshold)</td>
          <td>70%</td></tr>
      <tr><td><b>Calibration</b></td>
          <td>Whether the model's confidence token probabilities
              match its actual accuracy (low entropy on correct answers)</td>
          <td>15%</td></tr>
      <tr><td><b>Self-Consistency</b></td>
          <td>Mean pairwise cosine similarity of 3 answers sampled
              at temperature 0.7 — measures how stable the knowledge is</td>
          <td>15%</td></tr>
    </table>

    <p><b>Phase thresholds:</b></p>
    <table>
      <tr><th>Phase</th><th>Condition</th><th>Behaviour</th></tr>
      <tr><td>1</td><td>—</td><td>Standard RAG — text-in-context only</td></tr>
      <tr><td>2</td><td>PRS ≥ 0.75 (one round)</td><td>KV injection + selective parametric — queries that clear the known-good similarity eligibility gate are answered from weights</td></tr>
      <tr><td>3</td><td>PRS ≥ 0.80 (two consecutive rounds)</td>
          <td>Corpus-wide confidence gate — runs for every query, not just eligible ones</td></tr>
    </table>

    <div id="prs-live"></div>
  </div>
</div>

<!-- Chunk detail modal (coverage heatmap) -->
<div class="chunk-modal-overlay" id="chunk-modal" onclick="if(event.target===this)closeChunkModal()">
  <div class="chunk-modal-box">
    <span class="modal-close" onclick="closeChunkModal()">&#x2715;</span>
    <h2 id="cm-title">Chunk Detail</h2>
    <div class="chunk-meta" id="cm-meta"></div>
    <div class="chunk-text-body" id="cm-text"></div>
  </div>
</div>

<h1>KVForge Dashboard</h1>

<!-- Phase Progression -->
<div class="card" id="phase-stepper-card">
  <b>Phase Progression</b>
  <span class="help-btn" onclick="openPrsModal()">?</span>
  <div class="phase-stepper" id="phase-stepper">
    <div class="ps-node">
      <div class="ps-circle" id="ps1">1</div>
      <div class="ps-label" id="ps1-lbl">Text RAG</div>
    </div>
    <div class="ps-line"><div class="ps-line-fill" id="ps-line1"></div></div>
    <div class="ps-node">
      <div class="ps-circle" id="ps2">2</div>
      <div class="ps-label" id="ps2-lbl">KV Inject</div>
    </div>
    <div class="ps-line"><div class="ps-line-fill" id="ps-line2"></div></div>
    <div class="ps-node">
      <div class="ps-circle" id="ps3">3</div>
      <div class="ps-label" id="ps3-lbl">Parametric</div>
    </div>
  </div>
  <div style="font-size:0.82em;color:#888;margin-bottom:8px;">PRS History</div>
  <div class="prs-chart-wrap"><canvas id="prs-chart"></canvas></div>
</div>

<!-- Flywheel Analytics -->
<div class="card" id="flywheel-card">
  <b>Flywheel Analytics</b>
  <div class="fw-summary">
    <div class="fw-kv"><div class="fw-kv-label">Rounds</div><div class="fw-kv-val" id="fw-rounds">—</div></div>
    <div class="fw-kv"><div class="fw-kv-label">Last PRS</div><div class="fw-kv-val" id="fw-last-prs">—</div></div>
    <div class="fw-kv"><div class="fw-kv-label">Est. Cost</div><div class="fw-kv-val" id="fw-cost">—</div></div>
    <div class="fw-kv"><div class="fw-kv-label">ETA Phase 3</div><div class="fw-kv-val" id="fw-eta">—</div></div>
  </div>
  <div class="fw-chart-wrap"><canvas id="flywheel-chart"></canvas></div>
  <table class="fw-table" id="fw-table">
    <thead><tr><th>Round</th><th>PRS</th><th>Phase</th><th>Cost ($)</th></tr></thead>
    <tbody id="fw-tbody"></tbody>
  </table>
</div>

<div id="root">Loading…</div>

<!-- FAQ Coverage Heatmap -->
<div class="card" id="coverage-section">
  <b>FAQ Coverage Heatmap</b>
  <span style="font-size:0.82em;color:#888;margin-left:8px">— which chunks each FAQ maps to (top-5 by cosine similarity)</span>
  <button onclick="loadCoverage()"
    style="float:right;padding:4px 12px;background:#27a;color:#fff;border:none;cursor:pointer;font-family:monospace;font-size:0.85em">
    Refresh
  </button>
  <div class="heatmap-legend">
    <span><span class="legend-dot" style="background:#e74c3c"></span>≥ 0.85</span>
    <span><span class="legend-dot" style="background:#e67e22"></span>≥ 0.75</span>
    <span><span class="legend-dot" style="background:#f1c40f"></span>≥ 0.65</span>
    <span><span class="legend-dot" style="background:#3498db"></span>&lt; 0.65</span>
    <span style="color:#888;margin-left:8px">Click any cell to see full chunk text</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;margin:8px 0 12px;font-size:0.85em">
    <label for="hm-threshold" style="color:#aaa">Score threshold:</label>
    <input id="hm-threshold" type="range" min="0.60" max="1.00" step="0.01" value="0.60"
      style="width:200px;accent-color:#27a" oninput="onThresholdChange(this.value)"/>
    <span id="hm-threshold-val" style="color:#7af;font-weight:bold;min-width:38px">0.60</span>
    <span style="color:#666;font-size:0.9em">— cells below threshold are hidden</span>
  </div>
  <div id="heatmap-content" style="overflow-x:auto">
    <span style="color:#666">Click Refresh to load coverage data…</span>
  </div>
</div>

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
          <option value="claude">Claude (Anthropic)</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>
      <div class="param-group">
        <label>Model</label>
        <select id="b_model" style="background:#1a1a1a;color:#eee;border:1px solid #444;padding:4px 6px;font-family:monospace;font-size:0.9em;width:100%;box-sizing:border-box;"></select>
      </div>
      <div class="param-group">
        <label>API Key <span id="api-key-warning" style="color:#f87171;font-size:0.8em;display:none">⚠ required</span></label>
        <input id="b_api_key" type="password" placeholder="Paste API key…" oninput="onApiKeyInput()"/>
      </div>
      <div class="param-group" id="base-url-group" style="display:none">
        <label>Base URL <span style="color:#888;font-size:0.8em">(OpenAI-compatible)</span></label>
        <input id="b_base_url" type="text" placeholder="http://localhost:8090/v1" oninput="saveAndSyncModelB()"/>
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
  gemini: ['gemini-2.5-flash','gemini-2.5-pro','gemini-2.0-flash','gemini-1.5-pro'],
  claude: ['claude-sonnet-4-6','claude-opus-4-6','claude-haiku-4-5-20251001'],
  openai: ['gpt-4.1','gpt-4.1-mini','gpt-4o','gpt-4o-mini'],
};
const PROVIDER_LABELS = {
  gemini: 'Gemini',
  claude: 'Claude',
  openai: 'OpenAI',
};

function populateModelDropdown(provider) {
  const sel = document.getElementById('b_model');
  sel.innerHTML = MODELS_B[provider].map(m => `<option value="${m}">${m}</option>`).join('');
  const saved = localStorage.getItem(`modelb_${provider}_model`);
  if (saved && MODELS_B[provider].includes(saved)) sel.value = saved;
}

function onApiKeyInput() {
  const key = document.getElementById('b_api_key').value.trim();
  document.getElementById('api-key-warning').style.display = key ? 'none' : 'inline';
  document.getElementById('b_api_key').style.borderColor = key ? '#444' : '#f87171';
  saveAndSyncModelB();
}

function saveAndSyncModelB() {
  const provider = document.getElementById('b_provider').value;
  const model = document.getElementById('b_model').value;
  const apiKey = document.getElementById('b_api_key').value;
  const baseUrl = (document.getElementById('b_base_url') || {}).value || '';
  localStorage.setItem('modelb_provider', provider);
  localStorage.setItem(`modelb_${provider}_model`, model);
  localStorage.setItem(`modelb_${provider}_key`, apiKey);
  localStorage.setItem(`modelb_${provider}_base_url`, baseUrl);
  const baseUrlGroup = document.getElementById('base-url-group');
  if (baseUrlGroup) baseUrlGroup.style.display = provider === 'openai' ? '' : 'none';
  fetch('/api/set_model_b_config', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({provider, model, api_key: apiKey, base_url: baseUrl}),
  }).catch(e => console.error('[ModelB config sync failed]', e));
  const label = PROVIDER_LABELS[provider] || provider;
  document.getElementById('label-b').textContent = `Answer B — ${label} RAG`;
  document.getElementById('header-b').textContent = `Answer B — ${label} (${model})`;
  document.getElementById('model-info-b').innerHTML =
    `${label} model: <b style="color:#fa7">${model}</b>`;
}

document.getElementById('b_provider').addEventListener('change', function() {
  const provider = this.value;
  populateModelDropdown(provider);
  const savedKey = localStorage.getItem(`modelb_${provider}_key`) || '';
  document.getElementById('b_api_key').value = savedKey;
  // Show warning and highlight field if no key stored for this provider
  const warn = document.getElementById('api-key-warning');
  const keyInput = document.getElementById('b_api_key');
  if (!savedKey) {
    warn.style.display = 'inline';
    keyInput.style.borderColor = '#f87171';
    keyInput.focus();
  } else {
    warn.style.display = 'none';
    keyInput.style.borderColor = '#444';
  }
  saveAndSyncModelB();
});
document.getElementById('b_model').addEventListener('change', saveAndSyncModelB);
document.getElementById('b_api_key').addEventListener('change', saveAndSyncModelB);

async function loadConfig() {
  // Restore Model B config from localStorage (must run before fetch to pre-populate)
  const savedProvider = localStorage.getItem('modelb_provider') || 'gemini';
  document.getElementById('b_provider').value = savedProvider;
  populateModelDropdown(savedProvider);
  const savedKey = localStorage.getItem(`modelb_${savedProvider}_key`) || '';
  document.getElementById('b_api_key').value = savedKey;
  if (!savedKey) {
    document.getElementById('api-key-warning').style.display = 'inline';
    document.getElementById('b_api_key').style.borderColor = '#f87171';
  }
  const savedBaseUrl = localStorage.getItem(`modelb_${savedProvider}_base_url`) || '';
  const baseUrlInput = document.getElementById('b_base_url');
  if (baseUrlInput) baseUrlInput.value = savedBaseUrl;
  const baseUrlGroup = document.getElementById('base-url-group');
  if (baseUrlGroup) baseUrlGroup.style.display = savedProvider === 'openai' ? '' : 'none';
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
let _topChunks = [];
let _prsChart = null;

function renderPhaseStepper(phase) {
  [1,2,3].forEach(function(n) {
    var circle = document.getElementById('ps'+n);
    var label = document.getElementById('ps'+n+'-lbl');
    if (!circle) return;
    circle.className = 'ps-circle' + (n < phase ? ' done' : n === phase ? ' active' : '');
    label.className = 'ps-label' + (n < phase ? ' done' : n === phase ? ' active' : '');
  });
  var l1 = document.getElementById('ps-line1');
  var l2 = document.getElementById('ps-line2');
  if (l1) l1.style.width = phase >= 2 ? '100%' : '0%';
  if (l2) l2.style.width = phase >= 3 ? '100%' : '0%';
}

function renderPrsChart(history) {
  var canvas = document.getElementById('prs-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  var labels = history.map(function(h) { return 'Rnd ' + h.round; });
  var vals   = history.map(function(h) { return h.prs; });
  if (_prsChart) { _prsChart.destroy(); _prsChart = null; }
  _prsChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'PRS', data: vals,
        borderColor: '#27a', backgroundColor: 'rgba(34,119,170,0.12)',
        pointBackgroundColor: '#7af', tension: 0.35, fill: true,
      }]
    },
    options: {
      animation: { duration: 600 },
      scales: {
        y: { min: 0, max: 1, ticks: { color: '#888', stepSize: 0.25 },
             grid: { color: '#222' } },
        x: { ticks: { color: '#888' }, grid: { color: '#222' } }
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: function(ctx) { return 'PRS: ' + ctx.parsed.y.toFixed(3); } } }
      },
      responsive: true, maintainAspectRatio: false,
    }
  });
}

function openTopChunkModal(idx) {
  const c = _topChunks[idx];
  if (!c) return;
  const TIER_LABEL = { hot:'🔥 Hot', warm:'🌡 Warm', cold:'❄ Cold', frozen:'🧊 Frozen' };
  document.getElementById('cm-title').textContent =
    `Chunk #${c.chunk_id} — ${TIER_LABEL[c.tier] || c.tier}`;
  document.getElementById('cm-meta').innerHTML =
    `<span>Page: <b>${c.page}</b></span>` +
    `<span>Access count: <b>${c.access_count}</b></span>` +
    `<span>Parametric hits: <b>${c.parametric_hit_count}</b></span>` +
    `<span>KV version: <b>${c.kv_version !== null && c.kv_version !== undefined ? c.kv_version : '—'}</b></span>`;
  document.getElementById('cm-text').textContent = c.text || '(no text)';
  document.getElementById('chunk-modal').classList.add('open');
}

async function load(){
  const [stats, ver] = await Promise.all(
    [fetch('/api/stats').then(r=>r.json()), fetch('/api/version').then(r=>r.json())]
  );
  _prsHistory = ver.prs_history || [];
  renderPhaseStepper(ver.phase || 1);
  renderPrsChart(ver.prs_history || []);
  _topChunks = stats.top_chunks || [];
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
      ${stats.top_chunks.map((c,i)=>`<tr><td>${c.chunk_id}</td><td>${c.page}</td>
        <td class="${c.tier}">${c.tier}</td><td>${c.access_count}</td>
        <td>${c.parametric_hit_count}</td>
        <td style="cursor:pointer;color:#7af;text-decoration:underline dotted"
            title="Click to view full chunk text"
            onclick="openTopChunkModal(${i})">${c.text_preview}</td></tr>`).join('')}
      </table>
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
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closePrsModal(); closeChunkModal(); }
});

// ---------------------------------------------------------------------------
// FAQ Coverage Heatmap
// ---------------------------------------------------------------------------

const TIER_LABEL = { hot:'🔥 Hot', warm:'🌡 Warm', cold:'❄ Cold', frozen:'🧊 Frozen' };

function scoreClass(score) {
  if (score >= 0.85) return 'hm-hot';
  if (score >= 0.75) return 'hm-warm';
  if (score >= 0.65) return 'hm-cold';
  return 'hm-frozen';
}

let _coverageData = null;  // cached after first load
let _hmThreshold = 0.60;

function onThresholdChange(val) {
  _hmThreshold = parseFloat(val);
  document.getElementById('hm-threshold-val').textContent = _hmThreshold.toFixed(2);
  if (_coverageData) renderHeatmap(_coverageData, document.getElementById('heatmap-content'));
}

async function loadCoverage() {
  const el = document.getElementById('heatmap-content');
  el.innerHTML = '<span style="color:#888">Loading coverage data…</span>';
  try {
    const data = await fetch('/api/coverage?top_k=5').then(r => r.json());
    if (data.error) { el.innerHTML = `<span style="color:#f77">Error: ${data.error}</span>`; return; }
    _coverageData = data;
    renderHeatmap(data, el);
  } catch(e) {
    el.innerHTML = `<span style="color:#f77">Failed: ${e.message}</span>`;
  }
}

function renderHeatmap(data, el) {
  const { faqs, matches } = data;
  const n = faqs.length;
  if (!n) { el.innerHTML = '<span style="color:#666">No FAQs found.</span>'; return; }
  const topK = (matches[0] || []).length;
  // Only include rows where at least one match meets the threshold
  let rows = '';
  let visibleRows = 0;
  for (let i = 0; i < n; i++) {
    const ms = matches[i] || [];
    const hasMatch = ms.some(m => m.score >= _hmThreshold);
    if (!hasMatch) continue;
    visibleRows++;
    const q = faqs[i].question || '';
    const short = q.length > 55 ? q.slice(0,55)+'…' : q;
    const cells = ms.map((m, j) => {
      if (m.score < _hmThreshold) {
        return `<td class="hm-cell hm-frozen" style="opacity:0.2;cursor:default" title="score ${m.score.toFixed(3)} — below threshold">—</td>`;
      }
      const cls = scoreClass(m.score);
      const score = m.score.toFixed(3);
      return `<td class="hm-cell ${cls}" title="score ${score} | tier ${m.tier} | page ${m.page}"
        onclick="openChunkModal(${i},${j})">${score}</td>`;
    }).join('');
    rows += `<tr><td class="faq-label" title="${q.replace(/"/g,'&quot;')}">${short}</td>${cells}</tr>`;
  }
  if (!visibleRows) {
    el.innerHTML = `<span style="color:#888">No matches above threshold ${_hmThreshold.toFixed(2)}. Lower the slider.</span>`;
    return;
  }
  const html = `<div style="color:#666;font-size:0.8em;margin-bottom:6px">${visibleRows} of ${n} FAQs have at least one match ≥ ${_hmThreshold.toFixed(2)}</div>
  <table class="heatmap-table"><thead><tr>
    <th style="min-width:220px">FAQ</th>
    ${Array.from({length:topK},(_,i)=>`<th>Match ${i+1}</th>`).join('')}
  </tr></thead><tbody>${rows}</tbody></table>`;
  el.innerHTML = html;
}

function openChunkModal(faqIdx, matchIdx) {
  if (!_coverageData) return;
  const faq = _coverageData.faqs[faqIdx];
  const m = (_coverageData.matches[faqIdx] || [])[matchIdx];
  if (!m) return;
  document.getElementById('cm-title').textContent =
    `Chunk #${m.chunk_id} — ${TIER_LABEL[m.tier] || m.tier}`;
  document.getElementById('cm-meta').innerHTML =
    `<span>Score: <b style="color:#7af">${m.score}</b></span>` +
    `<span>Page: <b>${m.page}</b></span>` +
    `<span>Access count: <b>${m.access_count}</b></span>` +
    `<span>KV version: <b>${m.kv_version !== null && m.kv_version !== undefined ? m.kv_version : '—'}</b></span>` +
    `<br><span style="color:#888;margin-top:4px;display:block">FAQ: ${faq.question}</span>`;
  document.getElementById('cm-text').textContent = m.text || '(no text)';
  document.getElementById('chunk-modal').classList.add('open');
}

function closeChunkModal() {
  document.getElementById('chunk-modal').classList.remove('open');
}

let _fwChart = null;

function loadFlywheel() {
  fetch('/api/flywheel').then(function(r) { return r.json(); }).then(function(data) {
    if (data.error) return;
    document.getElementById('fw-rounds').textContent = data.rounds_completed != null ? data.rounds_completed : '—';
    document.getElementById('fw-last-prs').textContent = data.last_prs != null ? data.last_prs.toFixed(3) : '—';
    document.getElementById('fw-cost').textContent = data.estimated_cost_usd != null ? '$' + data.estimated_cost_usd.toFixed(4) : '—';
    document.getElementById('fw-eta').textContent = data.eta_to_phase3 || '—';

    var snapshots = data.round_snapshots || [];
    var labels = snapshots.map(function(s) { return 'R' + s.round; });
    var vals   = snapshots.map(function(s) { return s.prs; });
    var colors = vals.map(function(v) { return v >= 0.75 ? '#2ecc71' : v >= 0.60 ? '#f39c12' : '#e74c3c'; });

    if (_fwChart) { _fwChart.destroy(); _fwChart = null; }
    var canvas = document.getElementById('flywheel-chart');
    if (!canvas || typeof Chart === 'undefined') return;
    _fwChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'PRS per Round', data: vals,
          backgroundColor: colors, borderRadius: 3,
        }]
      },
      options: {
        animation: { duration: 500 },
        scales: {
          y: { min: 0, max: 1, ticks: { color: '#888', stepSize: 0.25 }, grid: { color: '#222' } },
          x: { ticks: { color: '#888' }, grid: { display: false } }
        },
        plugins: { legend: { display: false } },
        responsive: true, maintainAspectRatio: false,
      }
    });

    var tbody = document.getElementById('fw-tbody');
    tbody.innerHTML = snapshots.map(function(s) {
      return '<tr><td>' + s.round + '</td><td>' + (s.prs != null ? s.prs.toFixed(3) : '—') + '</td><td>' + (s.phase || '—') + '</td><td>' + (s.cost_usd != null ? '$' + s.cost_usd.toFixed(4) : '—') + '</td></tr>';
    }).join('');
  }).catch(function() {});
}

loadConfig();
load();
loadFlywheel();
setInterval(load, 30000);
setInterval(loadFlywheel, 60000);

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
    parser.add_argument("--gemini-key", default=None,
                        help="Gemini API key — pre-seeds Model B config so it persists across restarts")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash",
                        help="Gemini model for Model B (default: gemini-2.5-flash)")
    parser.add_argument("--openai-key", default=None,
                        help="OpenAI API key — sets Model B to OpenAI on startup")
    parser.add_argument("--openai-model", default="gpt-4.1",
                        help="OpenAI model for Model B (default: gpt-4.1)")
    parser.add_argument("--claude-key", default=None,
                        help="Anthropic API key — sets Model B to Claude on startup")
    parser.add_argument("--claude-model", default="claude-sonnet-4-6",
                        help="Claude model for Model B (default: claude-sonnet-4-6)")
    args = parser.parse_args()
    _config_path = args.config
    if args.claude_key:
        _model_b_config["provider"] = "claude"
        _model_b_config["model"] = args.claude_model
        _model_b_config["api_key"] = args.claude_key
    elif args.openai_key:
        _model_b_config["provider"] = "openai"
        _model_b_config["model"] = args.openai_model
        _model_b_config["api_key"] = args.openai_key
    elif args.gemini_key:
        _model_b_config["provider"] = "gemini"
        _model_b_config["model"] = args.gemini_model
        _model_b_config["api_key"] = args.gemini_key
    cfg = _load_cfg()
    port = args.port if args.port is not None else cfg.get("dashboard_port", 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    _main()
