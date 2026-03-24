"""
monitoring_dashboard.py — FastAPI dashboard at localhost:8080.

Start: python3 monitoring_dashboard.py
Or:    python3 monitoring_dashboard.py --config datasource_bedrock.json
Or:    uvicorn monitoring_dashboard:app --port 8080 --reload
"""

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure project root is on sys.path before any local imports (do once, not per-thread)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import version as ver
from qdrant_client import QdrantClient

app = FastAPI(title="Smart Qdrant Dashboard")
_cfg: dict = {}
_qdrant_client: QdrantClient | None = None
_query_executor = ThreadPoolExecutor(max_workers=2)

# Heavy modules (torch/transformers) loaded once at startup in main thread
# so worker threads never race on the lazy-import lock.
_model_loader = None
_kv_background = None
_kv_inference = None


@app.on_event("startup")
def _preload_inference_modules() -> None:
    """Import GPU modules and pre-warm the base model in the main thread at startup."""
    global _model_loader, _kv_background, _kv_inference
    try:
        import model_loader as _ml
        import kv_background as _kb
        import kv_inference as _ki
        _model_loader = _ml
        _kv_background = _kb
        _kv_inference = _ki
        # Pre-warm: load base model now so the first query hits the cache
        cfg = _load_cfg()
        _ml.init(cfg)
        print("[dashboard] pre-warming base model…", flush=True)
        _ml.load(None)   # loads + caches base model; subsequent load(None) calls are free
        print("[dashboard] base model ready", flush=True)
    except Exception as e:
        print(f"[dashboard] inference modules unavailable: {e}", flush=True)

# Config file path — overridden by --config CLI arg at startup (no annotation so global works)
_config_path = "my_config.json"


def _load_cfg() -> dict:
    global _cfg
    if not _cfg:
        with open(_config_path) as f:
            _cfg = json.load(f)
    return _cfg


def _get_client() -> QdrantClient:
    """Return a shared QdrantClient, creating it once from config."""
    global _qdrant_client
    if _qdrant_client is None:
        cfg = _load_cfg()
        _qdrant_client = QdrantClient(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
    return _qdrant_client


@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/api/version")
def get_version():
    return ver.load()


@app.get("/api/stats")
def get_stats():
    cfg = _load_cfg()
    v = ver.load()

    # Qdrant tier counts — paginate to handle >5000 chunks
    client = _get_client()
    tier_counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}
    top_chunks = []
    try:
        all_results = []
        offset = None
        while True:
            batch, offset = client.scroll(
                collection_name=cfg["collection"],
                limit=500,
                with_payload=["tier", "access_count", "page",
                               "parametric_hit_count", "text"],
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
        "gemini_model": cfg.get("gemini_model", "gemini-2.5-pro"),
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
    a_top_k: int = 3
    a_max_new_tokens: int = 256
    a_temperature: float = 0.7
    a_top_p: float = 0.9
    a_repetition_penalty: float = 1.2
    # Answer B (Gemini) params
    b_top_k: int = 5
    b_max_output_tokens: int = 1024
    b_temperature: float = 1.0


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _answer_smartqdrant(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "A:SmartQdrant"
    _log(tag, f"START query={query!r} top_k={params.a_top_k} max_new_tokens={params.a_max_new_tokens} temp={params.a_temperature}")
    if _model_loader is None or _kv_inference is None:
        _log(tag, "SKIP — inference modules not loaded (GPU required)")
        return {"answer": "SmartQdrant inference modules not available (GPU required)", "latency_ms": 0}
    retrieval_ms = 0
    generation_ms = 0
    chunks = []
    mode = "text_in_context"
    gate_info = {}
    try:
        import torch
        ver.init(cfg)
        _model_loader.init(cfg)
        _kv_background.start(cfg)

        phase = ver.get_phase()
        lora_ckpt = ver.load().get("checkpoint_path")
        model, tokenizer = _model_loader.load(lora_ckpt)
        model = model.half()

        # ── Phase 3: answer directly from fine-tuned weights, no retrieval ─
        if phase >= 3:
            mode = "parametric"
            _log(tag, "Phase 3 — answering from fine-tuned weights (no retrieval)…")
            t_gen = time.time()
            # Apply Llama instruction chat template so model stops cleanly at <|eot_id|>
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
        from qdrant_client import QdrantClient as _QC
        from bedrock_rag import _run_search, Config
        cfg_a = dict(cfg, top_k=params.a_top_k)

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        client = _QC(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        rag_cfg = Config(**{k: cfg_a[k] for k in Config.__dataclass_fields__ if k in cfg_a})

        _log(tag, "searching Qdrant…")
        t_ret = time.time()
        hits = _run_search(query, embedder, client, rag_cfg)
        retrieval_ms = int((time.time() - t_ret) * 1000)
        top_score_a = f"{hits[0].score:.4f}" if hits else "n/a"
        _log(tag, f"search done — {len(hits)} hits, top_score={top_score_a}, retrieval={retrieval_ms}ms")

        if not hits:
            answer = "No relevant chunks found."
        else:
            chunks = [{"chunk_id": h.id,
                       "text": h.payload["text"],
                       "page": h.payload["page"], "score": round(h.score, 4),
                       "kv_cache": h.payload.get("kv_cache"),
                       "kv_version": h.payload.get("kv_version")} for h in hits]
            current_ver = ver.get_lora_version()

            # Enqueue stale chunks for background KV recompute
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
            "mode": mode, "gate": gate_info}


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
        from qdrant_client import QdrantClient as _QC

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        client = _QC(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        q_vec = list(embedder.embed([query]))[0].tolist()

        _log(tag, "searching Qdrant…")
        from qdrant_client.models import NamedVector
        t_ret = time.time()
        result = client.query_points(
            collection_name=cfg["collection"],
            query=q_vec,
            limit=params.b_top_k,
            with_payload=True,
        )
        retrieval_ms = int((time.time() - t_ret) * 1000)
        hits = result.points
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
        api_key = cfg.get("gemini_api_key", "")
        model = cfg.get("gemini_model", "gemini-2.5-pro")
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


@app.post("/api/query")
async def run_query(req: QueryRequest):
    _log("query", f"received query={req.query!r}")
    cfg = _load_cfg()
    loop = asyncio.get_event_loop()
    fut_a = loop.run_in_executor(_query_executor, _answer_smartqdrant, req.query, cfg, req)
    fut_b = loop.run_in_executor(_query_executor, _answer_gemini, req.query, cfg, req)
    result_a, result_b = await asyncio.gather(fut_a, fut_b)
    return {
        "answer_a": result_a["answer"],
        "latency_a_ms": result_a["latency_ms"],
        "retrieval_a_ms": result_a.get("retrieval_ms", 0),
        "generation_a_ms": result_a.get("generation_ms", 0),
        "chunks_a": result_a.get("chunks", []),
        "mode_a": result_a.get("mode", "text_in_context"),
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
<head><meta charset="UTF-8"><title>Smart Qdrant Dashboard</title>
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

<h1>Smart Qdrant Dashboard</h1>
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

    <div class="section-label" id="label-a">Answer A — SmartQdrant</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Max new tokens</label>
        <input id="a_max_new_tokens" type="number" min="64" max="2048" value="256"/>
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

    <div class="section-label">Answer B — Gemini RAG</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Max output tokens</label>
        <input id="b_max_output_tokens" type="number" min="128" max="65536" value="1024"/>
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
        <b style="color:#7af" id="header-a">Answer A — SmartQdrant</b>
        <div id="latency-a" style="color:#888;font-size:0.85em;margin:4px 0;font-family:monospace"></div>
        <div id="mode-a" style="font-size:0.85em;margin:4px 0"></div>
        <pre id="answer-a" style="white-space:pre-wrap;margin:8px 0;color:#eee;font-size:0.9em"></pre>
        <details style="margin-top:8px">
          <summary style="cursor:pointer;color:#888">Retrieved context (reasoning)</summary>
          <div id="chunks-a" style="font-size:0.8em;margin-top:6px"></div>
        </details>
      </div>
      <div style="flex:1;border:1px solid #444;padding:12px">
        <b style="color:#fa7">Answer B — Gemini RAG</b>
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
async function loadConfig() {
  try {
    const cfg = await fetch('/api/config').then(r => r.json());
    document.getElementById('model-info-a').innerHTML =
      `LLM: <a href="${cfg.llm_model_url}" target="_blank" rel="noopener">${cfg.llm_model}</a>` +
      ` &nbsp;|&nbsp; Embedder: <a href="${cfg.embed_model_url}" target="_blank" rel="noopener">${cfg.embed_model}</a>` +
      ` &nbsp;|&nbsp; Collection: ${cfg.collection}`;
    document.getElementById('model-info-b').innerHTML =
      `Gemini model: <b style="color:#fa7">${cfg.gemini_model}</b>`;
    // set default top_k from server config (shared for both models)
    document.getElementById('top_k').value = cfg.top_k;
    // update section labels and answer headers with actual model name
    const shortName = cfg.llm_model.split('/').pop();
    document.getElementById('label-a').textContent = `Answer A — SmartQdrant (${shortName})`;
    document.getElementById('header-a').textContent = `Answer A — SmartQdrant (${shortName})`;
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
    parser = argparse.ArgumentParser(description="Smart Qdrant Dashboard")
    parser.add_argument(
        "--config",
        default="my_config.json",
        help="Path to JSON config file (default: my_config.json)",
    )
    args = parser.parse_args()
    _config_path = args.config
    cfg = _load_cfg()
    uvicorn.run(app, host="0.0.0.0", port=cfg.get("dashboard_port", 8080))


if __name__ == "__main__":
    _main()
