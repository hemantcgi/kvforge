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

app = FastAPI(title="RAG Intelligence Dashboard")
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
    """Import GPU modules in the main thread at startup to avoid thread-safety issues."""
    global _model_loader, _kv_background, _kv_inference
    try:
        import model_loader as _ml
        import kv_background as _kb
        import kv_inference as _ki
        _model_loader = _ml
        _kv_background = _kb
        _kv_inference = _ki
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
    # Answer A (TinyLlama) generation params
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
    try:
        ver.init(cfg)
        _model_loader.init(cfg)
        _kv_background.start(cfg)
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient as _QC
        from bedrock_rag import _run_search, Config
        cfg_a = dict(cfg, top_k=params.a_top_k)

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        client = _QC(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        rag_cfg = Config(**{k: cfg_a[k] for k in Config.__dataclass_fields__ if k in cfg_a})

        _log(tag, "searching Qdrant…")
        hits = _run_search(query, embedder, client, rag_cfg)
        top_score_a = f"{hits[0].score:.4f}" if hits else "n/a"
        _log(tag, f"search done — {len(hits)} hits, top_score={top_score_a}")

        if not hits:
            answer = "No relevant chunks found."
        else:
            chunks = [{"chunk_id": h.id,
                       "text": h.payload["text"][:500],
                       "page": h.payload["page"], "score": round(h.score, 4),
                       "kv_cache": None, "kv_version": None} for h in hits]
            _log(tag, "loading model…")
            model, tokenizer = _model_loader.load(None)
            model = model.half()
            _log(tag, "model ready — recording access + generating…")
            for rank, chunk in enumerate(chunks, start=1):
                _kv_background.record_access(chunk["chunk_id"], rank)
            answer = _kv_inference.generate_text_in_context(
                query, chunks, model, tokenizer,
                max_new_tokens=params.a_max_new_tokens,
                temperature=params.a_temperature,
                top_p=params.a_top_p,
                repetition_penalty=params.a_repetition_penalty,
            )
            _log(tag, f"generation done ({len(answer)} chars)")
    except Exception as e:
        import traceback
        _log(tag, f"ERROR: {e}\n{traceback.format_exc()}")
        answer = f"Error: {e}"
    elapsed = int((time.time() - t0) * 1000)
    _log(tag, f"DONE {elapsed}ms")
    return {"answer": answer, "latency_ms": elapsed}


def _answer_gemini(query: str, cfg: dict, params: QueryRequest) -> dict:
    t0 = time.time()
    tag = "B:Gemini"
    _log(tag, f"START query={query!r} top_k={params.b_top_k} max_tokens={params.b_max_output_tokens} temp={params.b_temperature}")
    chunks = []
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient as _QC

        _log(tag, "embedding query…")
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        client = _QC(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        q_vec = list(embedder.embed([query]))[0].tolist()

        _log(tag, "searching Qdrant…")
        from qdrant_client.models import NamedVector
        result = client.query_points(
            collection_name=cfg["collection"],
            query=q_vec,
            limit=params.b_top_k,
            with_payload=True,
        )
        hits = result.points
        top_score_b = f"{hits[0].score:.4f}" if hits else "n/a"
        _log(tag, f"search done — {len(hits)} hits, top_score={top_score_b}")
        chunks = [
            {
                "page": h.payload.get("page", 0),
                "score": round(h.score, 4),
                "text": h.payload.get("text", "")[:300],
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
        _log(tag, f"calling Gemini API ({model}, timeout=60s)…")
        resp = httpx.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": params.b_temperature,
                    "maxOutputTokens": params.b_max_output_tokens,
                },
            },
            timeout=60,
        )
        _log(tag, f"Gemini HTTP {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        _log(tag, f"generation done ({len(answer)} chars)")
    except Exception as e:
        import traceback
        _log(tag, f"ERROR: {e}\n{traceback.format_exc()}")
        answer = f"Error: {e}"
    elapsed = int((time.time() - t0) * 1000)
    _log(tag, f"DONE {elapsed}ms")
    return {"answer": answer, "latency_ms": elapsed, "chunks": chunks}


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
        "answer_b": result_b["answer"],
        "latency_b_ms": result_b["latency_ms"],
        "chunks_b": result_b.get("chunks", []),
    }


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>RAG Intelligence Dashboard</title>
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
</style>
</head>
<body>
<h1>RAG Intelligence Dashboard</h1>
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

    <div class="section-label">Answer A — SmartQdrant (TinyLlama RAG)</div>
    <div class="param-grid">
      <div class="param-group">
        <label>Chunks retrieved (top_k)</label>
        <input id="a_top_k" type="number" min="1" max="10" value="3"/>
      </div>
      <div class="param-group">
        <label>Max new tokens</label>
        <input id="a_max_new_tokens" type="number" min="64" max="512" value="256"/>
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
        <label>Chunks retrieved (top_k)</label>
        <input id="b_top_k" type="number" min="1" max="10" value="5"/>
      </div>
      <div class="param-group">
        <label>Max output tokens</label>
        <input id="b_max_output_tokens" type="number" min="128" max="8192" value="1024"/>
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
        <b style="color:#7af">Answer A — SmartQdrant (TinyLlama RAG)</b>
        <div id="latency-a" style="color:#888;font-size:0.85em;margin:4px 0"></div>
        <pre id="answer-a" style="white-space:pre-wrap;margin:8px 0;color:#eee;font-size:0.9em"></pre>
      </div>
      <div style="flex:1;border:1px solid #444;padding:12px">
        <b style="color:#fa7">Answer B — Gemini RAG</b>
        <div id="latency-b" style="color:#888;font-size:0.85em;margin:4px 0"></div>
        <pre id="answer-b" style="white-space:pre-wrap;margin:8px 0;color:#eee;font-size:0.9em"></pre>
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
    // set default top_k from server config
    document.getElementById('a_top_k').value = Math.min(cfg.top_k, 3);
    document.getElementById('b_top_k').value = cfg.top_k;
  } catch(e) {
    document.getElementById('model-info-a').textContent = 'Could not load config';
  }
}

async function load(){
  const [stats, ver] = await Promise.all(
    [fetch('/api/stats').then(r=>r.json()), fetch('/api/version').then(r=>r.json())]
  );
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
    <div class="card"><b>PRS history:</b>
      ${(ver.prs_history||[]).map(r=>`Round ${r.round}: ${r.prs}`).join(' \u2192 ') || 'No data yet'}
    </div>`;
}

function getNum(id) { return parseFloat(document.getElementById(id).value) || 0; }
function getInt(id) { return parseInt(document.getElementById(id).value) || 0; }

async function runQuery() {
  const q = document.getElementById('qinput').value.trim();
  if (!q) return;
  document.getElementById('query-loading').style.display = 'block';
  document.getElementById('ab-result').style.display = 'none';
  const payload = {
    query: q,
    a_top_k:               getInt('a_top_k'),
    a_max_new_tokens:      getInt('a_max_new_tokens'),
    a_temperature:         getNum('a_temperature'),
    a_top_p:               getNum('a_top_p'),
    a_repetition_penalty:  getNum('a_repetition_penalty'),
    b_top_k:               getInt('b_top_k'),
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
  document.getElementById('latency-a').textContent = `Latency: ${res.latency_a_ms}ms`;
  document.getElementById('answer-b').textContent = res.answer_b;
  document.getElementById('latency-b').textContent = `Latency: ${res.latency_b_ms}ms`;
  document.getElementById('chunks-b').innerHTML = (res.chunks_b||[]).map(c =>
    `<div style="margin:4px 0;border-top:1px solid #333;padding-top:4px">
      <span style="color:#888">page ${c.page} \u00b7 score ${c.score}</span><br>${c.text}\u2026</div>`
  ).join('');
  document.getElementById('ab-result').style.display = 'block';
}

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
    parser = argparse.ArgumentParser(description="RAG Intelligence Dashboard")
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
