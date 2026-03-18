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


def _answer_smartqdrant(query: str, cfg: dict) -> dict:
    t0 = time.time()
    if _model_loader is None or _kv_inference is None:
        return {"answer": "SmartQdrant inference modules not available (GPU required)", "latency_ms": 0}
    try:
        ver.init(cfg)
        _model_loader.init(cfg)
        _kv_background.start(cfg)
        answer = _kv_inference.answer_with_retrieval(query, cfg)
    except Exception as e:
        answer = f"Error: {e}"
    return {"answer": answer, "latency_ms": int((time.time() - t0) * 1000)}


def _answer_gemini(query: str, cfg: dict) -> dict:
    t0 = time.time()
    chunks = []
    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient as _QC

        # 1. embed + search Qdrant
        embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
        client = _QC(host=cfg["qdrant_host"], port=cfg["qdrant_port"])
        q_vec = list(embedder.embed([query]))[0].tolist()
        from qdrant_client.models import NamedVector
        result = client.query_points(
            collection_name=cfg["collection"],
            query=q_vec,
            limit=cfg.get("top_k", 5),
            with_payload=True,
        )
        hits = result.points
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
        resp = httpx.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        answer = f"Error: {e}"
        # preserve any chunks already retrieved before the error
    return {"answer": answer, "latency_ms": int((time.time() - t0) * 1000), "chunks": chunks}


@app.post("/api/query")
async def run_query(req: QueryRequest):
    cfg = _load_cfg()
    loop = asyncio.get_event_loop()
    fut_a = loop.run_in_executor(_query_executor, _answer_smartqdrant, req.query, cfg)
    fut_b = loop.run_in_executor(_query_executor, _answer_gemini, req.query, cfg)
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
  h1 { color:#7af; } .card { border:1px solid #333; padding:12px; margin:10px 0; }
  table { border-collapse:collapse; width:100%; }
  td,th { border:1px solid #333; padding:6px 10px; text-align:left; }
  .hot{color:#f90} .warm{color:#ff0} .cold{color:#0af} .frozen{color:#aaa}
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
  <div id="ab-result" style="display:none">
    <div style="display:flex;gap:16px;margin-top:12px">
      <div style="flex:1;border:1px solid #444;padding:12px">
        <b style="color:#7af">Answer A — SmartQdrant (KV-injected LLM)</b>
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
      ${(ver.prs_history||[]).map(r=>`Round ${r.round}: ${r.prs}`).join(' → ') || 'No data yet'}
    </div>`;
}
load();
setInterval(load, 30000);

async function runQuery() {
  const q = document.getElementById('qinput').value.trim();
  if (!q) return;
  document.getElementById('query-loading').style.display = 'block';
  document.getElementById('ab-result').style.display = 'none';
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query: q})
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
// also allow Enter key to submit
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
