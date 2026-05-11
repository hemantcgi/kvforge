"""KVForge main portal — landing page at port 8080.

Links to all 4 use-case dashboards and shows their live status.

Usage::

    python3 kvforge_portal.py [--port 8080]
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from starlette.middleware.sessions import SessionMiddleware

from auth.middleware import AuthMiddleware
from auth.routes import router as _auth_router
from auth.oauth import router as _oauth_router
from connectors.routes import connector_router, _sync_runs_router
from sync.scheduler import SyncScheduler
from connectors.sync_engine import make_default_engine

try:
    import anthropic as _anthropic_mod
    anthropic = _anthropic_mod
except ImportError:
    anthropic = None

USE_CASES = [
    {
        "id": "uc1",
        "title": "Use Case 1",
        "subtitle": "Customer Support Q&A",
        "description": "Bitext customer-support dataset · 2 000 Q&A pairs · Qdrant + bge-small",
        "port": 8081,
        "color": "#4a9eff",
        "vectordb": "Qdrant",
        "vectordb_url": "qdrant",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "ab_eval_dir": "examples/usecase1_customer_support",
    },
    {
        "id": "uc2",
        "title": "Use Case 2",
        "subtitle": "Biomedical Q&A",
        "description": "PubMedQA dataset · biomedical literature · ChromaDB + bge-small",
        "port": 8082,
        "color": "#4aff9e",
        "vectordb": "ChromaDB",
        "vectordb_url": "https://www.trychroma.com",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "ab_eval_dir": "examples/usecase2_pubmedqa",
    },
    {
        "id": "uc3",
        "title": "Use Case 3",
        "subtitle": "Reading Comprehension",
        "description": "SQuAD v2 dataset · Wikipedia passages · FAISS + bge-small",
        "port": 8083,
        "color": "#ff9e4a",
        "vectordb": "FAISS",
        "vectordb_url": "https://faiss.ai",
        "embed_model": "BAAI/bge-small-en-v1.5",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "ab_eval_dir": "examples/usecase3_squad",
    },
    {
        "id": "uc4",
        "title": "Use Case 4",
        "subtitle": "Amazon Bedrock User Guide",
        "description": "AWS Bedrock PDF (~500 pages) · Qdrant + mxbai-embed-large (1024-dim)",
        "port": 8084,
        "color": "#c97aff",
        "vectordb": "Qdrant",
        "vectordb_url": "qdrant",
        "embed_model": "mixedbread-ai/mxbai-embed-large-v1",
        "llm_model": "meta-llama/Llama-3.2-3B-Instruct",
        "ab_eval_dir": "examples/usecase4_bedrock_userguide",
    },
]


_scheduler: SyncScheduler | None = None

@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _scheduler
    import db.store as store
    import os as _os
    _db_override = _os.environ.get("KVFORGE_DB_PATH")
    if _db_override:
        store.DB_PATH = Path(_db_override)
    store.migrate()
    engine = make_default_engine()
    _scheduler = SyncScheduler(run_fn=engine.run)
    _scheduler.load_from_db()
    _scheduler.start()
    yield
    if _scheduler:
        _scheduler.shutdown()


app = FastAPI(title="KVForge Portal", lifespan=_lifespan)

# Middleware (added in reverse order — last added = outermost in Starlette)
# SessionMiddleware must be outermost so request.session is available to AuthMiddleware
app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("KVFORGE_SECRET_KEY", "dev-secret"))

from studio.routes import router as _studio_router
from sync.progress import progress_router
app.include_router(_studio_router, prefix="/studio")
app.include_router(_auth_router)
app.include_router(_oauth_router)
app.include_router(connector_router)
app.include_router(_sync_runs_router)
app.include_router(progress_router)

from sync.webhook import make_webhook_router as _make_wh
_engine_for_wh = make_default_engine()
app.include_router(_make_wh(run_fn=_engine_for_wh.run))


@app.get("/api/status")
async def get_status():
    """Check which use-case dashboards are reachable and return their phase and PRS."""
    results = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for uc in USE_CASES:
            base = f"http://localhost:{uc['port']}"
            try:
                r = await client.get(f"{base}/api/health")
                if r.status_code != 200:
                    results[uc["id"]] = {"status": "error", "phase": None, "prs": None}
                    continue
                try:
                    rv = await client.get(f"{base}/api/version")
                    if rv.status_code == 200:
                        vdata = rv.json()
                        phase = vdata.get("phase")
                        prs_history = vdata.get("prs_history", [])
                        prs = prs_history[-1]["prs"] if prs_history else None
                    else:
                        phase = None
                        prs = None
                except Exception:
                    phase = None
                    prs = None
                results[uc["id"]] = {"status": "online", "phase": phase, "prs": prs}
            except Exception:
                results[uc["id"]] = {"status": "offline", "phase": None, "prs": None}
    return results


# Build a lookup map for quick uc_id → USE_CASE resolution
_UC_MAP = {uc["id"]: uc for uc in USE_CASES}
# Also map by ab_eval_dir basename so /ab-eval/usecase1_customer_support works
for _uc in USE_CASES:
    _long_id = Path(_uc["ab_eval_dir"]).name
    if _long_id not in _UC_MAP:
        _UC_MAP[_long_id] = _uc


@app.get("/ab-eval/{uc_id}", response_class=HTMLResponse)
async def ab_eval_viewer(uc_id: str):
    """Serve the per-UC A/B evaluation viewer HTML."""
    uc = _UC_MAP.get(uc_id)
    if uc is None:
        raise HTTPException(status_code=404, detail=f"Unknown use case: {uc_id!r}")
    viewer = Path(uc["ab_eval_dir"]) / "ab_eval_viewer.html"
    if not viewer.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Report not yet generated for {uc_id}. "
                f"Run: python -m pipeline.ab_evaluator "
                f"--config {uc['ab_eval_dir']}/config.json "
                f"--dashboard-url http://localhost:{uc['port']} "
                f"--gemini-api-key <key>"
            ),
        )
    return HTMLResponse(viewer.read_text())


KVQ_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KVQ — Live Stats &amp; How It Works</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #0d1117; color: #e6edf3; padding: 32px 24px; }
  h1 { color: #7af; font-size: 1.6em; margin-bottom: 8px; }
  .back { color: #8b949e; font-size: 0.85em; margin-bottom: 32px; display: block; }
  .back a { color: #4a9eff; text-decoration: none; }
  h2 { color: #7af; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px; }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 32px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th { padding: 8px 12px; text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: normal; text-transform: uppercase; font-size: 0.75em; letter-spacing: 0.5px; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
  .tier-bar { display: flex; height: 16px; border-radius: 3px; overflow: hidden; gap: 1px; min-width: 120px; }
  .tier-hot { background: #ef4444; } .tier-warm { background: #f59e0b; }
  .tier-cold { background: #3b82f6; } .tier-frozen { background: #6b7280; }
  .phase-badge { display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 0.8em; font-weight: bold; }
  .p1 { background: #1f3a5f; color: #7ab8ff; } .p2 { background: #1f4a2f; color: #7aff9e; } .p3 { background: #3a1f4a; color: #c97aff; }
  .prs-good { color: #22c55e; } .prs-amber { color: #f59e0b; } .prs-none { color: #6b7280; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 2px solid #30363d; border-top-color: #7ab8ff; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #kvq-diagram { min-height: 60px; }
  .refresh-note { color: #484f58; font-size: 0.78em; margin-top: 12px; }
  .key-row { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
  .key-row input { flex: 1; background: #0d1117; border: 1px solid #30363d; color: #e6edf3; font-family: monospace; font-size: 0.85em; padding: 6px 10px; border-radius: 4px; outline: none; }
  .key-row input:focus { border-color: #4a9eff; }
  .key-row button { background: #1f3a5f; border: 1px solid #4a9eff; color: #7ab8ff; font-family: monospace; font-size: 0.85em; padding: 6px 14px; border-radius: 4px; cursor: pointer; white-space: nowrap; }
  .key-row button:hover { background: #2a4f7f; }
</style>
</head>
<body>
<h1>⚡ KVQ Live Stats</h1>
<span class="back"><a href="/">← Back to portal</a></span>

<div class="panel">
  <h2>KV Cache &amp; Phase Status</h2>
  <table>
    <thead>
      <tr>
        <th>Use Case</th>
        <th>Phase</th>
        <th>PRS</th>
        <th>KV Tiers (hot/warm/cold/frozen)</th>
        <th>Total Chunks</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody id="stats-tbody">
      <tr><td colspan="6" style="color:#6b7280;padding:16px">Loading…</td></tr>
    </tbody>
  </table>
  <div class="refresh-note">Auto-refreshes every 10 s</div>
</div>

<div class="panel">
  <h2>How KVQ Works</h2>
  <div class="key-row">
    <input id="api-key-input" type="password" placeholder="Anthropic API key (sk-ant-…) — required for diagram generation" />
    <button onclick="document.getElementById('kvq-diagram').innerHTML='<span class=\\'spinner\\'></span> Generating…'; loadDiagram();">Generate</button>
  </div>
  <div id="kvq-diagram"><p style="color:#484f58;font-family:monospace;font-size:0.85em">Enter your Anthropic API key above and click Generate to render the diagram.</p></div>
</div>

<script>
const UCS = """ + str([{"id": uc["id"], "title": uc["subtitle"], "port": uc["port"]} for uc in USE_CASES]).replace("'", '"') + """;

function phaseHtml(p) {
  if (!p) return '<span style="color:#6b7280">—</span>';
  const cls = {1:'p1',2:'p2',3:'p3'}[p] || '';
  return `<span class="phase-badge ${cls}">Phase ${p}</span>`;
}
function prsHtml(prs) {
  if (prs == null) return '<span class="prs-none">—</span>';
  const cls = prs >= 0.75 ? 'prs-good' : 'prs-amber';
  return `<span class="${cls}">${prs.toFixed(4)}</span>`;
}
function tierBarHtml(tc) {
  if (!tc) return '<span style="color:#6b7280">—</span>';
  const total = (tc.hot||0)+(tc.warm||0)+(tc.cold||0)+(tc.frozen||0);
  if (total === 0) return '<span style="color:#6b7280">empty</span>';
  const pct = k => ((tc[k]||0)/total*100).toFixed(1);
  return `<div class="tier-bar" title="hot:${tc.hot} warm:${tc.warm} cold:${tc.cold} frozen:${tc.frozen}">
    <div class="tier-hot" style="width:${pct('hot')}%" title="hot"></div>
    <div class="tier-warm" style="width:${pct('warm')}%" title="warm"></div>
    <div class="tier-cold" style="width:${pct('cold')}%" title="cold"></div>
    <div class="tier-frozen" style="width:${pct('frozen')}%" title="frozen"></div>
  </div>`;
}

async function refreshStats() {
  try {
    const r = await fetch('/api/kvq-stats');
    if (!r.ok) throw new Error('proxy error');
    const items = await r.json();
    const rows = items.map(item => {
      if (!item.online || !item.data) {
        return `<tr>
          <td>${item.title}</td>
          <td colspan="4" style="color:#6b7280">—</td>
          <td style="color:#ef4444">offline</td>
        </tr>`;
      }
      const d = item.data;
      const v = d.version || {};
      const prs_hist = v.prs_history || [];
      const prs = prs_hist.length ? prs_hist[prs_hist.length-1].prs : null;
      return `<tr>
        <td>${item.title}</td>
        <td>${phaseHtml(v.phase)}</td>
        <td>${prsHtml(prs)}</td>
        <td>${tierBarHtml(d.tier_counts)}</td>
        <td>${d.total_chunks ?? '—'}</td>
        <td style="color:#22c55e">online</td>
      </tr>`;
    });
    document.getElementById('stats-tbody').innerHTML = rows.join('');
  } catch {
    document.getElementById('stats-tbody').innerHTML =
      '<tr><td colspan="6" style="color:#ef4444;padding:12px">Failed to load stats</td></tr>';
  }
}

async function loadDiagram() {
  const key = document.getElementById('api-key-input').value.trim();
  if (key) localStorage.setItem('kvq_anthropic_key', key);
  const saved = key || localStorage.getItem('kvq_anthropic_key') || '';
  const url = saved ? `/kvq/diagram?key=${encodeURIComponent(saved)}` : '/kvq/diagram';
  try {
    const r = await fetch(url);
    const d = await r.json();
    document.getElementById('kvq-diagram').innerHTML = d.html;
  } catch(e) {
    document.getElementById('kvq-diagram').innerHTML = '<p style="color:#6b7280">Diagram unavailable</p>';
  }
}

// Restore saved key and auto-generate diagram if key is already stored
(function() {
  const saved = localStorage.getItem('kvq_anthropic_key');
  if (saved) {
    document.getElementById('api-key-input').value = saved;
    document.getElementById('kvq-diagram').innerHTML = '<span class="spinner"></span> Generating…';
    loadDiagram();
  }
})();

refreshStats();
setInterval(refreshStats, 10000);
</script>
</body>
</html>"""


@app.get("/api/kvq-stats")
async def kvq_stats():
    """Server-side proxy: fetch /api/stats from each dashboard and return aggregated data."""
    results = []
    async with httpx.AsyncClient(timeout=5.0) as client:
        for uc in USE_CASES:
            entry = {"id": uc["id"], "title": uc["subtitle"], "port": uc["port"]}
            try:
                r = await client.get(f"http://localhost:{uc['port']}/api/stats")
                r.raise_for_status()
                entry["data"] = r.json()
                entry["online"] = True
            except Exception:
                entry["data"] = None
                entry["online"] = False
            results.append(entry)
    return results


@app.get("/kvq", response_class=HTMLResponse)
async def kvq_page():
    """KVQ live stats page with Claude-generated architecture diagram."""
    return HTMLResponse(KVQ_HTML)


@app.get("/kvq/diagram")
async def kvq_diagram(key: str = ""):
    """Call Claude API to generate KVQ architecture diagram HTML snippet."""
    api_key = key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or anthropic is None:
        return {"html": "<p style='color:#6b7280;font-family:monospace'>Diagram unavailable — set ANTHROPIC_API_KEY env var</p>"}
    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Generate a clear HTML explanation with an inline SVG diagram showing how KVForge's "
            "3-phase progressive RAG system works:\n\n"
            "Phase 1 (Standard RAG): Query is embedded → top-K chunks retrieved from vector store "
            "→ chunks injected as text into LLM context → LLM generates answer. Typical latency ~6000ms.\n\n"
            "Phase 2 (KV Cache Injection): Same retrieval, but instead of text-in-context, "
            "pre-computed KV tensors for those chunks are injected directly into the LLM's attention "
            "layers, skipping re-encoding. Typical latency ~1500ms.\n\n"
            "Phase 3 (Parametric Gate): If PRS (Parametric Readiness Score) ≥ 0.75, skip retrieval "
            "entirely and answer from LoRA fine-tuned weights alone. Typical latency ~800ms.\n\n"
            "Also explain: what KVQ score measures (quality of pre-computed KV cache entries), "
            "and why pre-computing KV tensors saves memory bandwidth and latency vs re-encoding at query time.\n\n"
            "Use dark theme colors: background #0d1117, text #e6edf3, accent blue #7ab8ff, "
            "green #22c55e, purple #c97aff. Return ONLY the HTML snippet (SVG + explanation paragraphs), "
            "no DOCTYPE, no html/body tags. Use inline styles only."
        )
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        html = message.content[0].text
        return {"html": html}
    except Exception as e:
        return {"html": f"<p style='color:#6b7280;font-family:monospace'>Diagram unavailable — {str(e)[:120]}</p>"}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(PORTAL_HTML)


PORTAL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KVForge Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: monospace;
    background: #0d1117;
    color: #e6edf3;
    min-height: 100vh;
    padding: 32px 24px;
  }
  header {
    text-align: center;
    margin-bottom: 48px;
  }
  header h1 {
    font-size: 2.4em;
    color: #7af;
    letter-spacing: 2px;
  }
  header p {
    color: #8b949e;
    margin-top: 8px;
    font-size: 0.95em;
  }
  .badge {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75em;
    color: #9ca3af;
    margin-top: 6px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
    max-width: 1200px;
    margin: 0 auto;
  }
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 24px;
    transition: border-color 0.2s, transform 0.15s;
    cursor: pointer;
    text-decoration: none;
    color: inherit;
    display: block;
  }
  .card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
  }
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .card-title {
    font-size: 0.8em;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .card-subtitle {
    font-size: 1.1em;
    color: var(--accent);
    font-weight: bold;
    margin: 4px 0 10px;
  }
  .card-desc {
    font-size: 0.82em;
    color: #8b949e;
    line-height: 1.5;
  }
  .status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #374151;
    flex-shrink: 0;
    transition: background 0.3s;
  }
  .status-dot.online { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
  .status-dot.offline { background: #ef4444; }
  .status-dot.error  { background: #f59e0b; }
  .phase-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 0.72em;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-left: 6px;
    vertical-align: middle;
  }
  .phase-badge.p1 { background: #1f3a5f; color: #7ab8ff; }
  .phase-badge.p2 { background: #1f4a2f; color: #7aff9e; }
  .phase-badge.p3 { background: #3a1f4a; color: #c97aff; }
  .phase-badge.unknown { background: #2a2a2a; color: #666; }
  .card-meta {
    margin-top: 14px;
    border-top: 1px solid #21262d;
    padding-top: 10px;
    display: grid;
    grid-template-columns: 80px 1fr;
    gap: 4px 8px;
    font-size: 0.75em;
  }
  .meta-key { color: #6b7280; align-self: center; }
  .meta-val { color: #8b949e; text-decoration: none; word-break: break-all; }
  .meta-val:hover { color: #e6edf3; }
  .meta-val a { color: inherit; text-decoration: none; }
  .meta-val a:hover { color: #7ab8ff; }
  .prs-good { color: #22c55e !important; }
  .prs-amber { color: #f59e0b !important; }
  .prs-none { color: #6b7280 !important; }
  .footer {
    text-align: center;
    margin-top: 48px;
    color: #484f58;
    font-size: 0.8em;
  }
  .arch {
    max-width: 700px;
    margin: 0 auto 48px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    font-size: 0.82em;
    color: #8b949e;
    line-height: 1.7;
  }
  .arch h2 { color: #7af; margin-bottom: 8px; font-size: 0.95em; letter-spacing: 1px; text-transform: uppercase; }
  .phase { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; margin-right: 4px; }
  .p1 { background: #1f3a5f; color: #7ab8ff; }
  .p2 { background: #1f4a2f; color: #7aff9e; }
  .p3 { background: #3a1f4a; color: #c97aff; }
</style>
</head>
<body>

<header>
  <h1>⚡ KVForge Dashboard</h1>
  <p>Progressive RAG · KV-Cache Injection · LoRA Fine-Tuning · Confidence-Gated Parametric Answering</p>
  <span class="badge">4 use-case corpora · A10G GPU · Qdrant · ChromaDB · FAISS</span>
</header>

<div class="arch">
  <h2>How KVForge Works</h2>
  <span class="phase p1">Phase 1</span> Standard RAG — retrieval + text-in-context &nbsp;→&nbsp;
  <span class="phase p2">Phase 2</span> KV tensor injection (pre-computed, no re-encoding) &nbsp;→&nbsp;
  <span class="phase p3">Phase 3</span> Confidence gate: answer from LoRA weights when PRS ≥ 0.75
</div>

<div class="grid" id="grid">
""" + "".join(f"""
  <a class="card" href="#" data-port="{uc['port']}"
     id="{uc['id']}" style="--accent:{uc['color']}"
     data-vectordb-url="{uc['vectordb_url']}"
     data-uc-id="{uc['id']}">
    <div class="card-header">
      <span class="card-title">{uc['title']}</span>
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="phase-badge unknown" id="phase-{uc['id']}">…</span>
        <span class="status-dot" id="dot-{uc['id']}"></span>
      </div>
    </div>
    <div class="card-subtitle">{uc['subtitle']}</div>
    <div class="card-desc">{uc['description']}</div>
    <div class="card-meta">
      <span class="meta-key">Vector DB</span>
      <span class="meta-val"><a href="#" class="vectordb-link" target="_blank" rel="noopener">{uc['vectordb']} ↗</a></span>
      <span class="meta-key">Embed</span>
      <span class="meta-val"><a href="https://huggingface.co/{uc['embed_model']}" target="_blank" rel="noopener">{uc['embed_model'].split('/')[-1]} ↗</a></span>
      <span class="meta-key">LLM</span>
      <span class="meta-val"><a href="https://huggingface.co/{uc['llm_model']}" target="_blank" rel="noopener">{uc['llm_model'].split('/')[-1]} ↗</a></span>
      <span class="meta-key">PRS</span>
      <span class="meta-val prs-none" id="prs-{uc['id']}">—</span>
      <span class="meta-key">KVQ</span>
      <span class="meta-val"><a href="/kvq" target="_blank" rel="noopener">Live stats ↗</a></span>
    </div>
  </a>
""" for uc in USE_CASES) + """
</div>

<div class="footer">
  KVForge · hemantcgi/kvforge · status auto-refreshes every 10 s
</div>

<script>
// Build dashboard and vectordb links using current hostname
document.querySelectorAll('.card[data-port]').forEach(card => {
  const port = card.dataset.port;
  card.href = `http://${window.location.hostname}:${port}/`;
  card.target = '_blank';

  // Wire vectordb link
  const vdbUrl = card.dataset.vectordbUrl;
  const vdbLink = card.querySelector('.vectordb-link');
  if (vdbLink) {
    vdbLink.href = vdbUrl === 'qdrant'
      ? `http://${window.location.hostname}:6333/dashboard`
      : vdbUrl;
  }
});

const PHASE_LABELS = {1: 'Phase 1', 2: 'Phase 2', 3: 'Phase 3'};
const PHASE_CLASSES = {1: 'p1', 2: 'p2', 3: 'p3'};

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    for (const [id, info] of Object.entries(data)) {
      const dot = document.getElementById('dot-' + id);
      if (dot) { dot.className = 'status-dot ' + info.status; }

      const badge = document.getElementById('phase-' + id);
      if (badge) {
        const p = info.phase;
        badge.textContent = p ? PHASE_LABELS[p] || ('Phase ' + p) : (info.status === 'offline' ? 'offline' : '…');
        badge.className = 'phase-badge ' + (p ? (PHASE_CLASSES[p] || 'unknown') : 'unknown');
      }

      const prsEl = document.getElementById('prs-' + id);
      if (prsEl) {
        const prs = info.prs;
        if (prs == null) {
          prsEl.textContent = '—';
          prsEl.className = 'meta-val prs-none';
        } else {
          const color = prs >= 0.75 ? 'prs-good' : 'prs-amber';
          prsEl.className = `meta-val ${color}`;
          prsEl.innerHTML = `<a href="/ab-eval/${id}" target="_blank" rel="noopener">${prs.toFixed(4)} ↗</a>`;
        }
      }
    }
  } catch(e) {}
}
refreshStatus();
setInterval(refreshStatus, 10000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVForge main portal")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
