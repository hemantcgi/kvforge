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
from fastapi.staticfiles import StaticFiles
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
        "llm_model": "google/gemma-4-E2B-it",
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
        "llm_model": "google/gemma-4-E2B-it",
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
        "llm_model": "google/gemma-4-E2B-it",
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
        "llm_model": "google/gemma-4-E2B-it",
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

_docs_dir = Path(__file__).resolve().parent / "docs" / "demo-guides"
if _docs_dir.exists():
    app.mount("/docs/demo-guides", StaticFiles(directory=str(_docs_dir), html=True), name="demo-guides")

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
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVForge — Progressive RAG Platform</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0e14;color:#c9d1d9;min-height:100vh;}

/* NAV */
.nav{height:52px;display:flex;align-items:center;padding:0 40px;border-bottom:1px solid #1c2230;background:#0d1117;position:sticky;top:0;z-index:100;}
.nav-brand{font-size:17px;font-weight:800;color:#58a6ff;letter-spacing:-.3px;display:flex;align-items:center;gap:8px;}
.nav-logo{width:28px;height:28px;border-radius:6px;background:linear-gradient(135deg,#0052CC,#00875A);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:900;color:#fff;}
.nav-sp{flex:1;}
.nav-link{font-size:13px;color:#8b949e;text-decoration:none;padding:6px 12px;border-radius:5px;transition:color .15s,background .15s;}
.nav-link:hover{color:#e6edf3;background:#161b22;}
.btn-signin{font-size:13px;font-weight:600;color:#c9d1d9;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 16px;text-decoration:none;transition:all .15s;}
.btn-signin:hover{background:#30363d;border-color:#8b949e;}
.btn-signup{font-size:13px;font-weight:700;color:#0d1117;background:#58a6ff;border:none;border-radius:6px;padding:6px 18px;text-decoration:none;margin-left:8px;transition:opacity .15s;}
.btn-signup:hover{opacity:.88;}

/* HERO */
.hero{text-align:center;padding:96px 24px 80px;max-width:800px;margin:0 auto;}
.hero-tag{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#58a6ff;background:rgba(88,166,255,.1);border:1px solid rgba(88,166,255,.25);border-radius:20px;padding:4px 14px;margin-bottom:24px;}
.hero h1{font-size:52px;font-weight:800;line-height:1.1;color:#e6edf3;margin-bottom:20px;letter-spacing:-.5px;}
.hero h1 span{background:linear-gradient(90deg,#58a6ff,#3fb950);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-size:18px;color:#8b949e;line-height:1.6;max-width:600px;margin:0 auto 40px;}
.hero-cta{display:flex;align-items:center;justify-content:center;gap:12px;flex-wrap:wrap;}
.cta-primary{display:inline-block;font-size:15px;font-weight:700;color:#0d1117;background:#58a6ff;border-radius:8px;padding:12px 32px;text-decoration:none;transition:opacity .15s;}
.cta-primary:hover{opacity:.88;}
.cta-secondary{display:inline-block;font-size:15px;font-weight:600;color:#c9d1d9;background:#21262d;border:1px solid #30363d;border-radius:8px;padding:12px 28px;text-decoration:none;transition:all .15s;}
.cta-secondary:hover{background:#30363d;border-color:#8b949e;}

/* PHASES */
.phases{display:flex;align-items:center;justify-content:center;gap:0;max-width:760px;margin:0 auto 96px;padding:0 24px;flex-wrap:wrap;}
.phase-step{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:20px 22px;flex:1;min-width:200px;text-align:center;}
.phase-arrow{font-size:18px;color:#30363d;padding:0 8px;flex-shrink:0;}
.phase-num{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}
.p1 .phase-num{color:#58a6ff;} .p2 .phase-num{color:#3fb950;} .p3 .phase-num{color:#bc8cff;}
.phase-title{font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:6px;}
.phase-desc{font-size:12px;color:#8b949e;line-height:1.5;}

/* FEATURES */
.section{padding:0 24px 96px;max-width:1100px;margin:0 auto;}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#8b949e;text-align:center;margin-bottom:12px;}
.section-title{font-size:30px;font-weight:800;color:#e6edf3;text-align:center;margin-bottom:48px;}
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}
@media(max-width:768px){.feat-grid{grid-template-columns:1fr;}}
.feat-card{background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:24px;transition:border-color .15s,transform .15s;}
.feat-card:hover{border-color:#30363d;transform:translateY(-2px);}
.feat-icon{font-size:26px;margin-bottom:14px;}
.feat-title{font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:8px;}
.feat-desc{font-size:13px;color:#8b949e;line-height:1.6;}
.feat-tag{display:inline-block;font-size:10px;font-weight:700;color:#3fb950;background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.2);border-radius:10px;padding:2px 8px;margin-bottom:10px;}

/* GET STARTED */
.gs-section{background:#0d1117;border-top:1px solid #21262d;border-bottom:1px solid #21262d;padding:80px 24px;margin-bottom:80px;}
.gs-inner{max-width:840px;margin:0 auto;}
.gs-title{font-size:26px;font-weight:800;color:#e6edf3;margin-bottom:32px;text-align:center;}
.steps{display:flex;flex-direction:column;gap:16px;}
.step{display:flex;gap:20px;align-items:flex-start;}
.step-num{width:32px;height:32px;border-radius:50%;background:rgba(88,166,255,.15);border:1px solid rgba(88,166,255,.3);color:#58a6ff;font-size:13px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px;}
.step-body{}
.step-title{font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:4px;}
.step-desc{font-size:13px;color:#8b949e;line-height:1.5;}
.step-code{font-family:monospace;font-size:12px;background:#161b22;border:1px solid #21262d;border-radius:5px;padding:6px 12px;color:#79c0ff;margin-top:6px;display:inline-block;}

/* FOOTER */
footer{text-align:center;padding:32px 24px;color:#484f58;font-size:12px;border-top:1px solid #161b22;}
</style>
</head>
<body>

<nav class="nav">
  <div class="nav-brand"><div class="nav-logo">KV</div> KVForge</div>
  <div class="nav-sp"></div>
  <a class="nav-link" href="/studio">Studio</a>
  <a class="nav-link" href="/auth/signup">Docs</a>
  <a class="btn-signin" href="/auth/login">Sign In</a>
  <a class="btn-signup" href="/auth/signup">Get Started Free</a>
</nav>

<section class="hero">
  <div class="hero-tag">Progressive RAG · KV Cache Injection · LoRA Fine-Tuning</div>
  <h1>RAG that gets<br><span>smarter over time</span></h1>
  <p class="hero-sub">KVForge is an end-to-end RAG platform that starts on day one and automatically advances through three phases — from standard retrieval to full parametric answering — as your corpus matures.</p>
  <div class="hero-cta">
    <a class="cta-primary" href="/auth/signup">Create your workspace</a>
    <a class="cta-secondary" href="/auth/login">Sign in</a>
  </div>
</section>

<div class="phases">
  <div class="phase-step p1">
    <div class="phase-num">Phase 1</div>
    <div class="phase-title">Text RAG</div>
    <div class="phase-desc">Standard retrieval-augmented generation. Chunks retrieved and passed as context.</div>
  </div>
  <div class="phase-arrow">→</div>
  <div class="phase-step p2">
    <div class="phase-num">Phase 2</div>
    <div class="phase-title">KV Injection</div>
    <div class="phase-desc">Pre-computed KV tensors injected at query time — no re-encoding of matched chunks.</div>
  </div>
  <div class="phase-arrow">→</div>
  <div class="phase-step p3">
    <div class="phase-num">Phase 3</div>
    <div class="phase-title">Parametric</div>
    <div class="phase-desc">LoRA-fine-tuned model answers from weights when PRS confidence ≥ 0.75.</div>
  </div>
</div>

<div class="section">
  <div class="section-label">What's inside</div>
  <div class="section-title">Everything you need in one Studio</div>
  <div class="feat-grid">
    <div class="feat-card">
      <div class="feat-tag">Zero config</div>
      <div class="feat-icon">🧙</div>
      <div class="feat-title">Setup Wizard</div>
      <div class="feat-desc">5-step wizard configures your data source, vector DB, embedding model, and LLM. Supports HuggingFace datasets, PDFs, live APIs (Wikipedia, FDA, EDGAR, ESPN), and S3.</div>
    </div>
    <div class="feat-card">
      <div class="feat-tag">4.4× compression</div>
      <div class="feat-icon">⚡</div>
      <div class="feat-title">TurboQuant KV Cache</div>
      <div class="feat-desc">3-bit key / 4-bit value quantization stores pre-computed KV tensors at 4.4× compression with 0.91 recall — saving ~3.8 GB VRAM per 50k chunks.</div>
    </div>
    <div class="feat-card">
      <div class="feat-tag">Auto-advancing</div>
      <div class="feat-icon">🧠</div>
      <div class="feat-title">Corpus Intelligence</div>
      <div class="feat-desc">CIS scoring promotes hot chunks to enhanced storage, keeps active chunks in the vector store, and archives cold content — zero manual curation.</div>
    </div>
    <div class="feat-card">
      <div class="feat-tag">Pluggable</div>
      <div class="feat-icon">🧩</div>
      <div class="feat-title">7 Connector Types</div>
      <div class="feat-desc">Google Drive, Amazon S3, SharePoint, Wikipedia, openFDA, SEC EDGAR, and ESPN Sports — all configurable from the Studio Connectors panel.</div>
    </div>
    <div class="feat-card">
      <div class="feat-tag">Live sync</div>
      <div class="feat-icon">🔄</div>
      <div class="feat-title">Delta Sync Engine</div>
      <div class="feat-desc">Scheduled connector sync with section-hash diffing. Only changed chunks are re-indexed and re-computed, keeping your corpus fresh without full re-runs.</div>
    </div>
    <div class="feat-card">
      <div class="feat-tag">Observability</div>
      <div class="feat-icon">📊</div>
      <div class="feat-title">Phase Monitoring</div>
      <div class="feat-desc">Per-use-case dashboards show PRS history, tier distribution (hot/warm/cold/frozen), active jobs, and sync history — all in real time.</div>
    </div>
  </div>
</div>

<div class="gs-section">
  <div class="gs-inner">
    <div class="gs-title">Get started in minutes</div>
    <div class="steps">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <div class="step-title">Create an account</div>
          <div class="step-desc">Sign up at <a href="/auth/signup" style="color:#58a6ff">kvforge/signup</a>. No credit card required for local deployments.</div>
        </div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          <div class="step-title">Sign in and open Studio</div>
          <div class="step-desc">After signing in you land on the Studio dashboard. Click <strong style="color:#e6edf3">+ New Use Case</strong> to launch the setup wizard.</div>
        </div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          <div class="step-title">Choose your data source</div>
          <div class="step-desc">Pick a HuggingFace dataset, upload a PDF, connect a live API source, or point at an existing corpus. The wizard walks you through every setting.</div>
        </div>
      </div>
      <div class="step">
        <div class="step-num">4</div>
        <div class="step-body">
          <div class="step-title">Launch the pipeline</div>
          <div class="step-desc">The wizard generates your configuration and starts the index pipeline. KVForge streams live progress in the Studio — no CLI needed.</div>
          <div class="step-code">Setup · Index · Train · Evaluate — fully managed</div>
        </div>
      </div>
      <div class="step">
        <div class="step-num">5</div>
        <div class="step-body">
          <div class="step-title">Watch it advance</div>
          <div class="step-desc">KVForge automatically promotes your use case from Phase 1 → 2 → 3 as your corpus grows and PRS thresholds are met. Zero manual intervention.</div>
        </div>
      </div>
    </div>
  </div>
</div>

<footer>KVForge Studio &middot; Progressive RAG Platform &middot; <a href="/auth/login" style="color:#58a6ff">Sign in</a> &middot; <a href="/auth/signup" style="color:#58a6ff">Get started</a></footer>

</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVForge main portal")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
