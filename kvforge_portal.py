"""KVForge main portal — landing page at port 8080.

Links to all 4 use-case dashboards and shows their live status.

Usage::

    python3 kvforge_portal.py [--port 8080]
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

USE_CASES = [
    {
        "id": "uc1",
        "title": "Use Case 1",
        "subtitle": "Customer Support Q&A",
        "description": "Bitext customer-support dataset · 2 000 Q&A pairs · Qdrant + bge-small",
        "port": 8081,
        "color": "#4a9eff",
    },
    {
        "id": "uc2",
        "title": "Use Case 2",
        "subtitle": "Biomedical Q&A",
        "description": "PubMedQA dataset · biomedical literature · Qdrant + bge-small",
        "port": 8082,
        "color": "#4aff9e",
    },
    {
        "id": "uc3",
        "title": "Use Case 3",
        "subtitle": "Reading Comprehension",
        "description": "SQuAD v2 dataset · Wikipedia passages · Qdrant + bge-small",
        "port": 8083,
        "color": "#ff9e4a",
    },
    {
        "id": "uc4",
        "title": "Use Case 4",
        "subtitle": "Amazon Bedrock User Guide",
        "description": "AWS Bedrock PDF (~500 pages) · Qdrant + mxbai-embed-large (1024-dim)",
        "port": 8084,
        "color": "#c97aff",
    },
]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield


app = FastAPI(title="KVForge Portal", lifespan=_lifespan)


@app.get("/api/status")
async def get_status():
    """Check which use-case dashboards are reachable."""
    results = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for uc in USE_CASES:
            try:
                r = await client.get(f"http://localhost:{uc['port']}/api/health")
                results[uc["id"]] = "online" if r.status_code == 200 else "error"
            except Exception:
                results[uc["id"]] = "offline"
    return results


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
  <span class="badge">4 use-case corpora · A10G GPU · Qdrant vector store</span>
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
     id="{uc['id']}" style="--accent:{uc['color']}">
    <div class="card-header">
      <span class="card-title">{uc['title']}</span>
      <span class="status-dot" id="dot-{uc['id']}"></span>
    </div>
    <div class="card-subtitle">{uc['subtitle']}</div>
    <div class="card-desc">{uc['description']}</div>
  </a>
""" for uc in USE_CASES) + """
</div>

<div class="footer">
  KVForge · hemantcgi/kvforge · status auto-refreshes every 10 s
</div>

<script>
// Build dashboard links using current hostname (works on any host)
document.querySelectorAll('.card[data-port]').forEach(card => {
  const port = card.dataset.port;
  card.href = `http://${window.location.hostname}:${port}/`;
  card.target = '_blank';
});

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    for (const [id, status] of Object.entries(data)) {
      const dot = document.getElementById('dot-' + id);
      if (dot) { dot.className = 'status-dot ' + status; }
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
