"""
monitoring_dashboard.py — FastAPI dashboard at localhost:8080.

Start: python3 monitoring_dashboard.py
Or:    uvicorn monitoring_dashboard:app --port 8080 --reload
"""

import json
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import version as ver
from qdrant_client import QdrantClient

app = FastAPI(title="RAG Intelligence Dashboard")
_cfg: dict = {}
_qdrant_client: QdrantClient | None = None


def _load_cfg() -> dict:
    global _cfg
    if not _cfg:
        with open("my_config.json") as f:
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
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the self-contained monitoring dashboard."""
    return HTMLResponse(DASHBOARD_HTML)


if __name__ == "__main__":
    cfg = _load_cfg()
    uvicorn.run(app, host="0.0.0.0", port=cfg.get("dashboard_port", 8080))
