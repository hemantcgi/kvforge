"""Per-use-case A/B evaluation runner.

Queries a running KVForge dashboard for each FAQ question, computes
semantic similarity and ROUGE-L scores against ground truth, and writes
examples/<uc>/ab_eval_results.json + examples/<uc>/ab_eval_viewer.html.

Usage::

    python -m pipeline.ab_evaluator \\
        --config examples/usecase1_customer_support/config.json \\
        --dashboard-url http://localhost:8081 \\
        --gemini-api-key <key> \\
        --max-samples 200
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
from fastembed import TextEmbedding


def _rouge_l(hyp: str, ref: str) -> float:
    """Compute ROUGE-L F1 score (pure Python, no external deps)."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if r[i - 1] == h[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def run_eval(
    config_path: str,
    faqs_path: str,
    dashboard_url: str,
    gemini_api_key: str,
    max_samples: int,
) -> list[dict]:
    """Query dashboard for each FAQ, compute scores, return results list."""
    cfg = json.loads(Path(config_path).read_text())
    fq = Path(faqs_path)
    if not fq.exists():
        raise FileNotFoundError(
            f"{fq} not found — run the pipeline first to generate faqs.json"
        )

    faqs = json.loads(fq.read_text())
    q_key = cfg.get("faq_question_key", "question")
    a_key = cfg.get("faq_answer_key", "answer")

    if max_samples and max_samples < len(faqs):
        faqs = faqs[:max_samples]

    print(f"Evaluating {len(faqs)} questions against {dashboard_url} …")

    # Set Gemini API key on the dashboard before running queries
    with httpx.Client(timeout=10.0) as setup_client:
        setup_client.post(
            f"{dashboard_url}/api/set_model_b_config",
            json={"provider": "gemini", "model": "gemini-2.0-flash",
                  "api_key": gemini_api_key},
        )

    embedder = TextEmbedding(model_name=cfg["embed_model"], show_download_progress=False)
    results = []

    with httpx.Client(timeout=90.0) as client:
        for i, faq in enumerate(faqs):
            question = faq[q_key]
            ground_truth = faq[a_key]

            resp = client.post(
                f"{dashboard_url}/api/query",
                json={
                    "query": question,
                    "b_top_k": 5,
                    "b_max_output_tokens": 4096,
                    "b_temperature": 1.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            answer_a = data.get("answer_a", "")
            answer_b = data.get("answer_b", "")

            vecs = list(embedder.embed([answer_a, answer_b, ground_truth]))
            sem_sim_a = _cosine(np.array(vecs[0]), np.array(vecs[2]))
            sem_sim_b = _cosine(np.array(vecs[1]), np.array(vecs[2]))

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "mode_a": data.get("mode_a", ""),
                "prs_score": round(data.get("prs_score_a", 0.0), 4),
                "latency_a_ms": data.get("latency_a_ms", 0),
                "latency_b_ms": data.get("latency_b_ms", 0),
                "generation_a_ms": data.get("generation_a_ms", 0),
                "generation_b_ms": data.get("generation_b_ms", 0),
                "sem_sim_a": round(sem_sim_a, 4),
                "sem_sim_b": round(sem_sim_b, 4),
                "rouge_l_a": round(_rouge_l(answer_a, ground_truth), 4),
                "rouge_l_b": round(_rouge_l(answer_b, ground_truth), 4),
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(faqs):
                print(f"  {i + 1}/{len(faqs)} done")

    return results


def generate_html(results: list[dict], title: str) -> str:
    """Render a self-contained HTML viewer with results embedded as const AB_DATA."""
    n = len(results)
    # Win = query where fine-tuned model has mastered the answer (PRS >= 0.75)
    PRS_THRESHOLD = 0.75
    wins_a = sum(1 for r in results if (r.get("prs_score") or 0.0) >= PRS_THRESHOLD)
    wins_b = n - wins_a
    avg_lat_a = sum(r["latency_a_ms"] for r in results) / max(n, 1)
    avg_lat_b = sum(r["latency_b_ms"] for r in results) / max(n, 1)
    avg_sim_a = sum(r["sem_sim_a"] for r in results) / max(n, 1)
    avg_sim_b = sum(r["sem_sim_b"] for r in results) / max(n, 1)
    avg_rl_a = sum(r["rouge_l_a"] for r in results) / max(n, 1)
    avg_rl_b = sum(r["rouge_l_b"] for r in results) / max(n, 1)
    pct_a = wins_a / max(n, 1) * 100
    pct_b = wins_b / max(n, 1) * 100
    parametric_rows = [r for r in results if r.get("mode_a") == "parametric"]
    pct_parametric = len(parametric_rows) / max(n, 1) * 100
    _prs_vals = [r["prs_score"] for r in parametric_rows if r.get("prs_score") is not None]
    avg_prs_parametric = sum(_prs_vals) / len(_prs_vals) if _prs_vals else 0.0
    avg_lat_parametric = (
        sum(r["latency_a_ms"] for r in parametric_rows) / len(parametric_rows)
        if parametric_rows else 0.0
    )
    avg_lat_rag = (
        sum(r["latency_a_ms"] for r in results if r.get("mode_a") != "parametric")
        / max(n - len(parametric_rows), 1)
    )

    ab_data_js = json.dumps(results, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: monospace; background: #111; color: #eee; padding: 20px; }}
  h1 {{ color: #7af; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 0.88em; margin-bottom: 16px; }}
  .controls {{ margin: 12px 0; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  input[type=text] {{ background: #222; color: #eee; border: 1px solid #555; padding: 6px 10px; font-family: monospace; width: 300px; }}
  select {{ background: #222; color: #eee; border: 1px solid #555; padding: 6px; font-family: monospace; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85em; }}
  th {{ border: 1px solid #333; padding: 8px; background: #1a1a1a; color: #7af; text-align: left; cursor: pointer; user-select: none; }}
  th:hover {{ background: #222; }}
  td {{ border: 1px solid #222; padding: 8px; vertical-align: top; }}
  tr:hover td {{ background: #1a1a1a; }}
  .sim-high {{ color: #4d4; }} .sim-mid {{ color: #fa7; }} .sim-low {{ color: #f77; }}
  .q {{ color: #adf; max-width: 220px; }} .gt {{ color: #888; max-width: 200px; }}
  .ans-a {{ max-width: 280px; }} .ans-b {{ max-width: 280px; }}
  .trunc {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: inherit; cursor: pointer; }}
  .trunc.expanded {{ white-space: pre-wrap; overflow: visible; }}
  .winner-a {{ background: rgba(100,200,100,0.06); }} .winner-b {{ background: rgba(100,150,255,0.06); }}
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
  .badge-a {{ background: #2a4; color: #fff; }} .badge-b {{ background: #27a; color: #fff; }}
  #count {{ color: #888; font-size: 0.9em; }}
  .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .stats-card {{ background: #1a1a1a; border: 1px solid #333; padding: 16px; border-radius: 4px; }}
  .stats-card h3 {{ margin: 0 0 12px; font-size: 0.95em; }}
  .stats-card.a h3 {{ color: #4d4; }} .stats-card.b h3 {{ color: #47b; }}
  .stat-section {{ font-size: 0.75em; color: #555; text-transform: uppercase; margin: 10px 0 4px; padding-bottom: 2px; border-bottom: 1px solid #282828; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.88em; }}
  .stat-label {{ color: #888; }} .stat-val {{ font-weight: bold; }}
  .stat-val.good {{ color: #4d4; }} .stat-val.ok {{ color: #fa7; }} .stat-val.info {{ color: #7af; }}
  .wins-bar {{ display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 10px 0 4px; }}
  .wins-bar-a {{ background: #2a4; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #fff; }}
  .wins-bar-b {{ background: #27a; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #fff; }}
</style>
</head>
<body>
<h1>A/B Eval Results</h1>
<div class="subtitle">{n} Q&amp;A pairs &nbsp;|&nbsp; Model A: Llama 3.2 3B Parametric &nbsp;|&nbsp; Model B: Gemini Flash RAG</div>
<div class="stats-grid">
  <div class="stats-card a">
    <h3>Model A — Llama 3.2 3B (Parametric)</h3>
    <div class="stat-section">PRS Gate (per-query mastery)</div>
    <div class="stat-row"><span class="stat-label">Parametric (no retrieval)</span><span class="stat-val {'good' if pct_parametric >= 50 else 'ok'}">{len(parametric_rows)} / {n} ({pct_parametric:.1f}%)</span></div>
    <div class="stat-row"><span class="stat-label">Avg PRS (parametric)</span><span class="stat-val {'good' if avg_prs_parametric >= 0.8 else 'ok'}">{avg_prs_parametric:.4f}</span></div>
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End</span><span class="stat-val">{avg_lat_a:.0f} ms</span></div>
    <div class="stat-row"><span class="stat-label">Parametric (no retrieval)</span><span class="stat-val info">{avg_lat_parametric:.0f} ms</span></div>
    <div class="stat-row"><span class="stat-label">RAG (with retrieval)</span><span class="stat-val">{avg_lat_rag:.0f} ms</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val {'good' if avg_rl_a >= 0.4 else 'ok'}">{avg_rl_a:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Semantic Similarity</span><span class="stat-val {'good' if avg_sim_a >= 0.8 else 'ok'}">{avg_sim_a:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Mastered (PRS ≥ 0.75)</span><span class="stat-val {'good' if pct_a >= 75 else 'ok'}">{wins_a} / {n} ({pct_a:.1f}%)</span></div>
  </div>
  <div class="stats-card b">
    <h3>Model B — Gemini 2.0 Flash (RAG)</h3>
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End</span><span class="stat-val">{avg_lat_b:.0f} ms</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val {'good' if avg_rl_b >= 0.4 else 'ok'}">{avg_rl_b:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Semantic Similarity</span><span class="stat-val {'good' if avg_sim_b >= 0.8 else 'ok'}">{avg_sim_b:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Needs RAG (PRS &lt; 0.75)</span><span class="stat-val">{wins_b} / {n} ({pct_b:.1f}%)</span></div>
  </div>
</div>
<div class="wins-bar">
  <div class="wins-bar-a" style="width:{pct_a:.1f}%">Parametric {pct_a:.0f}%</div>
  <div class="wins-bar-b" style="width:{pct_b:.1f}%">RAG {pct_b:.0f}%</div>
</div>
<div class="controls">
  <input id="search" type="text" placeholder="Search questions…"/>
  <select id="filter">
    <option value="all">All</option>
    <option value="a">Model A wins</option>
    <option value="b">Model B wins</option>
    <option value="parametric">Parametric only (PRS ≥ 0.75)</option>
    <option value="rag">RAG only (PRS &lt; 0.75)</option>
  </select>
  <span id="count"></span>
</div>
<table>
  <thead>
    <tr>
      <th data-col="question">Question</th>
      <th data-col="ground_truth">Ground Truth</th>
      <th data-col="answer_a">Model A</th>
      <th data-col="answer_b">Model B</th>
      <th data-col="prs_score" title="Per-query PRS: cosine similarity to known-good queries. ≥0.75 = parametric (no retrieval)">PRS</th>
      <th data-col="sem_sim_a">Sim A</th>
      <th data-col="sem_sim_b">Sim B</th>
      <th data-col="rouge_l_a">RL A</th>
      <th data-col="rouge_l_b">RL B</th>
      <th data-col="latency_a_ms">Lat A</th>
      <th data-col="latency_b_ms">Lat B</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
<script>
const AB_DATA = {ab_data_js};
const DATA = AB_DATA;
let sortCol = 'sem_sim_a', sortDir = -1;
function sc(v) {{ return v >= 0.8 ? 'sim-high' : v >= 0.6 ? 'sim-mid' : 'sim-low'; }}
function trunc(s) {{ return '<div class="trunc">' + (s||'').replace(/</g,'&lt;') + '</div>'; }}
function render() {{
  const q = document.getElementById('search').value.toLowerCase();
  const f = document.getElementById('filter').value;
  let rows = AB_DATA.filter(r => {{
    if (q && !r.question.toLowerCase().includes(q)) return false;
    if (f === 'a' && r.sem_sim_a < r.sem_sim_b) return false;
    if (f === 'b' && r.sem_sim_b <= r.sem_sim_a) return false;
    if (f === 'parametric' && r.mode_a !== 'parametric') return false;
    if (f === 'rag' && r.mode_a === 'parametric') return false;
    return true;
  }});
  rows.sort((a,b) => (a[sortCol] > b[sortCol] ? 1 : -1) * sortDir);
  document.getElementById('count').textContent = rows.length + ' rows';
  const winner = r => (r.prs_score != null && r.prs_score >= 0.75) ? 'winner-a' : 'winner-b';
  document.getElementById('tbody').innerHTML = rows.map(r => `
    <tr class="${{winner(r)}}">
      <td class="q">${{trunc(r.question)}}</td>
      <td class="gt">${{trunc(r.ground_truth)}}</td>
      <td class="ans-a">${{trunc(r.answer_a)}}</td>
      <td class="ans-b">${{trunc(r.answer_b)}}</td>
      <td style="color:${{r.prs_score>=0.75?'#22c55e':r.prs_score>=0.5?'#f59e0b':'#6b7280'}}" title="${{r.mode_a}}">${{r.prs_score!=null?r.prs_score.toFixed(3):'—'}}</td>
      <td class="${{sc(r.sem_sim_a)}}">${{r.sem_sim_a}}</td>
      <td class="${{sc(r.sem_sim_b)}}">${{r.sem_sim_b}}</td>
      <td>${{r.rouge_l_a}}</td>
      <td>${{r.rouge_l_b}}</td>
      <td>${{r.latency_a_ms}}</td>
      <td>${{r.latency_b_ms}}</td>
    </tr>`).join('');
}}
document.querySelector('thead').addEventListener('click', e => {{
  const th = e.target.closest('th');
  if (!th) return;
  const col = th.dataset.col;
  if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}
  render();
}});
document.getElementById('tbody').addEventListener('click', e => {{
  let el = e.target;
  while (el) {{ if (el.classList && el.classList.contains('trunc')) {{ el.classList.toggle('expanded'); return; }} el = el.parentElement; }}
}});
document.getElementById('search').addEventListener('input', render);
document.getElementById('filter').addEventListener('change', render);
render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Run per-UC A/B evaluation")
    parser.add_argument("--config", required=True, help="Path to use case config.json")
    parser.add_argument("--dashboard-url", default="", help="Running dashboard base URL e.g. http://localhost:8081")
    parser.add_argument("--gemini-api-key", default="", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--max-samples", type=int, default=200, help="Max FAQ questions to evaluate")
    parser.add_argument("--regen-html", action="store_true",
                        help="Regenerate ab_eval_viewer.html from existing ab_eval_results.json without re-querying dashboard")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = config_path.parent
    json_out = out_dir / "ab_eval_results.json"
    html_out = out_dir / "ab_eval_viewer.html"
    uc_name = out_dir.name.replace("_", " ").title()

    if args.regen_html:
        if not json_out.exists():
            raise FileNotFoundError(f"{json_out} not found — run eval first")
        results = json.loads(json_out.read_text())
        # Backfill prs_score for older result files that predate the field
        for r in results:
            r.setdefault("prs_score", None)
        html_out.write_text(generate_html(results, title=f"A/B Eval — {uc_name}"))
        print(f"Regenerated {html_out} from {len(results)} existing results")
        return

    if not args.dashboard_url:
        parser.error("--dashboard-url is required unless --regen-html is set")

    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    faqs_path = config_path.parent / "faqs.json"

    results = run_eval(
        config_path=str(config_path),
        faqs_path=str(faqs_path),
        dashboard_url=args.dashboard_url,
        gemini_api_key=gemini_key,
        max_samples=args.max_samples,
    )

    json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {json_out}")

    html_out.write_text(generate_html(results, title=f"A/B Eval — {uc_name}"))
    print(f"Wrote {html_out}")


if __name__ == "__main__":
    main()
