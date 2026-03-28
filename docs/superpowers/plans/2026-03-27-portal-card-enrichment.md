# Portal Card Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich KVForge portal cards with vector DB, embed model, LLM, PRS score, and KVQ links; add per-UC AB eval runner and a live KVQ stats + Claude-diagram page.

**Architecture:** Three components — (1) `pipeline/ab_evaluator.py` runs AB evals per use case via the running dashboard's `/api/query`; (2) new routes on `kvforge_portal.py` serve `/ab-eval/{uc_id}`, `/kvq`, and `/kvq/diagram`; (3) portal card HTML extended with 5 metadata rows populated from USE_CASES + live `/api/status`.

**Tech Stack:** FastAPI, httpx, fastembed (all existing), anthropic (new), pure-Python ROUGE-L.

---

## File Map

| File | Role |
|------|------|
| `pipeline/ab_evaluator.py` | New — CLI that queries dashboard, scores answers, writes JSON + HTML |
| `tests/test_ab_evaluator.py` | New — unit tests for pure functions |
| `kvforge_portal.py` | Modified — USE_CASES metadata, new routes, card UI |
| `tests/test_portal_enrichment.py` | New — FastAPI TestClient tests for new routes |
| `requirements_gpu.txt` | Add `anthropic` |

---

### Task 1: AB Evaluator — pure scoring functions

**Files:**
- Create: `pipeline/ab_evaluator.py`
- Create: `tests/test_ab_evaluator.py`

- [ ] **Step 1: Write failing tests for `_rouge_l` and `_cosine`**

Create `tests/test_ab_evaluator.py`:

```python
# tests/test_ab_evaluator.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rouge_l_exact_match():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("the cat sat", "the cat sat") == 1.0


def test_rouge_l_partial():
    from pipeline.ab_evaluator import _rouge_l
    score = _rouge_l("the cat sat on the mat", "the cat sat")
    assert 0.5 < score < 1.0


def test_rouge_l_empty():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("", "reference") == 0.0
    assert _rouge_l("hypothesis", "") == 0.0


def test_rouge_l_no_overlap():
    from pipeline.ab_evaluator import _rouge_l
    assert _rouge_l("foo bar baz", "one two three") == 0.0


def test_cosine_identical():
    from pipeline.ab_evaluator import _cosine
    v = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    from pipeline.ab_evaluator import _cosine
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(_cosine(a, b)) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/hemant/Downloads/RoPE/qdrant
python -m pytest tests/test_ab_evaluator.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'pipeline.ab_evaluator'`

- [ ] **Step 3: Create `pipeline/ab_evaluator.py` with pure functions**

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_ab_evaluator.py -v 2>&1 | tail -10
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/ab_evaluator.py tests/test_ab_evaluator.py
git commit -m "feat: add ab_evaluator pure scoring functions with tests"
```

---

### Task 2: AB Evaluator — eval loop, JSON output, HTML viewer, CLI

**Files:**
- Modify: `pipeline/ab_evaluator.py` (add `run_eval`, `generate_html`, `main`)
- Modify: `tests/test_ab_evaluator.py` (add eval loop + HTML tests)

- [ ] **Step 1: Add tests for `run_eval` output schema and `generate_html`**

Append to `tests/test_ab_evaluator.py`:

```python
def test_run_eval_output_schema(tmp_path):
    """run_eval returns list of dicts with required keys."""
    from unittest.mock import patch, MagicMock
    import json
    from pipeline.ab_evaluator import run_eval

    # Minimal config
    cfg = {
        "embed_model": "BAAI/bge-small-en-v1.5",
        "faq_question_key": "question",
        "faq_answer_key": "answer",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))

    faqs = [{"question": "What is X?", "answer": "X is a thing."}]
    faqs_path = tmp_path / "faqs.json"
    faqs_path.write_text(json.dumps(faqs))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "answer_a": "X is a thing.",
        "answer_b": "X is something.",
        "mode_a": "parametric",
        "latency_a_ms": 100,
        "latency_b_ms": 500,
        "generation_a_ms": 100,
        "generation_b_ms": 450,
    }

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        results = run_eval(
            config_path=str(config_path),
            faqs_path=str(faqs_path),
            dashboard_url="http://localhost:9999",
            gemini_api_key="fake-key",
            max_samples=1,
        )

    assert len(results) == 1
    r = results[0]
    required = {"question", "ground_truth", "answer_a", "answer_b", "mode_a",
                "latency_a_ms", "latency_b_ms", "generation_a_ms", "generation_b_ms",
                "sem_sim_a", "sem_sim_b", "rouge_l_a", "rouge_l_b"}
    assert required.issubset(r.keys())
    assert 0.0 <= r["sem_sim_a"] <= 1.0
    assert 0.0 <= r["rouge_l_a"] <= 1.0


def test_run_eval_missing_faqs(tmp_path):
    """run_eval raises FileNotFoundError with helpful message when faqs.json absent."""
    import json
    from pipeline.ab_evaluator import run_eval

    cfg = {"embed_model": "BAAI/bge-small-en-v1.5",
           "faq_question_key": "question", "faq_answer_key": "answer"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    # faqs.json NOT created

    try:
        run_eval(str(config_path), str(tmp_path / "faqs.json"),
                 "http://localhost:9999", "", 1)
        assert False, "Should have raised"
    except FileNotFoundError as e:
        assert "run the pipeline first" in str(e)


def test_generate_html_contains_data():
    """generate_html produces valid HTML with const DATA embedded."""
    from pipeline.ab_evaluator import generate_html
    results = [{"question": "Q", "ground_truth": "A", "answer_a": "A", "answer_b": "B",
                "mode_a": "parametric", "latency_a_ms": 100, "latency_b_ms": 500,
                "generation_a_ms": 100, "generation_b_ms": 450,
                "sem_sim_a": 0.9, "sem_sim_b": 0.8, "rouge_l_a": 0.5, "rouge_l_b": 0.4}]
    html = generate_html(results, title="Test Eval")
    assert "const AB_DATA" in html
    assert "Test Eval" in html
    assert "<!DOCTYPE html>" in html
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_ab_evaluator.py::test_run_eval_output_schema -v 2>&1 | tail -5
```
Expected: `AttributeError: module 'pipeline.ab_evaluator' has no attribute 'run_eval'`

- [ ] **Step 3: Add `run_eval`, `generate_html`, and `main` to `pipeline/ab_evaluator.py`**

Append to the existing file after `_cosine`:

```python

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
    """Render a self-contained HTML viewer with results embedded as const DATA."""
    n = len(results)
    wins_a = sum(1 for r in results if r["sem_sim_a"] >= r["sem_sim_b"])
    wins_b = n - wins_a
    avg_lat_a = sum(r["latency_a_ms"] for r in results) / max(n, 1)
    avg_lat_b = sum(r["latency_b_ms"] for r in results) / max(n, 1)
    avg_sim_a = sum(r["sem_sim_a"] for r in results) / max(n, 1)
    avg_sim_b = sum(r["sem_sim_b"] for r in results) / max(n, 1)
    avg_rl_a = sum(r["rouge_l_a"] for r in results) / max(n, 1)
    avg_rl_b = sum(r["rouge_l_b"] for r in results) / max(n, 1)
    pct_a = wins_a / max(n, 1) * 100
    pct_b = wins_b / max(n, 1) * 100

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
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End</span><span class="stat-val">{avg_lat_a:.0f} ms</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val {'good' if avg_rl_a >= 0.4 else 'ok'}">{avg_rl_a:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Semantic Similarity</span><span class="stat-val {'good' if avg_sim_a >= 0.8 else 'ok'}">{avg_sim_a:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Questions Won</span><span class="stat-val">{wins_a} / {n} ({pct_a:.1f}%)</span></div>
  </div>
  <div class="stats-card b">
    <h3>Model B — Gemini 2.0 Flash (RAG)</h3>
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End</span><span class="stat-val">{avg_lat_b:.0f} ms</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val {'good' if avg_rl_b >= 0.4 else 'ok'}">{avg_rl_b:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Semantic Similarity</span><span class="stat-val {'good' if avg_sim_b >= 0.8 else 'ok'}">{avg_sim_b:.4f}</span></div>
    <div class="stat-row"><span class="stat-label">Questions Won</span><span class="stat-val">{wins_b} / {n} ({pct_b:.1f}%)</span></div>
  </div>
</div>
<div class="wins-bar">
  <div class="wins-bar-a" style="width:{pct_a:.1f}%">A {pct_a:.0f}%</div>
  <div class="wins-bar-b" style="width:{pct_b:.1f}%">B {pct_b:.0f}%</div>
</div>
<div class="controls">
  <input id="search" type="text" placeholder="Search questions…"/>
  <select id="filter">
    <option value="all">All</option>
    <option value="a">Model A wins</option>
    <option value="b">Model B wins</option>
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
    return true;
  }});
  rows.sort((a,b) => (a[sortCol] > b[sortCol] ? 1 : -1) * sortDir);
  document.getElementById('count').textContent = rows.length + ' rows';
  const winner = r => r.sem_sim_a >= r.sem_sim_b ? 'winner-a' : 'winner-b';
  document.getElementById('tbody').innerHTML = rows.map(r => `
    <tr class="${{winner(r)}}">
      <td class="q">${{trunc(r.question)}}</td>
      <td class="gt">${{trunc(r.ground_truth)}}</td>
      <td class="ans-a">${{trunc(r.answer_a)}}</td>
      <td class="ans-b">${{trunc(r.answer_b)}}</td>
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
    parser.add_argument("--dashboard-url", required=True, help="Running dashboard base URL e.g. http://localhost:8081")
    parser.add_argument("--gemini-api-key", default="", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--max-samples", type=int, default=200, help="Max FAQ questions to evaluate")
    args = parser.parse_args()

    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    config_path = Path(args.config)
    faqs_path = config_path.parent / "faqs.json"

    results = run_eval(
        config_path=str(config_path),
        faqs_path=str(faqs_path),
        dashboard_url=args.dashboard_url,
        gemini_api_key=gemini_key,
        max_samples=args.max_samples,
    )

    out_dir = config_path.parent
    json_out = out_dir / "ab_eval_results.json"
    html_out = out_dir / "ab_eval_viewer.html"
    uc_name = out_dir.name.replace("_", " ").title()

    json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Wrote {json_out}")

    html_out.write_text(generate_html(results, title=f"A/B Eval — {uc_name}"))
    print(f"Wrote {html_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all ab_evaluator tests — expect PASS**

```bash
python -m pytest tests/test_ab_evaluator.py -v 2>&1 | tail -15
```
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add pipeline/ab_evaluator.py tests/test_ab_evaluator.py
git commit -m "feat: add ab_evaluator — eval loop, HTML viewer, CLI"
```

---

### Task 3: Portal — USE_CASES metadata, /api/status PRS, description fixes

**Files:**
- Modify: `kvforge_portal.py` (lines 24–57 for USE_CASES, lines 68–88 for get_status)
- Create: `tests/test_portal_enrichment.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_portal_enrichment.py`:

```python
# tests/test_portal_enrichment.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient


def _make_portal_client():
    import kvforge_portal as portal
    return TestClient(portal.app)


def test_use_cases_have_required_fields():
    import kvforge_portal as portal
    required = {"id", "title", "subtitle", "description", "port", "color",
                "vectordb", "vectordb_url", "embed_model", "llm_model", "ab_eval_dir"}
    for uc in portal.USE_CASES:
        missing = required - uc.keys()
        assert not missing, f"{uc['id']} missing fields: {missing}"


def test_use_cases_vectordb_values():
    import kvforge_portal as portal
    uc_map = {uc["id"]: uc for uc in portal.USE_CASES}
    assert uc_map["uc1"]["vectordb"] == "Qdrant"
    assert uc_map["uc2"]["vectordb"] == "ChromaDB"
    assert uc_map["uc3"]["vectordb"] == "FAISS"
    assert uc_map["uc4"]["vectordb"] == "Qdrant"
    # Qdrant UCs use sentinel
    assert uc_map["uc1"]["vectordb_url"] == "qdrant"
    assert uc_map["uc4"]["vectordb_url"] == "qdrant"


def test_status_includes_prs():
    """GET /api/status returns prs field per UC."""
    import kvforge_portal as portal

    async def fake_get(url, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/health"):
            resp.json.return_value = {"status": "ok"}
        elif url.endswith("/api/version"):
            resp.json.return_value = {
                "phase": 3,
                "prs_history": [{"round": 1, "prs": 0.77}],
            }
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    with patch("kvforge_portal.httpx.AsyncClient", return_value=mock_client):
        client = _make_portal_client()
        r = client.get("/api/status")

    assert r.status_code == 200
    data = r.json()
    for uc_id, info in data.items():
        assert "prs" in info, f"{uc_id} missing prs field"
    assert data["uc1"]["prs"] == 0.77


def test_status_prs_null_when_no_history():
    """prs is null when prs_history is empty."""
    import kvforge_portal as portal

    async def fake_get(url, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/health"):
            resp.json.return_value = {"status": "ok"}
        elif url.endswith("/api/version"):
            resp.json.return_value = {"phase": 3, "prs_history": []}
        return resp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=fake_get)

    with patch("kvforge_portal.httpx.AsyncClient", return_value=mock_client):
        client = _make_portal_client()
        r = client.get("/api/status")

    data = r.json()
    assert data["uc1"]["prs"] is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_portal_enrichment.py::test_use_cases_have_required_fields -v 2>&1 | tail -5
```
Expected: `AssertionError: uc1 missing fields: {'vectordb', ...}`

- [ ] **Step 3: Update `USE_CASES` in `kvforge_portal.py`**

Replace the entire `USE_CASES` list (lines 24–57) with:

```python
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
```

Also update the header badge in `PORTAL_HTML` (near line 234). Change:
```
4 use-case corpora · A10G GPU · Qdrant vector store
```
to:
```
4 use-case corpora · A10G GPU · Qdrant · ChromaDB · FAISS
```

- [ ] **Step 4: Update `get_status` to return `prs`**

In `kvforge_portal.py`, replace the `get_status` function (lines 68–88) with:

```python
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
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
python -m pytest tests/test_portal_enrichment.py -v 2>&1 | tail -10
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add kvforge_portal.py tests/test_portal_enrichment.py
git commit -m "feat: add USE_CASES metadata and prs to /api/status"
```

---

### Task 4: Portal — /ab-eval/{uc_id} route

**Files:**
- Modify: `kvforge_portal.py` (add route after `get_status`)
- Modify: `tests/test_portal_enrichment.py` (add route tests)

- [ ] **Step 1: Add tests for /ab-eval route**

Append to `tests/test_portal_enrichment.py`:

```python
def test_ab_eval_unknown_uc():
    """Unknown uc_id returns 404 with 'Unknown use case' message."""
    client = _make_portal_client()
    r = client.get("/ab-eval/uc99")
    assert r.status_code == 404
    assert "Unknown use case" in r.text


def test_ab_eval_missing_file(tmp_path):
    """Known uc_id but no viewer file → 404 with generation instructions."""
    import kvforge_portal as portal
    original_dir = portal.USE_CASES[0]["ab_eval_dir"]
    portal.USE_CASES[0]["ab_eval_dir"] = str(tmp_path)  # no HTML file here
    try:
        client = _make_portal_client()
        r = client.get("/ab-eval/uc1")
        assert r.status_code == 404
        assert "ab_evaluator" in r.text
    finally:
        portal.USE_CASES[0]["ab_eval_dir"] = original_dir


def test_ab_eval_serves_html(tmp_path):
    """Known uc_id with viewer file → 200 HTML response."""
    import kvforge_portal as portal
    viewer = tmp_path / "ab_eval_viewer.html"
    viewer.write_text("<!DOCTYPE html><html><body>Test</body></html>")
    original_dir = portal.USE_CASES[0]["ab_eval_dir"]
    portal.USE_CASES[0]["ab_eval_dir"] = str(tmp_path)
    try:
        client = _make_portal_client()
        r = client.get("/ab-eval/uc1")
        assert r.status_code == 200
        assert "Test" in r.text
    finally:
        portal.USE_CASES[0]["ab_eval_dir"] = original_dir
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
python -m pytest tests/test_portal_enrichment.py::test_ab_eval_unknown_uc -v 2>&1 | tail -5
```
Expected: `404` or `FAILED` because route doesn't exist yet.

- [ ] **Step 3: Add the route to `kvforge_portal.py`**

`pathlib.Path` is already imported in `kvforge_portal.py`. Update the existing `from fastapi import FastAPI` line to also import `HTTPException`:
```python
from fastapi import FastAPI, HTTPException
```
(`FileResponse` is not needed — we use `HTMLResponse` which is already imported.)

Then add the route after `get_status`:

```python
# Build a lookup map for quick uc_id → USE_CASE resolution
_UC_MAP = {uc["id"]: uc for uc in USE_CASES}


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
```

- [ ] **Step 4: Run all portal tests — expect PASS**

```bash
python -m pytest tests/test_portal_enrichment.py -v 2>&1 | tail -10
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add kvforge_portal.py tests/test_portal_enrichment.py
git commit -m "feat: add /ab-eval/{uc_id} route to portal"
```

---

### Task 5: Portal — /kvq and /kvq/diagram routes

**Files:**
- Modify: `kvforge_portal.py` (add two routes + `anthropic` import)
- Modify: `requirements_gpu.txt`
- Modify: `tests/test_portal_enrichment.py`

- [ ] **Step 1: Add anthropic to requirements**

In `requirements_gpu.txt`, add after the last line:
```
anthropic>=0.30.0
```

- [ ] **Step 2: Add tests for /kvq routes**

Append to `tests/test_portal_enrichment.py`:

```python
def test_kvq_page_loads():
    """GET /kvq returns 200 HTML with spinner element."""
    client = _make_portal_client()
    r = client.get("/kvq")
    assert r.status_code == 200
    assert "kvq-diagram" in r.text  # spinner/diagram div id


def test_kvq_diagram_no_api_key():
    """GET /kvq/diagram with no API key returns 200 with unavailable message."""
    import os
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        client = _make_portal_client()
        r = client.get("/kvq/diagram")
    assert r.status_code == 200
    data = r.json()
    assert "html" in data
    assert "ANTHROPIC_API_KEY" in data["html"]


def test_kvq_diagram_api_error():
    """GET /kvq/diagram with API error returns 200 with unavailable message (no crash)."""
    import os
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}):
        with patch("kvforge_portal.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception("API error")
            client = _make_portal_client()
            r = client.get("/kvq/diagram")
    assert r.status_code == 200
    assert "html" in r.json()
```

- [ ] **Step 3: Run new tests to confirm they fail**

```bash
python -m pytest tests/test_portal_enrichment.py::test_kvq_page_loads -v 2>&1 | tail -5
```
Expected: `404` or `FAILED`

- [ ] **Step 4: Add /kvq routes to `kvforge_portal.py`**

Add `import os` and the anthropic import to `kvforge_portal.py`. Neither exists yet. Place after the existing `import time` line:
```python
import os
try:
    import anthropic as _anthropic_mod
    anthropic = _anthropic_mod
except ImportError:
    anthropic = None
```

Add routes after the `/ab-eval` route:

```python
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
  <div id="kvq-diagram"><span class="spinner"></span> Generating diagram…</div>
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
  const rows = await Promise.all(UCS.map(async uc => {
    try {
      const r = await fetch(`http://${window.location.hostname}:${uc.port}/api/stats`);
      if (!r.ok) throw new Error('offline');
      const d = await r.json();
      const v = d.version || {};
      const prs_hist = v.prs_history || [];
      const prs = prs_hist.length ? prs_hist[prs_hist.length-1].prs : null;
      return `<tr>
        <td>${uc.title}</td>
        <td>${phaseHtml(v.phase)}</td>
        <td>${prsHtml(prs)}</td>
        <td>${tierBarHtml(d.tier_counts)}</td>
        <td>${d.total_chunks ?? '—'}</td>
        <td style="color:#22c55e">online</td>
      </tr>`;
    } catch {
      return `<tr>
        <td>${uc.title}</td>
        <td colspan="4" style="color:#6b7280">—</td>
        <td style="color:#ef4444">offline</td>
      </tr>`;
    }
  }));
  document.getElementById('stats-tbody').innerHTML = rows.join('');
}

async function loadDiagram() {
  try {
    const r = await fetch('/kvq/diagram');
    const d = await r.json();
    document.getElementById('kvq-diagram').innerHTML = d.html;
  } catch(e) {
    document.getElementById('kvq-diagram').innerHTML = '<p style="color:#6b7280">Diagram unavailable</p>';
  }
}

refreshStats();
setInterval(refreshStats, 10000);
loadDiagram();
</script>
</body>
</html>"""


@app.get("/kvq", response_class=HTMLResponse)
async def kvq_page():
    """KVQ live stats page with Claude-generated architecture diagram."""
    return HTMLResponse(KVQ_HTML)


@app.get("/kvq/diagram")
async def kvq_diagram():
    """Call Claude API to generate KVQ architecture diagram HTML snippet."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
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
```

- [ ] **Step 5: Run all portal tests — expect PASS**

```bash
python -m pytest tests/test_portal_enrichment.py -v 2>&1 | tail -15
```
Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add kvforge_portal.py requirements_gpu.txt tests/test_portal_enrichment.py
git commit -m "feat: add /kvq page and /kvq/diagram route with Claude diagram generation"
```

---

### Task 6: Portal — card UI enrichment (meta rows)

**Files:**
- Modify: `kvforge_portal.py` (update PORTAL_HTML card template + CSS + JS)

- [ ] **Step 1: Add CSS for `.card-meta` to the `<style>` block in `PORTAL_HTML`**

In `kvforge_portal.py`, find the `.footer {` CSS block (around line 205) and insert before it:

```css
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
```

- [ ] **Step 2: Update the card template to include `.card-meta`**

Find the card template section (the `"".join(...)` block around lines 245–258). Replace it with:

```python
"".join(f"""
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
""" for uc in USE_CASES)
```

- [ ] **Step 3: Update the JavaScript in `PORTAL_HTML` to wire up vectordb links and PRS**

Find the `<script>` block (around lines 265–294). Replace it entirely with:

```javascript
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
```

- [ ] **Step 4: Smoke-test the portal locally**

```bash
cd /Users/hemant/Downloads/RoPE/qdrant
python3 kvforge_portal.py --port 8080 &
sleep 2
curl -s http://localhost:8080/ | grep -c "card-meta"
# Expected: 4 (one per use case)
curl -s http://localhost:8080/api/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(list(d.values())[0].keys())"
# Expected: dict_keys(['status', 'phase', 'prs'])
kill %1
```

- [ ] **Step 5: Run all tests**

```bash
python -m pytest tests/test_portal_enrichment.py tests/test_ab_evaluator.py -v 2>&1 | tail -15
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add kvforge_portal.py
git commit -m "feat: add card-meta rows with vectordb, model, PRS, and KVQ links"
```

---

### Task 7: Deploy to EC2, install anthropic, run AB eval for all 4 UCs

**Files:**
- EC2: `~/kvforge/kvforge_portal.py`, `~/kvforge/pipeline/ab_evaluator.py`, `~/kvforge/requirements_gpu.txt`

- [ ] **Step 1: Push to GitHub**

```bash
git push smartqdrant smartqdrant-main
```

- [ ] **Step 2: SCP changed files to EC2 (two separate commands)**

```bash
scp -i /Users/hemant/Downloads/RoPE/g5.x.pem \
  /Users/hemant/Downloads/RoPE/qdrant/pipeline/ab_evaluator.py \
  ubuntu@13.221.47.200:/home/ubuntu/kvforge/pipeline/ab_evaluator.py

scp -i /Users/hemant/Downloads/RoPE/g5.x.pem \
  /Users/hemant/Downloads/RoPE/qdrant/kvforge_portal.py \
  ubuntu@13.221.47.200:/home/ubuntu/kvforge/kvforge_portal.py
```

- [ ] **Step 3: Install anthropic on EC2**

```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 \
  "/home/ubuntu/qdrant/venv/bin/pip install 'anthropic>=0.30.0' -q && echo installed"
```
Expected: `installed`

- [ ] **Step 4: Restart portal on EC2**

```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  pkill -f kvforge_portal || true
  sleep 1
  cd ~/kvforge
  nohup /home/ubuntu/qdrant/venv/bin/python3 kvforge_portal.py --port 8080 >> logs/portal.log 2>&1 &
  sleep 3
  curl -s http://localhost:8080/api/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print({k: v[\"prs\"] for k,v in d.items()})'
"
```
Expected: `{'uc1': 0.7662, 'uc2': 0.8867, 'uc3': 0.8775, 'uc4': None}`

- [ ] **Step 5: Run AB eval for UC1 on EC2 (sample 50 questions)**

```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  cd ~/kvforge
  /home/ubuntu/qdrant/venv/bin/python3 -m pipeline.ab_evaluator \
    --config examples/usecase1_customer_support/config.json \
    --dashboard-url http://localhost:8081 \
    --gemini-api-key \$GEMINI_API_KEY \
    --max-samples 50 && echo UC1_DONE
"
```

- [ ] **Step 6: Verify UC1 output then run UC2, UC3, UC4**

After UC1 succeeds:
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  ls -lh ~/kvforge/examples/usecase1_customer_support/ab_eval_results.json
  ls -lh ~/kvforge/examples/usecase1_customer_support/ab_eval_viewer.html
"
```

Then run UC2, UC3, UC4:
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  cd ~/kvforge
  /home/ubuntu/qdrant/venv/bin/python3 -m pipeline.ab_evaluator \
    --config examples/usecase2_pubmedqa/config.json \
    --dashboard-url http://localhost:8082 \
    --gemini-api-key \$GEMINI_API_KEY \
    --max-samples 50 && echo UC2_DONE

  /home/ubuntu/qdrant/venv/bin/python3 -m pipeline.ab_evaluator \
    --config examples/usecase3_squad/config.json \
    --dashboard-url http://localhost:8083 \
    --gemini-api-key \$GEMINI_API_KEY \
    --max-samples 50 && echo UC3_DONE

  /home/ubuntu/qdrant/venv/bin/python3 -m pipeline.ab_evaluator \
    --config examples/usecase4_bedrock_userguide/config.json \
    --dashboard-url http://localhost:8084 \
    --gemini-api-key \$GEMINI_API_KEY \
    --max-samples 50 && echo UC4_DONE
"
```

- [ ] **Step 7: Final verification**

```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  for uc in usecase1_customer_support usecase2_pubmedqa usecase3_squad usecase4_bedrock_userguide; do
    f=~/kvforge/examples/\$uc/ab_eval_viewer.html
    [ -f \$f ] && echo \"\$uc: OK\" || echo \"\$uc: MISSING\"
  done
  curl -s http://localhost:8080/ab-eval/uc1 | head -2
  curl -s http://localhost:8080/kvq | grep -c 'kvq-diagram'
"
```
Expected: all 4 OKs, `<!DOCTYPE html>` for ab-eval, `1` for kvq diagram div count.
