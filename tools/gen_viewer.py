"""
gen_viewer.py — regenerate ab_eval_viewer.html with full stats panel
"""
import json, math

RESULTS = "ab_eval_results.json"
OUT     = "ab_eval_viewer.html"

with open(RESULTS) as f:
    data = json.load(f)

valid = [r for r in data if "error" not in r and r.get("answer_a") and r.get("answer_b")]
N = len(valid)

def avg(field):
    vals = [r[field] for r in valid if r.get(field) is not None]
    return sum(vals) / len(vals) if vals else 0

def wins(field_a, field_b):
    wa = sum(1 for r in valid if (r.get(field_a) or 0) > (r.get(field_b) or 0))
    wb = sum(1 for r in valid if (r.get(field_b) or 0) > (r.get(field_a) or 0))
    return wa, wb

win_a, win_b = wins("sem_sim_a", "sem_sim_b")

stats = {
    "lat_a":      f"{avg('latency_a_ms'):.0f} ms",
    "lat_b":      f"{avg('latency_b_ms'):.0f} ms",
    "gen_a":      f"{avg('generation_a_ms'):.0f} ms",
    "gen_b":      f"{avg('generation_b_ms'):.0f} ms",
    "ret_b":      f"{avg('retrieval_b_ms'):.0f} ms",
    "sim_a":      f"{avg('sem_sim_a'):.4f}",
    "sim_b":      f"{avg('sem_sim_b'):.4f}",
    "rouge_a":    f"{avg('rouge_l_a'):.4f}",
    "rouge_b":    f"{avg('rouge_l_b'):.4f}",
    "rsim_a":     f"{avg('ragas_sim_a'):.4f}",
    "rsim_b":     f"{avg('ragas_sim_b'):.4f}",
    "tok_a":      f"{avg('tokens_a'):.1f}",
    "tok_b":      f"{avg('tokens_b'):.1f}",
    "chunk_b":    f"{avg('chunk_tokens_b'):.0f} tokens",
    "win_a":      f"{win_a} / {N}",
    "win_b":      f"{win_b} / {N}",
    "win_a_pct":  f"{100*win_a/N:.1f}%",
    "win_b_pct":  f"{100*win_b/N:.1f}%",
    "N":          N,
}

# Embed JSON safely (escape </script>)
json_str = json.dumps(valid, ensure_ascii=False).replace("</script>", "<\\/script>")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A/B Eval Results — {N} Q&A pairs</title>
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
  .stats-card h3 {{ margin: 0 0 12px; font-size: 0.95em; letter-spacing: 0.03em; }}
  .stats-card.a h3 {{ color: #4d4; }}
  .stats-card.b h3 {{ color: #47b; }}
  .stat-section {{ font-size: 0.75em; color: #555; text-transform: uppercase; letter-spacing: 0.08em; margin: 10px 0 4px; padding-bottom: 2px; border-bottom: 1px solid #282828; }}
  .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.88em; }}
  .stat-label {{ color: #888; }}
  .stat-val {{ font-weight: bold; }}
  .stat-val.good {{ color: #4d4; }} .stat-val.ok {{ color: #fa7; }} .stat-val.info {{ color: #7af; }}
  .wins-bar {{ display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 10px 0 4px; }}
  .wins-bar-a {{ background: #2a4; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #fff; }}
  .wins-bar-b {{ background: #27a; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #fff; }}
  .wins-legend {{ display: flex; gap: 16px; font-size: 0.82em; color: #888; margin-bottom: 16px; }}
  .note {{ color: #555; font-size: 0.78em; margin-top: 8px; }}
</style>
</head>
<body>
<h1>A/B Eval Results</h1>
<div class="subtitle">{N} Q&A pairs &nbsp;|&nbsp; Model A: Llama 3.2 3B Parametric &nbsp;|&nbsp; Model B: Gemini Flash RAG</div>

<div class="stats-grid">
  <div class="stats-card a">
    <h3>Model A — Llama 3.2 3B (Parametric)</h3>
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End Latency</span><span class="stat-val">{stats["lat_a"]}</span></div>
    <div class="stat-row"><span class="stat-label">Avg Generation Time</span><span class="stat-val">{stats["gen_a"]}</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val good">{stats["rouge_a"]}</span></div>
    <div class="stat-row"><span class="stat-label">Ragas Semantic Sim (Gemini Embed)</span><span class="stat-val good">{stats["rsim_a"]}</span></div>
    <div class="stat-row"><span class="stat-label">Fastembed Semantic Sim</span><span class="stat-val">{stats["sim_a"]}</span></div>
    <div class="stat-section">Output</div>
    <div class="stat-row"><span class="stat-label">Avg Output Tokens</span><span class="stat-val info">{stats["tok_a"]}</span></div>
    <div class="stat-row"><span class="stat-label">Questions Won (sem_sim)</span><span class="stat-val">{stats["win_a"]} ({stats["win_a_pct"]})</span></div>
    <div class="stat-row"><span class="stat-label">Inference Mode</span><span class="stat-val good">Parametric (no retrieval)</span></div>
  </div>
  <div class="stats-card b">
    <h3>Model B — Gemini 2.0 Flash (RAG)</h3>
    <div class="stat-section">Latency</div>
    <div class="stat-row"><span class="stat-label">Avg End-to-End Latency</span><span class="stat-val">{stats["lat_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Avg Generation Time</span><span class="stat-val">{stats["gen_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Avg Retrieval Time (Qdrant)</span><span class="stat-val info">{stats["ret_b"]}</span></div>
    <div class="stat-section">Accuracy vs Ground Truth</div>
    <div class="stat-row"><span class="stat-label">ROUGE-L F1</span><span class="stat-val ok">{stats["rouge_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Ragas Semantic Sim (Gemini Embed)</span><span class="stat-val ok">{stats["rsim_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Fastembed Semantic Sim</span><span class="stat-val">{stats["sim_b"]}</span></div>
    <div class="stat-section">Output</div>
    <div class="stat-row"><span class="stat-label">Avg Output Tokens</span><span class="stat-val info">{stats["tok_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Avg Retrieved Chunk Tokens</span><span class="stat-val">{stats["chunk_b"]}</span></div>
    <div class="stat-row"><span class="stat-label">Questions Won (sem_sim)</span><span class="stat-val">{stats["win_b"]} ({stats["win_b_pct"]})</span></div>
    <div class="stat-row"><span class="stat-label">Inference Mode</span><span class="stat-val info">RAG (Qdrant + Gemini)</span></div>
  </div>
</div>

<div class="wins-bar">
  <div class="wins-bar-a" style="width:{100*win_a/N:.1f}%">A {stats["win_a_pct"]}</div>
  <div class="wins-bar-b" style="width:{100*win_b/N:.1f}%">B {stats["win_b_pct"]}</div>
</div>
<div class="wins-legend">
  <span><span style="color:#4d4">&#9632;</span> Model A wins: {stats["win_a"]} questions</span>
  <span><span style="color:#47b">&#9632;</span> Model B wins: {stats["win_b"]} questions</span>
  <span style="color:#555">Tie: {N - win_a - win_b} questions</span>
</div>
<div class="note">ROUGE-L F1: longest-common-subsequence overlap vs ground truth. Ragas Semantic Sim: cosine similarity via Gemini Embedding-001. Fastembed: BAAI/bge-small-en-v1.5 cosine sim.</div>

<div class="controls">
  <input type="text" id="search" placeholder="Search questions or answers…">
  <select id="filter">
    <option value="all">All results</option>
    <option value="a_wins">Model A wins (sim_a &gt; sim_b)</option>
    <option value="b_wins">Model B wins (sim_b &gt; sim_a)</option>
    <option value="high_b">High sim_b (&#8805; 0.90)</option>
    <option value="low_b">Low sim_b (&lt; 0.60)</option>
    <option value="a_rouge">Model A higher ROUGE-L</option>
    <option value="b_rouge">Model B higher ROUGE-L</option>
  </select>
  <span id="count"></span>
</div>

<table>
  <thead><tr>
    <th data-col="idx">#</th>
    <th data-col="question">Question</th>
    <th data-col="ground_truth">Ground Truth</th>
    <th data-col="answer_a">Answer A (Llama)</th>
    <th data-col="answer_b">Answer B (Gemini RAG)</th>
    <th data-col="sem_sim_a">sim_a</th>
    <th data-col="sem_sim_b">sim_b</th>
    <th data-col="rouge_l_a">rouge_a</th>
    <th data-col="rouge_l_b">rouge_b</th>
    <th data-col="ragas_sim_a">rsim_a</th>
    <th data-col="ragas_sim_b">rsim_b</th>
    <th data-col="latency_a_ms">lat_a</th>
    <th data-col="latency_b_ms">lat_b</th>
    <th data-col="tokens_a">tok_a</th>
    <th data-col="tokens_b">tok_b</th>
  </tr></thead>
  <tbody id="tbody"></tbody>
</table>

<script>
const DATA = {json_str};

function esc(s) {{
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}}
function simClass(v) {{
  if (v == null) return '';
  return v >= 0.85 ? 'sim-high' : v >= 0.65 ? 'sim-mid' : 'sim-low';
}}
function fmt(v, dec) {{
  if (v == null || v === '') return '—';
  if (typeof v === 'number') return dec != null ? v.toFixed(dec) : v;
  return v;
}}

let sortCol = null, sortDir = 1;

function filtered() {{
  var q = document.getElementById('search').value.toLowerCase();
  var f = document.getElementById('filter').value;
  return DATA.map((r,i) => ({{...r, idx: i+1}})).filter(r => {{
    if (q && !r.question.toLowerCase().includes(q) &&
           !(r.answer_a||'').toLowerCase().includes(q) &&
           !(r.answer_b||'').toLowerCase().includes(q)) return false;
    if (f === 'a_wins' && !((r.sem_sim_a||0) > (r.sem_sim_b||0))) return false;
    if (f === 'b_wins' && !((r.sem_sim_b||0) > (r.sem_sim_a||0))) return false;
    if (f === 'high_b' && !((r.sem_sim_b||0) >= 0.90)) return false;
    if (f === 'low_b'  && !((r.sem_sim_b||0) <  0.60)) return false;
    if (f === 'a_rouge' && !((r.rouge_l_a||0) > (r.rouge_l_b||0))) return false;
    if (f === 'b_rouge' && !((r.rouge_l_b||0) > (r.rouge_l_a||0))) return false;
    return true;
  }});
}}

function render() {{
  var rows = filtered();
  if (sortCol) {{
    rows.sort((a,b) => {{
      var av = a[sortCol], bv = b[sortCol];
      if (av == null) av = -Infinity;
      if (bv == null) bv = -Infinity;
      return sortDir * (av > bv ? 1 : av < bv ? -1 : 0);
    }});
  }}
  document.getElementById('count').textContent = rows.length + ' / ' + DATA.length + ' shown';
  var html = '';
  rows.forEach(function(r) {{
    var wa = (r.sem_sim_a||0) > (r.sem_sim_b||0);
    var wb = (r.sem_sim_b||0) > (r.sem_sim_a||0);
    var rowCls = wa ? 'winner-a' : wb ? 'winner-b' : '';
    html += '<tr class="' + rowCls + '">';
    html += '<td>' + r.idx + '</td>';
    html += '<td class="q"><div class="trunc">' + esc(r.question) + '</div></td>';
    html += '<td class="gt"><div class="trunc">' + esc(r.ground_truth) + '</div></td>';
    html += '<td class="ans-a"><div class="trunc">' + esc(r.answer_a) + '</div></td>';
    html += '<td class="ans-b"><div class="trunc">' + esc(r.answer_b) + '</div></td>';
    html += '<td class="' + simClass(r.sem_sim_a) + '">' + fmt(r.sem_sim_a,3) + '</td>';
    html += '<td class="' + simClass(r.sem_sim_b) + '">' + fmt(r.sem_sim_b,3) + '</td>';
    html += '<td class="' + simClass(r.rouge_l_a) + '">' + fmt(r.rouge_l_a,3) + '</td>';
    html += '<td class="' + simClass(r.rouge_l_b) + '">' + fmt(r.rouge_l_b,3) + '</td>';
    html += '<td class="' + simClass(r.ragas_sim_a) + '">' + fmt(r.ragas_sim_a,3) + '</td>';
    html += '<td class="' + simClass(r.ragas_sim_b) + '">' + fmt(r.ragas_sim_b,3) + '</td>';
    html += '<td>' + fmt(r.latency_a_ms,0) + 'ms</td>';
    html += '<td>' + fmt(r.latency_b_ms,0) + 'ms</td>';
    html += '<td>' + fmt(r.tokens_a,0) + '</td>';
    html += '<td>' + fmt(r.tokens_b,0) + '</td>';
    html += '</tr>';
  }});
  document.getElementById('tbody').innerHTML = html;
}}

// Sort on header click
document.querySelector('thead').addEventListener('click', function(e) {{
  var th = e.target.closest('th');
  if (!th) return;
  var col = th.dataset.col;
  if (sortCol === col) {{ sortDir *= -1; }} else {{ sortCol = col; sortDir = 1; }}
  document.querySelectorAll('th').forEach(function(t) {{ t.textContent = t.textContent.replace(/ [▲▼]$/,''); }});
  th.textContent += sortDir === 1 ? ' ▲' : ' ▼';
  render();
}});

// Expand on row click
document.getElementById('tbody').addEventListener('click', function(e) {{
  var el = e.target;
  while (el && el !== this) {{
    if (el.classList.contains('trunc')) {{ el.classList.toggle('expanded'); return; }}
    el = el.parentElement;
  }}
}});

document.getElementById('search').addEventListener('input', render);
document.getElementById('filter').addEventListener('change', render);

render();
</script>
</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Written {OUT} ({len(html)//1024}KB), {N} rows")
print(f"Stats: ROUGE-L A={stats['rouge_a']} B={stats['rouge_b']} | Ragas sim A={stats['rsim_a']} B={stats['rsim_b']}")
print(f"Wins: A={win_a} B={win_b} Tie={N-win_a-win_b}")
