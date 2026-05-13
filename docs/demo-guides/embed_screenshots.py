#!/usr/bin/env python3
"""
Regenerate all 8 demo-guide HTML files with real embedded screenshots.
Run from repo root:  python docs/demo-guides/embed_screenshots.py
"""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SHOTS = ROOT / "tests" / "screenshots"
GUIDES = ROOT / "docs" / "demo-guides"


def b64(name: str) -> str:
    """Return data-URI for a screenshot file.

    Accepts:
    - bare slug like '01_login_page'  → looks for demo_guide__<slug>.png
    - full walkthrough name           → looks for <name>.png directly
    """
    # Try exact filename first (for full walkthrough_ names passed as slug)
    exact = SHOTS / f"{name}.png"
    if exact.exists():
        return "data:image/png;base64," + base64.b64encode(exact.read_bytes()).decode()
    # Fall back to prefixed search for short slugs
    for prefix in ("demo_guide__", "walkthrough_"):
        p = SHOTS / f"{prefix}{name}.png"
        if p.exists():
            return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return ""


def img(slug: str, caption: str) -> str:
    uri = b64(slug)
    if not uri:
        return f'<div class="screenshot-missing">📸 {caption}</div>'
    return (
        f'<div class="screenshot-wrap">'
        f'<img src="{uri}" alt="{caption}" loading="lazy">'
        f'<div class="screenshot-cap">{caption}</div>'
        f'</div>'
    )


# ── Shared CSS (same across all guides) ───────────────────────────────────────
CSS = """
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#d4d4d4;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:0 0 80px}
header{background:#111;border-bottom:1px solid #1e1e1e;padding:24px 40px;position:sticky;top:0;z-index:10}
.header-top{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.logo{width:28px;height:28px;border-radius:7px;background:#0052CC;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:900;color:#fff;flex-shrink:0}
h1{color:#e0e0e0;font-size:18px;font-weight:700}
.demo-badge{font-size:10px;font-weight:700;padding:3px 9px;border-radius:3px;text-transform:uppercase;letter-spacing:.05em}
.header-sub{color:#555;font-size:12px;margin-top:6px}
.tech-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.tech-pill{font-size:10px;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:4px;padding:3px 9px;color:#888}
.tech-pill span{color:#4ec9b0;font-weight:700}
.guide{max-width:1000px;margin:0 auto;padding:0 40px}
.step{margin-top:40px;border:1px solid #1e1e1e;border-radius:12px;overflow:hidden}
.step-header{padding:16px 20px;background:#111;border-bottom:1px solid #1e1e1e;display:flex;align-items:flex-start;gap:14px}
.step-num{width:32px;height:32px;border-radius:50%;background:rgba(78,201,176,.15);border:1.5px solid rgba(78,201,176,.35);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#4ec9b0;flex-shrink:0}
.step-title{font-size:14px;font-weight:700;color:#e0e0e0;margin-bottom:3px}
.step-desc{font-size:12px;color:#888;line-height:1.6}
.instructions{padding:16px 20px;background:#0e0e0e;border-bottom:1px solid #1a1a1a}
.instr-row{display:flex;gap:10px;align-items:flex-start;margin-bottom:10px}
.instr-row:last-child{margin-bottom:0}
.instr-icon{font-size:14px;flex-shrink:0;margin-top:1px}
.instr-text{font-size:13px;color:#c8c8c8;line-height:1.6}
.instr-text code{background:#1e1e1e;border:1px solid #2a2a2a;border-radius:3px;padding:1px 6px;font-family:monospace;font-size:12px;color:#4ec9b0}
.instr-text strong{color:#e0e0e0}
.screenshot-wrap{position:relative;background:#000}
.screenshot-wrap img{width:100%;display:block}
.screenshot-cap{padding:10px 20px;background:#0a0a0a;font-size:11px;color:#555;border-top:1px solid #1a1a1a;font-style:italic}
.screenshot-missing{color:#444;padding:16px 20px;font-style:italic;background:#0a0a0a;border-top:1px solid #1a1a1a}
.todo-box{margin:0 20px 16px;padding:14px 18px;background:rgba(255,153,0,.06);border:1px solid rgba(255,153,0,.25);border-radius:8px;border-left:3px solid #FF991F}
.todo-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#FF991F;margin-bottom:6px}
.todo-item{display:flex;gap:8px;align-items:flex-start;margin-bottom:5px;font-size:12px;color:#b08030;line-height:1.5}
.todo-item:last-child{margin-bottom:0}
.todo-bullet{color:#FF991F;flex-shrink:0}
.note-box{margin:0 20px 16px;padding:12px 16px;background:rgba(156,220,254,.05);border:1px solid rgba(156,220,254,.2);border-radius:6px}
.note-box p{font-size:12px;color:#7a9ab8;line-height:1.6}
.note-box strong{color:#9cdcfe}
.cli-box{margin:0 20px 16px;background:#0a0a0a;border:1px solid #1a2a1a;border-radius:6px;padding:12px 16px}
.cli-box pre{font-family:monospace;font-size:12px;color:#4ec9b0;white-space:pre-wrap;line-height:1.7}
.cli-label{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#4ec9b0;margin-bottom:6px;opacity:.7}
footer{margin:60px 40px 0;color:#333;font-size:12px;text-align:center}
"""

# ── Shared step blocks (HTML fragments) ───────────────────────────────────────

def step_signin():
    return f"""
  <div class="step">
    <div class="step-header">
      <div class="step-num">1</div>
      <div>
        <div class="step-title">Sign In to KVForge Studio</div>
        <div class="step-desc">Open Studio in your browser and sign in. The landing page is public; the Studio hub requires authentication.</div>
      </div>
    </div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">▶</span><div class="instr-text">Start the server: <code>python kvforge_portal.py --port 8080</code></div></div>
      <div class="instr-row"><span class="instr-icon">🌐</span><div class="instr-text">Navigate to <code>http://localhost:8080/auth/login</code> (or open <code>http://localhost:8080</code> and click <strong>Sign In</strong>).</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Enter your email and password, then click <strong>Sign In</strong>. You are redirected to the Studio Hub on success.</div></div>
    </div>
    {img("01_login_page", "Login page — all unauthenticated users land here when navigating to /studio/")}
  </div>"""


def step_gpu_connect():
    return f"""
  <div class="step">
    <div class="step-header">
      <div class="step-num">2</div>
      <div>
        <div class="step-title">Connect a Remote GPU</div>
        <div class="step-desc">KVForge Studio can run on your laptop while the GPU work happens on a remote EC2 instance. If no GPU is detected locally, the hub shows a warning banner.</div>
      </div>
    </div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">⚠️</span><div class="instr-text">If you see the <strong>No GPU Detected</strong> banner, click <strong>Connect GPU →</strong> to open the Remote GPU wizard.</div></div>
      <div class="instr-row"><span class="instr-icon">🔑</span><div class="instr-text">On the <strong>Connect Remote GPU</strong> page: enter the <strong>Hostname / IP</strong>, <strong>SSH User</strong> (e.g. <code>ubuntu</code>), and upload your <strong>.pem key file</strong>.</div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">Click <strong>Test Connection →</strong>. Studio SSHes into the host, runs <code>nvidia-smi</code>, and shows CUDA driver info. Click <strong>Install Dependencies →</strong> to set up the Python environment, then <strong>Save Profile</strong>.</div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">If your GPU is local and <code>nvidia-smi</code> is in your PATH, the hub will detect it automatically and skip this step.</div></div>
    </div>
    {img("11_hub_gpu_warning", "Studio Hub showing 'No GPU Detected' warning — click 'Connect GPU' to launch the SSH wizard")}
    {img("04_gpu_connect", "Remote GPU Connection wizard — enter host details and upload your EC2 PEM key")}
    {img("05_gpu_connect_filled", "GPU connect form filled in with hostname, SSH user, and display name")}
  </div>"""


def step_hub():
    return f"""
  <div class="step">
    <div class="step-header">
      <div class="step-num">3</div>
      <div>
        <div class="step-title">Open the Studio Hub</div>
        <div class="step-desc">After sign-in you land on the Studio Hub — the main dashboard for managing all your use cases.</div>
      </div>
    </div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🗂</span><div class="instr-text">The hub shows all registered use cases as cards. A fresh installation shows an empty state with a <strong>+ Create Your First Use Case</strong> button.</div></div>
      <div class="instr-row"><span class="instr-icon">📊</span><div class="instr-text">The top system status bar shows live GPU utilisation, Qdrant connection status, active jobs, and overall system health.</div></div>
    </div>
    {img("03_hub_empty", "Studio Hub — empty state on first login, with system status bar and 'Create Your First Use Case' CTA")}
  </div>"""


def step_wizard_open():
    return f"""
  <div class="step">
    <div class="step-header">
      <div class="step-num">4</div>
      <div>
        <div class="step-title">Open the New Use Case Wizard</div>
        <div class="step-desc">The 6-step wizard configures data source, vector DB, model, GPU, training params, and launches the pipeline.</div>
      </div>
    </div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click <strong>+ New Use Case</strong> in the top-right of the hub, or click the dashed <strong>+ Create</strong> card in the UC grid.</div></div>
      <div class="instr-row"><span class="instr-icon">📋</span><div class="instr-text">The wizard opens at <strong>Step 1 — Data Source</strong>. You will see source-type cards: HuggingFace, PDF / File Upload, Connector, and Existing Corpus.</div></div>
    </div>
    {img("08_wizard_step1", "New Use Case Wizard — Step 1: Data Source, showing source-type selection cards")}
  </div>"""


def step_monitor():
    return f"""
  <div class="step">
    <div class="step-header">
      <div class="step-num">9</div>
      <div>
        <div class="step-title">Monitor Pipeline Progress</div>
        <div class="step-desc">After launch the wizard streams live logs. The UC detail page shows phase, PRS history, and metrics once the pipeline completes.</div>
      </div>
    </div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📊</span><div class="instr-text">Watch the 5 pipeline stages: <strong>Indexing</strong> → <strong>KV Cache</strong> → <strong>FAQ Gen</strong> → <strong>LoRA Train</strong> → <strong>PRS Eval</strong>. Each turns green when complete.</div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">When all stages are green, navigate to the UC detail page to see the PRS score and start querying.</div></div>
      <div class="instr-row"><span class="instr-icon">📈</span><div class="instr-text">The UC detail page shows Phase (1 → 3), current PRS, latency comparison (KVForge vs baseline), and sync history.</div></div>
    </div>
    {img("walkthrough_monitoring__01_studio_hub_overview", "Studio Hub showing a running use case with live status indicators")}
    {img("walkthrough_monitoring__02_uc_detail_page", "UC detail page — phase badge, PRS history chart, and pipeline log stream")}
  </div>"""


# ── Per-use-case guide builders ───────────────────────────────────────────────

def build_customer_support():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Support RAG — KVForge Setup Guide</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo">K</div>
    <h1>Customer Support RAG Demo</h1>
    <span class="demo-badge" style="background:#1C3353;color:#579DFF">CUSTOMER SUPPORT</span>
  </div>
  <div class="header-sub">KVForge Studio — Step-by-step setup guide</div>
  <div class="tech-row">
    <div class="tech-pill">Dataset: <span>bitext/Bitext-customer-support-llm-chatbot-training-dataset</span></div>
    <div class="tech-pill">VectorDB: <span>Qdrant</span></div>
    <div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>
    <div class="tech-pill">Embedding: <span>BAAI/bge-small-en-v1.5</span></div>
    <div class="tech-pill">Phases: <span>1 → 3</span></div>
  </div>
</header>
<div class="guide">

  <div class="step">
    <div class="step-header"><div class="step-num">0</div><div>
      <div class="step-title">What this demo does</div>
      <div class="step-desc">Builds a customer support RAG system from a HuggingFace intent-labeled Q&amp;A dataset. Advances through all 3 KVForge phases: standard RAG → KV-cache injection → confidence-gated parametric answering.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📦</span><div class="instr-text"><strong>Dataset:</strong> <code>bitext/Bitext-customer-support-llm-chatbot-training-dataset</code> — 26,872 intent-labeled support exchanges across 27 intents.</div></div>
      <div class="instr-row"><span class="instr-icon">🎯</span><div class="instr-text"><strong>Goal:</strong> Answer queries like <em>"How do I cancel my subscription?"</em> and <em>"Where is my order?"</em> using indexed support history, improving after LoRA fine-tuning.</div></div>
      <div class="instr-row"><span class="instr-icon">⏱</span><div class="instr-text"><strong>Time:</strong> 5 min setup · 10 min indexing (A10G) · 20–40 min LoRA training</div></div>
    </div>
  </div>

  {step_signin()}
  {step_gpu_connect()}
  {step_hub()}
  {step_wizard_open()}

  <div class="step">
    <div class="step-header"><div class="step-num">5</div><div>
      <div class="step-title">Wizard Step 1 — Select HuggingFace Data Source</div>
      <div class="step-desc">Select the HuggingFace source card and configure the customer support dataset.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click the <strong>HuggingFace 🤗</strong> source card — it highlights blue when selected.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Dataset ID:</strong> <code>bitext/Bitext-customer-support-llm-chatbot-training-dataset</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Split:</strong> <code>train</code> &nbsp;|&nbsp; <strong>Text Column:</strong> <code>instruction</code> &nbsp;|&nbsp; <strong>Max Rows:</strong> <code>5000</code> (full dataset: 26,872)</div></div>
      <div class="instr-row"><span class="instr-icon">➡</span><div class="instr-text">Click <strong>Next</strong> to proceed to Vector DB configuration.</div></div>
    </div>
    {img("09_wizard_hf_selected", "Wizard Step 1 — HuggingFace source card selected with Dataset ID field visible")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">6</div><div>
      <div class="step-title">Wizard Step 2 — Configure Qdrant</div>
      <div class="step-desc">Select Qdrant as the vector store and configure the collection for customer support data.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🐳</span><div class="instr-text">Start Qdrant: <code>docker run -d --name qdrant -p 6333:6333 qdrant/qdrant</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>URL:</strong> <code>http://localhost:6333</code> &nbsp;|&nbsp; <strong>Collection:</strong> <code>customer_support</code> &nbsp;|&nbsp; <strong>Dimensions:</strong> <code>384</code></div></div>
      <div class="instr-row"><span class="instr-icon">🔵</span><div class="instr-text">Click <strong>Test Connection</strong> to verify Qdrant is reachable.</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">7</div><div>
      <div class="step-title">Wizard Steps 3–5 — Model, GPU, Training</div>
      <div class="step-desc">Select the LLM, assign a GPU, and set LoRA training parameters.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🤖</span><div class="instr-text"><strong>Model:</strong> Select <strong>Llama-3.2-3B-Instruct</strong> (4-bit). VRAM badge should show green ✓ on an A10G.</div></div>
      <div class="instr-row"><span class="instr-icon">🖥</span><div class="instr-text"><strong>GPU:</strong> Click any free GPU card (green border). On g5.xlarge you will see one NVIDIA A10G (24 GB).</div></div>
      <div class="instr-row"><span class="instr-icon">⚙</span><div class="instr-text"><strong>Training:</strong> Leave defaults — LoRA Rank <code>16</code>, LR <code>2e-4</code>, Epochs <code>3</code>. Set FAQ count to <code>50</code> with Cloud LLM (requires <code>GOOGLE_API_KEY</code> or <code>ANTHROPIC_API_KEY</code>).</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">8</div><div>
      <div class="step-title">Wizard Step 6 — Review &amp; Launch</div>
      <div class="step-desc">Verify the configuration summary and launch the pipeline.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👁</span><div class="instr-text">Confirm: Dataset = <code>bitext/…</code>, Collection = <code>customer_support</code>, Model = Llama-3.2-3B, GPU = A10G.</div></div>
      <div class="instr-row"><span class="instr-icon">🚀</span><div class="instr-text">Click <strong>Launch Pipeline</strong>. The wizard transitions to the live log stream.</div></div>
    </div>
  </div>

  {step_monitor()}

  <div class="step">
    <div class="step-header"><div class="step-num">10</div><div>
      <div class="step-title">Alternative: CLI Setup</div>
      <div class="step-desc">Set up the Customer Support demo entirely from the command line.</div>
    </div></div>
    <div class="cli-box">
      <div class="cli-label">Terminal</div>
      <pre>python kvforge.py init --name customer_support

python -m pipeline.kv_indexer \\
  --config datasource_customer_support.json \\
  index hf://bitext/Bitext-customer-support-llm-chatbot-training-dataset

python -m pipeline.sleep_faq_generator \\
  --config datasource_customer_support.json \\
  --output customer_support_faqs.json --count 50

python -m pipeline.lora_trainer \\
  --config datasource_customer_support.json \\
  --faqs customer_support_faqs.json

python -m pipeline.kv_inference \\
  --config datasource_customer_support.json \\
  "How do I cancel my subscription?"</pre>
    </div>
  </div>

</div>
<footer>KVForge Studio · Branch kvforge-demos · docs/demo-guides/customer-support-setup.html</footer>
</body></html>"""


def build_pubmedqa():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PubMedQA RAG — KVForge Setup Guide</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo">K</div>
    <h1>PubMedQA Biomedical RAG Demo</h1>
    <span class="demo-badge" style="background:#1a2e28;color:#4ec9b0">PUBMEDQA</span>
  </div>
  <div class="header-sub">KVForge Studio — Step-by-step setup guide</div>
  <div class="tech-row">
    <div class="tech-pill">Dataset: <span>qiaojin/PubMedQA (pqa_labeled)</span></div>
    <div class="tech-pill">VectorDB: <span>Qdrant</span></div>
    <div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>
    <div class="tech-pill">Embedding: <span>BAAI/bge-small-en-v1.5</span></div>
    <div class="tech-pill">Phases: <span>1 → 3</span></div>
  </div>
</header>
<div class="guide">

  <div class="step">
    <div class="step-header"><div class="step-num">0</div><div>
      <div class="step-title">What this demo does</div>
      <div class="step-desc">Builds a biomedical question-answering RAG system from the PubMedQA dataset. 1,000 expert-annotated PubMed research questions with long-form answers derived from abstracts. The model learns to answer yes/no/maybe research questions with clinical evidence.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📦</span><div class="instr-text"><strong>Dataset:</strong> <code>qiaojin/PubMedQA</code> config <code>pqa_labeled</code> — 1,000 expert-labeled biomedical Q&amp;A pairs.</div></div>
      <div class="instr-row"><span class="instr-icon">🎯</span><div class="instr-text"><strong>Goal:</strong> Answer queries like <em>"Does aspirin reduce cardiovascular mortality?"</em> with yes/no/maybe + evidence from PubMed abstracts.</div></div>
      <div class="instr-row"><span class="instr-icon">⏱</span><div class="instr-text"><strong>Time:</strong> 5 min setup · 5 min indexing (small dataset) · 20–30 min LoRA training</div></div>
    </div>
  </div>

  {step_signin()}
  {step_gpu_connect()}
  {step_hub()}
  {step_wizard_open()}

  <div class="step">
    <div class="step-header"><div class="step-num">5</div><div>
      <div class="step-title">Wizard Step 1 — Select HuggingFace Data Source</div>
      <div class="step-desc">Select HuggingFace source and configure the PubMedQA dataset.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click the <strong>HuggingFace 🤗</strong> source card.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Dataset ID:</strong> <code>qiaojin/PubMedQA</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Config:</strong> <code>pqa_labeled</code> &nbsp;|&nbsp; <strong>Split:</strong> <code>train</code> &nbsp;|&nbsp; <strong>Text Column:</strong> <code>question</code> &nbsp;|&nbsp; <strong>Max Rows:</strong> <code>1000</code></div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">PubMedQA has 8 dataset configs. The <code>pqa_labeled</code> config is the expert-annotated subset used for benchmarking.</div></div>
    </div>
    {img("09_wizard_hf_selected", "Wizard Step 1 — HuggingFace card selected, ready for Dataset ID entry")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">6</div><div>
      <div class="step-title">Wizard Step 2 — Configure Qdrant</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>URL:</strong> <code>http://localhost:6333</code> &nbsp;|&nbsp; <strong>Collection:</strong> <code>pubmedqa</code> &nbsp;|&nbsp; <strong>Dimensions:</strong> <code>384</code></div></div>
      <div class="instr-row"><span class="instr-icon">🔵</span><div class="instr-text">Click <strong>Test Connection</strong> to verify Qdrant is reachable.</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">7</div><div>
      <div class="step-title">Wizard Steps 3–5 — Model, GPU, Training</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🤖</span><div class="instr-text"><strong>Model:</strong> <strong>Llama-3.2-3B-Instruct</strong> (4-bit). For highest biomedical accuracy, a BioMistral-7B preset would be ideal — see TODO below.</div></div>
      <div class="instr-row"><span class="instr-icon">⚙</span><div class="instr-text"><strong>Training:</strong> LoRA Rank <code>16</code>, LR <code>2e-4</code>, Epochs <code>3</code>, FAQ count <code>30</code>. The dataset is small so training is fast.</div></div>
    </div>
    <div class="todo-box">
      <div class="todo-title">TODO — Domain Model Presets</div>
      <div class="todo-item"><span class="todo-bullet">▶</span>Add domain presets to the model step: selecting <strong>Biomedical</strong> domain automatically suggests <code>BioMistral-7B</code>, which is pre-trained on 1M+ biomedical texts and outperforms Llama on PubMedQA.</div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">8</div><div>
      <div class="step-title">Wizard Step 6 — Review &amp; Launch</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🚀</span><div class="instr-text">Confirm dataset = <code>qiaojin/PubMedQA</code>, collection = <code>pubmedqa</code>, then click <strong>Launch Pipeline</strong>.</div></div>
    </div>
  </div>

  {step_monitor()}

  <div class="step">
    <div class="step-header"><div class="step-num">10</div><div>
      <div class="step-title">Alternative: CLI Setup</div>
    </div></div>
    <div class="cli-box">
      <div class="cli-label">Terminal</div>
      <pre>python kvforge.py init --name pubmedqa

python -m pipeline.kv_indexer \\
  --config datasource_pubmedqa.json \\
  index hf://qiaojin/PubMedQA?config=pqa_labeled

python -m pipeline.sleep_faq_generator \\
  --config datasource_pubmedqa.json \\
  --output pubmedqa_faqs.json --count 30

python -m pipeline.lora_trainer \\
  --config datasource_pubmedqa.json \\
  --faqs pubmedqa_faqs.json

python -m pipeline.kv_inference \\
  --config datasource_pubmedqa.json \\
  "Does low-dose aspirin reduce cardiovascular mortality?"</pre>
    </div>
  </div>

</div>
<footer>KVForge Studio · Branch kvforge-demos · docs/demo-guides/pubmedqa-setup.html</footer>
</body></html>"""


def build_squad():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SQuAD Reading Comprehension — KVForge Setup Guide</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo">K</div>
    <h1>SQuAD Reading Comprehension Demo</h1>
    <span class="demo-badge" style="background:#2a1a3e;color:#c084fc">SQUAD</span>
  </div>
  <div class="header-sub">KVForge Studio — Step-by-step setup guide</div>
  <div class="tech-row">
    <div class="tech-pill">Dataset: <span>rajpurkar/squad</span></div>
    <div class="tech-pill">VectorDB: <span>Qdrant</span></div>
    <div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>
    <div class="tech-pill">Embedding: <span>BAAI/bge-small-en-v1.5</span></div>
    <div class="tech-pill">Phases: <span>1 → 3</span></div>
  </div>
</header>
<div class="guide">

  <div class="step">
    <div class="step-header"><div class="step-num">0</div><div>
      <div class="step-title">What this demo does</div>
      <div class="step-desc">Builds a reading comprehension RAG system using the Stanford Question Answering Dataset (SQuAD). 87,599 question–context–answer triples from Wikipedia articles. Tests the model's ability to extract precise spans from indexed passages.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📦</span><div class="instr-text"><strong>Dataset:</strong> <code>rajpurkar/squad</code> — 87,599 reading comprehension pairs from 536 Wikipedia articles.</div></div>
      <div class="instr-row"><span class="instr-icon">🎯</span><div class="instr-text"><strong>Goal:</strong> Answer factual questions like <em>"In what year was the Eiffel Tower built?"</em> by retrieving the right Wikipedia passage and extracting the answer span.</div></div>
      <div class="instr-row"><span class="instr-icon">⏱</span><div class="instr-text"><strong>Time:</strong> 5 min setup · 10 min indexing · 30 min LoRA training</div></div>
    </div>
  </div>

  {step_signin()}
  {step_gpu_connect()}
  {step_hub()}
  {step_wizard_open()}

  <div class="step">
    <div class="step-header"><div class="step-num">5</div><div>
      <div class="step-title">Wizard Step 1 — Select HuggingFace Data Source</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click the <strong>HuggingFace 🤗</strong> source card.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Dataset ID:</strong> <code>rajpurkar/squad</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Split:</strong> <code>train</code> &nbsp;|&nbsp; <strong>Text Column:</strong> <code>context</code> &nbsp;|&nbsp; <strong>Max Rows:</strong> <code>5000</code> (full: 87,599)</div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">Indexing the <code>context</code> column embeds the Wikipedia passage text. The <code>question</code> column is used for FAQ generation at training time.</div></div>
    </div>
    {img("09_wizard_hf_selected", "Wizard Step 1 — HuggingFace card selected")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">6</div><div>
      <div class="step-title">Wizard Step 2 — Configure Qdrant</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>URL:</strong> <code>http://localhost:6333</code> &nbsp;|&nbsp; <strong>Collection:</strong> <code>squad_rag</code> &nbsp;|&nbsp; <strong>Dimensions:</strong> <code>384</code></div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">7</div><div>
      <div class="step-title">Wizard Steps 3–5 — Model, GPU, Training</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🤖</span><div class="instr-text"><strong>Model:</strong> <strong>Llama-3.2-3B-Instruct</strong> (4-bit) — good factual recall for span extraction.</div></div>
      <div class="instr-row"><span class="instr-icon">⚙</span><div class="instr-text"><strong>Training:</strong> LoRA Rank <code>16</code>, LR <code>2e-4</code>, Epochs <code>3</code>, FAQ count <code>50</code>.</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">8</div><div>
      <div class="step-title">Wizard Step 6 — Review &amp; Launch</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🚀</span><div class="instr-text">Confirm dataset = <code>rajpurkar/squad</code>, collection = <code>squad_rag</code>, then click <strong>Launch Pipeline</strong>.</div></div>
    </div>
  </div>

  {step_monitor()}

  <div class="step">
    <div class="step-header"><div class="step-num">10</div><div>
      <div class="step-title">Alternative: CLI Setup</div>
    </div></div>
    <div class="cli-box">
      <div class="cli-label">Terminal</div>
      <pre>python kvforge.py init --name squad_rag

python -m pipeline.kv_indexer \\
  --config datasource_squad_rag.json \\
  index hf://rajpurkar/squad

python -m pipeline.sleep_faq_generator \\
  --config datasource_squad_rag.json \\
  --output squad_faqs.json --count 50

python -m pipeline.lora_trainer \\
  --config datasource_squad_rag.json \\
  --faqs squad_faqs.json

python -m pipeline.kv_inference \\
  --config datasource_squad_rag.json \\
  "In what year was the Eiffel Tower built?"</pre>
    </div>
  </div>

</div>
<footer>KVForge Studio · Branch kvforge-demos · docs/demo-guides/squad-setup.html</footer>
</body></html>"""


def build_bedrock():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AWS Bedrock User Guide RAG — KVForge Setup Guide</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo">K</div>
    <h1>AWS Bedrock User Guide Demo</h1>
    <span class="demo-badge" style="background:#2a1e10;color:#fb923c">BEDROCK PDF</span>
  </div>
  <div class="header-sub">KVForge Studio — Step-by-step setup guide</div>
  <div class="tech-row">
    <div class="tech-pill">Source: <span>PDF files (AWS Bedrock docs)</span></div>
    <div class="tech-pill">VectorDB: <span>Qdrant</span></div>
    <div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>
    <div class="tech-pill">Embedding: <span>BAAI/bge-small-en-v1.5</span></div>
    <div class="tech-pill">Phases: <span>1 → 3</span></div>
  </div>
</header>
<div class="guide">

  <div class="step">
    <div class="step-header"><div class="step-num">0</div><div>
      <div class="step-title">What this demo does</div>
      <div class="step-desc">Builds a RAG system over the AWS Bedrock User Guide PDFs. Demonstrates PDF ingestion, chunking, and semantic indexing of technical documentation. The model learns to answer Bedrock API questions from the official docs.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📄</span><div class="instr-text"><strong>Source:</strong> PDF files in <code>examples/usecase4_bedrock_userguide/data/</code> — AWS Bedrock User Guide chapters.</div></div>
      <div class="instr-row"><span class="instr-icon">🎯</span><div class="instr-text"><strong>Goal:</strong> Answer queries like <em>"How do I invoke a foundation model using the Bedrock API?"</em> or <em>"What models does Bedrock support?"</em></div></div>
      <div class="instr-row"><span class="instr-icon">⏱</span><div class="instr-text"><strong>Time:</strong> 5 min setup · 15 min indexing (PDF parsing) · 30 min LoRA training</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">0b</div><div>
      <div class="step-title">Prerequisites — Download the PDF Files</div>
      <div class="step-desc">Download the AWS Bedrock User Guide PDFs into the data directory before starting Studio.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📁</span><div class="instr-text">Create the data directory: <code>mkdir -p examples/usecase4_bedrock_userguide/data/</code></div></div>
      <div class="instr-row"><span class="instr-icon">⬇</span><div class="instr-text">Download the Bedrock User Guide PDF from <strong>docs.aws.amazon.com/bedrock/latest/userguide/bedrock-ug.pdf</strong> and place it in the <code>data/</code> folder.</div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">You can add multiple PDFs — the loader processes all <code>*.pdf</code> files in the directory.</div></div>
    </div>
  </div>

  {step_signin()}
  {step_gpu_connect()}
  {step_hub()}
  {step_wizard_open()}

  <div class="step">
    <div class="step-header"><div class="step-num">5</div><div>
      <div class="step-title">Wizard Step 1 — Select PDF / File Upload Source</div>
      <div class="step-desc">Select the PDF source type and point it to the Bedrock User Guide data directory.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click the <strong>PDF / File Upload</strong> source card.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Source Path:</strong> <code>examples/usecase4_bedrock_userguide/data/</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>Chunk Size:</strong> <code>512</code> tokens &nbsp;|&nbsp; <strong>Chunk Overlap:</strong> <code>64</code> tokens</div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">Smaller chunk sizes work better for technical documentation with dense information per paragraph.</div></div>
    </div>
    {img("10_wizard_pdf_selected", "Wizard Step 1 — PDF / File Upload source card selected")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">6</div><div>
      <div class="step-title">Wizard Step 2 — Configure Qdrant</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>URL:</strong> <code>http://localhost:6333</code> &nbsp;|&nbsp; <strong>Collection:</strong> <code>bedrock_userguide</code> &nbsp;|&nbsp; <strong>Dimensions:</strong> <code>384</code></div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">7</div><div>
      <div class="step-title">Wizard Steps 3–5 — Model, GPU, Training</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🤖</span><div class="instr-text"><strong>Model:</strong> <strong>Llama-3.2-3B-Instruct</strong> (4-bit) — strong instruction following for technical Q&amp;A.</div></div>
      <div class="instr-row"><span class="instr-icon">⚙</span><div class="instr-text"><strong>Training:</strong> LoRA Rank <code>16</code>, LR <code>2e-4</code>, Epochs <code>3</code>, FAQ count <code>50</code> (generated from the PDF content).</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">8</div><div>
      <div class="step-title">Wizard Step 6 — Review &amp; Launch</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🚀</span><div class="instr-text">Confirm source path = <code>examples/usecase4_bedrock_userguide/data/</code>, collection = <code>bedrock_userguide</code>, then click <strong>Launch Pipeline</strong>.</div></div>
    </div>
  </div>

  {step_monitor()}

  <div class="step">
    <div class="step-header"><div class="step-num">10</div><div>
      <div class="step-title">Alternative: CLI Setup</div>
    </div></div>
    <div class="cli-box">
      <div class="cli-label">Terminal</div>
      <pre>python kvforge.py init --name bedrock_userguide

python -m pipeline.kv_indexer \\
  --config datasource_bedrock_userguide.json \\
  index examples/usecase4_bedrock_userguide/data/

python -m pipeline.sleep_faq_generator \\
  --config datasource_bedrock_userguide.json \\
  --output bedrock_faqs.json --count 50

python -m pipeline.lora_trainer \\
  --config datasource_bedrock_userguide.json \\
  --faqs bedrock_faqs.json

python -m pipeline.kv_inference \\
  --config datasource_bedrock_userguide.json \\
  "How do I invoke a foundation model using the Bedrock runtime API?"</pre>
    </div>
  </div>

</div>
<footer>KVForge Studio · Branch kvforge-demos · docs/demo-guides/bedrock-userguide-setup.html</footer>
</body></html>"""


def build_connector_guide(title, badge_bg, badge_fg, badge_label,
                          pills, overview_desc, goal, dataset_desc,
                          collection, connector_type, connector_config_steps,
                          sample_query, cli_name, cli_index_cmd):
    """Generic builder for connector-based guides (Wikipedia, FDA, SEC, Sports)."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — KVForge Setup Guide</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <div class="header-top">
    <div class="logo">K</div>
    <h1>{title}</h1>
    <span class="demo-badge" style="background:{badge_bg};color:{badge_fg}">{badge_label}</span>
  </div>
  <div class="header-sub">KVForge Studio — Step-by-step setup guide</div>
  <div class="tech-row">{pills}</div>
</header>
<div class="guide">

  <div class="step">
    <div class="step-header"><div class="step-num">0</div><div>
      <div class="step-title">What this demo does</div>
      <div class="step-desc">{overview_desc}</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">📦</span><div class="instr-text"><strong>Data source:</strong> {dataset_desc}</div></div>
      <div class="instr-row"><span class="instr-icon">🎯</span><div class="instr-text"><strong>Goal:</strong> {goal}</div></div>
      <div class="instr-row"><span class="instr-icon">⏱</span><div class="instr-text"><strong>Time:</strong> 10 min setup (including connector) · 10–20 min indexing · 20–40 min LoRA training</div></div>
    </div>
  </div>

  {step_signin()}
  {step_gpu_connect()}
  {step_hub()}

  <div class="step">
    <div class="step-header"><div class="step-num">4</div><div>
      <div class="step-title">Set Up the {connector_type} Connector</div>
      <div class="step-desc">Before creating the use case, configure the data source connector in the Addons page. KVForge Studio will use this connector to pull and sync data.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🧩</span><div class="instr-text">Click <strong>Addons</strong> in the left sidebar to open the connector management page.</div></div>
      {connector_config_steps}
    </div>
    {img("walkthrough_create_connector_and_sync__01_connectors_page_initial", "Connectors page — empty state before adding the first connector")}
    {img("walkthrough_create_connector_and_sync__02_connector_added_in_list", "Connector added to the list and ready to sync")}
  </div>

  {step_wizard_open()}

  <div class="step">
    <div class="step-header"><div class="step-num">6</div><div>
      <div class="step-title">Wizard Step 1 — Select Connector as Data Source</div>
      <div class="step-desc">Connect the use case to the {connector_type} connector you just created.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">👆</span><div class="instr-text">Click the <strong>Connector</strong> source card in the wizard.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Select your <strong>{connector_type}</strong> connector from the dropdown. It pulls data from the configured source automatically.</div></div>
      <div class="instr-row"><span class="instr-icon">➡</span><div class="instr-text">Click <strong>Next</strong> to proceed to Vector DB configuration.</div></div>
    </div>
    {img("08_wizard_step1", "Wizard Step 1 — Connector source type selected")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">7</div><div>
      <div class="step-title">Wizard Step 2 — Configure Qdrant</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text"><strong>URL:</strong> <code>http://localhost:6333</code> &nbsp;|&nbsp; <strong>Collection:</strong> <code>{collection}</code> &nbsp;|&nbsp; <strong>Dimensions:</strong> <code>384</code></div></div>
      <div class="instr-row"><span class="instr-icon">🔵</span><div class="instr-text">Click <strong>Test Connection</strong> to verify Qdrant is reachable.</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">8</div><div>
      <div class="step-title">Wizard Steps 3–5 — Model, GPU, Training</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🤖</span><div class="instr-text"><strong>Model:</strong> <strong>Llama-3.2-3B-Instruct</strong> (4-bit). Fits on a single A10G.</div></div>
      <div class="instr-row"><span class="instr-icon">🖥</span><div class="instr-text"><strong>GPU:</strong> Click any free GPU (green border).</div></div>
      <div class="instr-row"><span class="instr-icon">⚙</span><div class="instr-text"><strong>Training:</strong> LoRA Rank <code>16</code>, LR <code>2e-4</code>, Epochs <code>3</code>, FAQ count <code>50</code>.</div></div>
    </div>
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">9</div><div>
      <div class="step-title">Wizard Step 6 — Review &amp; Launch</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🚀</span><div class="instr-text">Confirm collection = <code>{collection}</code> and click <strong>Launch Pipeline</strong>. The connector sync runs first, then indexing starts automatically.</div></div>
    </div>
  </div>

  {step_monitor()}

  <div class="step">
    <div class="step-header"><div class="step-num">11</div><div>
      <div class="step-title">Monitor Sync History</div>
      <div class="step-desc">Connector syncs run on the schedule you configured. You can also trigger a manual sync from the Addons page at any time.</div>
    </div></div>
    <div class="instructions">
      <div class="instr-row"><span class="instr-icon">🔄</span><div class="instr-text">Click <strong>Addons</strong> in the sidebar → select your connector → click <strong>Sync Now</strong> to trigger an immediate sync.</div></div>
      <div class="instr-row"><span class="instr-icon">📋</span><div class="instr-text">The sync history table shows each sync run: timestamp, records fetched, chunks added/updated/deleted, and status.</div></div>
    </div>
    {img("walkthrough_create_connector_and_sync__03_sync_triggered", "Connector sync triggered — sync status updates in real time")}
    {img("walkthrough_monitoring__03_connectors_with_sync_history", "Monitoring view — connector with full sync history")}
  </div>

  <div class="step">
    <div class="step-header"><div class="step-num">12</div><div>
      <div class="step-title">Alternative: CLI Setup</div>
    </div></div>
    <div class="cli-box">
      <div class="cli-label">Terminal</div>
      <pre>python kvforge.py init --name {cli_name}

{cli_index_cmd}

python -m pipeline.sleep_faq_generator \\
  --config datasource_{cli_name}.json \\
  --output {cli_name}_faqs.json --count 50

python -m pipeline.lora_trainer \\
  --config datasource_{cli_name}.json \\
  --faqs {cli_name}_faqs.json

python -m pipeline.kv_inference \\
  --config datasource_{cli_name}.json \\
  "{sample_query}"</pre>
    </div>
  </div>

</div>
<footer>KVForge Studio · Branch kvforge-demos</footer>
</body></html>"""


def wikipedia_connector_steps():
    return """
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Click <strong>+ Add Connector</strong> → select <strong>Wikipedia API</strong> as the connector type.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Topics / Categories</strong>: e.g. <code>Machine learning, Natural language processing, Large language model</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Max Articles</strong>: <code>500</code> &nbsp;|&nbsp; <strong>Language</strong>: <code>en</code> &nbsp;|&nbsp; <strong>Sync Schedule</strong>: <code>Weekly</code></div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">Click <strong>Save</strong>. The connector appears in the list. Click <strong>Sync Now</strong> to pull the initial batch of articles.</div></div>"""


def fda_connector_steps():
    return """
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Click <strong>+ Add Connector</strong> → select <strong>openFDA API</strong> as the connector type.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Endpoint</strong>: <code>drug/label</code> &nbsp;|&nbsp; <strong>Search Query</strong>: leave blank to index all labels, or filter by e.g. <code>openfda.product_type:OTC</code></div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Max Records</strong>: <code>10000</code> &nbsp;|&nbsp; <strong>Sync Schedule</strong>: <code>Monthly</code> (FDA labels update infrequently).</div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">No API key is required for openFDA. Rate limit is 1,000 requests/day without a key — request a free key at <code>open.fda.gov/apis/authentication/</code> for higher limits.</div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">Click <strong>Save</strong>, then <strong>Sync Now</strong>.</div></div>"""


def sec_connector_steps():
    return """
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Click <strong>+ Add Connector</strong> → select <strong>SEC EDGAR</strong> as the connector type.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Form Types</strong>: <code>10-K, 10-Q</code> (annual and quarterly reports). Add <code>8-K</code> for current event filings.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Tickers / CIKs</strong>: e.g. <code>AAPL, MSFT, AMZN, GOOGL</code> &nbsp;|&nbsp; <strong>Date Range</strong>: last 2 years &nbsp;|&nbsp; <strong>Sync Schedule</strong>: <code>Quarterly</code></div></div>
      <div class="instr-row"><span class="instr-icon">ℹ</span><div class="instr-text">SEC EDGAR is a public API — no key required. Set the <code>User-Agent</code> header to your email per SEC guidelines: <code>company@example.com</code></div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">Click <strong>Save</strong>, then <strong>Sync Now</strong>.</div></div>"""


def sports_connector_steps():
    return """
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Click <strong>+ Add Connector</strong> → select <strong>Sports Data API</strong> as the connector type.</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Select <strong>Sports</strong>: e.g. <code>NFL, NBA, MLB</code> &nbsp;|&nbsp; Set <strong>Data Types</strong>: <code>Rosters, Team Stats, Game Results</code></div></div>
      <div class="instr-row"><span class="instr-icon">🔑</span><div class="instr-text">Enter your <strong>API Key</strong> from your sports data provider (e.g. SportsData.io, API-Sports, or ESPN API).</div></div>
      <div class="instr-row"><span class="instr-icon">✏</span><div class="instr-text">Set <strong>Sync Schedule</strong>: <code>Daily</code> (roster and stats change frequently during the season).</div></div>
      <div class="instr-row"><span class="instr-icon">✅</span><div class="instr-text">Click <strong>Save</strong>, then <strong>Sync Now</strong> to pull the initial dataset.</div></div>"""


# ── Write all 8 guides ─────────────────────────────────────────────────────────

def main():
    guides = {
        "customer-support-setup.html": build_customer_support(),
        "pubmedqa-setup.html": build_pubmedqa(),
        "squad-setup.html": build_squad(),
        "bedrock-userguide-setup.html": build_bedrock(),
        "wikipedia-setup.html": build_connector_guide(
            title="Wikipedia RAG Demo",
            badge_bg="#1a2a20", badge_fg="#86efac", badge_label="WIKIPEDIA",
            pills='<div class="tech-pill">Source: <span>Wikipedia API Connector</span></div>'
                  '<div class="tech-pill">VectorDB: <span>Qdrant</span></div>'
                  '<div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>'
                  '<div class="tech-pill">Phases: <span>1 → 3</span></div>',
            overview_desc="Builds a general-knowledge RAG system by syncing Wikipedia articles on configurable topics. The connector fetches and incrementally updates articles, enabling always-fresh knowledge retrieval.",
            goal='Answer open-domain factual questions like <em>"What is quantum entanglement?"</em> or <em>"Who founded OpenAI?"</em> using live Wikipedia content.',
            dataset_desc="Wikipedia REST API — articles for specified topics/categories, synced on a configurable schedule.",
            collection="wikipedia_rag",
            connector_type="Wikipedia",
            connector_config_steps=wikipedia_connector_steps(),
            sample_query="What is the transformer architecture in machine learning?",
            cli_name="wikipedia_rag",
            cli_index_cmd='python -m pipeline.kv_indexer \\\n  --config datasource_wikipedia_rag.json \\\n  index connector://wikipedia',
        ),
        "fda-drug-labels-setup.html": build_connector_guide(
            title="FDA Drug Labels RAG Demo",
            badge_bg="#1a2e28", badge_fg="#4ec9b0", badge_label="FDA DRUG LABELS",
            pills='<div class="tech-pill">Source: <span>openFDA API Connector</span></div>'
                  '<div class="tech-pill">VectorDB: <span>Qdrant</span></div>'
                  '<div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>'
                  '<div class="tech-pill">Phases: <span>1 → 3</span></div>',
            overview_desc="Builds a pharmaceutical RAG system using FDA drug label data from the openFDA API. Enables natural-language querying of drug indications, contraindications, dosage, and adverse reactions.",
            goal='Answer queries like <em>"What are the contraindications for ibuprofen?"</em> or <em>"What is the maximum daily dose of acetaminophen?"</em>',
            dataset_desc="openFDA drug/label endpoint — structured drug label data (indications, contraindications, dosage, warnings). Public API, no key required.",
            collection="fda_labels",
            connector_type="openFDA",
            connector_config_steps=fda_connector_steps(),
            sample_query="What are the contraindications for metformin?",
            cli_name="fda_labels",
            cli_index_cmd='python -m pipeline.kv_indexer \\\n  --config datasource_fda_labels.json \\\n  index connector://openfda',
        ),
        "sec-edgar-setup.html": build_connector_guide(
            title="SEC EDGAR Filings RAG Demo",
            badge_bg="#1a1a2e", badge_fg="#818cf8", badge_label="SEC EDGAR",
            pills='<div class="tech-pill">Source: <span>SEC EDGAR API Connector</span></div>'
                  '<div class="tech-pill">VectorDB: <span>Qdrant</span></div>'
                  '<div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>'
                  '<div class="tech-pill">Phases: <span>1 → 3</span></div>',
            overview_desc="Builds a financial document RAG system over SEC 10-K/10-Q filings. Enables semantic search across annual reports, earnings discussions, and risk factor sections for a configurable set of public companies.",
            goal='Answer investor questions like <em>"What were Apple\'s main risk factors in 2023?"</em> or <em>"What is Amazon\'s cloud revenue trend?"</em>',
            dataset_desc="SEC EDGAR EDGAR Full-Text Search API — 10-K, 10-Q, 8-K filings for specified ticker symbols. Public API, no key required.",
            collection="sec_filings",
            connector_type="SEC EDGAR",
            connector_config_steps=sec_connector_steps(),
            sample_query="What were Microsoft's primary risk factors in their most recent 10-K filing?",
            cli_name="sec_filings",
            cli_index_cmd='python -m pipeline.kv_indexer \\\n  --config datasource_sec_filings.json \\\n  index connector://sec-edgar',
        ),
        "sports-setup.html": build_connector_guide(
            title="Sports Roster &amp; Stats RAG Demo",
            badge_bg="#2a1a1a", badge_fg="#f87171", badge_label="SPORTS STATS",
            pills='<div class="tech-pill">Source: <span>Sports Data API Connector</span></div>'
                  '<div class="tech-pill">VectorDB: <span>Qdrant</span></div>'
                  '<div class="tech-pill">LLM: <span>Llama-3.2-3B-Instruct (4-bit)</span></div>'
                  '<div class="tech-pill">Sync: <span>Daily</span></div>'
                  '<div class="tech-pill">Phases: <span>1 → 3</span></div>',
            overview_desc="Builds a sports knowledge RAG system syncing roster, team stats, and game results daily. Enables natural-language queries over up-to-date sports data — ideal for fantasy sports assistants or sports journalism tools.",
            goal='Answer queries like <em>"Who are the top scorers for the Lakers this season?"</em> or <em>"What is the current NFL standings?"</em>',
            dataset_desc="Sports Data API (SportsData.io or equivalent) — rosters, game results, season stats for NFL/NBA/MLB/NHL. Requires a paid API key.",
            collection="sports_stats",
            connector_type="Sports Data",
            connector_config_steps=sports_connector_steps(),
            sample_query="What are the current standings in the NBA Western Conference?",
            cli_name="sports_stats",
            cli_index_cmd='python -m pipeline.kv_indexer \\\n  --config datasource_sports_stats.json \\\n  index connector://sports-data',
        ),
    }

    for filename, html in guides.items():
        path = GUIDES / filename
        path.write_text(html, encoding="utf-8")
        lines = html.count('\n')
        missing = html.count('<div class="screenshot-missing">')
        screenshots = html.count('data:image/png;base64')
        print(f"  ✓ {filename} ({lines} lines, {screenshots} screenshots, {missing} missing)")

    print(f"\nAll {len(guides)} guides written to {GUIDES}")


if __name__ == "__main__":
    main()
