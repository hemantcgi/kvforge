#!/usr/bin/env python3
"""
Generate tests/ui_walkthrough.html from walkthrough_manifest.json + screenshot PNGs.
Run after: pytest tests/ui/test_walkthroughs.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "tests" / "walkthrough_manifest.json"
SCREENSHOTS = ROOT / "tests" / "screenshots"
OUTPUT = ROOT / "tests" / "ui_walkthrough.html"

WORKFLOW_DESCRIPTIONS = {
    "walkthrough_login_to_dashboard": (
        "Login to Dashboard",
        "Full authentication flow: unauthenticated redirect → login page → "
        "invalid credentials error → successful login → studio hub."
    ),
    "walkthrough_create_connector_and_sync": (
        "Create Connector & Trigger Sync",
        "Connector management lifecycle: empty page → add GDrive connector → "
        "configure UC scope → trigger sync → sync history appears."
    ),
    "walkthrough_role_enforcement": (
        "Role-Based Access Control",
        "RBAC enforcement: admin accesses connectors page → logout → "
        "viewer gets 403 on connectors → viewer can still see the hub."
    ),
    "walkthrough_monitoring": (
        "Monitoring & Observability",
        "Monitoring flow: studio hub overview → UC detail page → "
        "connectors page with sync history → raw sync-runs API response."
    ),
}


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def step_indicator(current: int, total: int) -> str:
    dots = []
    for i in range(1, total + 1):
        if i == current:
            dots.append(f'<span class="sip sip-active">{i}</span>')
        elif i < current:
            dots.append(f'<span class="sip sip-done">{i}</span>')
        else:
            dots.append(f'<span class="sip sip-future">{i}</span>')
        if i < total:
            dots.append('<span class="sip-line"></span>')
    return '<div class="step-indicator">' + ''.join(dots) + '</div>'


def build_html(manifest: list[dict]) -> str:
    # Group by test name, preserving order
    by_test: dict[str, list[dict]] = {}
    for entry in manifest:
        by_test.setdefault(entry["test"], []).append(entry)

    sections = []
    for test_name, steps in by_test.items():
        title, intent = WORKFLOW_DESCRIPTIONS.get(test_name, (test_name, ""))
        sorted_steps = sorted(steps, key=lambda x: x["step"])
        total_steps = len(sorted_steps)

        step_html = []
        for s in sorted_steps:
            png = SCREENSHOTS / s["file"]
            if not png.exists():
                continue
            b64 = img_b64(png)
            action_html = ""
            if s.get("action"):
                action_html = f'<div class="step-action"><span class="sa-label">Action</span>{s["action"]}</div>'

            step_html.append(f"""
            <div class="step">
              <div class="step-meta">
                <div class="step-meta-top">
                  <span class="wf-badge">{title}</span>
                  {step_indicator(s['step'], total_steps)}
                </div>
                {action_html}
                <div class="step-observation">
                  <span class="so-label">Shows</span>{s['description']}
                </div>
              </div>
              <img src="data:image/png;base64,{b64}" alt="{s['slug']}" loading="lazy"/>
            </div>""")

        sections.append(f"""
        <section class="workflow" id="{test_name}">
          <div class="wf-header">
            <h2>{title}</h2>
            <span class="wf-count">{total_steps} steps</span>
          </div>
          <p class="intent">{intent}</p>
          <div class="steps">{''.join(step_html)}</div>
        </section>""")

    body = "\n".join(sections)
    total = sum(len(v) for v in by_test.values())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVForge Studio — UI Walkthrough Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #d4d4d4; font-family: -apple-system, BlinkMacSystemFont,
          'Segoe UI', sans-serif; padding: 0 0 80px; }}

  /* ── Header ── */
  header {{ background: #111; border-bottom: 1px solid #1e1e1e; padding: 24px 40px;
            position: sticky; top: 0; z-index: 10; }}
  header h1 {{ color: #4ec9b0; font-size: 20px; font-weight: 700; }}
  header p {{ color: #6a6a6a; font-size: 13px; margin-top: 4px; }}

  /* ── TOC ── */
  .toc {{ background: #111; border: 1px solid #1e1e1e; border-radius: 10px;
          margin: 32px 40px 0; padding: 20px 24px; }}
  .toc h3 {{ color: #9cdcfe; font-size: 11px; font-weight: 700; text-transform: uppercase;
             letter-spacing: .08em; margin-bottom: 12px; }}
  .toc-row {{ display: flex; align-items: center; justify-content: space-between;
              padding: 7px 0; border-bottom: 1px solid #1a1a1a; }}
  .toc-row:last-child {{ border-bottom: none; }}
  .toc-row a {{ color: #4ec9b0; font-size: 13px; text-decoration: none; font-weight: 600; }}
  .toc-row a:hover {{ color: #9cdcfe; }}
  .toc-desc {{ color: #555; font-size: 12px; max-width: 520px; line-height: 1.5; }}

  /* ── Workflow section ── */
  .workflow {{ margin: 48px 40px 0; }}
  .wf-header {{ display: flex; align-items: center; gap: 12px; padding-bottom: 12px;
                border-bottom: 1px solid #1e1e1e; margin-bottom: 12px; }}
  .wf-header h2 {{ color: #4ec9b0; font-size: 17px; font-weight: 700; }}
  .wf-count {{ font-size: 11px; color: #555; background: #1a1a1a; border: 1px solid #2a2a2a;
               border-radius: 4px; padding: 3px 9px; font-weight: 600; }}
  .intent {{ color: #6a6a6a; font-size: 13px; margin-bottom: 24px; line-height: 1.6; }}
  .steps {{ display: flex; flex-direction: column; gap: 32px; }}

  /* ── Step card ── */
  .step {{ background: #111; border: 1px solid #1e1e1e; border-radius: 12px; overflow: hidden; }}
  .step-meta {{ padding: 16px 20px; border-bottom: 1px solid #1e1e1e; }}
  .step-meta-top {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }}

  /* Workflow badge on each step */
  .wf-badge {{ font-size: 10px; font-weight: 700; color: #4ec9b0; background: rgba(78,201,176,.1);
               border: 1px solid rgba(78,201,176,.25); border-radius: 4px; padding: 3px 9px;
               letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; }}

  /* Step indicator dots */
  .step-indicator {{ display: flex; align-items: center; gap: 0; }}
  .sip {{ width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center;
          justify-content: center; font-size: 10px; font-weight: 700; flex-shrink: 0; }}
  .sip-done {{ background: rgba(78,201,176,.2); color: #4ec9b0; border: 1.5px solid rgba(78,201,176,.4); }}
  .sip-active {{ background: #4ec9b0; color: #0d0d0d; border: 1.5px solid #4ec9b0; }}
  .sip-future {{ background: transparent; color: #3a3a3a; border: 1.5px solid #2a2a2a; }}
  .sip-line {{ width: 20px; height: 1.5px; background: #2a2a2a; flex-shrink: 0; }}

  /* Action row */
  .step-action {{ display: flex; gap: 10px; align-items: flex-start; margin-bottom: 10px;
                  background: #0e1218; border: 1px solid #1e2a3a; border-radius: 6px;
                  padding: 10px 14px; color: #94a3b8; font-size: 13px; line-height: 1.5; }}
  .sa-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
               color: #9cdcfe; background: rgba(156,220,254,.1); border: 1px solid rgba(156,220,254,.2);
               border-radius: 3px; padding: 2px 7px; white-space: nowrap; margin-top: 1px; flex-shrink: 0; }}

  /* Observation row */
  .step-observation {{ display: flex; gap: 10px; align-items: flex-start;
                       color: #c8c8c8; font-size: 13px; line-height: 1.6; }}
  .so-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
               color: #c8c8c8; background: rgba(200,200,200,.08); border: 1px solid rgba(200,200,200,.15);
               border-radius: 3px; padding: 2px 7px; white-space: nowrap; margin-top: 2px; flex-shrink: 0; }}

  /* Screenshot */
  .step img {{ width: 100%; display: block; }}

  footer {{ margin: 60px 40px 0; color: #333; font-size: 12px; }}

  @media (max-width: 700px) {{
    .workflow, .toc {{ margin-left: 16px; margin-right: 16px; }}
    header {{ padding: 16px 20px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>KVForge Studio — UI Walkthrough Report</h1>
  <p>{len(by_test)} workflows · {total} screenshots · Playwright headless Chromium</p>
</header>

<div class="toc">
  <h3>Workflows</h3>
  {''.join(
    f'<div class="toc-row"><a href="#{k}">{WORKFLOW_DESCRIPTIONS.get(k,(k,""))[0]}</a>'
    f'<span class="toc-desc">{WORKFLOW_DESCRIPTIONS.get(k,("",""))[1]}</span></div>'
    for k in by_test
  )}
</div>

{body}

<footer>Generated by tests/ui/generate_report.py · KVForge Studio · Branch kvforge-demos</footer>
</body>
</html>"""


def main():
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run pytest tests/ui/test_walkthroughs.py first.")
        raise SystemExit(1)
    manifest = json.loads(MANIFEST.read_text())
    if not manifest:
        print("Manifest is empty — no walkthrough screenshots were captured.")
        raise SystemExit(1)
    html = build_html(manifest)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Report written to: {OUTPUT}")
    print(f"Open with: open {OUTPUT}")


if __name__ == "__main__":
    main()
