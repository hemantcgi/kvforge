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

STATUS_COLORS = {
    "passed": "#4ec9b0",
    "failed": "#ce9178",
    "unknown": "#9cdcfe",
}


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_html(manifest: list[dict]) -> str:
    # Group by test name, preserving order
    by_test: dict[str, list[dict]] = {}
    for entry in manifest:
        by_test.setdefault(entry["test"], []).append(entry)

    sections = []
    for test_name, steps in by_test.items():
        title, intent = WORKFLOW_DESCRIPTIONS.get(
            test_name, (test_name, "")
        )
        step_html = []
        for s in sorted(steps, key=lambda x: x["step"]):
            png = SCREENSHOTS / s["file"]
            if not png.exists():
                continue
            b64 = img_b64(png)
            step_html.append(f"""
            <div class="step">
              <div class="step-header">
                <span class="step-num">Step {s['step']}</span>
                <span class="step-desc">{s['description']}</span>
              </div>
              <img src="data:image/png;base64,{b64}" alt="{s['slug']}" loading="lazy"/>
            </div>""")

        sections.append(f"""
        <section class="workflow" id="{test_name}">
          <h2>{title}</h2>
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
          'Segoe UI', sans-serif; padding: 0 0 60px; }}
  header {{ background: #111; border-bottom: 1px solid #1e1e1e; padding: 24px 40px;
            position: sticky; top: 0; z-index: 10; }}
  header h1 {{ color: #4ec9b0; font-size: 20px; font-weight: 700; }}
  header p {{ color: #6a6a6a; font-size: 13px; margin-top: 4px; }}
  .toc {{ background: #111; border: 1px solid #1e1e1e; border-radius: 10px;
          margin: 32px 40px 0; padding: 20px 24px; }}
  .toc h3 {{ color: #9cdcfe; font-size: 12px; font-weight: 700; text-transform: uppercase;
             letter-spacing: .07em; margin-bottom: 12px; }}
  .toc a {{ display: block; color: #4ec9b0; font-size: 13px; text-decoration: none;
            padding: 4px 0; border-bottom: 1px solid #1e1e1e; }}
  .toc a:last-child {{ border-bottom: none; }}
  .toc a:hover {{ color: #9cdcfe; }}
  .workflow {{ margin: 40px 40px 0; }}
  .workflow h2 {{ color: #4ec9b0; font-size: 17px; font-weight: 700; padding-bottom: 10px;
                  border-bottom: 1px solid #1e1e1e; }}
  .intent {{ color: #6a6a6a; font-size: 13px; margin: 10px 0 20px; line-height: 1.6; }}
  .steps {{ display: flex; flex-direction: column; gap: 28px; }}
  .step {{ background: #111; border: 1px solid #1e1e1e; border-radius: 10px; overflow: hidden; }}
  .step-header {{ padding: 14px 18px; display: flex; align-items: flex-start; gap: 12px;
                   border-bottom: 1px solid #1e1e1e; }}
  .step-num {{ background: rgba(78,201,176,.15); color: #4ec9b0; font-size: 11px;
               font-weight: 700; padding: 3px 9px; border-radius: 4px; white-space: nowrap;
               flex-shrink: 0; margin-top: 1px; }}
  .step-desc {{ color: #c8c8c8; font-size: 13px; line-height: 1.6; }}
  .step img {{ width: 100%; display: block; border-top: 1px solid #1e1e1e; }}
  footer {{ margin: 48px 40px 0; color: #444; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>KVForge Studio — UI Walkthrough Report</h1>
  <p>{len(by_test)} workflows · {total} screenshots · Generated from Playwright headless Chromium</p>
</header>
<div class="toc">
  <h3>Workflows</h3>
  {''.join(f'<a href="#{k}">{WORKFLOW_DESCRIPTIONS.get(k,(k,""))[0]}</a>' for k in by_test)}
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
