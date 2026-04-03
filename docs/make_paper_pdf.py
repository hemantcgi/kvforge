"""
Convert docs/KVForge_Research_Paper.md to a 2-column academic PDF
using Chrome headless --print-to-pdf.
"""

import subprocess
import sys
import os
import re
import markdown as md_lib

ROOT = "/Users/hemant/Downloads/RoPE/qdrant"
SRC  = os.path.join(ROOT, "docs/KVForge_Research_Paper.md")
HTML = "/tmp/kvforge_paper.html"
PDF  = os.path.join(ROOT, "docs/KVForge_Research_Paper.pdf")

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ── read markdown ──────────────────────────────────────────────────────────────
with open(SRC) as f:
    raw = f.read()

# ── markdown → HTML (tables, fenced code, nl2br) ─────────────────────────────
body_html = md_lib.markdown(
    raw,
    extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
)

# ── post-process: wrap abstract specially ────────────────────────────────────
# Abstract section gets full-width styling
body_html = body_html.replace(
    "<h2>Abstract</h2>",
    '<h2 class="fullwidth-heading">Abstract</h2><div class="abstract-block">',
)
# Close after the abstract paragraphs (before Introduction)
body_html = body_html.replace(
    "<h2>1. Introduction</h2>",
    '</div><h2>1. Introduction</h2>',
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Linux+Libertine:ital,wght@0,400;0,600;1,400&family=Linux+Biolinum&display=swap');

/* ── page ── */
@page {
    size: letter;
    margin: 19mm 16mm 22mm 16mm;
}

*, *::before, *::after { box-sizing: border-box; }

body {
    font-family: "Linux Libertine", "Times New Roman", Times, Georgia, serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #000;
    background: #fff;
    max-width: 100%;
    margin: 0;
    padding: 0;
}

/* ── title block – full width ── */
.title-block {
    text-align: center;
    margin-bottom: 10pt;
    page-break-after: avoid;
}
.title-block h1 {
    font-family: "Linux Biolinum", "Helvetica Neue", Arial, sans-serif;
    font-size: 17pt;
    font-weight: bold;
    line-height: 1.25;
    margin: 0 0 7pt;
    text-align: center;
    border: none;
}
.title-block .authors {
    font-size: 10pt;
    margin: 3pt 0;
}
.title-block .affil {
    font-size: 9pt;
    color: #333;
    font-style: italic;
}
.title-block .repo {
    font-size: 8.5pt;
    color: #1a5276;
    margin-top: 4pt;
}

/* ── abstract – full width, boxed ── */
.abstract-block {
    margin: 0 auto 10pt;
    padding: 7pt 10pt;
    border: 1px solid #bbb;
    font-size: 9pt;
    line-height: 1.4;
    text-align: justify;
    column-count: 1;
    max-width: 95%;
}
.fullwidth-heading {
    font-size: 10pt;
    font-weight: bold;
    font-variant: small-caps;
    text-align: center;
    margin: 0 0 4pt;
    border: none;
}

/* ── 2-column body ── */
.two-col {
    column-count: 2;
    column-gap: 14pt;
    column-rule: 0.4pt solid #ccc;
    text-align: justify;
    hyphens: auto;
    -webkit-hyphens: auto;
}

/* ── headings ── */
h1 { display: none; }  /* already in title block */

h2 {
    font-family: "Linux Biolinum", Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: bold;
    font-variant: small-caps;
    border-bottom: 0.5pt solid #555;
    margin: 9pt 0 3pt;
    padding-bottom: 1pt;
    page-break-after: avoid;
    column-span: none;
    letter-spacing: 0.03em;
}

h3 {
    font-family: "Linux Biolinum", Arial, sans-serif;
    font-size: 9.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 7pt 0 2pt;
    page-break-after: avoid;
}

h4 {
    font-size: 9pt;
    font-weight: bold;
    margin: 5pt 0 1pt;
    page-break-after: avoid;
}

/* ── paragraphs ── */
p {
    margin: 0 0 5pt;
    text-indent: 0;
}

/* ── tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin: 6pt 0;
    page-break-inside: avoid;
}
thead tr {
    background: #2c3e50;
    color: #fff;
}
th {
    padding: 3pt 5pt;
    text-align: center;
    font-weight: bold;
    border: 0.5pt solid #555;
}
td {
    padding: 2.5pt 5pt;
    border: 0.5pt solid #ccc;
    vertical-align: top;
}
tr:nth-child(even) { background: #f5f5f5; }
tr:nth-child(odd)  { background: #fff; }

/* ── code blocks ── */
pre {
    background: #f4f4f4;
    border: 0.5pt solid #ccc;
    border-left: 2.5pt solid #2c3e50;
    font-family: "Courier New", Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.35;
    padding: 5pt 7pt;
    margin: 4pt 0;
    overflow-wrap: break-word;
    white-space: pre-wrap;
    page-break-inside: avoid;
}
code {
    font-family: "Courier New", Courier, monospace;
    font-size: 8pt;
    background: #f0f0f0;
    padding: 0 2pt;
    border-radius: 1pt;
}
pre code {
    background: none;
    padding: 0;
    font-size: 7.5pt;
}

/* ── lists ── */
ul, ol {
    margin: 2pt 0 5pt 14pt;
    padding: 0;
}
li { margin-bottom: 1.5pt; }

/* ── blockquote ── */
blockquote {
    margin: 4pt 0 4pt 10pt;
    padding-left: 8pt;
    border-left: 2pt solid #888;
    color: #444;
    font-style: italic;
}

/* ── links ── */
a { color: #1a5276; text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── horizontal rule ── */
hr {
    border: none;
    border-top: 0.5pt solid #aaa;
    margin: 8pt 0;
}

/* ── references section – smaller ── */
.references p {
    font-size: 8pt;
    line-height: 1.35;
    margin: 0 0 3pt;
    text-indent: -14pt;
    margin-left: 14pt;
}

/* ── print media – ensure no browser chrome leaks in ── */
@media print {
    body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
"""

# ── extract title and author lines ───────────────────────────────────────────
# The markdown starts with "# Title", then author lines as plain paragraphs
lines = raw.split('\n')
title = lines[0].lstrip('#').strip()

# Author block is lines 2-4 (author, affil, repo link)
# lines[2] = "**Hemant Joshi**"
# lines[3] = "Independent Research"
# lines[4] = "[https://github.com...]"
author_lines = [l.strip() for l in lines[2:8] if l.strip() and not l.startswith('#') and not l.startswith('---')]

author_html = ""
for i, al in enumerate(author_lines):
    # convert **bold** and [text](url)
    al = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', al)
    al = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', al)
    if i == 0:
        author_html += f'<div class="authors">{al}</div>'
    elif 'github' in al.lower() or 'http' in al.lower():
        author_html += f'<div class="repo">{al}</div>'
    else:
        author_html += f'<div class="affil">{al}</div>'

# ── strip the title + author block from body_html ────────────────────────────
# Everything up to and including the first <hr /> is the title/author block.
# Drop it — those elements are rendered in .title-block instead.
hr_match = re.search(r'<hr\s*/?>', body_html)
if hr_match:
    body_html = body_html[hr_match.end():].lstrip()

# ── wrap references section ───────────────────────────────────────────────────
body_html = re.sub(
    r'(<h2[^>]*>References</h2>)(.*?)(<hr\s*/?>.*)?$',
    r'<div class="references">\1\2</div>',
    body_html,
    flags=re.DOTALL,
)

# ── full HTML ─────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{CSS}
</style>
</head>
<body>

<div class="title-block">
  <h1>{title}</h1>
  {author_html}
</div>

<div class="two-col">
{body_html}
</div>

</body>
</html>
"""

with open(HTML, "w") as f:
    f.write(html)

print(f"HTML written to {HTML}")

# ── Chrome headless PDF ───────────────────────────────────────────────────────
cmd = [
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    f"--print-to-pdf={PDF}",
    "--print-to-pdf-no-header",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    f"file://{HTML}",
]

print("Running Chrome headless…")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
if result.returncode != 0:
    print("STDERR:", result.stderr[:500])
    sys.exit(1)

size = os.path.getsize(PDF)
print(f"PDF written to {PDF}  ({size:,} bytes)")
