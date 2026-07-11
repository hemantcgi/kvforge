#!/usr/bin/env python3
"""
Convert KVForge_Research_Paper.md → KVForge_Research_Paper.tex
and compile with tectonic → KVForge_Research_Paper.pdf

Produces a proper two-column LaTeX paper with:
  • clickable \cite{} → bibliography hyperlinks via hyperref
  • Times-like fonts (mathptmx)
  • booktabs tables, listings code blocks
  • embedded figures from docs/figures/
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
PAPER_MD = ROOT / "KVForge_Research_Paper.md"
TEX_FILE = ROOT / "KVForge_Research_Paper.tex"
OUT_PDF  = ROOT / "KVForge_Research_Paper.pdf"

# ── LaTeX preamble ─────────────────────────────────────────────────────────────
PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}

%% ─── Packages ─────────────────────────────────────────────────────────────────
\usepackage[margin=0.72in,columnsep=0.26in,top=0.80in,bottom=0.80in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
%% mathptmx removed — causes missing-character and font errors in BasicTeX
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}  %% auto-width wrapping column
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
%% microtype removed — requires scalable fonts not available in BasicTeX
\usepackage{xcolor}
\usepackage{url}
\usepackage{fancyhdr}
%% enumitem removed — not available in BasicTeX/TeX Live Basic
\usepackage{listings}
\usepackage[normalem]{ulem}     %% for \sout if needed
\usepackage{hyperref}
\definecolor{linkblue}{RGB}{25,85,155}
\hypersetup{
  colorlinks = true,
  linkcolor  = linkblue,
  citecolor  = linkblue,
  urlcolor   = linkblue,
  pdftitle   = {From Retrieval to Recall: Autonomous Phase-Adaptive Question Answering with Persistent KV-Cache Injection},
  pdfauthor  = {Hemant Joshi}
}

%% ─── Code listing style ───────────────────────────────────────────────────────
\definecolor{codebg}{RGB}{245,246,247}
\lstset{
  basicstyle   = \footnotesize\ttfamily,
  breaklines   = true,
  keepspaces   = true,
  backgroundcolor = \color{codebg},
  frame        = single,
  framerule    = 0.4pt,
  rulecolor    = \color{gray!50},
  xleftmargin  = 4pt,
  xrightmargin = 4pt,
  aboveskip    = 6pt,
  belowskip    = 6pt,
  numbers      = none,
  literate     = {×}{{$\times$}}1 {·}{{$\cdot$}}1 {≥}{{$\geq$}}1
                 {≤}{{$\leq$}}1 {≠}{{$\neq$}}1 {≈}{{$\approx$}}1
                 {→}{{$\to$}}1 {←}{{$\leftarrow$}}1 {∈}{{$\in$}}1
                 {−}{{$-$}}1 {≪}{{$\ll$}}1 {≫}{{$\gg$}}1
                 {ε}{{$\varepsilon$}}1 {α}{{$\alpha$}}1 {β}{{$\beta$}}1
                 {γ}{{$\gamma$}}1 {ᵀ}{{$^\top$}}1 {√}{{$\sqrt{}$}}1,
}

%% ─── Header / footer ─────────────────────────────────────────────────────────
\pagestyle{fancy}
\fancyhf{}
\lhead{\small\textit{KVForge -- Progressive KV-Cache Persistence}}
\rhead{\small Dr.\ Hemant Joshi $\cdot$ 2025}
\cfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

%% ─── Section spacing (tighten a little) ──────────────────────────────────────
\setlength{\parskip}{2pt}
\setlength{\parindent}{1em}

%% ─── Compact lists (default LaTeX list spacing; enumitem removed) ─────────────

%% ─── Title ───────────────────────────────────────────────────────────────────
\title{\Large\textbf{From Retrieval to Recall: Autonomous Phase-Adaptive\\
       Question Answering with Persistent KV-Cache Injection}}
\author{Dr.\ Hemant Joshi\\[4pt]
        {\small GitHub: \url{https://github.com/hemantcgi/kvforge}}}
\date{2025}

\begin{document}

\maketitle
\thispagestyle{fancy}
"""

POSTAMBLE = r"""
\end{document}
"""

# ── Markdown block parser (same as generate_paper_pdf.py) ─────────────────────
def parse_md(path: Path) -> list:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        s   = raw.strip()
        if not s:
            i += 1; continue
        if raw.startswith("#### "): blocks.append(("h4", raw[5:].strip())); i += 1; continue
        if raw.startswith("### "):  blocks.append(("h3", raw[4:].strip())); i += 1; continue
        if raw.startswith("## "):   blocks.append(("h2", raw[3:].strip())); i += 1; continue
        if raw.startswith("# ") and not raw.startswith("## "):
            blocks.append(("h1", raw[2:].strip())); i += 1; continue
        if s in ("---", "***", "___"):
            blocks.append(("hr", None)); i += 1; continue
        if raw.startswith("```"):
            lang = raw[3:].strip(); i += 1
            code = []
            while i < n and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", (lang, "\n".join(code)))); continue
        if raw.startswith("!["):
            m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', raw)
            if m: blocks.append(("image", (m.group(1), m.group(2))))
            i += 1; continue
        if raw.startswith("|"):
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            blocks.append(("table", rows)); continue
        if raw.startswith("> "):
            bq = []
            while i < n and lines[i].startswith("> "):
                bq.append(lines[i][2:]); i += 1
            blocks.append(("blockquote", " ".join(bq))); continue
        if re.match(r'^[-*] ', raw):
            items = []
            while i < n and re.match(r'^[-*] ', lines[i]):
                items.append(lines[i][2:].strip()); i += 1
            blocks.append(("bullets", items)); continue
        if re.match(r'^\d+\. ', raw):
            items = []
            while i < n and re.match(r'^\d+\. ', lines[i]):
                items.append(re.sub(r'^\d+\. ', '', lines[i]).strip()); i += 1
            blocks.append(("numlist", items)); continue
        para = []
        while i < n:
            l = lines[i]
            if (not l.strip()
                    or l.startswith("#")
                    or l.strip() in ("---", "***", "___")
                    or l.startswith("```")
                    or l.startswith("![")
                    or l.startswith("|")
                    or l.startswith("> ")
                    or re.match(r'^[-*] ', l)
                    or re.match(r'^\d+\. ', l)):
                break
            para.append(l.rstrip()); i += 1
        if para:
            blocks.append(("para", " ".join(para)))
    return blocks

# ── LaTeX text helpers ─────────────────────────────────────────────────────────

_LATEX_SPECIAL = str.maketrans({
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
})

# Unicode → LaTeX math substitutions (applied in text context)
_UNICODE_MATH = [
    ("×",  r"$\times$"),
    ("÷",  r"$\div$"),
    ("≥",  r"$\geq$"),
    ("≤",  r"$\leq$"),
    ("≪",  r"$\ll$"),
    ("≫",  r"$\gg$"),
    ("−",  r"$-$"),
    ("≠",  r"$\neq$"),
    ("≈",  r"$\approx$"),
    ("→",  r"$\to$"),
    ("←",  r"$\leftarrow$"),
    ("↑",  r"$\uparrow$"),
    ("↓",  r"$\downarrow$"),
    ("∈",  r"$\in$"),
    ("∉",  r"$\notin$"),
    ("⊆",  r"$\subseteq$"),
    ("∑",  r"$\sum$"),
    ("∏",  r"$\prod$"),
    ("∫",  r"$\int$"),
    ("√",  r"$\sqrt{\cdot}$"),
    ("·",  r"$\cdot$"),
    ("•",  r"$\bullet$"),
    ("ε",  r"$\varepsilon$"),
    ("α",  r"$\alpha$"),
    ("β",  r"$\beta$"),
    ("γ",  r"$\gamma$"),
    ("δ",  r"$\delta$"),
    ("λ",  r"$\lambda$"),
    ("μ",  r"$\mu$"),
    ("σ",  r"$\sigma$"),
    ("τ",  r"$\tau$"),
    ("π",  r"$\pi$"),
    ("ᵀ",  r"${}^\top$"),
    ("ℝ",  r"$\mathbb{R}$"),
    ("ℕ",  r"$\mathbb{N}$"),
    ("∞",  r"$\infty$"),
    # Subscript/superscript digits
    ("₀",  r"$_0$"),  ("₁", r"$_1$"),  ("₂", r"$_2$"),  ("₃", r"$_3$"),
    ("₄",  r"$_4$"),  ("₅", r"$_5$"),  ("₆", r"$_6$"),  ("₇", r"$_7$"),
    ("₈",  r"$_8$"),  ("₉", r"$_9$"),
    ("⁰",  r"$^0$"),  ("¹", r"$^1$"),  ("²", r"$^2$"),  ("³", r"$^3$"),
    ("⁴",  r"$^4$"),  ("⁵", r"$^5$"),
    ("—",  r"---"),
    ("–",  r"--"),
    ("…",  r"\ldots{}"),
    (" ", "~"),   # non-breaking space
    ("✓",  r"\checkmark"),
    ("✗",  r"$\times$"),
    ("★",  r"$\star$"),
    ("■",  r"$\blacksquare$"),
    ("▼",  r"$\blacktriangledown$"),
    ("╱",  r"/"),
    ("│",  r"|"),
]

# Citation pattern: [N] or [N,M,...] where all N ≤ 50
_CITE_RE = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')


def _apply_unicode(text: str) -> str:
    for uc, ltx in _UNICODE_MATH:
        text = text.replace(uc, ltx)
    return text


def _apply_citations(text: str) -> str:
    """Replace [N] or [N,M] citation patterns with \\cite{refN}."""
    def repl(m):
        nums = [x.strip() for x in m.group(1).split(",")]
        if all(n.isdigit() and int(n) <= 50 for n in nums):
            keys = ",".join(f"ref{n}" for n in nums)
            return f"\\cite{{{keys}}}"
        return m.group(0)  # leave unchanged if not a citation
    return _CITE_RE.sub(repl, text)


# Table-ref pattern: [T:tab-slug] → Table~\ref{tab:tab-slug}
_TREF_RE = re.compile(r'\[T:([\w:+-]+)\]')


def _apply_table_refs(text: str) -> str:
    """Replace [T:tab-slug] with Table~\\ref{tab:tab-slug}."""
    return _TREF_RE.sub(lambda m: f"Table~\\ref{{{m.group(1)}}}", text)


def escape(text: str) -> str:
    """Escape raw text for LaTeX (no markdown processing)."""
    t = text.translate(_LATEX_SPECIAL)
    t = _apply_unicode(t)
    return t


def tex(text: str) -> str:
    """Convert inline markdown + escape for LaTeX body text."""
    # Protect inline code spans BEFORE the global LaTeX-special escape below,
    # otherwise `past_key_values` gets its underscore escaped once here and
    # again inside code_repl, corrupting it to "past\{}_key\{}_values".
    code_spans: list[str] = []
    def stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"
    t = re.sub(r'`([^`\n]+?)`', stash_code, text)

    # Escape LaTeX specials first (before adding LaTeX commands)
    t = t.translate(_LATEX_SPECIAL)
    # Apply unicode math
    t = _apply_unicode(t)
    # Bold+italic
    t = re.sub(r'\*\*\*(.+?)\*\*\*', r'\\textbf{\\textit{\1}}', t)
    # Bold
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    # Italic
    t = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'\\textit{\1}', t)
    # Restore code spans, escaping each exactly once
    def restore_code(m):
        c = code_spans[int(m.group(1))]
        c = c.replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}")
        c = c.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        c = c.replace("#", r"\#").replace("^", r"\textasciicircum{}")
        # Unicode math symbols (×, ≥, ·, − etc.) inside inline code were
        # previously left as raw Unicode, which the monospace font can't
        # render (e.g. `q × scale` corrupting into "q Πscale"). Convert them
        # to the same $...$ LaTeX forms used in body text; these introduce
        # their own "$" delimiters, so we do NOT separately escape "$" here.
        c = _apply_unicode(c)
        return f"\\texttt{{{c}}}"
    t = re.sub(r'\x00CODE(\d+)\x00', restore_code, t)
    # Markdown links [text](url) → \href{url}{text}
    def link_repl(m):
        link_text = m.group(1)
        url = m.group(2)
        safe_url = url.replace("%", "\\%").replace("_", "\\_")
        return f"\\href{{{url}}}{{{link_text}}}"
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link_repl, t)
    # Citations [N]
    t = _apply_citations(t)
    # Table cross-references [T:tab-slug]
    t = _apply_table_refs(t)
    return t


def _is_display_math(text: str) -> bool:
    """Heuristic: is a code block actually a display-math formula?"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) > 6:
        return False  # too long to be a formula
    joined = " ".join(lines)
    math_signals = ["=", "×", "≥", "≤", "∑", "→", "ε", "α", "β", "γ", "ᵀ",
                    "√", "·", "÷", "∈", "ℝ", "−", "≪", "≫"]
    hits = sum(1 for s in math_signals if s in joined)
    # Also flag single-line things like `size = 2 × L × ...`
    return hits >= 2 and len(lines) <= 4


def _math_to_latex(text: str) -> str:
    """Convert a display-math code block to LaTeX \\[ ... \\]."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    result = []
    for line in lines:
        l = line
        # Common replacements for display math
        l = l.replace("×", r"\times ")
        l = l.replace("÷", r"\div ")
        l = l.replace("≥", r"\geq ")
        l = l.replace("≤", r"\leq ")
        l = l.replace("≠", r"\neq ")
        l = l.replace("≈", r"\approx ")
        l = l.replace("→", r"\to ")
        l = l.replace("←", r"\leftarrow ")
        l = l.replace("∈", r"\in ")
        l = l.replace("·", r"\cdot ")
        l = l.replace("−", "-")
        l = l.replace("≪", r"\ll ")
        l = l.replace("≫", r"\gg ")
        l = l.replace("ε", r"\varepsilon ")
        l = l.replace("α", r"\alpha ")
        l = l.replace("β", r"\beta ")
        l = l.replace("γ", r"\gamma ")
        l = l.replace("ᵀ", r"^\top")
        l = l.replace("ℝ", r"\mathbb{R}")
        l = l.replace("√", r"\sqrt{}")
        # Subscripts: word_suffix -> word_{suffix}. Identifiers with multiple
        # underscores (e.g. self_confidence_normalized) must become ONE
        # subscript group with escaped internal underscores, not chained
        # _{}_{} groups -- LaTeX math mode errors ("Double subscript") on
        # the latter.
        def _subscript_repl(m):
            base, rest = m.group(1), m.group(2)
            return base + "_{" + rest.replace("_", r"\_") + "}"
        l = re.sub(r'([a-zA-Z]+)_([a-zA-Z0-9_]+)', _subscript_repl, l)
        # Superscripts: ^(d×k) etc.
        l = re.sub(r'\^\(([^)]+)\)', r'^{\1}', l)
        # Bare-word superscripts: 2^bits, r^2 etc. (no parens)
        l = re.sub(r'\^([a-zA-Z0-9]+)', r'^{\1}', l)
        # Text functions: softmax, min, max, round, log, sim
        for fn in ["softmax", "min", "max", "round", "log", "sim",
                   "argmax", "entropy", "Attention"]:
            l = l.replace(fn, f"\\mathrm{{{fn}}}")
        # Escape remaining LaTeX specials (but NOT math ones we just added)
        l = l.replace("%", r"\%").replace("#", r"\#").replace("&", r"\&")
        result.append(l)
    body = r" \\ ".join(result)
    if len(result) > 1:
        return f"\\[\n  \\begin{{aligned}}\n    {body}\n  \\end{{aligned}}\n\\]"
    return f"\\[\n  {body}\n\\]"


def code_to_latex(lang: str, text: str) -> str:
    if _is_display_math(text):
        return _math_to_latex(text) + "\n"
    # Regular code block
    # Escape for listings (listings handles most things, but { } \ need care)
    return f"\\begin{{lstlisting}}\n{text}\n\\end{{lstlisting}}\n"


def figure_to_latex(alt: str, rel_path: str, fig_num: list) -> str:
    fig_num[0] += 1
    n = fig_num[0]
    img_path = ROOT / rel_path
    # Check if it's a wide figure (fig02, fig03 etc. are wide)
    wide = any(x in rel_path for x in ["fig02", "fig03", "fig07"])
    env  = "figure*" if wide else "figure"
    caption_clean = alt.strip()
    # Markdown alt-text already reads "Figure N: ..." (written for readers of
    # the .md source); LaTeX's \caption prepends "Figure N:" again via the
    # figure counter, so strip the redundant leading prefix here to avoid
    # "Figure 15: Figure 15: ..." in the compiled PDF.
    caption_clean = re.sub(r'^Figure\s+\d+\s*:\s*', '', caption_clean)
    # Shorten caption: everything before the first period or the full caption if short
    return (
        f"\\begin{{{env}}}[htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=\\linewidth]{{{rel_path}}}\n"
        f"  \\caption{{{tex(caption_clean)}}}\n"
        f"  \\label{{fig:{n}}}\n"
        f"\\end{{{env}}}\n\n"
    )


def table_to_latex(rows: list, caption: str = "", label: str = "") -> str:
    """Convert a markdown table to LaTeX using one of three strategies:

    1. many columns (>=4)  → table* + \\resizebox  (scales wide multi-col tables)
    2. long cell content   → table* + tabularx L   (wraps text, no distortion)
    3. compact table       → table  + tabular       (stays in one column)

    "Long content" is triggered when any single cell exceeds 35 characters,
    which reliably overflows \\columnwidth (~3.5 in) in a twocolumn layout.
    """
    def parse_row(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    parsed = [parse_row(r) for r in rows]
    parsed = [r for r in parsed if not all(re.match(r'^[-:= ]+$', c) for c in r)]
    if not parsed:
        return ""

    ncols = max(len(r) for r in parsed)
    for r in parsed:
        while len(r) < ncols:
            r.append("")

    # Build LaTeX rows
    rows_tex = []
    for ri, row in enumerate(parsed):
        cells   = [tex(c) for c in row]
        row_tex = " & ".join(cells) + r" \\"
        rows_tex.append(("  \\midrule\n  " if ri == 1 else "  ") + row_tex)
    header = rows_tex[0]
    body   = "\n".join(rows_tex[1:])

    # Caption + label lines (inserted after \centering\small)
    cap_lines = ""
    if caption:
        cap_lines += f"  \\caption{{{tex(caption)}}}\n"
    if label:
        cap_lines += f"  \\label{{{label}}}\n"

    # Measure content width per column
    col_max = [max(len(row[i]) for row in parsed) for i in range(ncols)]
    max_any = max(col_max)

    # ── Strategy 1: many columns ─────────────────────────────────────────────
    # sum(col_max) > 55 means the row is genuinely wide → shrink with resizebox.
    # A compact 4-col table (e.g. datasets) must NOT be resized — that would
    # expand it to full text width and make the font look huge vs other tables.
    if ncols >= 4:
        col_spec  = "l" + "c" * (ncols - 1)
        needs_resize = sum(col_max) > 55 or max_any > 35
        if needs_resize:
            return (
                f"\\begin{{table*}}[htbp]\n"
                f"  \\centering\\small\n"
                f"{cap_lines}"
                f"  \\resizebox{{\\textwidth}}{{!}}{{%\n"
                f"  \\begin{{tabular}}{{{col_spec}}}\n"
                f"  \\toprule\n  {header}\n  \\midrule\n{body}\n  \\bottomrule\n"
                f"  \\end{{tabular}}%\n  }}\n"
                f"\\end{{table*}}\n\n"
            )
        else:
            # Compact multi-column table: centre at natural size inside table*
            return (
                f"\\begin{{table*}}[htbp]\n"
                f"  \\centering\\small\n"
                f"{cap_lines}"
                f"  \\begin{{tabular}}{{{col_spec}}}\n"
                f"  \\toprule\n  {header}\n  \\midrule\n{body}\n  \\bottomrule\n"
                f"  \\end{{tabular}}\n"
                f"\\end{{table*}}\n\n"
            )

    # ── Strategy 2: long content → tabularx with wrapping L columns ──────────
    if max_any > 35:
        # Columns wider than 20 chars get auto-fill wrapping (L); others fixed (l)
        spec = "".join("L" if w > 20 else "l" for w in col_max)
        return (
            f"\\begin{{table*}}[htbp]\n"
            f"  \\centering\\small\n"
            f"{cap_lines}"
            f"  \\begin{{tabularx}}{{\\textwidth}}{{{spec}}}\n"
            f"  \\toprule\n  {header}\n  \\midrule\n{body}\n  \\bottomrule\n"
            f"  \\end{{tabularx}}\n"
            f"\\end{{table*}}\n\n"
        )

    # ── Strategy 3: compact table ─────────────────────────────────────────────
    col_spec = "l" + "c" * (ncols - 1)
    return (
        f"\\begin{{table}}[htbp]\n"
        f"  \\centering\\small\n"
        f"{cap_lines}"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        f"  \\toprule\n  {header}\n  \\midrule\n{body}\n  \\bottomrule\n"
        f"  \\end{{tabular}}\n"
        f"\\end{{table}}\n\n"
    )


def _strip_sec_num(title: str) -> str:
    """Remove leading section numbers like '7.1 ' or '3.' from heading text."""
    return re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', title)


# Known table captions keyed by the subsection heading (normalised: lower, no numbers)
# Maps stripped-lower subsection text → (caption_text, label_slug)
_TABLE_META: dict[str, tuple[str, str]] = {
    "tier-weighted replay buffer": (
        "Tier-Weighted Replay Buffer: access tier conditions and replay weights",
        "tab:replay-buffer",
    ),
    "phase transition thresholds": (
        "Phase Transition Thresholds for autonomous advancement",
        "tab:phase-thresholds",
    ),
    "pluggable backends": (
        "Pluggable Backend Options supported by KVForge",
        "tab:backends",
    ),
    "hardware and software setup": (
        "Experimental Hardware and Software Setup",
        "tab:hardware",
    ),
    "datasets": (
        "Use-Case Configurations: Datasets, Vector Stores, Embedders, LoRA ranks, and Base LLMs",
        "tab:datasets",
    ),
    "pipeline timing": (
        "Pipeline Timing for UC4 Reference Run",
        "tab:timing",
    ),
    "comparison with alternative systems": (
        "Comparison with Alternative RAG and KV-Cache Systems",
        "tab:comparison",
    ),
    "positioning kvforge: a combinatorial gap, not yet a benchmarked one": (
        "Related Systems: Comparative Overview",
        "tab:related",
    ),
}


def _caption_for_section(heading: str, table_num: int) -> tuple[str, str]:
    """Return (caption, label) for a table appearing under *heading*."""
    key = re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', heading).strip().lower()
    # Direct match
    if key in _TABLE_META:
        return _TABLE_META[key]
    # Partial / substring match (handles headings with extra words)
    for k, v in _TABLE_META.items():
        if k in key or key in k:
            return v
    # Fallback: derive from heading text
    slug = re.sub(r'[^a-z0-9]+', '-', key).strip('-')
    return (heading.strip(), f"tab:{slug}")


# ── Main converter ─────────────────────────────────────────────────────────────
def blocks_to_latex(blocks: list) -> str:
    parts = []
    fig_num = [0]
    tab_num = [0]
    in_ref_section = False
    current_heading = [""]   # mutable container so inner blocks can reference it

    for kind, payload in blocks:

        if kind == "h1":
            pass  # title is in \maketitle

        elif kind == "h2":
            label = _strip_sec_num(payload.strip())
            current_heading[0] = label
            if label.lower() == "abstract":
                parts.append("\\section*{Abstract}\n")
            elif label.lower() in ("acknowledgements", "acknowledgments"):
                parts.append("\\section*{Acknowledgements}\n")
            elif label.lower() == "references":
                in_ref_section = True
                parts.append("% === References start ===\n")
            else:
                parts.append(f"\\section{{{tex(label)}}}\n")

        elif kind == "h3":
            stripped = _strip_sec_num(payload.strip())
            current_heading[0] = stripped
            parts.append(f"\\subsection{{{tex(stripped)}}}\n")

        elif kind == "h4":
            stripped = _strip_sec_num(payload.strip())
            current_heading[0] = stripped
            parts.append(f"\\subsubsection{{{tex(stripped)}}}\n")

        elif kind == "hr":
            parts.append("\n\\smallskip\n")

        elif kind == "para":
            if in_ref_section:
                # Parse reference lines: [N] Author... → \bibitem
                m = re.match(r'^\[(\d+)\]\s+(.+)', payload.strip())
                if m:
                    num  = m.group(1)
                    body = m.group(2)
                    # Bold title: **title** → just title (remove bold)
                    body = re.sub(r'\*\*(.+?)\*\*', r'\1', body)
                    body = escape(body)
                    parts.append(f"\\bibitem{{ref{num}}} {body}\n\n")
                else:
                    parts.append(f"{tex(payload)}\n\n")
            else:
                parts.append(f"{tex(payload)}\n\n")

        elif kind == "blockquote":
            parts.append(f"\\begin{{quote}}\n\\textit{{{tex(payload)}}}\n\\end{{quote}}\n\n")

        elif kind == "bullets":
            items = "\n".join(f"  \\item {tex(item)}" for item in payload)
            parts.append(f"\\begin{{itemize}}\n{items}\n\\end{{itemize}}\n\n")

        elif kind == "numlist":
            items = "\n".join(f"  \\item {tex(item)}" for item in payload)
            parts.append(f"\\begin{{enumerate}}\n{items}\n\\end{{enumerate}}\n\n")

        elif kind == "code":
            lang, text = payload
            parts.append(code_to_latex(lang, text))

        elif kind == "image":
            alt, rel = payload
            parts.append(figure_to_latex(alt, rel, fig_num))

        elif kind == "table":
            tab_num[0] += 1
            cap, lbl = _caption_for_section(current_heading[0], tab_num[0])
            parts.append(table_to_latex(payload, caption=cap, label=lbl))

    return "\n".join(parts)


# ── Split header (title+author) from body ──────────────────────────────────────
def split_header(blocks: list):
    """Return (author_info_str, body_blocks) where body starts at ## Abstract."""
    # Find title
    for b in blocks:
        if b[0] == "h1":
            break

    # Find first body section (Abstract or section 1)
    body_start = 0
    for i, (kind, payload) in enumerate(blocks):
        if kind == "h2":  # first h2 is Abstract
            body_start = i
            break

    return blocks[body_start:]


# ── Build references environment ───────────────────────────────────────────────
def build_bibliography(blocks: list) -> str:
    """Extract references section and build thebibliography."""
    in_refs = False
    ref_blocks = []
    for kind, payload in blocks:
        if kind == "h2" and payload.strip().lower() == "references":
            in_refs = True
            continue
        if in_refs:
            ref_blocks.append((kind, payload))

    if not ref_blocks:
        return ""

    items = []
    for kind, payload in ref_blocks:
        if kind != "para":
            continue
        m = re.match(r'^\[(\d+)\]\s+(.+)', payload.strip())
        if m:
            num  = m.group(1)
            body = m.group(2)
            # Remove markdown bold
            body = re.sub(r'\*\*(.+?)\*\*', r'\1', body)
            body = escape(body)
            items.append(f"  \\bibitem{{ref{num}}} {body}\n")

    if not items:
        return ""
    return (
        "\n\\begin{thebibliography}{99}\n"
        + "".join(items)
        + "\\end{thebibliography}\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Parsing markdown …")
    blocks = parse_md(PAPER_MD)
    print(f"  {len(blocks)} blocks")

    # Separate references from body (they're handled by \bibitem)
    body_blocks = []
    in_refs = False
    for b in blocks:
        kind, payload = b
        if kind == "h2" and payload.strip().lower() == "references":
            in_refs = True
        if not in_refs:
            body_blocks.append(b)

    # Strip title (handled by \maketitle) and author para blocks
    body_start = 0
    for i, (kind, payload) in enumerate(body_blocks):
        if kind == "h2" and payload.strip().lower() == "abstract":
            body_start = i
            break
    body_blocks = body_blocks[body_start:]

    print("Converting to LaTeX …")
    body_tex = blocks_to_latex(body_blocks)
    bib_tex  = build_bibliography(blocks)

    tex_content = PREAMBLE + body_tex + bib_tex + POSTAMBLE
    TEX_FILE.write_text(tex_content, encoding="utf-8")
    print(f"  ✓ {TEX_FILE.name}  ({TEX_FILE.stat().st_size // 1024} KB)")

    print("Compiling with tectonic …")
    try:
        result = subprocess.run(
            ["tectonic", "--outdir", str(ROOT), str(TEX_FILE)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        if result.returncode == 0:
            size = OUT_PDF.stat().st_size // 1024
            print(f"  ✓ {OUT_PDF.name}  ({size} KB)")
        else:
            print("  ✗ tectonic errors:")
            # Show last 30 lines of stderr (tectonic output)
            err = (result.stdout + result.stderr).splitlines()
            for l in err[-30:]:
                print("   ", l)
    except FileNotFoundError:
        print("  tectonic not found — open KVForge_Research_Paper.tex in Overleaf")


if __name__ == "__main__":
    main()
