#!/usr/bin/env python3
"""Assemble main.tex from preamble + title/abstract + processed body, injecting
the hand-crafted figures and tables from parts.tex."""
import re
from pathlib import Path

B = Path(__file__).parent
preamble = (B / "preamble.tex").read_text()
abstract = (B / "abstract.tex").read_text().strip()
body = (B / "body_raw.tex").read_text()
parts_raw = (B / "parts.tex").read_text()

# (4) Table 10 PRS provenance: §5.1 (Setup) reports no PRS runs; the
#     version.json PRS example is §4.1 and the definition §4.7. Fix the locus.
_t10prs = r"\S4.7, \S5.1 PRS runs, \S6.2"
assert parts_raw.count(_t10prs) == 1, "Table 10 PRS locus not found"
parts_raw = parts_raw.replace(_t10prs, r"\S4.1 example, \S4.7 definition, \S6.2")

# --- parse parts.tex into {key: latex} ---
parts = {}
cur, buf = None, []
for line in parts_raw.splitlines():
    m = re.match(r'^%%%PART (\S+)\s*$', line)
    if m:
        if cur:
            parts[cur] = "\n".join(buf).strip()
        cur, buf = m.group(1), []
    else:
        buf.append(line)
if cur:
    parts[cur] = "\n".join(buf).strip()
assert len([k for k in parts if k.startswith("FIG:")]) == 8, parts.keys()
assert len([k for k in parts if k.startswith("TABLE:")]) == 10, parts.keys()

# --- starred (unnumbered) headings ---
for h in ["Acknowledgements", "Appendix B: Metric and Judge Provenance", "References"]:
    body = body.replace(f"\\section{{{h}}}", f"\\section*{{{h}}}")
body = body.replace("\\subsection{Contributions}", "\\subsection*{Contributions}")
# References: last section -> set it \small so the bibliography is dense and the
# tail page isn't near-empty (the group runs to \end{document}, nothing else after).
body = body.replace("\\section*{References}", "\\section*{References}\n\\small")

# Relocate the wide Table 10 (a table* can never appear on its defining page).
# Placed at the end it floated onto the References page; anchored in the last
# Discussion subsection (6.5, which typesets on the page *before* Appendix B) it
# top-floats onto the Appendix B / Conclusion page, ahead of the References list.
_t10 = r"\%\%TABLE:10\%\%"
assert body.count(_t10) == 1, "TABLE:10 placeholder count"
body = body.replace(_t10 + "\n\n", "").replace("\n\n" + _t10, "").replace(_t10, "")
_anchor = r"\hypertarget{limitations-and-the-validation-roadmap}{%"
assert _anchor in body, "limitations anchor missing"
body = body.replace(_anchor, _t10 + "\n\n" + _anchor, 1)

# --- fix the LoRA dimension math mangled by pandoc ---
body = body.replace(
    r"\textbf{W₀ ∈ ℝ\^{}(d×k)}", r"$\mathbf{W_0}\in\mathbb{R}^{d\times k}$")
body = body.replace(
    r"\textbf{B ∈ ℝ\^{}(d×r)}", r"$\mathbf{B}\in\mathbb{R}^{d\times r}$")
body = body.replace(
    r"\textbf{A ∈ ℝ\^{}(r×k)}", r"$\mathbf{A}\in\mathbb{R}^{r\times k}$")
body = body.replace("r ≪ min(d,k)", r"$r\ll\min(d,k)$")

# --- reflow the long prs_history JSON line so verbatim fits the column width ---
body = body.replace(
    '  "prs_history": [{"round": 1, "prs": 0.727}, {"round": 2, "prs": 0.783}, {"round": 3, "prs": 0.863}],',
    '  "prs_history": [\n    {"round": 1, "prs": 0.727},\n    {"round": 2, "prs": 0.783},\n    {"round": 3, "prs": 0.863}],')

# --- scrub unicode × and break the PRS formula so it fits one column (verbatim: no wrap, no newunicodechar) ---
body = body.replace(
    "PRS = 0.5 × accuracy_ratio + 0.3 × calibration + 0.2 × consistency",
    "PRS = 0.5 * accuracy_ratio\n    + 0.3 * calibration\n    + 0.2 * consistency")

# --- section cross-reference accuracy fixes (content corrections vs the V3 md) ---
# (1) amortized-cost pointer was wrong: §5.9 is the Phase-2 negative result,
#     Amortized Cost is §5.13. Split the cite so each clause points correctly.
_old = ("for mean-pooled KV injection and a corpus-dependent amortized-cost "
        "analysis (§5.9) that tempers naive latency claims.")
assert body.count(_old) == 1, "amortized-cost (5.9) cite not found"
body = body.replace(_old, ("for mean-pooled KV injection (§5.9) and a "
        "corpus-dependent amortized-cost analysis (§5.13) that tempers naive "
        "latency claims."))

# (2) diversity-is-the-lever range §§5.3--5.6 over-claims (5.4 is the leak
#     caveat, 5.5 the superseded n=89 precursors). Narrow to the two sections
#     that actually support it (matches the Discussion's own §5.3, §5.6 set).
_old = ("\\textbf{Training-signal diversity is the dominant lever for "
        "parametric absorption} (§§5.3--5.6).")
assert body.count(_old) == 1, "diversity-lever range not found"
body = body.replace(_old, ("\\textbf{Training-signal diversity is the dominant "
        "lever for parametric absorption} (§5.3, §5.6)."))

# (3) reading-guide banner: §§5.3--5.8 is NOT one judge/regime (5.5 n=89
#     deepseek-flash; 5.7/5.8 coverage@0.5 blind same-session; 5.6 a 2nd
#     architecture) -- contradicts the next sentence and Table 10. Reword to
#     separate the regimes, and order 5.9 before 5.10 (as the paper runs them).
_old = ("the training-signal and configuration studies on UC4 at n=500 with a "
        "single fixed judge (§§5.3--5.8), then reports the broader but weaker "
        "cross-corpus crossover (§5.10) and the Phase-2 negative result (§5.9).")
assert body.count(_old) == 1, "reading-guide banner not found"
body = body.replace(_old, ("the training-signal and configuration studies "
        "(§§5.3--5.8): the UC4 n=500 single-judge data-lever and "
        "architecture-transfer ablations (§5.3, §5.6; §5.4 characterizes the "
        "held-out set's residual leak, and the n=89 precursors of §5.5 are "
        "reported only for continuity), and the pre-registered, blind "
        "same-session-judged placement and RAFT studies (§§5.7--5.8). It then "
        "reports the Phase-2 negative result (§5.9) and the broader but weaker "
        "cross-corpus crossover (§5.10)."))

# --- shrink verbatim blocks to footnotesize so their lines fit the narrow column ---
body = body.replace("\\begin{verbatim}", "{\\footnotesize\n\\begin{verbatim}")
body = body.replace("\\end{verbatim}", "\\end{verbatim}\n}")

# --- inject figures ---
def fig_sub(m):
    name = m.group(1).replace("\\", "")
    key = "FIG:" + name
    return parts[key]
body = re.sub(r'\\%\\%FIG:(.+?)\\%\\%', fig_sub, body)

# --- inject tables ---
def tab_sub(m):
    return parts["TABLE:" + m.group(1)]
body = re.sub(r'\\%\\%TABLE:(\d+)\\%\\%', tab_sub, body)

# --- drop abstract placeholder (abstract goes in the title block) ---
body = re.sub(r'\\%\\%ABSTRACT\\%\\%\s*', '', body)

# sanity: no placeholders left
leftover = re.findall(r'\\%\\%[A-Z]', body)
assert not leftover, f"unreplaced placeholders: {leftover}"

header = r"""\twocolumn[{%
\begin{center}
{\LARGE\bfseries KVForge: Progressive Parametric Absorption\\[2pt] in Small Language Models\par}
\vspace{0.6em}
{\itshape Training-signal diversity, not scale, drives the crossover from retrieval to parameters.\par}
\vspace{0.9em}
{\large Dr.~Hemant Joshi\par}\vspace{0.25em}
{Independent Researcher \quad\textbullet\quad \url{https://github.com/hemantcgi/kvforge}\par}
\end{center}
\vspace{0.7em}
\begin{minipage}{\textwidth}
{\small\noindent\textbf{Abstract.}\quad @@ABSTRACT@@\par}
\end{minipage}
\vspace{1.3em}
}]
""".replace("@@ABSTRACT@@", abstract)

doc = preamble + "\n\\begin{document}\n" + header + "\n" + body + "\n\\end{document}\n"
(B / "main.tex").write_text(doc)
print("wrote main.tex:", len(doc.splitlines()), "lines")
print("figures injected:", len(re.findall(r'\\includegraphics', doc)))
print("tables injected:", doc.count("\\begin{table}") + doc.count("\\begin{table*}"))
