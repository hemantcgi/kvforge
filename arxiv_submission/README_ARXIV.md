# KVForge V3 — arXiv submission package (two-column)

Two-column, arXiv-ready LaTeX build of the V3 paper
*"KVForge: Progressive Parametric Absorption in Small Language Models."*

## Contents
```
main.tex            # single-file source (self-contained; inline references)
figures/            # all 8 figures referenced by main.tex
  fig1_state_machine.pdf     fig2_phase_flow.pdf    fig3_tier_weights.pdf
  fig4_prs_breakdown.pdf     fig5_prs_scores.pdf    fig6_confidence_gate.png
  fig7_e2b_accuracy.pdf      fig8_crossover.png
main.pdf            # compiled reference PDF
README_ARXIV.md     # this file
```

## Build
```
pdflatex main.tex
pdflatex main.tex      # second pass resolves \ref / \label
```
No BibTeX/Biber step is needed: the references are a self-contained inline list
and every citation uses the paper's own hand-numbered `[N]` scheme (there is no
`.bib`, so nothing to compile — this is independent of whether arXiv runs BibTeX).

## arXiv submission notes
(per arXiv's TeX help, https://info.arxiv.org/help/submit_tex.html)
- **Engine:** pdfLaTeX. `\pdfoutput=1` is the first line — needed here because the
  figures are PNG/PDF, which only the pdfLaTeX path can embed.
- **Figures:** PNG + PDF (arXiv accepts JPEG/PNG/PDF with pdfLaTeX). They sit in
  `figures/`; compilation runs from the submission root and `\graphicspath{{figures/}}`
  resolves them. arXiv's help does not explicitly bless figure subfolders, so **if
  the submission checker objects, flatten `figures/*` to the top level and delete
  the `\graphicspath` line** (figure filenames are already unique).
- **Bibliography:** inline — no `.bib`/`.bbl` to upload.
- **Upload:** `main.tex` + the `figures/` folder **only** (zip those two — do
  *not* zip the whole directory, which would also ship `main.pdf` and this
  README). arXiv auto-detects `main.tex`.
- **Fonts:** all embedded (Type 1, via `lmodern`).
- **Portability:** unicode is handled with inputenc `\DeclareUnicodeCharacter`
  (no `newunicodechar`), and `microtype` loads only if present, so it builds on a
  minimal TeX install and on arXiv's full TeX Live alike.

## Abstract for the arXiv metadata field
The web form's abstract box is length-limited (reported ~1920 characters —
**verify on the form**); the paper's own abstract is longer, so paste this
plain-text version (ASCII, no LaTeX) into the metadata field:

Retrieval-Augmented Generation (RAG) keeps knowledge outside model weights,
re-encoding context on every query. We study progressive parametric absorption:
fine-tuning a small language model (SLM) on a corpus until it answers an
increasing share of queries from its weights, with retrieval retained as a
fallback. Diversity of the training signal, not model capacity, is the dominant
lever. In matched-rank ablations on a 500-question held-out set (one corpus,
single judge), adding diverse rewrites of source chunks to a fixed QA pool
raises held-out factual accuracy by +0.030 to +0.042, versus a non-significant
+0.009 from a 180x increase in question volume; the effect transfers to a second
architecture (a 3.2B Llama-3.2-3B reaches parity, 0.254, with a 5.1B
Gemma-4-E2B, 0.249). Two pre-registered, blind-judged studies sharpen the
picture: adapter placement (all-linear vs attention-only, matched budget) is the
dominant retention lever (+0.067, per-fact Wilcoxon p=0.0001; +0.123 full-scale),
exposure is marginal and relational structure inert; and a RAFT-trained
generator complements retrieval where it fails (difference-in-differences
+0.135, 95% CI [0.062, 0.206]). These results are produced on KVForge, a
platform that advances an SLM-backed QA deployment through three gated phases:
standard RAG, per-chunk KV-cache injection, and confidence-gated parametric
answering. The KV-injection phase, in its mean-pooled form, is a characterized
negative result, not a validated speedup. This paper reports no head-to-head
comparison against an external system under matched hardware, base model, and
query set.

## What changed from the V3 markdown source
- **Layout:** single-column markdown -> two-column `article` (10pt), arXiv style.
  Compiles to **14 pages** (pdfLaTeX, 0 errors / 0 overfull boxes).
- **Spacing pass (2026-10-31):** removed the stretched inter-paragraph whitespace
  (`\raggedbottom`, since `twocolumn` defaults to `\flushbottom`); compacted the
  section/subsection headings; tightened float-to-text separation (~9pt); and
  relocated the wide Appendix-B table (Table 10) earlier in the source (a `table*`
  cannot sit on its defining page, so its placeholder lives in the last Discussion
  subsection) so it now top-floats onto the Appendix B / Conclusion page, ahead of
  the References list, instead of stranding on its own float page or landing amid
  the references. References are set `\small` so the bibliography is dense. Net:
  17 -> 14 pages, tables now sit at the top of a text page within one page of
  their first mention.
- **Figures — every baked-in in-image title removed** (they duplicated the LaTeX
  captions, and some contradicted the figure numbering). Per figure:
  - **fig1** (state machine): regenerated — corrected PRS thresholds to 0.75
    (advance) / 0.60 (regression guard), fixed arrowheads, start-node and label
    overlaps.
  - **fig2** (phase flow): regenerated — removed the latency "2.7x faster" badges
    (the paper disavows a Phase-2 speedup), restored the "Standard RAG" headers,
    and moved the "fallback to Phase 2" note clear of the box.
  - **fig3** (tier weights): regenerated — relabeled the bars "% of total weight"
    (was the misleading "of budget", which read as contradicting the pie), and
    moved the sub-5% wedge label outside the pie with a leader line.
  - **fig4, fig5** (PRS): regenerated at true single-column size (values verified
    against the figure generator).
  - **fig7** (E2B/E4B accuracy): regenerated — fixed the clipped title, the value
    labels colliding with error-bar caps, and the legend overlapping the bars.
  - **fig6** (confidence gate), **fig8** (crossover): already correct; only the
    top title band was cropped off the original high-resolution PNG.
- **Tables — all 10 reformatted** with booktabs; wide tables (3, 4, 8, 10) span
  both columns (`table*`), the rest are single-column with wrapping cells.
- **Section cross-references (§):** normalized for consistency — contiguous
  ranges use the plural sign (`§§5.3--5.8`), and sentence-initial references are
  spelled out ("Section 5.8 …") rather than opening with the symbol. Comma-separated
  lists of non-adjacent sections keep an individual `§` each (`§3.1, §5.9`).
- **Section cross-references — accuracy corrections (content, not just style).**
  Four pointers in the V3 markdown were wrong or over-broad; corrected here (the
  markdown source on `main` still carries the originals and should be back-ported):
  - Contributions: the amortized-cost pointer read `(§5.9)`, but §5.9 is the
    Phase-2 *negative result*; Amortized Cost is §5.13. Split into
    "…mean-pooled KV injection **(§5.9)** and a corpus-dependent amortized-cost
    analysis **(§5.13)**…".
  - Contributions: "training-signal diversity is the dominant lever
    **(§§5.3--5.6)**" swept in §5.4 (leak caveat) and §5.5 (superseded n=89
    precursors); narrowed to **(§5.3, §5.6)** — the sections that support it, and
    the set the Discussion already cites.
  - Reading guide: the banner called **§§5.3--5.8** "UC4 at n=500 with a single
    fixed judge", but that span mixes judges/regimes (§5.5 is n=89 deepseek-flash;
    §5.7/§5.8 use blind coverage@0.5; §5.6 is a second architecture) — contradicting
    the very next sentence. Reworded to separate the two judge regimes, and the
    Phase-2 negative result (§5.9) now precedes the crossover (§5.10), matching the
    order the sections appear.
  - Table 10 (metric provenance): the PRS row listed "§4.7, **§5.1 PRS runs**, §6.2",
    but §5.1 (Setup) reports no PRS runs — the version.json PRS example is in §4.1.
    Corrected to "§4.1 example, §4.7 definition, §6.2".
- **Fidelity:** every numeric result in the V3 source appears in the PDF (checked
  automatically); no personal email or other private strings appear in the
  source, the PDF text, or the PDF metadata.
