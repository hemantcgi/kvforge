# arXiv package build pipeline

Regenerates `../main.tex` (the two-column, arXiv-ready build) from the paper
source. The compiled `../main.pdf` is 14 pages, 0 errors / 0 overfull boxes
(pdfLaTeX, BasicTeX or full TeX Live).

## Source of truth
The paper text is `docs/KVForge_Research_Paper_V3.md` (repo root). This build
does **not** read it directly at assembly time — it reads the already
pandoc-converted `body_raw.tex` (see "Content corrections" below).

## Stages
```
preprocess.py   V3 markdown -> cleaned markdown (strips baked-in figure titles etc.)
pandoc          cleaned markdown -> body_raw.tex          (one-time; see note)
assemble.py     preamble.tex + abstract.tex + body_raw.tex + parts.tex -> ../main.tex
pdflatex x2     ../main.tex -> ../main.pdf                (2nd pass resolves refs)
```
Figures are produced by `make_figs.py` / `make_figs2.py` / `crop_titles.py` into
`../figures/` (8 files). `preamble.tex` holds the two-column layout / float /
spacing tuning; `parts.tex` holds the hand-formatted figures and all 10 tables.

## Content corrections (important)
Section edits and numeric/pandoc fixes are applied as **count-asserted string
replacements inside `assemble.py`** (e.g. the LoRA math, the PRS formula reflow,
and the four section cross-reference accuracy fixes), **not** by editing
`body_raw.tex`. Re-running `preprocess.py` + pandoc would regenerate
`body_raw.tex` and discard those fixes, so the corrections live in `assemble.py`
where they survive a re-convert. Each has an `assert ... == 1` guard so a drifted
source string fails the build loudly instead of silently no-op-ing.

## Rebuild
```
cd build
python3 assemble.py
cd ..
pdflatex main.tex && pdflatex main.tex
```
