#!/usr/bin/env python3
"""Crop the top main-title band off figures that are otherwise correct.
Detects: [title ink block] [white gap] [content block], crops above content."""
import sys
import numpy as np
from PIL import Image
from pathlib import Path

SRC = Path(__file__).parent / "figsrc"
OUT = Path(__file__).parent / "figures"
# source -> output name (logical paper figure numbers)
JOBS = {
    "fig03_rag_vs_kv_injection.png": "fig2_phase_flow.png",
    "fig04_tier_weights.png":        "fig3_tier_weights.png",
    "fig06_confidence_gate.png":     "fig6_confidence_gate.png",
    "gemma4_crossover_consolidated.png": "fig8_crossover.png",
}

def crop_top_title(src, dst):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im.convert("L"))
    h, w = a.shape
    darkfrac = (a < 160).mean(axis=1)        # fraction of dark pixels per row
    inky = darkfrac > 0.003                   # row has meaningful ink
    # walk: skip leading white, title block, gap, then content start
    i = 0
    while i < h and not inky[i]: i += 1       # top white margin
    title_start = i
    while i < h and inky[i]: i += 1           # title block
    title_end = i
    gap_start = i
    while i < h and not inky[i]: i += 1        # gap after title
    content_start = i
    gap = content_start - gap_start
    # sanity: a real title band is in the top ~18% and followed by a clear gap
    if title_start < h * 0.18 and gap >= 6 and content_start < h * 0.22:
        top = max(0, content_start - int(0.012 * h))   # keep a thin margin
        im.crop((0, top, w, h)).save(dst)
        print(f"ok  {src.name}: cropped {top}px of {h}  (title {title_start}-{title_end}, gap {gap}) -> {dst.name}")
    else:
        im.save(dst)
        print(f"WARN {src.name}: no clear title band (title_start={title_start}, gap={gap}, content={content_start}); copied as-is")

for s, d in JOBS.items():
    crop_top_title(SRC / s, OUT / d)
