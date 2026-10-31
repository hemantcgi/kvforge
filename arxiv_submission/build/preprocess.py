#!/usr/bin/env python3
"""Pre-process V3.md for pandoc: drop byline, strip heading numbers, pull out
the abstract, and replace figures/tables with placeholders (raw tables saved
for hand-conversion). Produces body_clean.md + abstract.md + tables_raw/*.md."""
import re
from pathlib import Path

B = Path(__file__).parent
src = (B / "V3.md").read_text().splitlines()
(tbl_dir := B / "tables_raw").mkdir(exist_ok=True)

out, i, n = [], 0, len(src)
tnum = 0
in_abstract = False
abstract = []

# figure file basenames in order -> our logical figure files
FIGMAP = {
    "fig01_phase_state_machine_v2.png": "fig1_state_machine",
    "fig03_rag_vs_kv_injection.png":    "fig2_phase_flow",
    "fig04_tier_weights.png":           "fig3_tier_weights",
    "fig05a_prs_waterfall.png":         "fig4_prs_breakdown",
    "fig05b_prs_scores.png":            "fig5_prs_scores",
    "fig06_confidence_gate.png":        "fig6_confidence_gate",
    "fig_e2b_comparison_accuracy.png":  "fig7_e2b_accuracy",
    "gemma4_crossover_consolidated.png":"fig8_crossover",
}

def strip_heading_number(line):
    m = re.match(r'^(#{1,6})\s+(.*)$', line)
    if not m:
        return line
    hashes, title = m.groups()
    title = re.sub(r'^\d+(\.\d+)*\.?\s+', '', title)   # "1." / "2.1" / "5.10"
    return f"{hashes} {title}"

while i < n:
    line = src[i]

    # byline block: title (#) then until first '---'
    if i == 0 and line.startswith('# '):
        out.append(strip_heading_number(line))  # keep title line (used? we set title in TeX, but keep for ref)
        i += 1
        # skip everything up to and including the first standalone '---'
        while i < n and src[i].strip() != '---':
            i += 1
        i += 1  # skip the '---'
        continue

    # abstract: capture between '## Abstract' and next '---'
    if re.match(r'^##\s+Abstract\s*$', line):
        i += 1
        while i < n and src[i].strip() != '---':
            if src[i].strip():
                abstract.append(src[i])
            i += 1
        i += 1  # skip '---'
        out.append('%%ABSTRACT%%')
        continue

    # image line -> figure placeholder
    m = re.match(r'^!\[.*\]\((?:figures/)?([^)]+)\)\s*$', line)
    if m:
        base = m.group(1).split('/')[-1]
        key = FIGMAP.get(base, base)
        out.append(f'%%FIG:{key}%%')
        i += 1
        continue

    # table block -> placeholder, raw saved
    if line.lstrip().startswith('|'):
        block = []
        while i < n and src[i].lstrip().startswith('|'):
            block.append(src[i]); i += 1
        tnum += 1
        (tbl_dir / f"t{tnum:02d}.md").write_text("\n".join(block) + "\n")
        out.append(f'%%TABLE:{tnum:02d}%%')
        continue

    # horizontal rules -> drop (section breaks)
    if line.strip() == '---':
        i += 1
        continue

    out.append(strip_heading_number(line))
    i += 1

(B / "body_clean.md").write_text("\n".join(out) + "\n")
(B / "abstract.md").write_text("\n".join(abstract) + "\n")
print(f"tables extracted: {tnum}")
print(f"abstract lines: {len(abstract)}")
print(f"body_clean lines: {len(out)}")
