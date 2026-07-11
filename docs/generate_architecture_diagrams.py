#!/usr/bin/env python3
"""Generate KVForge V1 and V2 architecture SVG diagrams.

Run: python docs/generate_architecture_diagrams.py
Outputs: docs/KVForge_V1_Architecture.svg
         docs/KVForge_V2_Architecture.svg
"""
import math
from pathlib import Path

OUT = Path(__file__).parent
FONT = "Inter, system-ui, -apple-system, sans-serif"
BG   = "#0a0f1e"
LANE = "#0c1628"

S = {  # (fill, stroke, text_color)
    "neutral":  ("#1e293b", "#475569", "#cbd5e1"),
    "model":    ("#1e1b4b", "#6366f1", "#a5b4fc"),
    "vdb":      ("#042f2e", "#0d9488", "#5eead4"),
    "sleep":    ("#052e16", "#16a34a", "#86efac"),
    "query":    ("#2e1065", "#9333ea", "#d8b4fe"),
    "enhanced": ("#1c1002", "#d97706", "#fcd34d"),
    "archive":  ("#0f172a", "#64748b", "#94a3b8"),
    "dash":     ("#1c0a02", "#ea580c", "#fdba74"),
    "cis":      ("#061020", "#1e3a5f", "#60a5fa"),
}

# ─── low-level SVG primitives ────────────────────────────────────────────────

def _esc(t): return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def rect(x, y, w, h, fill, stroke="none", sw=1.5, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')

def text(x, y, label, fill, size=11, weight="500", anchor="middle", dy=0):
    return (f'<text x="{x}" y="{y+dy}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" font-family="{FONT}" text-anchor="{anchor}" '
            f'dominant-baseline="middle">{_esc(label)}</text>')

def arrow(x1, y1, x2, y2, color="#475569", mid="arr"):
    dx, dy = x2-x1, y2-y1
    ln = math.sqrt(dx*dx+dy*dy)
    if ln < 1: return ""
    ex = x2 - (dx/ln)*10
    ey = y2 - (dy/ln)*10
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{color}" stroke-width="1.5" marker-end="url(#{mid})"/>')

def h_arr(x1, x2, y, color="#475569", mid="arr"):
    return arrow(x1, y, x2, y, color, mid)

def v_arr(x, y1, y2, color="#475569", mid="arr"):
    return arrow(x, y1, x, y2, color, mid)

def note(x, y, label, fill="#475569", size=10):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-style="italic" '
            f'fill="{fill}" font-family="{FONT}" text-anchor="middle">{_esc(label)}</text>')

def defs_block():
    def mk(id_, color):
        return (f'<marker id="{id_}" markerWidth="8" markerHeight="6" '
                f'refX="7" refY="3" orient="auto">'
                f'<polygon points="0 0, 8 3, 0 6" fill="{color}"/></marker>')
    return ("<defs>"
            + mk("arr",          "#475569")
            + mk("arr-enhanced", "#d97706")
            + mk("arr-active",   "#0d9488")
            + mk("arr-archive",  "#64748b")
            + "</defs>")

def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">\n'
            + defs_block()
            + f'\n{rect(0, 0, w, h, BG, rx=0)}')

# ─── composite helpers ───────────────────────────────────────────────────────

def swimlane(x, y, w, h, label, tag=None):
    parts = [rect(x, y, w, h, LANE, rx=10)]
    lx = x + 16
    parts.append(text(lx, y-10, label.upper(), "#475569", size=10, weight="700", anchor="start"))
    if tag == "unchanged":
        tx = lx + len(label)*6.5 + 16
        parts.append(rect(tx, y-20, 80, 16, "#172554", rx=3))
        parts.append(text(tx+40, y-12, "unchanged", "#93c5fd", size=9, weight="700"))
    elif tag == "new":
        tx = lx + len(label)*6.5 + 16
        parts.append(rect(tx, y-20, 70, 16, "#14532d", rx=3))
        parts.append(text(tx+35, y-12, "new in V2", "#86efac", size=9, weight="700"))
    return "\n".join(parts)

def node(cx, cy, w, h, label, sub, style):
    f, k, t = S[style]
    parts = [rect(cx-w//2, cy-h//2, w, h, f, k)]
    if sub:
        parts.append(text(cx, cy, label, t, size=11, dy=-7))
        parts.append(text(cx, cy, sub,   t, size=9,  weight="400", dy=8))
        parts[-1] = parts[-1].replace('dominant-baseline="middle"',
                                      'dominant-baseline="middle" opacity="0.75"')
    else:
        parts.append(text(cx, cy, label, t))
    return "\n".join(parts)

def legend_row(items, x, y):
    parts = []
    cx = x
    for color, label in items:
        parts.append(rect(cx, y-7, 12, 12, color, rx=2))
        parts.append(text(cx+18, y, label, "#64748b", size=10, anchor="start"))
        cx += len(label)*6.5 + 36
    return "\n".join(parts)

# ═══════════════════════════════════════════════════════════════════════════════
#  V1 diagram
# ═══════════════════════════════════════════════════════════════════════════════

def build_v1():
    W, H = 1200, 535
    out = [svg_open(W, H)]

    # Title
    out.append(text(W//2, 28, "KVForge V1 — Mean-pooled KV Injection", "#f8fafc", size=20, weight="700"))
    out.append(text(W//2, 48, "All chunks stored identically · Single storage tier · Fixed top-K retrieval", "#64748b", size=12, weight="400"))

    # ── Swimlane 1: INDEX TIME ─────────────────────────────────────────────
    SL1_Y, SL1_H = 72, 95
    out.append(swimlane(30, SL1_Y, W-60, SL1_H, "Index Time"))
    cy1 = SL1_Y + SL1_H//2

    nodes1 = [
        (105,  "Documents",        None,              "neutral"),
        (245,  "Chunker",          None,              "neutral"),
        (385,  "Embedder",         None,              "neutral"),
        (555,  "LLM forward pass", "one pass/chunk",  "model"),
        (725,  "mean_pool_kv",     "collapse seq_len","neutral"),
        (905,  "Qdrant payload",   "embed + KV blob", "vdb"),
    ]
    widths1 = [100,100,100,135,120,140]
    for (cx,lbl,sub,sty),w in zip(nodes1, widths1):
        out.append(node(cx, cy1, w, 44, lbl, sub, sty))

    # arrows between nodes1
    rights1 = [cx+w//2 for (cx,*_),w in zip(nodes1, widths1)]
    lefts1  = [cx-w//2 for (cx,*_),w in zip(nodes1, widths1)]
    for r,l in zip(rights1, lefts1[1:]):
        out.append(h_arr(r, l, cy1))

    out.append(note(W//2, SL1_Y+SL1_H+12,
        "Stored per chunk: [num_layers, 2, num_kv_heads, head_dim] float16 — token dimension collapsed by mean-pooling"))

    # ── Swimlane 2: QUERY TIME ─────────────────────────────────────────────
    SL2_Y, SL2_H = 200, 85
    out.append(swimlane(30, SL2_Y, W-60, SL2_H, "Query Time"))
    cy2 = SL2_Y + SL2_H//2

    nodes2 = [
        (95,   "Query",               None,                "query"),
        (230,  "Embedder",            None,                "neutral"),
        (385,  "Qdrant top-K",        "fixed K",           "vdb"),
        (575,  "Inject mean-pool KVs","all chunks equal",  "model"),
        (760,  "LLM",                 None,                "model"),
        (890,  "Answer",              None,                "query"),
    ]
    widths2 = [90,100,110,155,80,90]
    for (cx,lbl,sub,sty),w in zip(nodes2, widths2):
        out.append(node(cx, cy2, w, 44, lbl, sub, sty))

    rights2 = [cx+w//2 for (cx,*_),w in zip(nodes2, widths2)]
    lefts2  = [cx-w//2 for (cx,*_),w in zip(nodes2, widths2)]
    for r,l in zip(rights2, lefts2[1:]):
        out.append(h_arr(r, l, cy2))

    # ── Swimlane 3: BACKGROUND / SLEEP TIME ────────────────────────────────
    SL3_Y, SL3_H = 315, 155
    out.append(swimlane(30, SL3_Y, W-60, SL3_H, "Background / Sleep Time"))

    # Row A: access tracking → phase
    cyA = SL3_Y + 45
    nodesA = [
        (120, "Access Tracker", None,               "neutral"),
        (295, "Tier labels",    "hot/warm/cold",    "neutral"),
        (475, "LoRA Training",  "tier-weighted",    "model"),
        (645, "PRS Eval",       None,               "neutral"),
        (820, "Phase 1→2→3",   None,               "neutral"),
    ]
    widthsA = [120,120,120,110,120]
    for (cx,lbl,sub,sty),w in zip(nodesA, widthsA):
        out.append(node(cx, cyA, w, 44, lbl, sub, sty))
    rightsA = [cx+w//2 for (cx,*_),w in zip(nodesA, widthsA)]
    leftsA  = [cx-w//2 for (cx,*_),w in zip(nodesA, widthsA)]
    for r,l in zip(rightsA, leftsA[1:]):
        out.append(h_arr(r, l, cyA))

    # Row B: FAQ generator + KV recomputation
    cyB = SL3_Y + 110
    nodesB = [
        (120, "FAQ Generator",     "cloud LLM",     "sleep"),
        (310, "KV Recomputation",  "post-LoRA",     "neutral"),
        (490, "Update Qdrant",     "refresh blobs", "vdb"),
    ]
    widthsB = [120,130,120]
    for (cx,lbl,sub,sty),w in zip(nodesB, widthsB):
        out.append(node(cx, cyB, w, 44, lbl, sub, sty))

    # FAQ → LoRA (diagonal arrow)
    out.append(arrow(120+60, cyB-22, 475-60, cyA+22, "#16a34a"))
    # KV recompute → Update Qdrant
    out.append(h_arr(310+65, 490-60, cyB))

    out.append(note(W//2, SL3_Y+SL3_H+14,
        "Tier labels are used only for LoRA training sample weighting — they do not affect storage format or injection quality"))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        ("#6366f1","Model / LLM"), ("#0d9488","Vector Store"),
        ("#16a34a","Sleep-time compute"), ("#9333ea","User / Query"),
    ]
    out.append(legend_row(legend_items, 60, H-22))

    out.append("</svg>")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
#  V2 diagram
# ═══════════════════════════════════════════════════════════════════════════════

def build_v2():
    W, H = 1400, 970
    out = [svg_open(W, H)]

    # Title
    out.append(text(W//2, 28, "KVForge V2 — Corpus Intelligence System", "#f8fafc", size=20, weight="700"))
    out.append(text(W//2, 48, "Three storage tiers · CIS-driven curation · TurboQuant compression · User-confirmed archival", "#64748b", size=12, weight="400"))

    # ── Swimlane 1: INDEX TIME (unchanged) ────────────────────────────────
    SL1_Y, SL1_H = 72, 95
    out.append(swimlane(30, SL1_Y, W-60, SL1_H, "Index Time", tag="unchanged"))
    cy1 = SL1_Y + SL1_H//2

    nodes1 = [
        (105,  "Documents",        None,              "neutral"),
        (245,  "Chunker",          None,              "neutral"),
        (395,  "Embedder",         None,              "neutral"),
        (575,  "LLM forward pass", None,              "model"),
        (755,  "mean_pool_kv",     "collapse seq_len","neutral"),
        (960,  "Active Tier",      "Qdrant: embed + KV blob · kv_token_path=null", "vdb"),
    ]
    widths1 = [100,100,100,135,120,200]
    for (cx,lbl,sub,sty),w in zip(nodes1, widths1):
        out.append(node(cx, cy1, w, 44, lbl, sub, sty))
    rights1 = [cx+w//2 for (cx,*_),w in zip(nodes1, widths1)]
    lefts1  = [cx-w//2 for (cx,*_),w in zip(nodes1, widths1)]
    for r,l in zip(rights1, lefts1[1:]):
        out.append(h_arr(r, l, cy1))
    out.append(note(W//2, SL1_Y+SL1_H+12,
        "All new chunks land in Active Tier. Enhanced tier promotion happens asynchronously via the sleep-time curation pass."))

    # ── Swimlane 2: SLEEP TIME CURATION ───────────────────────────────────
    SL2_Y, SL2_H = 198, 365
    out.append(swimlane(30, SL2_Y, W-60, SL2_H, "Sleep Time — Corpus Curation Pass", tag="new"))

    # Three inputs → CIS (converging arrows)
    inp_y  = SL2_Y + 55
    cis_cx = W//2
    cis_cy = SL2_Y + 110
    cis_w, cis_h = 420, 80

    inp_nodes = [
        (175,  "FAQ Generator",      "cloud LLM · user's provider",   "sleep"),
        (cis_cx, "Embedding distances","cosine sim to nearest neighbors","neutral"),
        (W-175,"Access Tracker",     "retrieval frequency per chunk",  "neutral"),
    ]
    inp_widths = [160, 175, 160]
    for (cx,lbl,sub,sty),w in zip(inp_nodes, inp_widths):
        out.append(node(cx, inp_y, w, 44, lbl, sub, sty))

    # CIS box
    out.append(rect(cis_cx-cis_w//2, cis_cy-cis_h//2, cis_w, cis_h, "#061020", "#1e3a5f", rx=10))
    out.append(text(cis_cx, cis_cy-22, "Corpus Importance Score (CIS)", "#60a5fa", size=12, weight="700"))
    out.append(text(cis_cx, cis_cy-5,  "CIS = α × access_score  +  β × uniqueness_score  +  γ × coverage_score", "#34d399", size=11, weight="500"))

    # signal labels inside CIS box
    sig_y = cis_cy + 15
    out.append(text(cis_cx-130, sig_y, "access_score", "#fbbf24", size=10, weight="600"))
    out.append(text(cis_cx,     sig_y, "uniqueness_score", "#34d399", size=10, weight="600"))
    out.append(text(cis_cx+130, sig_y, "coverage_score", "#a78bfa", size=10, weight="600"))

    # arrows from inputs into CIS box top
    for (cx,*_),w in zip(inp_nodes, inp_widths):
        out.append(v_arr(cx, inp_y+22, cis_cy-cis_h//2))

    # CIS → branches label
    out.append(text(cis_cx, cis_cy+cis_h//2+16, "CIS score determines tier action ↓", "#475569", size=10))

    # Three branch boxes
    br_y   = cis_cy + cis_h//2 + 40
    br_h   = 155
    brs = [
        (W//2-390, "High CIS → Enhanced Tier",
         ["Background daemon triggers", "Full LLM forward pass", "preserve token sequence",
          "TurboQuant compress (3b keys + 2b values)", "Write compressed bytes to disk",
          "Update kv_token_path in Qdrant"],
         "enhanced", "#d97706"),
        (W//2,      "Mid CIS → Stay Active",
         ["No action taken", "Remains in Qdrant with", "mean-pooled KV blob", "(V1 behavior unchanged)",
          "CIS re-evaluated each", "sleep cycle"],
         "vdb", "#0d9488"),
        (W//2+390, "Low CIS + Low Uniqueness → Archive",
         ["Dashboard recommendation:", "similarity, FAQ appearances,", "days since last retrieval",
          "User reviews & confirms", "Text → archival backend", "Clear kv_cache from Qdrant"],
         "archive", "#64748b"),
    ]
    br_w = 250
    for (cx, title, steps, sty, stroke_c) in brs:
        f, k, tc = S[sty]
        out.append(rect(cx-br_w//2, br_y, br_w, br_h, f, stroke_c, rx=9))
        out.append(text(cx, br_y+14, title, tc, size=10, weight="700"))
        for i, step in enumerate(steps):
            out.append(text(cx, br_y+32+i*19, step, tc, size=9, weight="400"))
        # vertical arrow from CIS bottom to branch top
        out.append(v_arr(cx, cis_cy+cis_h//2+32, br_y))

    # Reinstatement note
    rst_y = br_y + br_h + 18
    out.append(rect(cis_cx-450, rst_y, 900, 28, "#061020", "#1e3a5f", rx=6))
    out.append(text(cis_cx, rst_y+14,
        "Archive retrieval count > user threshold  →  Dashboard recommends reinstatement  →  User confirms  →  Chunk returns to Active Tier",
        "#93c5fd", size=10))

    # ── Swimlane 3: STORAGE TIERS ──────────────────────────────────────────
    SL3_Y = SL2_Y + SL2_H + 22
    SL3_H = 115
    out.append(swimlane(30, SL3_Y, W-60, SL3_H, "Storage Tiers", tag="new"))

    tier_defs = [
        (W//2-390, "Enhanced Tier", "Per-token KV on disk · TurboQuant compressed",
         "~15 MB / chunk at 4.4× compression", "enhanced", "#d97706"),
        (W//2,      "Active Tier",  "Mean-pooled KV blob in Qdrant · V1 behavior",
         "~131 KB / chunk · kv_token_path = null", "vdb", "#0d9488"),
        (W//2+390, "Archive Tier", "Embedding + pointer in Qdrant · text in archival backend",
         "~8 KB in Qdrant · kv_cache blob cleared", "archive", "#64748b"),
    ]
    tier_w = 280
    tier_h = 90
    for (cx, title, body, size_note, sty, stroke_c) in tier_defs:
        f, k, tc = S[sty]
        out.append(rect(cx-tier_w//2, SL3_Y+12, tier_w, tier_h, f, stroke_c, rx=9))
        out.append(text(cx, SL3_Y+30, title, tc, size=11, weight="700"))
        out.append(text(cx, SL3_Y+50, body,      tc, size=9,  weight="400"))
        out.append(text(cx, SL3_Y+75, size_note,  stroke_c, size=9, weight="600"))

    # ── Swimlane 4: QUERY TIME ────────────────────────────────────────────
    SL4_Y = SL3_Y + SL3_H + 22
    SL4_H = 200
    out.append(swimlane(30, SL4_Y, W-60, SL4_H, "Query Time — Dynamic Routing Across Three Tiers", tag="new"))

    # Left: query → qdrant
    q_nodes_l = [
        (80,  "Query",    None,              "query"),
        (215, "Embedder", None,              "neutral"),
        (385, "Qdrant",   "MMR + CIS weighted","vdb"),
    ]
    q_widths_l = [80,100,130]
    q_cy = SL4_Y + SL4_H//2
    for (cx,lbl,sub,sty),w in zip(q_nodes_l, q_widths_l):
        out.append(node(cx, q_cy, w, 44, lbl, sub, sty))
    q_rights_l = [cx+w//2 for (cx,*_),w in zip(q_nodes_l, q_widths_l)]
    q_lefts_l  = [cx-w//2 for (cx,*_),w in zip(q_nodes_l, q_widths_l)]
    for r,l in zip(q_rights_l, q_lefts_l[1:]):
        out.append(h_arr(r, l, q_cy))

    # Three paths (y offsets around center)
    path_start_x = 460
    path_end_x   = 1060
    path_ys = [q_cy-60, q_cy, q_cy+60]
    path_defs = [
        # (dot_color, step_labels, style, marker)
        ("#d97706", ["Enhanced chunk","Load per-token KV (disk)","TurboQuant attention","Full-fidelity inject"], "enhanced", "arr-enhanced"),
        ("#0d9488", ["Active chunk","Load mean-pool KV (Qdrant)","Mean-pool inject (V1)"], "vdb", "arr-active"),
        ("#64748b", ["Archive chunk","Fetch text (archive backend)","Text-in-context","Track retrieval freq → reinstate"], "archive", "arr-archive"),
    ]

    # Fanout from Qdrant right edge
    qdrant_rx = 385+65
    llm_cx    = 1150
    llm_w     = 80
    ans_cx    = 1285

    for py, (dot_c, steps, sty, mid) in zip(path_ys, path_defs):
        f, k, tc = S[sty]
        # line from Qdrant to first step
        out.append(arrow(qdrant_rx, q_cy, path_start_x, py, dot_c, mid))
        step_w = 120
        gap    = 10
        total  = len(steps)*step_w + (len(steps)-1)*gap
        x0 = path_start_x + (path_end_x - path_start_x - total)//2
        for i, step in enumerate(steps):
            cx_s = x0 + i*(step_w+gap) + step_w//2
            out.append(rect(cx_s-step_w//2, py-18, step_w, 36, f, k, rx=6))
            out.append(text(cx_s, py, step, tc, size=9))
            if i < len(steps)-1:
                out.append(h_arr(cx_s+step_w//2, cx_s+step_w//2+gap, py, dot_c, mid))
        # line from last step to LLM
        last_cx = x0 + (len(steps)-1)*(step_w+gap) + step_w
        out.append(arrow(last_cx, py, llm_cx-llm_w//2, q_cy, dot_c, mid))

    # LLM + Answer
    out.append(node(llm_cx, q_cy, llm_w, 44, "LLM", None, "model"))
    out.append(h_arr(llm_cx+llm_w//2, ans_cx-50, q_cy))
    out.append(node(ans_cx, q_cy, 90, 44, "Answer", None, "query"))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        ("#6366f1","Model / LLM"), ("#0d9488","Vector Store / Active Tier"),
        ("#d97706","Enhanced Tier (TurboQuant)"), ("#16a34a","Sleep-time compute"),
        ("#64748b","Archive Tier"), ("#9333ea","User / Query"), ("#ea580c","Dashboard"),
    ]
    out.append(legend_row(legend_items, 40, H-22))

    out.append("</svg>")
    return "\n".join(out)


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    v1_path = OUT / "KVForge_V1_Architecture.svg"
    v2_path = OUT / "KVForge_V2_Architecture.svg"

    v1_path.write_text(build_v1(), encoding="utf-8")
    print(f"✓  {v1_path}")

    v2_path.write_text(build_v2(), encoding="utf-8")
    print(f"✓  {v2_path}")
