#!/usr/bin/env python3
"""
Generate all research figures for the KVForge arXiv paper.
Output: docs/figures/fig_*.png  (300 DPI, publication quality)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,   # preview; saved at 300
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# Palette (colour-blind friendly, suitable for print)
P = {
    "blue":   "#2563EB",
    "teal":   "#0891B2",
    "green":  "#16A34A",
    "orange": "#D97706",
    "red":    "#DC2626",
    "purple": "#7C3AED",
    "gray":   "#6B7280",
    "light":  "#F1F5F9",
    "dark":   "#1E293B",
    "phase1": "#6B7280",
    "phase2": "#2563EB",
    "phase3": "#16A34A",
}

SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Phase State Machine
# ─────────────────────────────────────────────────────────────────────────────
def fig_phase_state_machine():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5)
    ax.axis("off")

    def box(cx, cy, w, h, color, label, sub="", radius=0.35):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle=f"round,pad=0.15,rounding_size={radius}",
                               fc=color, ec="white", lw=2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy + (0.18 if sub else 0), label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", zorder=4)
        if sub:
            ax.text(cx, cy - 0.28, sub, ha="center", va="center",
                    fontsize=8.5, color="white", alpha=0.92, zorder=4)

    def arrow(x1, y1, x2, y2, col=P["dark"], cs="arc3,rad=0.0"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=18,
                                     color=col, connectionstyle=cs), zorder=2)

    # Boxes
    box(1.5, 2.2, 2.4, 1.5, P["phase1"], "Phase 1", "Text-in-context RAG")
    box(5.0, 2.2, 2.4, 1.5, P["phase2"], "Phase 2", "KV Cache Injection")
    box(8.5, 2.2, 2.4, 1.5, P["phase3"], "Phase 3", "Parametric Answering")

    # Start node
    circ = plt.Circle((0.25, 2.2), 0.18, color=P["dark"], zorder=3)
    ax.add_patch(circ)

    # Forward arrows
    arrow(0.44, 2.2, 3.8, 2.2)
    arrow(3.72, 2.9, 6.3, 2.9, P["phase2"])
    arrow(6.25, 2.9, 7.3, 2.9, P["phase3"])

    # Forward label: Phase 1 → Phase 2
    ax.text(2.85, 3.25, "PRS ≥ 0.75", ha="center", va="bottom",
            fontsize=9, color=P["phase2"], fontweight="bold")
    ax.annotate("", xy=(3.78, 2.85), xytext=(2.72, 2.95),
                 arrowprops=dict(arrowstyle="-", lw=0.8, color=P["phase2"]))

    # Forward label: Phase 2 → Phase 3
    ax.text(6.75, 3.25, "PRS ≥ 0.80 (×2 rounds)", ha="center", va="bottom",
            fontsize=9, color=P["phase3"], fontweight="bold")

    # Regression arrow: Phase 3 → Phase 2
    arrow(7.28, 1.6, 6.25, 1.6, col=P["red"])
    ax.text(6.77, 1.2, "PRS < 0.75  (regression guard)", ha="center",
            fontsize=9, color=P["red"], fontstyle="italic")

    # Stale KV fallback: Phase 2 → Phase 1
    arrow(3.78, 1.6, 2.72, 1.6, col=P["gray"])
    ax.text(3.25, 1.08, "stale KV tensors →\nPhase 1 fallback", ha="center",
            fontsize=8.5, color=P["gray"], fontstyle="italic")

    # Self-loop on Phase 3 (LoRA round)
    ax.annotate("", xy=(9.25, 3.15), xytext=(8.8, 3.45),
                 arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                 connectionstyle="arc3,rad=-0.8",
                                 color=P["phase3"], mutation_scale=14))
    ax.text(9.55, 3.6, "LoRA\nround N", ha="center", fontsize=8.5, color=P["phase3"])

    ax.set_title("Figure 1 — KVForge Phase Transition State Machine",
                 pad=14, fontsize=13)
    fig.savefig(OUT / "fig01_phase_state_machine.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig01_phase_state_machine.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Text RAG vs KV Injection comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_rag_vs_kv():
    """Figure 3 — three-panel flowchart: Phase 1 (Text RAG), Phase 2 (KV Injection), Phase 3 (Parametric).

    Design goals:
    - Corresponding steps sit at the same vertical position across all three panels
    - Uniform inter-box spacing throughout
    - Key/highlighted box uses phase accent colour; others use light fill
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5))
    fig.patch.set_facecolor("white")

    # ── Step definitions (label, is_key) for each phase ──────────────────────
    steps_rag = [
        ("Embed Query", False),
        ("Vector Search → Top-K Chunks", False),
        ("Concatenate chunk text into\ncontext window (~3,000 tokens)", False),
        ("LLM full forward pass over all\ncontext tokens — REPEATED EVERY QUERY", True),
        ("Autoregressive token generation", False),
    ]
    steps_kv = [
        ("Embed Query", False),
        ("Vector Search → Top-K Chunks", False),
        ("Load pre-computed kv_cache tensors\nfrom vector store payload", False),
        ("Inject past_key_values directly into\nattention cache — SKIPS RE-ENCODING", True),
        ("Autoregressive token generation", False),
    ]
    steps_param = [
        ("Embed Query", False),
        ("Confidence Gate: entropy + hedging\n+ query similarity ≥ 0.75", False),
        ("NO retrieval, NO KV injection —\ncorpus knowledge encoded in weights", True),
        ("Fine-tuned LLM generates answer\nfrom parametric memory only", True),
        ("Autoregressive token generation", False),
    ]

    # ── Layout constants ─────────────────────────────────────────────────────
    # FS=20 compensates for ~0.41× print scale (18in figure → ~7.3in on page)
    # so text renders at ~8pt in the final PDF, matching readable body text.
    BOX_H  = [0.09, 0.11, 0.15, 0.15, 0.09]   # taller key boxes to hold larger text
    GAP    = 0.048
    MARGIN = 0.05
    TOP_Y  = 0.93

    y_tops = []
    y = TOP_Y
    for h in BOX_H:
        y_tops.append(y)
        y -= (h + GAP)
    last_bot = y_tops[-1] - BOX_H[-1]
    badge_y  = last_bot / 2

    phases = [
        ("Phase 1 — Text RAG  (Standard)",       P["phase1"], steps_rag,   "~1,840 ms  (baseline)",   P["red"]),
        ("Phase 2 — KV Injection  (KVForge)",     P["phase2"], steps_kv,    "~680 ms  (2.7× faster)",  P["green"]),
        ("Phase 3 — Parametric  (KVForge)",       P["phase3"], steps_param, "~510 ms  (3.6× faster)",  P["teal"]),
    ]

    FS = 14   # ~5.5pt after 0.41× print scale — balanced between 10 (tiny) and 20 (oversized)

    for ax, (title, col, steps, latency, badge_col) in zip(axes, phases):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=FS, fontweight="bold", color=col, pad=12)

        for i, ((label, is_key), h, ytop) in enumerate(zip(steps, BOX_H, y_tops)):
            ybot = ytop - h
            ycen = (ytop + ybot) / 2

            fc = col if is_key else P["light"]
            tc = "white" if is_key else P["dark"]
            lw = 2.5 if is_key else 1.2

            r = FancyBboxPatch((MARGIN, ybot), 1 - 2*MARGIN, h,
                                boxstyle="round,pad=0.015,rounding_size=0.02",
                                fc=fc, ec=col, lw=lw, zorder=2)
            ax.add_patch(r)
            ax.text(0.5, ycen, label,
                    ha="center", va="center",
                    fontsize=FS, color=tc,
                    fontweight="bold" if is_key else "normal",
                    multialignment="center")

            if i < len(steps) - 1:
                a_tail = ybot - 0.007
                a_head = y_tops[i + 1] + 0.007
                ax.annotate("", xy=(0.5, a_head), xytext=(0.5, a_tail),
                             arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                             color=P["gray"], mutation_scale=13))

        # Phase 3 fallback note
        if col == P["phase3"]:
            ax.text(0.5, badge_y + 0.075,
                    "Low confidence → fallback to Phase 2",
                    ha="center", va="center", fontsize=FS,
                    color=P["gray"], style="italic")

        # Latency badge (taller to hold FS=20 text)
        ax.add_patch(FancyBboxPatch((MARGIN, badge_y - 0.045), 1 - 2*MARGIN, 0.090,
                                    boxstyle="round,pad=0.01,rounding_size=0.01",
                                    fc=badge_col + "18", ec=badge_col, lw=1.2, zorder=2))
        ax.text(0.5, badge_y, latency,
                ha="center", va="center", fontsize=FS,
                color=badge_col, fontweight="bold")

    fig.suptitle(
        "Figure 3 — KVForge Phase Progression: Text RAG → KV Injection → Parametric Answering",
        y=1.01, fontsize=FS, fontweight="bold",
    )
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / "fig03_rag_vs_kv_injection.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig03_rag_vs_kv_injection.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Generation Latency by Phase (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_latency():
    fig, ax = plt.subplots(figsize=(8, 3.8))

    phases   = ["Phase 1\n(Text RAG)", "Phase 2\n(KV Injection)", "Phase 3\n(Parametric)"]
    retrieval= [12,  12,  0]
    generate = [1840, 680, 510]
    speedup  = ["baseline", "2.7× faster", "3.6× faster"]
    cols_r   = [P["gray"],  P["blue"],  P["green"]]
    cols_g   = [P["phase1"], P["phase2"], P["phase3"]]

    y = np.arange(len(phases))
    h = 0.45

    bars_r = ax.barh(y, retrieval, h, color=cols_r, alpha=0.5, label="Retrieval")
    bars_g = ax.barh(y, generate, h, left=retrieval, color=cols_g, label="Generation")

    # Speedup annotations
    for i, (r, g, sp) in enumerate(zip(retrieval, generate, speedup)):
        total = r + g
        ax.text(total + 30, i, f"  {total:,} ms  ({sp})",
                va="center", fontsize=10, color=P["dark"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(phases, fontsize=11)
    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_xlim(0, 2600)
    ax.set_title("Figure 11 — Query-Response Latency by Phase\n"
                 "Llama-3.2-3B · UC4 Amazon Bedrock Docs · NVIDIA A10G · n = 50 queries",
                 fontsize=12, pad=10)

    legend = ax.legend(loc="lower right", framealpha=0.9)
    ax.axvline(x=0, color=P["dark"], lw=0.8)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_latency_by_phase.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig11_latency_by_phase.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – PRS Results by Use Case
# ─────────────────────────────────────────────────────────────────────────────
def fig_prs_by_uc():
    fig, ax = plt.subplots(figsize=(8, 4.2))

    ucs   = ["UC1 — Customer Support", "UC2 — PubMedQA\n(Biomedical)",
              "UC3 — SQuAD 2.0\n(Reading Comprehension)", "UC4 — Bedrock Docs\n(Technical)"]
    prs   = [0.755, 0.852, 0.800, 0.863]
    cols  = [P["teal"], P["purple"], P["blue"], P["green"]]
    y     = np.arange(len(ucs))

    bars = ax.barh(y, prs, 0.55, color=cols, zorder=3)
    ax.axvline(x=0.75, color=P["red"], lw=2, ls="--", zorder=4, label="Phase 3 threshold (0.75)")
    ax.axvline(x=0.80, color=P["orange"], lw=1.5, ls=":", zorder=4, label="Phase 3 stable threshold (0.80)")

    for bar, p in zip(bars, prs):
        ax.text(p + 0.003, bar.get_y() + bar.get_height()/2,
                f"  {p:.3f}", va="center", fontsize=11, fontweight="bold", color=P["dark"])

    # Phase 3 checkmarks
    for i, p in enumerate(prs):
        ax.text(0.91, i, "✓ Phase 3", va="center", fontsize=10,
                color=P["green"], fontweight="bold")

    ax.set_yticks(y); ax.set_yticklabels(ucs, fontsize=10.5)
    ax.set_xlabel("Parametric Readiness Score (PRS)", fontsize=11)
    ax.set_xlim(0.68, 0.95)
    ax.set_title("Figure 12 — Parametric Readiness Score by Use Case\n"
                 "Sleep-time FAQ generation via Gemini 2.5 Flash · best training round",
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig12_prs_by_usecase.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig12_prs_by_usecase.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Effect of Sleep-Time FAQ Generation (grouped bars)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sleep_time_effect():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    rounds   = ["Round 1", "Round 2"]
    heuristic = [0.727, 0.783]
    sleeptime = [0.783, 0.863]
    x = np.arange(len(rounds)); w = 0.32

    b1 = ax.bar(x - w/2, heuristic, w, label="Heuristic FAQs",
                color=P["gray"], alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, sleeptime, w, label="Sleep-time FAQs (Gemini 2.5 Flash)",
                color=P["green"], alpha=0.92, zorder=3)

    ax.axhline(y=0.75, color=P["red"], lw=2, ls="--", zorder=4, label="Phase 3 gate (0.75)")
    ax.axhline(y=0.80, color=P["orange"], lw=1.5, ls=":", zorder=4, label="Phase 3 stable (0.80)")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # Delta annotation
    delta = sleeptime[1] - heuristic[1]
    ax.annotate("", xy=(x[1] + w/2, sleeptime[1] + 0.005),
                 xytext=(x[1] - w/2, heuristic[1] + 0.005),
                 arrowprops=dict(arrowstyle="<->", lw=2, color=P["purple"]))
    ax.text(x[1], (sleeptime[1] + heuristic[1])/2 + 0.015,
            f"+{delta:.3f}\n(+{delta/heuristic[1]*100:.1f}%)",
            ha="center", fontsize=10, color=P["purple"], fontweight="bold")

    # Phase outcome badges
    ax.text(0.97, 0.71, "Heuristic → Phase 2 ✗", transform=ax.transAxes,
            ha="right", fontsize=9.5, color=P["gray"], fontstyle="italic")
    ax.text(0.97, 0.97, "Sleep-time → Phase 3 ✓", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=P["green"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(rounds, fontsize=11)
    ax.set_ylabel("Parametric Readiness Score (PRS)", fontsize=11)
    ax.set_ylim(0.68, 0.94)
    ax.set_title("Figure 13 — Effect of Training Signal Quality on PRS\n"
                 "UC4 — Amazon Bedrock User Guide · Llama-3.2-3B-Instruct",
                 fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig13_sleep_time_effect.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig13_sleep_time_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – PRS Progression over Training Rounds
# ─────────────────────────────────────────────────────────────────────────────
def fig_prs_progression():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    rounds = [1, 2, 3]
    uc1 = [0.698, 0.735, 0.755]
    uc2 = [0.791, 0.830, 0.852]
    uc3 = [0.745, 0.778, 0.800]
    uc4_h = [0.727, 0.783, 0.783]   # heuristic plateau
    uc4_s = [0.727, 0.783, 0.863]   # sleep-time breakthrough

    kw = dict(marker="o", markersize=7, linewidth=2.2)
    ax.plot(rounds, uc1, color=P["teal"],   label="UC1 Customer Support",          **kw)
    ax.plot(rounds, uc2, color=P["purple"], label="UC2 PubMedQA",                  **kw)
    ax.plot(rounds, uc3, color=P["blue"],   label="UC3 SQuAD 2.0",                 **kw)
    ax.plot(rounds, uc4_h, color=P["gray"], label="UC4 Heuristic FAQs",
            linestyle="--", marker="s", markersize=7, linewidth=2)
    ax.plot(rounds, uc4_s, color=P["green"],label="UC4 Sleep-time FAQs (Gemini)",  **kw)

    ax.axhline(0.75, color=P["red"],    lw=2,   ls="--", alpha=0.8, label="Phase 3 threshold (0.75)")
    ax.axhline(0.80, color=P["orange"], lw=1.5, ls=":",  alpha=0.8, label="Phase 3 stable (0.80)")

    # Breakthrough annotation on UC4 sleep-time
    ax.annotate("Sleep-time FAQs\nunlock Phase 3",
                 xy=(3, 0.863), xytext=(2.45, 0.895),
                 fontsize=9, color=P["green"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=P["green"], lw=1.5))

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=11)
    ax.set_ylabel("Parametric Readiness Score (PRS)", fontsize=11)
    ax.set_ylim(0.66, 0.93)
    ax.set_title("Figure 14 — PRS Progression Over Training Rounds\n"
                 "All use-cases · AWS g5.12xlarge (4× A10G) · Llama-3.2-3B-Instruct",
                 fontsize=12, pad=10)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig14_prs_progression.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig14_prs_progression.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – PRS Component Breakdown (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_prs_components():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: contribution waterfall for UC4 final round
    ax = axes[0]
    components = ["Accuracy\n(weight 0.50)", "Calibration\n(weight 0.30)",
                  "Consistency\n(weight 0.20)"]
    scores     = [0.88,  0.82,  0.88]
    weights    = [0.50,  0.30,  0.20]
    contribs   = [s * w for s, w in zip(scores, weights)]
    cols_comp  = [P["blue"], P["orange"], P["green"]]

    bars = ax.bar(components, contribs, color=cols_comp, width=0.55, zorder=3)
    for bar, score, contrib in zip(bars, scores, contribs):
        ax.text(bar.get_x() + bar.get_width()/2, contrib + 0.003,
                f"score={score:.2f}\ncontrib={contrib:.3f}",
                ha="center", va="bottom", fontsize=9.5)

    ax.axhline(sum(contribs), color=P["red"], lw=2, ls="--")
    ax.text(2.35, sum(contribs) + 0.005, f"PRS = {sum(contribs):.3f}",
            va="bottom", ha="right", fontsize=11, color=P["red"], fontweight="bold")

    ax.set_ylim(0, 0.58)
    ax.set_ylabel("PRS Contribution", fontsize=11)
    ax.set_title("PRS Component Breakdown\n(UC4 · Round 3)", fontsize=11, pad=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor("white")

    # Right: component scores across all UCs
    ax2 = axes[1]
    uc_labels = ["UC1\nCust. Support", "UC2\nPubMedQA",
                 "UC3\nSQuAD", "UC4\nBedrock"]
    acc_scores  = [0.80, 0.88, 0.83, 0.88]
    cal_scores  = [0.75, 0.82, 0.78, 0.82]
    con_scores  = [0.77, 0.85, 0.80, 0.88]

    x = np.arange(len(uc_labels)); w = 0.24
    ax2.bar(x - w, acc_scores, w, label="Accuracy",     color=P["blue"],   alpha=0.9, zorder=3)
    ax2.bar(x,     cal_scores, w, label="Calibration",  color=P["orange"], alpha=0.9, zorder=3)
    ax2.bar(x + w, con_scores, w, label="Consistency",  color=P["green"],  alpha=0.9, zorder=3)

    ax2.set_xticks(x); ax2.set_xticklabels(uc_labels, fontsize=10)
    ax2.set_ylim(0.68, 0.96)
    ax2.set_ylabel("Component Score", fontsize=11)
    ax2.set_title("PRS Component Scores Across Use Cases", fontsize=11, pad=8)
    ax2.legend(fontsize=9.5)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_facecolor("white")

    fig.suptitle("Figure 5 — PRS Decomposition: Component Scores and Contributions",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig05_prs_components.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig05_prs_components.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 – Storage Tier Comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_storage_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: per-chunk size by model and format
    ax = axes[0]
    formats = ["Float16\n(uncompressed)", "TurboQuant\n(4.4× compressed)", "Mean-pool\n(V1 Active)"]
    llama3b  = [np.nan, np.nan, 0.057]         # MB
    llama8b  = [67.0,   15.2,   0.131]
    cols_fmt = [P["red"], P["green"], P["blue"]]

    x = np.arange(len(formats)); w = 0.32
    m3 = ax.bar(x - w/2, [v if v == v else 0 for v in llama3b], w,
                label="Llama-3.2-3B", color=P["teal"],  alpha=0.85, zorder=3)
    m8 = ax.bar(x + w/2, [v if v == v else 0 for v in llama8b], w,
                label="Llama-3.1-8B", color=P["purple"], alpha=0.85, zorder=3)

    for i, (v3, v8) in enumerate(zip(llama3b, llama8b)):
        if v3 == v3:  # not NaN
            ax.text(i - w/2, v3 + 0.5, f"{v3:.3f}", ha="center", fontsize=8.5)
        if v8 == v8:
            ax.text(i + w/2, v8 + 0.5, f"{v8:.1f}", ha="center", fontsize=8.5)

    ax.set_xticks(x); ax.set_xticklabels(formats, fontsize=10)
    ax.set_ylabel("Per-Chunk KV Storage (MB)", fontsize=11)
    ax.set_title("Per-Chunk Storage by Format", fontsize=11, pad=8)
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor("white")

    # Right: total storage for 10K chunks (Llama-8B)
    ax2 = axes[1]
    categories = ["Float16\n(full-token)", "TurboQuant\n(Enhanced Tier)",
                  "Mean-pool\n(Active Tier)"]
    total_gb = [670_000 / 1000, 152_000 / 1000, 1.3]   # GB for 10K chunks
    cols2 = [P["red"], P["green"], P["blue"]]
    bars = ax2.bar(categories, total_gb, color=cols2, width=0.5, zorder=3)

    for bar, gb in zip(bars, total_gb):
        label = f"{gb:.0f} GB" if gb >= 1 else f"{gb:.2f} GB"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 label, ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    ax2.set_ylabel("Total Storage for 10K Chunks (GB)", fontsize=11)
    ax2.set_title("Total Storage at Scale (Llama-3.1-8B)", fontsize=11, pad=8)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_facecolor("white")

    fig.suptitle("Figure 10 — KV Storage Requirements: Format and Scale Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_storage_comparison.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig10_storage_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 – Tier Weight System (replay buffer sampling)
# ─────────────────────────────────────────────────────────────────────────────
def fig_tier_weights():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: weight bar chart
    ax = axes[0]
    tiers   = ["Frozen", "Cold", "Warm", "Hot"]
    weights = [1, 2, 4, 8]
    pcts    = [f"{int(100*w/sum(weights))}%" for w in weights]
    cols_t  = [P["gray"], "#64748B", P["blue"], P["orange"]]

    bars = ax.barh(tiers, weights, color=cols_t, height=0.55, zorder=3)
    for bar, w, pct in zip(bars, weights, pcts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"weight {w}  ({pct} of budget)",
                va="center", fontsize=10.5)

    ax.set_xlim(0, 12.5)
    ax.set_xlabel("Sampling Weight", fontsize=11)
    ax.set_title("Replay Buffer Tier Weights", fontsize=11, pad=8)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_facecolor("white")

    # Right: expected sample distribution pie
    ax2 = axes[1]
    labels2  = ["Hot (top 15%)", "Warm (next 50%)", "Cold", "Frozen"]
    sizes2   = [8*0.15, 4*0.50, 2*0.25, 1*0.10]   # proportional corpus × weight
    sizes2   = [s / sum(sizes2) for s in sizes2]
    explode  = (0.05, 0.02, 0, 0)
    cols_pie = [P["orange"], P["blue"], "#64748B", P["gray"]]

    wedges, texts, autotexts = ax2.pie(
        sizes2, labels=labels2, colors=cols_pie, explode=explode,
        autopct=lambda p: f"{p:.0f}%", startangle=140,
        textprops={"fontsize": 10},
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(10)
    ax2.set_title("Expected Mini-batch Distribution\n(Replay Fraction = 20%)", fontsize=11, pad=8)

    fig.suptitle("Figure 4 — Tier-Weighted Replay Buffer: Weights and Expected Sampling Distribution",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig04_tier_weights.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig04_tier_weights.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 – TurboQuant compression ratio comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig_turboquant_compression():
    fig, ax = plt.subplots(figsize=(8, 4.2))

    methods = ["Float16\n(baseline)", "KIVI\n2-bit [31]", "KVQuant\n4-bit [30]",
               "TurboQuant Keys\n(3-bit, ours)", "TurboQuant Keys+Vals\n(3+4-bit, ours)"]
    compression = [1.0, 2.2, 2.0, 4.9, 4.4]
    cols_m = [P["gray"], "#94A3B8", "#64748B", P["blue"], P["green"]]
    is_ours = [False, False, False, True, True]

    bars = ax.bar(methods, compression, color=cols_m,
                  width=0.55, zorder=3,
                  edgecolor=["white"]*3 + [P["dark"]]*2,
                  linewidth=[0]*3 + [2]*2)

    for bar, c, ours in zip(bars, compression, is_ours):
        va_adj = 0.08 if c < 1.5 else 0.06
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + va_adj,
                f"{c}×", ha="center", va="bottom",
                fontsize=12 if ours else 10.5,
                fontweight="bold" if ours else "normal",
                color=P["green"] if ours else P["dark"])

    # "Ours" bracket
    ax.annotate("", xy=(3.28, 5.0), xytext=(4.28, 5.0),
                 arrowprops=dict(arrowstyle="<->", lw=2, color=P["green"]))
    ax.text(3.78, 5.12, "TurboQuant (ours)", ha="center", fontsize=10,
            color=P["green"], fontweight="bold")

    ax.set_ylabel("Compression Ratio vs Float16", fontsize=11)
    ax.set_ylim(0, 6.0)
    ax.set_title("Figure 9 — TurboQuant Compression vs Prior KV Quantization Methods\n"
                 "Llama model · head_dim = 128 · 512-token sequence",
                 fontsize=12, pad=10)
    ax.axhline(1.0, color=P["red"], lw=1.2, ls="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig09_turboquant_compression.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig09_turboquant_compression.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 – CIS Component Radar Chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_cis_radar():
    fig = plt.figure(figsize=(9, 4.5))

    # Left: radar for example chunks
    ax1 = fig.add_subplot(121, polar=True)
    cats  = ["Access\nScore", "Uniqueness\nScore", "Coverage\nScore"]
    N     = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    examples = {
        "High-CIS chunk (Enhanced Tier)":    [0.95, 0.82, 0.88],
        "Mid-CIS chunk (Active Tier)":       [0.45, 0.68, 0.40],
        "Low-CIS chunk (Archive Candidate)": [0.08, 0.12, 0.05],
    }
    ex_cols = [P["green"], P["blue"], P["gray"]]

    for (label, vals), col in zip(examples.items(), ex_cols):
        v = vals + vals[:1]
        ax1.plot(angles, v, "o-", lw=2, color=col, label=label, markersize=5)
        ax1.fill(angles, v, alpha=0.08, color=col)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(cats, size=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax1.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=8)
    ax1.set_title("CIS Component Profiles\nby Storage Tier", size=11, pad=14)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15), fontsize=8.5)

    # Right: distribution of CIS scores across a hypothetical corpus
    ax2 = fig.add_subplot(122)
    np.random.seed(42)
    n_chunks = 2520
    access   = np.random.beta(0.7, 3.5, n_chunks)
    unique   = np.random.beta(2.5, 2.0, n_chunks)
    coverage = np.random.beta(0.5, 4.0, n_chunks)
    cis = (access + unique + coverage) / 3

    ax2.hist(cis, bins=40, color=P["blue"], alpha=0.75, edgecolor="white", zorder=3)
    ax2.axvline(np.percentile(cis, 90), color=P["orange"], lw=2, ls="--",
                label=f"90th pct (Enhanced Tier) = {np.percentile(cis, 90):.2f}")
    ax2.axvline(np.percentile(cis, 20), color=P["red"], lw=2, ls=":",
                label=f"20th pct (Archive Candidates) = {np.percentile(cis, 20):.2f}")
    ax2.set_xlabel("CIS Score", fontsize=11)
    ax2.set_ylabel("Number of Chunks", fontsize=11)
    ax2.set_title("Simulated CIS Distribution\n(n = 2,520 chunks)", fontsize=11, pad=8)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25, zorder=0)
    ax2.set_facecolor("white")

    fig.suptitle("Figure 8 — Corpus Importance Score (CIS): Component Profiles and Distribution",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig08_cis_analysis.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig08_cis_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 – System Architecture (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def fig_system_architecture():
    """Figure 2 — Full KVForge system architecture.

    Restores the proven layout (offline pipeline full-width at top, VDB
    centre-right, query path left, training loop right) and adds:
      - Two lightweight dashed machine labels (no instance names)
      - Larger arrowheads (mutation_scale=22)
      - Clearly separated VDB ↔ Retrieval arrows (out above, return below)
      - VDB label updated to "Pluggable Backend"
    """
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    WHITE = "#FFFFFF"
    DBORD = "#475569"
    FS_LBL = 16
    FS_SUB = 13
    FS_ARR = 12
    FS_SEC = 15
    MS     = 22   # arrowhead mutation_scale

    # ── Helpers ───────────────────────────────────────────────────────────

    def box(cx, cy, w, h, label, sub="", fc=WHITE, ec=DBORD, tc=None):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.25",
                                    fc=fc, ec=ec, lw=2.2, zorder=3))
        if tc is None:
            r, g, b = int(fc[1:3], 16), int(fc[3:5], 16), int(fc[5:7], 16)
            tc = "white" if r + g + b < 400 else P["dark"]
        sc = "white" if tc == "white" else "#475569"
        if sub:
            ax.text(cx, cy + 0.20, label, ha="center", va="center",
                    fontsize=FS_LBL, fontweight="bold", color=tc, zorder=4)
            ax.text(cx, cy - 0.26, sub, ha="center", va="center",
                    fontsize=FS_SUB, color=sc, zorder=4, multialignment="center")
        else:
            ax.text(cx, cy, label, ha="center", va="center",
                    fontsize=FS_LBL, fontweight="bold", color=tc, zorder=4,
                    multialignment="center")

    def arr(x1, y1, x2, y2, col=P["gray"], lw=2.0, cs="arc3,rad=0.0"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", lw=lw, color=col,
                                     connectionstyle=cs, mutation_scale=MS),
                     zorder=2)

    def lbl(x, y, t, col=P["gray"], ha="center", bold=False):
        ax.text(x, y, t, ha=ha, va="center", fontsize=FS_ARR, color=col,
                fontweight="bold" if bold else "normal", zorder=5,
                multialignment="center",
                bbox=dict(fc="white", ec="none", pad=1.5))

    def sec_label(x, y, t, col):
        ax.text(x, y, t, ha="left", va="center", fontsize=FS_SEC,
                color=col, fontweight="bold", fontstyle="italic", zorder=5)

    VDB_CX, VDB_CY, VDB_W, VDB_H = 13.3, 7.9, 8.5, 1.10

    # ═════════════════════════════════════════════════════════════════════
    # 1. OFFLINE PIPELINE  (full-width top row)
    # ═════════════════════════════════════════════════════════════════════
    sec_label(0.3, 11.65, "① Offline Pipeline — Index & KV Pre-compute", P["blue"])

    off_specs = [
        ("Documents\n& Connectors", "",                2.6, P["blue"]),
        ("Chunker",                 "≤ 512 tokens",    2.1, DBORD),
        ("Embedder",                "BGE / mxbai",     2.2, DBORD),
        ("LLM Forward Pass",        "use_cache=True",  2.7, P["purple"]),
        ("KV Encoder",              "mean-pool → b64", 2.4, DBORD),
    ]
    off_w   = [s[2] for s in off_specs]
    gap_off = (19.0 - sum(off_w)) / (len(off_w) - 1)
    off_cx  = []
    xc = 0.5
    for w in off_w:
        off_cx.append(xc + w / 2)
        xc += w + gap_off

    OFF_Y, OFF_H = 10.55, 1.00
    for cx, (lt, st, w, ec) in zip(off_cx, off_specs):
        box(cx, OFF_Y, w, OFF_H, lt, st, fc=WHITE, ec=ec)
    for i in range(len(off_cx) - 1):
        arr(off_cx[i] + off_w[i]/2, OFF_Y,
            off_cx[i+1] - off_w[i+1]/2, OFF_Y, col=DBORD, lw=1.8)

    # ═════════════════════════════════════════════════════════════════════
    # 2. VECTOR DATABASE  (Vector Store Server)
    # ═════════════════════════════════════════════════════════════════════
    box(VDB_CX, VDB_CY, VDB_W, VDB_H,
        "Vector Database  (Pluggable Backend)",
        "embedding · text · kv_cache · access_count · tier",
        fc=P["blue"], ec=P["blue"])

    # Embedder → VDB
    emb_cx = off_cx[2]
    arr(emb_cx, OFF_Y - OFF_H/2, emb_cx, VDB_CY + VDB_H/2, col=P["purple"], lw=2.0)
    lbl(emb_cx - 1.2, (OFF_Y - OFF_H/2 + VDB_CY + VDB_H/2) / 2,
        "embedding\nvector", col=P["purple"])

    # KV Encoder → VDB
    kv_cx = off_cx[4]
    arr(kv_cx, OFF_Y - OFF_H/2, VDB_CX + VDB_W/2 - 0.5, VDB_CY + VDB_H/2,
        col=DBORD, lw=2.0)
    lbl(kv_cx + 1.1, (OFF_Y - OFF_H/2 + VDB_CY + VDB_H/2) / 2,
        "kv_cache\n(base64)", col=DBORD)

    # ═════════════════════════════════════════════════════════════════════
    # 3. ONLINE QUERY PATH  (left column, GPU Server)
    # ═════════════════════════════════════════════════════════════════════
    sec_label(0.3, 9.25, "② Online Query Path", P["green"])

    QX, QW, QH = 4.5, 3.0, 0.88
    box(QX, 7.90, QW, QH, "Embed Query",     fc=WHITE, ec=P["green"])
    box(QX, 6.75, QW, QH, "Top-K Retrieval", fc=WHITE, ec=P["green"])
    arr(QX, 7.90 - QH/2, QX, 6.75 + QH/2, col=P["green"])

    PW, PH, PY = 2.10, 0.88, 5.50
    box(1.60, PY, PW, PH, "Phase 1\nText RAG",    fc=P["phase1"], ec=P["phase1"])
    box(4.50, PY, PW, PH, "Phase 2\nKV Injection", fc=P["phase2"], ec=P["phase2"])
    box(7.40, PY, PW, PH, "Phase 3\nParametric",  fc=P["phase3"], ec=P["phase3"])

    box(QX, 4.10, QW, QH, "LLM Generation", fc=WHITE, ec=DBORD)

    arr(QX - 0.15, 6.75 - QH/2, 1.60, PY + PH/2, col=P["phase1"])
    arr(QX,        6.75 - QH/2, 4.50, PY + PH/2, col=P["phase2"])
    arr(QX + 0.15, 6.75 - QH/2, 7.40, PY + PH/2, col=P["phase3"])
    arr(1.60, PY - PH/2, QX - 0.30, 4.10 + QH/2, col=P["phase1"])
    arr(4.50, PY - PH/2, QX,        4.10 + QH/2, col=P["phase2"])
    arr(7.40, PY - PH/2, QX + 0.30, 4.10 + QH/2, col=P["phase3"])

    # ── VDB ↔ Top-K Retrieval: two clearly separated arrows ──────────────
    VDB_LEFT  = VDB_CX - VDB_W / 2
    RETR_RIGHT = QX + QW / 2
    # OUT: search query — departs top-right of retrieval, arrives top-left of VDB
    arr(RETR_RIGHT, 6.75 + 0.20,
        VDB_LEFT,   VDB_CY + 0.20,
        col=P["green"], lw=2.5)
    lbl((RETR_RIGHT + VDB_LEFT) / 2,
        (6.75 + 0.20 + VDB_CY + 0.20) / 2 + 0.35,
        "search query →", col=P["green"], bold=True)
    # IN: chunks + KV tensors — departs bottom-left of VDB, arrives bottom-right of retrieval
    arr(VDB_LEFT,   VDB_CY - 0.20,
        RETR_RIGHT, 6.75 - 0.20,
        col=P["teal"], lw=2.5, cs="arc3,rad=0.25")
    lbl((RETR_RIGHT + VDB_LEFT) / 2,
        (VDB_CY - 0.20 + 6.75 - 0.20) / 2 - 0.45,
        "← top-K chunks + KV tensors", col=P["teal"], bold=True)

    # ═════════════════════════════════════════════════════════════════════
    # 4. TRAINING LOOP  (right column, GPU Server)
    # ═════════════════════════════════════════════════════════════════════
    sec_label(9.7, 9.25, "③ Training Loop  (Sleep-time)", P["orange"])

    TX, TW, TH = 13.0, 3.2, 0.85
    t_specs = [
        ("Access Tracker",     "record_access(chunk_id)"),
        ("Tier Classifier",    "hot · warm · cold · frozen"),
        ("Replay Buffer",      "tier-weighted  8×/4×/2×/1×"),
        ("Sleep-time FAQ Gen", "Cloud LLM — offline"),
        ("LoRA Fine-tuning",   "r=16 · QLoRA 4-bit"),
    ]
    T_GAP = 0.40
    T_CY  = []
    yc = VDB_CY - VDB_H/2 - 0.55 - TH/2
    for _ in t_specs:
        T_CY.append(yc)
        yc -= TH + T_GAP

    for (lt, st), yc in zip(t_specs, T_CY):
        box(TX, yc, TW, TH, lt, st, fc=WHITE, ec=P["orange"])
    for i in range(len(T_CY) - 1):
        arr(TX, T_CY[i] - TH/2, TX, T_CY[i+1] + TH/2, col=P["orange"])

    SX, SW = 17.8, 2.40
    box(SX, T_CY[-1], SW, TH, "PRS Evaluator",
        "≥0.75→Ph2\n≥0.80×2→Ph3", fc=WHITE, ec=P["red"])
    box(SX, T_CY[2],  SW, TH, "KV Recompute\nDaemon",
        "hot-first priority", fc=WHITE, ec=P["blue"])

    arr(TX + TW/2, T_CY[-1], SX - SW/2, T_CY[-1], col=P["red"],  lw=2.2)
    arr(SX, T_CY[-1] + TH/2, SX, T_CY[2] - TH/2,  col=P["blue"], lw=2.2)
    arr(SX, T_CY[2]  + TH/2, VDB_CX + VDB_W/2,
        VDB_CY, col=P["blue"], lw=2.2, cs="arc3,rad=-0.20")

    # VDB → Access Tracker
    vdb_bot     = VDB_CY - VDB_H/2
    tracker_top = T_CY[0] + TH/2
    arr(TX, vdb_bot, TX, tracker_top, col=P["gray"], lw=2.0)
    lbl(TX - 1.9, (vdb_bot + tracker_top) / 2,
        "access events", col=P["gray"])

    # ── Machine annotations (plain text in empty areas, no bounding boxes) ──
    # GPU Server label — bottom-left corner, well clear of all boxes
    ax.text(0.40, 0.50,
            "[ GPU / Inference Server ]",
            ha="left", va="center", fontsize=FS_ARR,
            color="#1E3A8A", fontweight="bold", fontstyle="italic",
            alpha=0.75, zorder=5)
    # Vector Store label — sits just below the VDB box in the gap above training boxes
    ax.text(VDB_CX, VDB_CY - VDB_H/2 - 0.28,
            "[ Vector Store Server — any host, any supported backend ]",
            ha="center", va="center", fontsize=FS_ARR,
            color=P["teal"], fontweight="bold", fontstyle="italic",
            alpha=0.85, zorder=5)

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title("Figure 2 — KVForge System Architecture",
                 fontsize=18, fontweight="bold", pad=16)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_system_architecture.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig02_system_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 – Confidence Gate signal weights (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────
def fig_confidence_gate():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: signal weights
    ax = axes[0]
    signals = ["Token Entropy\n(inverted)", "Hedging Score\n(inverted)", "Query Similarity\nto Known-Good"]
    weights_gate = [0.4, 0.3, 0.3]
    cols_g = [P["blue"], P["orange"], P["green"]]
    wedges, texts, autotexts = ax.pie(
        weights_gate, labels=signals, colors=cols_g,
        autopct="%1.0f%%", startangle=90,
        pctdistance=0.65, textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontweight("bold"); at.set_fontsize(11)
    ax.set_title("P(no_retrieval) Signal Weights", fontsize=11, pad=10)

    # Right: gate decision surface (entropy vs similarity, hedging fixed = 0.1)
    ax2 = axes[1]
    entropy_vals = np.linspace(0, 1, 100)
    sim_vals     = np.linspace(0, 1, 100)
    E, S = np.meshgrid(entropy_vals, sim_vals)
    hedging = 0.10
    P_gate   = 0.4 * (1 - E) + 0.3 * (1 - hedging) + 0.3 * S

    cf = ax2.contourf(entropy_vals, sim_vals, P_gate, levels=20,
                       cmap="RdYlGn", vmin=0, vmax=1)
    cs = ax2.contour(entropy_vals, sim_vals, P_gate, levels=[0.75],
                      colors=P["dark"], linewidths=2.5)
    ax2.clabel(cs, fmt="threshold 0.75", fontsize=9, colors=P["dark"])
    plt.colorbar(cf, ax=ax2, label="P(no_retrieval)")

    ax2.set_xlabel("Token Entropy (high = uncertain)", fontsize=11)
    ax2.set_ylabel("Query Similarity to Known-Good", fontsize=11)
    ax2.set_title("Decision Surface (hedging fixed = 0.10)\nGreen = Phase 3, Red = Phase 2 fallback",
                  fontsize=10, pad=8)
    ax2.set_facecolor("white")

    fig.suptitle("Figure 6 — Phase 3 Confidence Gate: Signal Weights and Decision Surface",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig06_confidence_gate.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig06_confidence_gate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14 – Pipeline timing breakdown (UC4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pipeline_timing():
    fig, ax = plt.subplots(figsize=(9, 4.2))

    steps = ["Chunk + Embed\n+ Upsert", "KV Tensor\nComputation",
              "LoRA Training\n(3 epochs)", "KV Recompute\n(post-training)",
              "PRS Evaluation"]
    times_s = [45, 498, 474, 510, 90]    # seconds
    times_m = [t / 60 for t in times_s]
    cols_p  = [P["teal"], P["blue"], P["orange"], P["blue"], P["green"]]

    bars = ax.bar(steps, times_m, color=cols_p, width=0.6, zorder=3)
    for bar, tm, ts in zip(bars, times_m, times_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{ts}s\n({tm:.1f}m)", ha="center", va="bottom", fontsize=9.5)

    ax.axhline(sum(times_m), color=P["red"], lw=2, ls="--",
               label=f"Total: {sum(times_m):.0f} min")
    ax.text(4.45, sum(times_m) + 0.3, f"Total: ~{sum(times_m):.0f} min",
            ha="right", fontsize=10, color=P["red"], fontweight="bold")

    ax.set_ylabel("Duration (minutes)", fontsize=11)
    ax.set_title("Figure — UC4 Pipeline Timing (one training round)\n"
                 "Amazon Bedrock User Guide · 2,520 chunks · AWS g5.12xlarge (4× A10G)",
                 fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline_timing.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig_pipeline_timing.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 15 – KV recompute hot-first benefit
# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 – V2 Storage Tier Architecture
# ─────────────────────────────────────────────────────────────────────────────
def fig_storage_tier_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tier_data = [
        {
            "label": "ENHANCED TIER",
            "cis": "HIGH CIS  (top ~10% of corpus)",
            "color": P["green"],
            "y0": 6.0,
            "height": 2.4,
            "attrs": [
                "Full per-token KV sequences stored on disk",
                "TurboQuant compressed: 3-bit keys + 4-bit values",
                "4.4× smaller than float16 · full attention fidelity",
                "Llama-3.1-8B: ~15 MB / chunk  (vs 67 MB float16)",
                "Direct attention estimation — no decompression required",
            ],
        },
        {
            "label": "ACTIVE TIER",
            "cis": "MID CIS  (all new chunks; retained mid-CIS)",
            "color": P["phase2"],
            "y0": 3.3,
            "height": 2.4,
            "attrs": [
                "Mean-pooled KV blob in Qdrant payload  [V1 behavior]",
                "Shape: [L × 2 × H × d_h] float16 → base64 string",
                "Llama-3.2-3B: ~57 KB / chunk   (Llama-3.1-8B: ~131 KB)",
                "Instant availability — no extra disk I/O at query time",
            ],
        },
        {
            "label": "ARCHIVE TIER",
            "cis": "LOW CIS  (low uniqueness; user-confirmed)",
            "color": P["gray"],
            "y0": 0.6,
            "height": 2.4,
            "attrs": [
                "Embedding vector + metadata pointer only in Qdrant (~8 KB)",
                "Chunk text archived to S3 / local filesystem backend",
                "Still retrievable: falls back to Phase 1 text-in-context RAG",
                "Re-access tracking triggers reinstatement alert to admin",
            ],
        },
    ]

    for td in tier_data:
        # Background box
        col = td["color"]
        rect = FancyBboxPatch((0.6, td["y0"]), 9.0, td["height"],
                               boxstyle="round,pad=0.18,rounding_size=0.22",
                               fc=col + "18", ec=col, lw=2.5, zorder=2)
        ax.add_patch(rect)
        # Tier label
        ax.text(1.05, td["y0"] + td["height"] - 0.25, td["label"],
                fontsize=13, fontweight="bold", color=col, va="top", zorder=4)
        ax.text(1.05, td["y0"] + td["height"] - 0.60, td["cis"],
                fontsize=9.5, color=col, va="top", fontstyle="italic", zorder=4)
        for i, attr in enumerate(td["attrs"]):
            ax.text(1.2, td["y0"] + td["height"] - 1.0 - i * 0.34,
                    f"• {attr}", fontsize=9.5, color=P["dark"], va="top", zorder=4)

    # CIS gradient arrow on left
    ax.annotate("", xy=(0.22, tier_data[2]["y0"] + 0.6),
                xytext=(0.22, tier_data[0]["y0"] + tier_data[0]["height"] - 0.3),
                arrowprops=dict(arrowstyle="-|>", color=P["dark"], lw=2.0,
                                mutation_scale=16))
    ax.text(0.14, (tier_data[0]["y0"] + tier_data[2]["y0"]) / 2 + 1.5,
            "CIS\ndecreasing", ha="center", fontsize=9, color=P["dark"],
            rotation=90, style="italic")

    ax.set_title("Figure 7 — KVForge V2: Three-Tier Storage Architecture",
                 pad=14, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig07_storage_tier_architecture.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig07_storage_tier_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
def fig_recompute_strategy():
    fig, ax = plt.subplots(figsize=(9, 4.0))

    # Cumulative traffic coverage vs recompute progress
    progress = np.linspace(0, 100, 200)
    # Hot-first: 80% traffic covered by 20% of recompute (Pareto)
    hot_first = np.where(
        progress <= 20,
        progress * 4.0,
        80 + (progress - 20) * (20/80)
    ).clip(0, 100)
    # Uniform order: linear
    uniform = progress

    ax.fill_between(progress, hot_first, uniform, alpha=0.18, color=P["green"],
                    label="Advantage of hot-first recompute")
    ax.plot(progress, hot_first, lw=2.5, color=P["green"], label="Hot-first order (KVForge)")
    ax.plot(progress, uniform,   lw=2,   color=P["gray"],  ls="--", label="Uniform order (baseline)")

    # Annotation: 80/20 point
    ax.annotate("80% of traffic has\nfresh KV tensors\nat 20% recompute",
                 xy=(20, 80), xytext=(35, 70),
                 fontsize=9, color=P["green"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=P["green"], lw=1.5))
    ax.axvline(20, color=P["green"], lw=1.5, ls=":", alpha=0.7)
    ax.axhline(80, color=P["green"], lw=1.5, ls=":", alpha=0.7)

    ax.set_xlabel("% of Corpus Recomputed (by chunk count)", fontsize=11)
    ax.set_ylabel("% of Live Traffic with Fresh KV Tensors", fontsize=11)
    ax.set_title("Figure — Tier-Ordered KV Recomputation: Traffic Coverage vs Progress\n"
                 "Assuming Pareto distribution (20% chunks → 80% traffic)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(OUT / "fig_recompute_strategy.png", **SAVE_KW)
    plt.close(fig)
    print("✓ fig_recompute_strategy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating KVForge paper figures...")
    fig_phase_state_machine()
    fig_system_architecture()
    fig_rag_vs_kv()
    fig_tier_weights()
    fig_prs_components()
    fig_confidence_gate()
    fig_storage_tier_architecture()
    fig_turboquant_compression()
    fig_cis_radar()
    fig_storage_comparison()
    fig_turboquant_compression()
    fig_latency()
    fig_prs_by_uc()
    fig_sleep_time_effect()
    fig_prs_progression()
    fig_pipeline_timing()
    fig_recompute_strategy()

    figs = sorted(Path("docs/figures").glob("*.png"))
    print(f"\nGenerated {len(figs)} figures in docs/figures/:")
    for f in figs:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.0f} KB)")
