#!/usr/bin/env python3
"""Regenerate the 4 figures that need rebuilding for the arXiv V3 draft:
  fig1 state machine (correct 0.75/0.60 thresholds, no overlaps, no title)
  fig4 PRS component breakdown (UC4 R3, 1-column)
  fig5 PRS component scores across use cases (1-column)
  fig7 E2B/E4B accuracy comparison (fixed labels/legend, no title, full-width)
All title-less (LaTeX caption carries the title) and saved as vector PDF.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from pathlib import Path

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
})
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
P = {"phase1": "#6B7280", "phase2": "#2563EB", "phase3": "#16A34A",
     "red": "#DC2626", "gray": "#6B7280", "dark": "#1E293B",
     "blue": "#2563EB", "green": "#16A34A", "orange": "#D97706", "purple": "#7C3AED"}
SAVE = dict(bbox_inches="tight", facecolor="white")


def fig1_state_machine():
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3); ax.axis("off")

    def box(cx, cy, w, h, color, label, sub):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0.12,rounding_size=0.3",
                     fc=color, ec="white", lw=1.5, zorder=3))
        ax.text(cx, cy + 0.22, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)
        ax.text(cx, cy - 0.32, sub, ha="center", va="center",
                fontsize=7.5, color="white", zorder=4)

    def arr(x1, y1, x2, y2, col, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, mutation_scale=12,
                                    color=col, connectionstyle=f"arc3,rad={rad}"), zorder=2)

    cy = 2.15; w = 2.5; h = 1.35
    xs = [1.65, 5.0, 8.35]
    box(xs[0], cy, w, h, P["phase1"], "Phase 1", "Text-in-context RAG")
    box(xs[1], cy, w, h, P["phase2"], "Phase 2", "KV Cache Injection")
    box(xs[2], cy, w, h, P["phase3"], "Phase 3", "Parametric Answering")

    # start node, clearly left of Phase 1 (left edge x = 0.40)
    ax.add_patch(plt.Circle((0.18, cy), 0.12, color=P["dark"], zorder=3))
    arr(0.30, cy, 0.38, cy, P["dark"])

    rt1, lf2, rt2, lf3 = xs[0] + w/2, xs[1] - w/2, xs[1] + w/2, xs[2] - w/2  # box edges
    # forward transitions (upper band)
    arr(rt1, cy + 0.33, lf2, cy + 0.33, P["phase2"])
    arr(rt2, cy + 0.33, lf3, cy + 0.33, P["phase3"])
    ax.text((rt1 + lf2) / 2, cy + 0.95, "PRS ≥ 0.75", ha="center", fontsize=8,
            color=P["phase2"], fontweight="bold")
    ax.text((rt2 + lf3) / 2, cy + 0.95, "PRS ≥ 0.75\n(×2 rounds)", ha="center",
            fontsize=8, color=P["phase3"], fontweight="bold")
    # backward transitions (lower band)
    arr(lf2, cy - 0.33, rt1, cy - 0.33, P["gray"])
    arr(lf3, cy - 0.33, rt2, cy - 0.33, P["red"])
    ax.text((rt1 + lf2) / 2, cy - 1.25, "stale KV →\nPhase 1 fallback", ha="center",
            fontsize=7, color=P["gray"], fontstyle="italic")
    ax.text((rt2 + lf3) / 2, cy - 1.15, "PRS < 0.60\n(regression guard)", ha="center",
            fontsize=7.5, color=P["red"], fontstyle="italic")
    # LoRA self-loop on Phase 3
    arr(xs[2] + 0.55, cy + h/2 - 0.05, xs[2] + 0.55, cy + h/2 + 0.02, P["phase3"], rad=-0.9)
    ax.annotate("", xy=(xs[2] + 0.35, cy + h/2 + 0.12), xytext=(xs[2] - 0.1, cy + h/2 + 0.12),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, mutation_scale=10,
                                color=P["phase3"], connectionstyle="arc3,rad=-0.7"), zorder=2)
    ax.text(xs[2] + 0.75, cy + h/2 + 0.45, "LoRA\nround N", ha="center",
            fontsize=7.5, color=P["phase3"])
    fig.savefig(OUT / "fig1_state_machine.pdf", **SAVE)
    plt.close(fig); print("ok fig1_state_machine.pdf")


def fig4_prs_breakdown():
    comps = ["Accuracy\n(weight 0.50)", "Calibration\n(weight 0.30)", "Consistency\n(weight 0.20)"]
    score = [0.88, 0.82, 0.88]; weight = [0.50, 0.30, 0.20]
    contrib = [s * w for s, w in zip(score, weight)]  # 0.440, 0.246, 0.176
    cols = [P["blue"], P["orange"], P["green"]]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    bars = ax.bar(comps, contrib, color=cols, width=0.62)
    for b, s, c in zip(bars, score, contrib):
        ax.text(b.get_x() + b.get_width()/2, c + 0.008,
                f"score={s:.2f}\ncontrib={c:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("PRS contribution")
    ax.set_ylim(0, 0.52)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.savefig(OUT / "fig4_prs_breakdown.pdf", **SAVE)
    plt.close(fig); print("ok fig4_prs_breakdown.pdf")


def fig5_prs_scores():
    ucs = ["UC1\nCust. Support", "UC2\nPubMedQA", "UC3\nSQuAD", "UC4\nBedrock"]
    acc = [0.80, 0.88, 0.83, 0.88]; cal = [0.75, 0.82, 0.78, 0.82]; con = [0.77, 0.85, 0.80, 0.88]
    import numpy as np
    x = np.arange(len(ucs)); bw = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.bar(x - bw, acc, bw, label="Accuracy", color=P["blue"])
    ax.bar(x,       cal, bw, label="Calibration", color=P["orange"])
    ax.bar(x + bw,  con, bw, label="Consistency", color=P["green"])
    ax.set_ylabel("Component score"); ax.set_ylim(0.68, 0.95)
    ax.set_xticks(x); ax.set_xticklabels(ucs)
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=6.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#e8e8e8", lw=0.5); ax.set_axisbelow(True)
    fig.savefig(OUT / "fig5_prs_scores.pdf", **SAVE)
    plt.close(fig); print("ok fig5_prs_scores.pdf")


def fig7_e2b():
    # (label, FA, FA_sem, group) — embedded data from generate_e2b_comparison_charts.py
    RUNS = [
        ("E4B r64 · 60k QA + rewrites", 0.286, 0.013, "E4B"),
        ("Llama-3.2-3B · 60k QA + rewrites", 0.254, 0.012, "Llama"),
        ("E2B r64 · 60k QA + rewrites", 0.249, 0.012, "E2B"),
        ("E4B r64 · 60k QA", 0.244, 0.011, "E4B"),
        ("E4B r64 · 337 QA", 0.235, 0.009, "E4B"),
        ("E2B r64 · 60k QA", 0.219, 0.011, "E2B"),
        ("E2B RAG (untrained)", 0.175, 0.011, "RAG"),
        ("E4B RAG (untrained)", 0.161, 0.010, "RAG"),
        ("E2B baseline · r16 · 337 QA", 0.147, 0.007, "baseline"),
    ]
    C = {"E2B": "#2563EB", "E4B": "#16A34A", "Llama": "#7C3AED",
         "RAG": "#9E9E9E", "baseline": "#DC2626"}
    acc = sorted(RUNS, key=lambda r: r[1])  # ascending -> highest at top of barh
    labels = [r[0] for r in acc]; fa = [r[1] for r in acc]
    err = [r[2] for r in acc]; cols = [C[r[3]] for r in acc]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    bars = ax.barh(labels, fa, xerr=err, color=cols, capsize=2.5,
                   error_kw={"ecolor": "#333", "linewidth": 0.7})
    for b, v, e in zip(bars, fa, err):
        ax.text(v + e + 0.006, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", ha="left", fontsize=7.5)
    ax.set_xlabel("Factual accuracy  (FA = 0.5·token-F1 + 0.5·LLM-judge)")
    ax.set_xlim(0, 0.345)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", lw=0.5); ax.set_axisbelow(True)
    legend = [Patch(color=C["baseline"], label="E2B baseline (r16, 337 QA)"),
              Patch(color=C["E2B"], label="E2B trained"),
              Patch(color=C["E4B"], label="E4B trained"),
              Patch(color=C["Llama"], label="Llama-3.2-3B trained"),
              Patch(color=C["RAG"], label="RAG (untrained + retrieval)")]
    ax.legend(handles=legend, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=7)
    fig.savefig(OUT / "fig7_e2b_accuracy.pdf", **SAVE)
    plt.close(fig); print("ok fig7_e2b_accuracy.pdf")


if __name__ == "__main__":
    fig1_state_machine()
    fig4_prs_breakdown()
    fig5_prs_scores()
    fig7_e2b()
