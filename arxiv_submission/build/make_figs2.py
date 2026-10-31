#!/usr/bin/env python3
"""Regenerate fig2 (phase flow) and fig3 (tier weights) properly — matching the
V3 content, title-less, column-size vector PDFs, with the box-overlap and
mislabel defects fixed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "DejaVu Sans"})
OUT = Path(__file__).parent / "figures"
P = {"phase1": "#6B7280", "phase2": "#2563EB", "phase3": "#16A34A",
     "gray": "#6B7280", "dark": "#1E293B", "light": "#F1F5F9",
     "blue": "#2563EB", "orange": "#D97706"}
SAVE = dict(bbox_inches="tight", facecolor="white")


def fig2_phase_flow():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.4))
    steps_rag = [("Embed Query", False),
                 ("Vector Search →\nTop-K Chunks", False),
                 ("Concatenate chunk text into\ncontext window (~3,000 tokens)", False),
                 ("LLM full forward pass over all\ncontext tokens —\nREPEATED EVERY QUERY", True),
                 ("Autoregressive token generation", False)]
    steps_kv = [("Embed Query", False),
                ("Vector Search →\nTop-K Chunks", False),
                ("Load pre-computed kv_cache\ntensors from vector store", False),
                ("Inject past_key_values into\nattention cache —\nSKIPS RE-ENCODING", True),
                ("Autoregressive token generation", False)]
    steps_param = [("Embed Query", False),
                   ("Confidence Gate: entropy +\nhedging + query sim. ≥ 0.75", False),
                   ("NO retrieval, NO KV injection —\ncorpus knowledge in weights", True),
                   ("Fine-tuned LLM generates answer\nfrom parametric memory only", True),
                   ("Autoregressive token generation", False)]
    BOX_H = [0.085, 0.115, 0.145, 0.165, 0.085]
    GAP, MARGIN, TOP_Y = 0.045, 0.04, 0.95
    y_tops, y = [], TOP_Y
    for h in BOX_H:
        y_tops.append(y); y -= (h + GAP)
    phases = [("Phase 1 — Standard RAG", P["phase1"], steps_rag),
              ("Phase 2 — KV Injection  (KVForge)", P["phase2"], steps_kv),
              ("Phase 3 — Parametric  (KVForge)", P["phase3"], steps_param)]
    FS = 6.6
    for ax, (title, col, steps) in zip(axes, phases):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=7.6, fontweight="bold", color=col, pad=4)
        for i, ((label, key), h, ytop) in enumerate(zip(steps, BOX_H, y_tops)):
            ybot = ytop - h
            ax.add_patch(FancyBboxPatch((MARGIN, ybot), 1 - 2*MARGIN, h,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         fc=col if key else P["light"], ec=col,
                         lw=1.6 if key else 0.9, zorder=2))
            ax.text(0.5, (ytop + ybot)/2, label, ha="center", va="center",
                    fontsize=FS, color="white" if key else P["dark"],
                    fontweight="bold" if key else "normal")
            if i < len(steps) - 1:
                ax.annotate("", xy=(0.5, y_tops[i+1] + 0.006), xytext=(0.5, ybot - 0.006),
                            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=P["gray"],
                                            mutation_scale=8))
        if col == P["phase3"]:
            last_bot = y_tops[-1] - BOX_H[-1]
            ax.text(0.5, last_bot - 0.055, "Low confidence →\nfallback to Phase 2",
                    ha="center", va="top", fontsize=6.2, color=P["gray"], style="italic")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig2_phase_flow.pdf", **SAVE)
    plt.close(fig); print("ok fig2_phase_flow.pdf")


def fig3_tier_weights():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axes[0]
    tiers = ["Frozen", "Cold", "Warm", "Hot"]; weights = [1, 2, 4, 8]
    tot = sum(weights)
    pcts = [f"{round(100*w/tot)}%" for w in weights]
    cols_t = [P["gray"], "#64748B", P["blue"], P["orange"]]
    bars = ax.barh(tiers, weights, color=cols_t, height=0.58)
    for bar, w, pct in zip(bars, weights, pcts):
        ax.text(w + 0.15, bar.get_y() + bar.get_height()/2,
                f"weight {w}  ({pct} of total weight)", va="center", fontsize=7.3)
    ax.set_xlim(0, 13); ax.set_xlabel("Sampling weight", fontsize=8)
    ax.set_title("Replay-buffer tier weights", fontsize=8.5, pad=6)
    ax.grid(axis="x", alpha=0.3); ax.tick_params(labelsize=7.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    labels2 = ["Hot\n(top 15%)", "Warm\n(next 50%)", "Cold", "Frozen"]
    sizes = [8*0.15, 4*0.50, 2*0.25, 1*0.10]
    sizes = [s/sum(sizes) for s in sizes]
    cols_pie = [P["orange"], P["blue"], "#64748B", P["gray"]]
    explode = (0.04, 0.02, 0.0, 0.10)
    wedges, texts, auto = ax2.pie(sizes, labels=labels2, colors=cols_pie, explode=explode,
                                  autopct=lambda p: f"{p:.0f}%", startangle=140,
                                  textprops={"fontsize": 7.6}, pctdistance=0.72)
    for i, (wg, at) in enumerate(zip(wedges, auto)):
        at.set_fontsize(7.6); at.set_fontweight("bold")
        if sizes[i]*100 < 8:   # move tiny-wedge % outside with a leader line
            ang = np.deg2rad((wg.theta1 + wg.theta2)/2)
            x, y = np.cos(ang), np.sin(ang)
            at.set_position((1.28*x, 1.28*y)); at.set_color("#333")
            ax2.annotate("", xy=(0.98*x, 0.98*y), xytext=(1.2*x, 1.2*y),
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="#999"))
    ax2.set_title("Expected mini-batch distribution\n(replay fraction = 20%)",
                  fontsize=8.5, pad=6)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / "fig3_tier_weights.pdf", **SAVE)
    plt.close(fig); print("ok fig3_tier_weights.pdf")


if __name__ == "__main__":
    fig2_phase_flow()
    fig3_tier_weights()
