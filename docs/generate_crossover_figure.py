"""Generate Gemma 4 crossover summary figure for all datasets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ── Data: (N, text_rag_FA, text_rag_sem, param_FA, param_sem) ──────────────

datasets = {
    "UC1: Customer Support": {
        "text_rag": [(200, 0.215, 0.011), (500, 0.220, 0.037), (1000, 0.192, 0.025)],
        "parametric": [(200, 0.360, 0.004), (500, 0.340, 0.027), (1000, 0.356, 0.006)],
    },
    "UC2: PubMedQA Biomedical": {
        "text_rag": [(50, 0.229, 0.033), (100, 0.191, 0.021), (200, 0.187, 0.014), (297, 0.195, 0.012)],
        "parametric": [(50, 0.201, 0.022), (100, 0.207, 0.016), (200, 0.203, 0.011), (297, 0.190, 0.009)],
    },
    "UC3: SQuAD 2.0": {
        "text_rag": [(500, 0.144, 0.016), (1000, 0.161, 0.016), (2000, 0.163, 0.016)],
        "parametric": [(500, 0.221, 0.015), (1000, 0.221, 0.015), (2000, 0.221, 0.015)],
    },
    "UC4: Bedrock User Guide": {
        "text_rag": [(500, 0.219, None), (1000, 0.198, None), (2000, 0.136, None), (4000, 0.116, None), (6000, 0.109, None)],
        "parametric": [(500, 0.234, None), (1000, 0.228, None), (2000, 0.234, None), (4000, 0.234, None), (6000, 0.234, None)],
    },
}

COLOR_TR = "#1f77b4"      # text_rag blue
COLOR_PR = "#d62728"      # parametric red
COLOR_CR = "#2ca02c"      # crossover marker green

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (name, data) in zip(axes, datasets.items()):
    tr = np.array(data["text_rag"])
    pr = np.array(data["parametric"])

    tr_n, tr_fa, tr_sem = tr[:, 0], tr[:, 1], tr[:, 2]
    pr_n, pr_fa, pr_sem = pr[:, 0], pr[:, 1], pr[:, 2]

    # Lines
    ax.plot(tr_n, tr_fa, "o-", color=COLOR_TR, linewidth=2, markersize=7, label="text_rag (retrieval)")
    ax.plot(pr_n, pr_fa, "s-", color=COLOR_PR, linewidth=2, markersize=7, label="parametric (Gemma 4)")

    # Error bars
    for i in range(len(tr_n)):
        if tr_sem[i] is not None:
            ax.errorbar(tr_n[i], tr_fa[i], yerr=tr_sem[i], fmt="none", color=COLOR_TR, capsize=4, alpha=0.5)
    for i in range(len(pr_n)):
        if pr_sem[i] is not None:
            ax.errorbar(pr_n[i], pr_fa[i], yerr=pr_sem[i], fmt="none", color=COLOR_PR, capsize=4, alpha=0.5)

    # Find and mark crossover points
    for i in range(len(tr_n)):
        if pr_fa[i] > tr_fa[i]:
            prev_was_text_rag = (i > 0 and pr_fa[i - 1] <= tr_fa[i - 1])
            if prev_was_text_rag or i == 0:
                ax.axvline(x=pr_n[i], color=COLOR_CR, linestyle="--", linewidth=1.2, alpha=0.7)
                ax.annotate(
                    "crossover",
                    xy=(pr_n[i], pr_fa[i]),
                    xytext=(0, -22),
                    textcoords="offset points",
                    fontsize=9,
                    color=COLOR_CR,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=COLOR_CR, lw=1.2),
                )

    # Shade region where parametric > text_rag
    for i in range(1, len(tr_n)):
        x_start = tr_n[i - 1]
        x_end = tr_n[i]
        if pr_fa[i - 1] > tr_fa[i - 1] and pr_fa[i] > tr_fa[i]:
            ax.axvspan(x_start, x_end, alpha=0.06, color=COLOR_CR)

    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Training FAQ Pairs (N)")
    ax.set_ylabel("Factual Accuracy (FA)")
    ax.set_ylim(0.05, 0.42)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    # Nice x-axis ticks
    all_n = sorted(set(int(n) for n in tr_n))
    ax.set_xticks(all_n)
    ax.set_xticklabels([str(n) for n in all_n])

fig.suptitle(
    "Gemma 4 Crossover: Parametric vs. Text-RAG Factual Accuracy by Training Scale",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()

outpath = "docs/figures/gemma4_crossover_summary.png"
plt.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
