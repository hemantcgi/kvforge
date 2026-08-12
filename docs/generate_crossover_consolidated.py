"""Single consolidated Gemma 4 crossover chart — all 4 datasets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "legend.fontsize": 10.5,
    "figure.dpi": 200,
})

datasets = [
    {
        "name": "UC1: Customer Support",
        "color": "#1b9e77",
        "text_rag": [(200, 0.215, 0.011), (500, 0.220, 0.037), (1000, 0.192, 0.025)],
        "parametric": [(200, 0.360, 0.004), (500, 0.340, 0.027), (1000, 0.356, 0.006)],
    },
    {
        "name": "UC2: PubMedQA",
        "color": "#d95f02",
        "text_rag": [(50, 0.229, 0.033), (100, 0.191, 0.021), (200, 0.187, 0.014), (297, 0.195, 0.012)],
        "parametric": [(50, 0.201, 0.022), (100, 0.207, 0.016), (200, 0.203, 0.011), (297, 0.190, 0.009)],
    },
    {
        "name": "UC3: SQuAD 2.0",
        "color": "#7570b3",
        "text_rag": [(500, 0.144, 0.016), (1000, 0.161, 0.016), (2000, 0.163, 0.016)],
        "parametric": [(500, 0.221, 0.015), (1000, 0.221, 0.015), (2000, 0.221, 0.015)],
    },
    {
        "name": "UC4: Bedrock User Guide",
        "color": "#e7298a",
        "text_rag": [(500, 0.219, None), (1000, 0.198, None), (2000, 0.136, None), (4000, 0.116, None), (6000, 0.109, None)],
        "parametric": [(500, 0.234, None), (1000, 0.228, None), (2000, 0.234, None), (4000, 0.234, None), (6000, 0.234, None)],
    },
]

fig, ax = plt.subplots(figsize=(12, 7))

# Parity line
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
ax.axhline(y=0.10, color="gray", linewidth=0.3, linestyle=":", alpha=0.2)
ax.axhline(y=0.20, color="gray", linewidth=0.3, linestyle=":", alpha=0.2)
ax.axhline(y=0.30, color="gray", linewidth=0.3, linestyle=":", alpha=0.2)
ax.axhline(y=0.40, color="gray", linewidth=0.3, linestyle=":", alpha=0.2)

legend_handles = []
crossover_markers = []

for ds in datasets:
    c = ds["color"]
    tr = np.array(ds["text_rag"])
    pr = np.array(ds["parametric"])

    # text_rag: dashed line with X markers
    (l1,) = ax.plot(tr[:, 0], tr[:, 1], "X--", color=c, linewidth=2, markersize=8, alpha=0.7)
    # parametric: solid line with filled circles
    (l2,) = ax.plot(pr[:, 0], pr[:, 1], "o-", color=c, linewidth=2.5, markersize=9, label=ds["name"])

    # Error bars for text_rag
    for i in range(len(tr)):
        if tr[i, 2] is not None:
            ax.errorbar(tr[i, 0], tr[i, 1], yerr=tr[i, 2], fmt="none", color=c, capsize=4, alpha=0.35)
    for i in range(len(pr)):
        if pr[i, 2] is not None:
            ax.errorbar(pr[i, 0], pr[i, 1], yerr=pr[i, 2], fmt="none", color=c, capsize=4, alpha=0.35)

    # Crossover annotations
    for i in range(len(tr)):
        if pr[i, 1] > tr[i, 1]:
            is_crossover = (i == 0) or (pr[i - 1, 1] <= tr[i - 1, 1])
            if is_crossover and i > 0:
                # Draw vertical span showing crossover region
                ax.axvline(x=pr[i, 0], color=c, linestyle=":", linewidth=1.5, alpha=0.5)
                ax.annotate(
                    f"N={int(pr[i,0])}",
                    xy=(pr[i, 0], max(pr[i, 1], tr[i, 1])),
                    xytext=(0, 10),
                    textcoords="offset points",
                    fontsize=8.5, color=c, ha="center", fontweight="bold",
                )

    # Dataset name label at last data point
    ax.annotate(
        ds["name"].split(":")[0],
        xy=(pr[-1, 0], pr[-1, 1]),
        xytext=(15, 3),
        textcoords="offset points",
        fontsize=9, color=c, fontweight="bold", va="center",
    )

    legend_handles.append((l2, l1))

# Legend: parametric (solid) vs text_rag (dashed)
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color="black", linewidth=2.5, marker="o", markersize=8, label="parametric (Gemma 4 fine-tuned)"),
    Line2D([0], [0], color="black", linewidth=2, marker="x", markersize=8, linestyle="--", label="text_rag (retrieval baseline)"),
]
# Add dataset color legend
for ds in datasets:
    custom_lines.append(Line2D([0], [0], color=ds["color"], linewidth=3, label=ds["name"]))

ax.legend(handles=custom_lines, loc="upper left", framealpha=0.92, ncol=1)

ax.set_xscale("log")
ax.set_xlabel("Training FAQ Pairs (N, log scale)")
ax.set_ylabel("Factual Accuracy (FA)")
ax.set_title(
    "Gemma 4 Crossover: Parametric vs. Text-RAG Factual Accuracy\n"
    "Solid = parametric (fine-tuned)   ·   Dashed = text_rag (retrieval)",
    fontsize=14, fontweight="bold", pad=18
)
ax.set_ylim(0.06, 0.42)
ax.grid(True, alpha=0.25)

# Nicer x ticks
ticks = [50, 100, 200, 300, 500, 1000, 2000, 4000, 6000]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], rotation=0)

# Add a note about crossover regions
ax.text(
    0.99, 0.02,
    "Vertical dotted lines = crossover points (parametric > text_rag)",
    transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
    color="gray", style="italic",
)

plt.tight_layout()

outpath = "docs/figures/gemma4_crossover_consolidated.png"
plt.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
