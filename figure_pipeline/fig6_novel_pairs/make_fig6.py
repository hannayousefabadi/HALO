#!/usr/bin/env python3
"""
Generate Figure 6:

Reporting top synergistic novel pairs predicted by HALO. plotting similarity features in HALO training dataset
and predicted novel pairs.

Panel outputs:
- ‌B: 
- C: 
"""

import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halo.paths import FIGURE_PIPELINE, MODEL_RESULTS

OUT_DIR = FIGURE_PIPELINE / "fig6_novel_pairs" / "fig6_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = MODEL_RESULTS / "external_validation" / "novel_pairs" / "cc_similarity_summary.csv"
df = pd.read_csv(summary_path)
df["Group"] = df["Set"].astype(str) + "-" + df["Interaction Type"].astype(str)

DPI = 600


# Normalize interaction/type labels
df["Interaction Type"] = df["Interaction Type"].str.lower()
df["Set"] = df["Set"].str.lower()

# Clean legend groups
df["LegendGroup"] = df["Set"].map({
    "training": "Training drug pairs",
    "novel": "Novel drug pairs"
})

# Strong contrasting colors
colors = {
    "Training drug pairs": "#1F77B4",   # blue
    "Novel drug pairs": "#E78181"       # orange
}

# ---- FIGURE ----
fig = plt.figure(figsize=(11, 5), dpi=DPI)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# ==========================
# Panel B – Synergy Only
# ==========================
axB = fig.add_subplot(gs[0, 0])
syn_df = df[df["Interaction Type"] == "synergy"]

for legend_name, subdf in syn_df.groupby("LegendGroup"):
    axB.scatter(
        subdf["cc_cosine_sd"],
        subdf["cc_euclidean_sd"],
        s=22,
        alpha=0.30,
        label=legend_name,
        color=colors[legend_name]
    )

axB.set_title(r"$\mathbf{B.}$  Synergy Similarity Spread")
axB.set_xlabel("Cosine Similarity SD")
axB.set_ylabel("Euclidean Similarity SD")
axB.legend(frameon=False, fontsize=8, title="")

# ==========================
# Panel C – Antagonism Only
# ==========================
axC = fig.add_subplot(gs[0, 1])
ant_df = df[df["Interaction Type"] == "antagonism"]

for legend_name, subdf in ant_df.groupby("LegendGroup"):
    axC.scatter(
        subdf["cc_cosine_sd"],
        subdf["cc_euclidean_sd"],
        s=22,
        alpha=0.30,
        color=colors[legend_name],
        label=legend_name
    )

axC.set_title(r"$\mathbf{C.}$  Antagonism Similarity Spread")
axC.set_xlabel("Cosine Similarity SD")
axC.set_ylabel("Euclidean Similarity SD")
axC.legend(frameon=False, fontsize=8, title="")

for ax in [axB, axC]:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-2, 2))

# --------------------------
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_panelB_C.png", dpi=DPI)
plt.close()
