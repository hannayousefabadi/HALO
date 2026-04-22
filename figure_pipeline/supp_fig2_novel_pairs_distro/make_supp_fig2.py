#!/usr/bin/env python3
"""
Generate Supplementary Figure 2:

Reporting top synergistic novel pairs predicted by HALO. plotting similarity features in HALO training dataset
and predicted novel pairs.

Panel outputs:
- ‌A: Synergy Similarity Spread
- B: Antagonism Similarity Spread
"""

import pandas as pd
from matplotlib.ticker import ScalarFormatter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from halo.paths import MODEL_RESULTS, FIGURES

OUT_DIR = FIGURES / "supplementary" 
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = MODEL_RESULTS / "external_validation" / "novel_pairs" / "cc_similarity_summary.csv"
df = pd.read_csv(summary_path)
df["Group"] = df["Set"].astype(str) + "-" + df["Interaction Type"].astype(str)

DPI = 600


df["Interaction Type"] = df["Interaction Type"].str.lower()
df["Set"] = df["Set"].str.lower()

# legend groups
df["LegendGroup"] = df["Set"].map({
    "training": "Training drug pairs",
    "novel": "Novel drug pairs"
})

# colors
colors = {
    "Training drug pairs": "#1F77B4",
    "Novel drug pairs": "#E78181"       
}

fig = plt.figure(figsize=(11, 5), dpi=DPI)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# ==========================
# Panel A – synergy pairs
# ==========================
axA = fig.add_subplot(gs[0, 0])
syn_df = df[df["Interaction Type"] == "synergy"]

for legend_name, subdf in syn_df.groupby("LegendGroup"):
    axA.scatter(
        subdf["cc_cosine_sd"],
        subdf["cc_euclidean_sd"],
        s=22,
        alpha=0.30,
        label=legend_name,
        color=colors[legend_name]
    )

axA.set_title(r"$\mathbf{A.}$  Synergy Similarity Spread")
axA.set_xlabel("Cosine Similarity SD")
axA.set_ylabel("Euclidean Similarity SD")
axA.legend(frameon=False, fontsize=12, title="")

# ==========================
# Panel B – antagonism pair
# ==========================
axB = fig.add_subplot(gs[0, 1])
ant_df = df[df["Interaction Type"] == "antagonism"]

for legend_name, subdf in ant_df.groupby("LegendGroup"):
    axB.scatter(
        subdf["cc_cosine_sd"],
        subdf["cc_euclidean_sd"],
        s=22,
        alpha=0.30,
        color=colors[legend_name],
        label=legend_name
    )

axB.set_title(r"$\mathbf{C.}$  Antagonism Similarity Spread")
axB.set_xlabel("Cosine Similarity SD")
axB.set_ylabel("Euclidean Similarity SD")
axB.legend(frameon=False, fontsize=12, title="")

for ax in [axA, axB]:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-2, 2))


plt.tight_layout()
plt.savefig(OUT_DIR / "supp_fig2.png", dpi=DPI)
plt.close()
