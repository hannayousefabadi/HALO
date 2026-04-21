#!/usr/bin/env python3
"""
Generate Figure 4:

Feature analysis for HALO (exp06d) and CC vs SS (M2, exp06b)

Output panels:
- A: Top-20 features by gain-based importance (HALO, exp06d)
- B: Grouped importance (CC sublevels + strain-space) – HALO
- C: Grouped importance by CC top level + strain-space – HALO
- D: CC vs strain-space contributions – M2 (exp06b)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================
# Paths
# ==========================
from pathlib import Path
from halo.paths import MODEL_RESULTS, FEATURES, FIGURES

RESULT_DIR_EXP06D = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv"
FI_PATH_EXP06D = RESULT_DIR_EXP06D / "feature_importances_cv1.csv"

RESULT_DIR_EXP06B = MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv"
CC_VS_SS_SUMMARY_PATH = RESULT_DIR_EXP06B / "cc_vs_ss_importance_summary.csv"

PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG4_PNG = PLOT_DIR / "fig4.png"

FEATURE_META_PATH = FEATURES / "feature_metadata_cc_s_full.csv"

# ==========================
# Global style + palette
# ==========================

TITLE_SIZE = 16
LABEL_SIZE = 16 
TICK_SIZE = 12
ANNOT_SIZE = 12
BAR_LABEL_SIZE = 14

plt.rcParams.update(
    {
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)

# colors
MAIN_BLUE = "#1f77b4"
PASTEL_GREEN = "#A7D7A0"
PASTEL_PEACH = "#F7C9A9"
PASTEL_BLUE   = "#7BAFD4"

BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.4

# helpers
def short_label(text):
    replacements = {
        "Chemical genetics": "Chem. genetics",
        "Mechanisms of action": "Mech. of action",
        "Small molecule roles": "Small mol. roles",
        "Small molecule pathways": "Small mol. pathways",
        "Structural keys": "Struct. keys",
        "Metabolic genes": "Metab. genes",
        "Side effects": "Side effects",
        "2D fingerprints": "2D fingerprints",
        "Indications": "Indications",
        "Transcription": "Transcript.",
        "Therapeutic areas": "Therap. areas",
        "Diseases & toxicology": "Disease/tox.",
        "Cancer cell lines": "Cancer cell lines",
        "Signaling pathways": "Signaling paths.",
    }
    return replacements.get(text, text)

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ==========================
# Load data
# ==========================

def load_importances(fi_path: Path) -> pd.DataFrame:
    if not fi_path.exists():
        raise FileNotFoundError(fi_path)

    df = pd.read_csv(fi_path).copy()

    if "importance_gain_norm" not in df.columns:
        total_gain = df["importance_gain"].sum()
        if total_gain <= 0:
            raise ValueError("Total importance_gain is non-positive; cannot normalize.")
        df["importance_gain_norm"] = df["importance_gain"] / total_gain

    df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return df


def load_feature_meta():
    """
    Load base CC + strain-space metadata and return:
    - meta: indexed by original_name (dim_i, s_j)
    - n_cc_dims: number of CC dims, used to split CC vs strain in elementwise indexing
    """
    if not FEATURE_META_PATH.exists():
        raise FileNotFoundError(FEATURE_META_PATH)

    meta = pd.read_csv(FEATURE_META_PATH).copy()

    if "original_name" not in meta.columns:
        raise ValueError("feature_metadata_cc_s_full.csv must have 'original_name' column")

    meta = meta.set_index("original_name")

    cc_dims = meta[meta["space"] == "CC"]["dimension"].dropna()
    if cc_dims.empty:
        raise ValueError("No CC dimensions found in metadata (space == 'CC').")

    n_cc_dims = int(cc_dims.max()) + 1
    return meta, n_cc_dims


# ==========================
# Decode elementwise features using metadata
# ==========================

def decode_elementwise_feature(feat_name: str, meta: pd.DataFrame, n_cc_dims: int):
    """
    Map 'cos_elem_i' / 'euc_elem_i' to base feature metadata via feature_metadata_cc_s_full.csv.
    """
    name = feat_name.lower()

    if name.startswith("cos_elem_"):
        metric = "cosine"
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            idx = None
    elif name.startswith("euc_elem_"):
        metric = "euclidean"
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            idx = None
    else:
        return {
            "metric": "unknown",
            "space": "Unknown",
            "space_name": "Unknown",
            "cc_level": None,
            "cc_sublevel": None,
            "cc_level_name": None,
            "cc_sublevel_name": None,
            "group_label": "Unknown",
            "base_feature": None,
        }

    if idx is None:
        return {
            "metric": metric,
            "space": "Unknown",
            "space_name": "Unknown",
            "cc_level": None,
            "cc_sublevel": None,
            "cc_level_name": None,
            "cc_sublevel_name": None,
            "group_label": "Unknown",
            "base_feature": None,
        }

    # Map elementwise index -> original base feature name
    if idx < n_cc_dims:
        base_feature = f"dim_{idx}"
    else:
        s_idx = idx - n_cc_dims
        base_feature = f"s_{s_idx}"

    if base_feature not in meta.index:
        return {
            "metric": metric,
            "space": "Unknown",
            "space_name": "Unknown",
            "cc_level": None,
            "cc_sublevel": None,
            "cc_level_name": None,
            "cc_sublevel_name": None,
            "group_label": "Unknown",
            "base_feature": base_feature,
        }

    row = meta.loc[base_feature]

    space = row.get("space", "Unknown")
    space_name = row.get("space_name", "Unknown")
    cc_level = row.get("cc_level", None)
    cc_sublevel = row.get("cc_sublevel", None)
    cc_level_name = row.get("cc_level_name", None)
    cc_sublevel_name = row.get("cc_sublevel_name", None)
    group_label = row.get("group_label", space_name)

    return {
        "metric": metric,
        "space": space,
        "space_name": space_name,
        "cc_level": cc_level,
        "cc_sublevel": cc_sublevel,
        "cc_level_name": cc_level_name,
        "cc_sublevel_name": cc_sublevel_name,
        "group_label": group_label,
        "base_feature": base_feature,
    }


# ==========================
# Panel A – top-15 features
# ==========================

def plot_panel_A(ax, df_importance, meta, n_cc_dims, top_n=12):
    df = df_importance.copy()

    decoded = df["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    decoded_df = pd.DataFrame(list(decoded))
    df = pd.concat([df, decoded_df], axis=1)

    def build_label(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return f"{row['metric']} / Strain-space"

        sub = row.get("cc_sublevel_name")
        if isinstance(sub, str) and sub != "":
            return f"{row['metric']} / {short_label(sub)}"

        return f"{row['metric']} / {short_label(row['group_label'])}"

    df["nice_label"] = df.apply(build_label, axis=1)

    agg = (
        df.groupby("nice_label")["importance_gain"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    agg = agg.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(agg))

    ax.barh(
        y,
        agg["importance_gain"],
        color=MAIN_BLUE,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(agg["nice_label"], fontsize=BAR_LABEL_SIZE)
    ax.set_xlabel("Gain-based importance")

    ax.set_title(
        r"$\mathbf{A.}$  Top similarity feature groups by gain",
        fontsize=TITLE_SIZE,
        pad=8,
    )

    clean_axis(ax)

# ==========================
# Panel B – grouped importance (sublevels + strain)
# ==========================
def plot_panel_B(ax, df_importance, meta, n_cc_dims):
    """
    Panel B:
    CC domain contributions to normalized gain (HALO)
    """

    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))]

    decoded = df_imp["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    df_imp = pd.concat([df_imp, pd.DataFrame(list(decoded))], axis=1)

    def level_group(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return "Strain-space"

        if isinstance(row["cc_level_name"], str) and row["cc_level_name"]:
            return row["cc_level_name"]

        if row["space_name"] == "Chemical Checker":
            return "Chemical Checker"
        return "Unknown"

    df_imp["level_group"] = df_imp.apply(level_group, axis=1)

    grouped = (
        df_imp.groupby("level_group")["importance_gain_norm"]
        .sum()
        .reset_index()
    )

    grouped = grouped[grouped["level_group"] != "Strain-space"].copy()

    desired_order = ["Chemistry", "Targets", "Networks", "Cells", "Clinics"]
    grouped["level_group"] = pd.Categorical(
        grouped["level_group"], categories=desired_order, ordered=True
    )
    grouped = grouped.sort_values("level_group")

    x = np.arange(len(grouped))

    ax.bar(
        x,
        grouped["importance_gain_norm"],
        color=PASTEL_BLUE,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(grouped["level_group"], rotation=20, ha="right", fontsize=14)
    ax.set_ylabel("Total normalized gain importance")
    ax.set_ylim(0, grouped["importance_gain_norm"].max() * 1.22)

    ax.set_title(
        r"$\mathbf{B.}$  CC domain contributions",
        fontsize=TITLE_SIZE,
        pad=8,
    )

    for xi, yi in zip(x, grouped["importance_gain_norm"]):
        ax.text(xi, yi + 0.004, f"{yi:.1%}", ha="center", va="bottom", fontsize=14)

    clean_axis(ax)


# ==========================
# Panel C – grouped importance by CC top level + strain
# ==========================

def plot_panel_C(ax, df_importance, meta, n_cc_dims, top_n=10):
    """
    Panel C:
    Top CC sublevels by normalized gain (HALO)
    """

    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))]

    decoded = df_imp["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    df_imp = pd.concat([df_imp, pd.DataFrame(list(decoded))], axis=1)

    def sub_group(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return "Strain-space"

        sub = row.get("cc_sublevel_name")
        if isinstance(sub, str) and sub:
            return sub

        return row["group_label"]

    df_imp["sub_group"] = df_imp.apply(sub_group, axis=1)

    grouped = (
        df_imp.groupby("sub_group")["importance_gain_norm"]
        .sum()
        .sort_values(ascending=False)
    )

    grouped = grouped[grouped.index != "Strain-space"].head(top_n)

    plot_df = grouped.sort_values(ascending=True).reset_index()
    plot_df.columns = ["sub_group", "importance_gain_norm"]
    plot_df["sub_group"] = plot_df["sub_group"].map(short_label)

    ax.barh(
        plot_df["sub_group"],
        plot_df["importance_gain_norm"],
        color=PASTEL_GREEN,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    xmax = plot_df["importance_gain_norm"].max()
    ax.set_xlim(0, xmax * 1.16)

    for y, v in enumerate(plot_df["importance_gain_norm"]):
        ax.text(
            v + xmax * 0.02,
            y,
            f"{v:.1%}",
            va="center",
            ha="left",
            fontsize=12,
        )

    ax.set_xlabel("Total normalized gain importance")
    ax.set_title(
        r"$\mathbf{C.}$  Top CC sublevels by normalized gain",
        fontsize=TITLE_SIZE,
        pad=8,
    )
    ax.tick_params(axis="y", labelsize=14)

    clean_axis(ax)

# ==========================
# Panel D – CC vs SS (exp06b, M2)
# ==========================

def plot_panel_D(ax, cc_vs_ss_summary: pd.DataFrame):
    df = cc_vs_ss_summary.copy()

    desired_order = ["CC", "SS"]
    value_map = dict(zip(df["group"], df["fraction_of_total_gain"]))

    plot_df = pd.DataFrame({
        "group": desired_order,
        "fraction_of_total_gain": [value_map.get(g, 0.0) for g in desired_order],
    })

    colors = []
    for g, v in zip(plot_df["group"], plot_df["fraction_of_total_gain"]):
        if g == "SS" and v == 0:
            colors.append("white")
        elif g == "CC":
            colors.append(MAIN_BLUE)
        else:
            colors.append(PASTEL_PEACH)

    bars = ax.bar(
        plot_df["group"],
        plot_df["fraction_of_total_gain"],
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )

    ax.set_ylabel("Fraction of total gain importance")
    ax.set_ylim(0, 1.10)

    for bar, v in zip(bars, plot_df["fraction_of_total_gain"]):
        x = bar.get_x() + bar.get_width() / 2
        y_text = v + 0.025 if v > 0 else 0.025
        ax.text(
            x,
            y_text,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_SIZE,
        )

    ax.set_title(
        r"$\mathbf{D.}$  CC vs strain-space gain",
        fontsize=TITLE_SIZE,
        pad=8,
    )

    clean_axis(ax)


# ==========================
# Assemble Fig 4
# ==========================

def main():
    df_imp_exp06d = load_importances(FI_PATH_EXP06D)
    meta, n_cc_dims = load_feature_meta()

    if not CC_VS_SS_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"CC vs SS summary not found: {CC_VS_SS_SUMMARY_PATH} "
            "(run exp06b_cc_vs_ss_analysis.py first)."
        )
    cc_vs_ss_summary = pd.read_csv(CC_VS_SS_SUMMARY_PATH).copy()

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.55, 1.00],
    height_ratios=[1.00, 1.00],
    wspace=0.35,
    hspace=0.60,
)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    plot_panel_A(axA, df_imp_exp06d, meta, n_cc_dims, top_n=10)
    plot_panel_B(axB, df_imp_exp06d, meta, n_cc_dims)
    plot_panel_C(axC, df_imp_exp06d, meta, n_cc_dims, top_n=10)
    plot_panel_D(axD, cc_vs_ss_summary)

    fig.subplots_adjust(left=0.24, right=0.98, top=0.94, bottom=0.09)

    fig.savefig(FIG4_PNG, dpi=600)
    plt.close(fig)

    print("Saved Fig 4 PNG to:", FIG4_PNG)


if __name__ == "__main__":
    main()
