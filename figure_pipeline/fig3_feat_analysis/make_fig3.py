#!/usr/bin/env python3
"""
Generate Figure 3:

Feature analysis for HALO-CV1 (exp06d) and CC vs SS (HALO-S-CV1, exp06b)

Output panels:
- A: Top-20 features by gain-based importance (HALO-CV1, exp06d)
- B: Grouped importance (CC sublevels + strain-space) – HALO-CV1
- C: Grouped importance by CC top level + strain-space – HALO-CV1
- D: CC vs strain-space contributions – HALO-S-CV1 (exp06b)
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

RESULT_DIR_EXP06D = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005"
FI_PATH_EXP06D = RESULT_DIR_EXP06D / "feature_importances_cv1.csv"

RESULT_DIR_EXP06B = MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005"
CC_VS_SS_SUMMARY_PATH = RESULT_DIR_EXP06B / "cc_vs_ss_importance_summary.csv"

PLOT_DIR = FIGURES / "main"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIG3_PNG = PLOT_DIR / "fig3.png"

FEATURE_META_PATH = FEATURES / "feature_metadata_cc_s_full.csv"

# ==========================
# Global style + palette
# ==========================

TITLE_SIZE = 14
LABEL_SIZE = 14
TICK_SIZE = 12
ANNOT_SIZE = 12
BAR_LABEL_SIZE = 12

plt.rcParams.update(
    {
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
    }
)

MAIN_BLUE = "#1f77b4"
PASTEL_GREEN = "#A7D7A0"
PASTEL_PEACH = "#F7C9A9"

BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.4

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
# Panel A – top-20 features
# ==========================

def plot_panel_A(ax, df_importance, meta, n_cc_dims, top_n=20):
    top = df_importance.head(top_n).copy()

    decoded = top["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    decoded_df = pd.DataFrame(list(decoded))
    top = pd.concat([top, decoded_df], axis=1)

    def build_label(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return f"{row['metric']} / Strain-space"

        sub = row.get("cc_sublevel_name")
        if isinstance(sub, str) and sub != "":
            return f"{row['metric']} / {sub}"
        return f"{row['metric']} / {row['group_label']}"

    top["nice_label"] = top.apply(build_label, axis=1)
    top = top.iloc[::-1]

    colors = []
    for _, row in top.iterrows():
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            colors.append(PASTEL_PEACH)
        else:
            colors.append(MAIN_BLUE)

    ax.barh(
        top["nice_label"],
        top["importance_gain"],
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_xlabel("Gain-based importance", fontsize=LABEL_SIZE)

    ax.set_title(
        r"$\mathbf{A.}$" + "  Top 20 similarity features by gain (HALO-CV1)",
        fontsize=TITLE_SIZE,
        loc="center",
        pad=10,
    )

    ax.tick_params(axis="y", labelsize=BAR_LABEL_SIZE)
    # plt.subplots_adjust(left=0.4)

# ==========================
# Panel B – grouped importance (sublevels + strain)
# ==========================

def plot_panel_B(ax, df_importance, meta, n_cc_dims):
    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))].copy()

    decoded = df_imp["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    decoded_df = pd.DataFrame(list(decoded))
    df_imp = pd.concat([df_imp, decoded_df], axis=1)

    def sub_group(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return "Strain-space"
        sub = row.get("cc_sublevel_name")
        if isinstance(sub, str) and sub != "":
            return sub
        return row["group_label"]

    df_imp["sub_group"] = df_imp.apply(sub_group, axis=1)

    grouped = (
        df_imp.groupby("sub_group")["importance_gain_norm"]
        .sum()
        .reset_index()
        .sort_values("importance_gain_norm", ascending=False)
    )

    x = np.arange(len(grouped))
    colors = [
        PASTEL_PEACH if label == "Strain-space" else PASTEL_GREEN
        for label in grouped["sub_group"]
    ]

    ax.bar(
        x,
        grouped["importance_gain_norm"],
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["sub_group"], rotation=60, ha="right", fontsize=BAR_LABEL_SIZE)
    ax.set_ylabel("Total normalized gain importance")

    ax.set_title(
        r"$\mathbf{B.}$" + "  CC sublevels and strain-space (HALO-CV1)",
        fontsize=TITLE_SIZE,
        loc="center",
        pad=10,
    )

    # Vertical % labels to avoid collision
    for i, v in enumerate(grouped["importance_gain_norm"]):
        if v <= 0.015:
            continue
        ax.text(
            i,
            v + 0.003,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            rotation=90,
        )

    # ymax = grouped["importance_gain_norm"].max() * 1.5
    # ax.set_ylim(0, ymax)
    ax.set_ylim(0, 0.1)
    # plt.subplots_adjust(bottom=0.10)
    ax.margins(x=0.1)



# ==========================
# Panel C – grouped importance by CC top level + strain
# ==========================

def plot_panel_C(ax, df_importance, meta, n_cc_dims):
    df_imp = df_importance.copy()
    df_imp = df_imp[df_imp["feature"].str.startswith(("cos_elem_", "euc_elem_"))].copy()

    decoded = df_imp["feature"].apply(
        lambda f: decode_elementwise_feature(f, meta, n_cc_dims)
    )
    decoded_df = pd.DataFrame(list(decoded))
    df_imp = pd.concat([df_imp, decoded_df], axis=1)

    def level_group(row):
        if (
            row["space_name"] == "Strain-space"
            or row["group_label"] == "Strain-space"
            or row["cc_level"] == "Strain"
        ):
            return "Strain-space"
        if row["cc_level_name"] is not None:
            return row["cc_level_name"]
        if row["space_name"] == "Chemical Checker":
            return "Chemical Checker (unspecified)"
        return "Unknown"

    df_imp["level_group"] = df_imp.apply(level_group, axis=1)

    grouped = (
        df_imp.groupby("level_group")["importance_gain_norm"]
        .sum()
        .reset_index()
        .sort_values("importance_gain_norm", ascending=False)
    )

    x = np.arange(len(grouped))
    colors = [
        PASTEL_PEACH if label == "Strain-space" else PASTEL_GREEN
        for label in grouped["level_group"]
    ]

    ax.bar(
        x,
        grouped["importance_gain_norm"],
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["level_group"], rotation=20, ha="right")
    ax.set_ylabel("Total normalized gain importance")

    ax.set_title(
        r"$\mathbf{C.}$" + "  CC level contributions to normalized gain (HALO-CV1)",
        fontsize=TITLE_SIZE,
        loc="center",
        pad=10,
    )

    for i, v in enumerate(grouped["importance_gain_norm"]):
        ax.text(
            i,
            v + 0.004,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
        )

    ymax = grouped["importance_gain_norm"].max() + 0.06
    ax.set_ylim(0, ymax)


# ==========================
# Panel D – CC vs SS (exp06b, HALO-S-CV1)
# ==========================

def plot_panel_D(ax, cc_vs_ss_summary: pd.DataFrame):
    df = cc_vs_ss_summary.copy()

    order = ["CC", "SS"]
    present = [g for g in order if g in df["group"].values]
    others = [g for g in df["group"] if g not in present]
    ordered_groups = present + others

    df["group"] = pd.Categorical(df["group"], categories=ordered_groups, ordered=True)
    df = df.sort_values("group")

    colors_map = {
        "CC": MAIN_BLUE,
        "SS": PASTEL_PEACH,
        "Unknown": PASTEL_GREEN,
    }
    colors = [colors_map.get(g, PASTEL_GREEN) for g in df["group"]]

    y = df["fraction_of_total_gain"]

    ax.bar(
        df["group"],
        y,
        color=colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_WIDTH,
    )
    ax.set_ylabel("Fraction of total gain importance")

    for i, v in enumerate(y):
        ax.text(
            i,
            v + 0.03,
            f"{v*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
        )

    ax.set_ylim(0, 1.15)

    ax.set_title(
        r"$\mathbf{D.}$" + "  CC vs strain-space normalized gain (HALO-S-CV1)",
        fontsize=TITLE_SIZE,
        loc="center",
        pad=10,
    )


# ==========================
# Assemble Fig 3
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

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.0, 1.9],
        height_ratios=[1.3, 1.0],
        wspace=0.40,
        hspace=0.9,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    plot_panel_A(axA, df_imp_exp06d, meta, n_cc_dims, top_n=20)
    plot_panel_B(axB, df_imp_exp06d, meta, n_cc_dims)
    plot_panel_C(axC, df_imp_exp06d, meta, n_cc_dims)
    plot_panel_D(axD, cc_vs_ss_summary)

    fig.tight_layout()
    fig.subplots_adjust(left=0.16, bottom=0.07)

    fig.savefig(FIG3_PNG, dpi=600)
    plt.close(fig)

    print("Saved Fig 3 PNG to:", FIG3_PNG)


if __name__ == "__main__":
    main()
