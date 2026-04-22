#!/usr/bin/env python3
"""
Generate Figure 1:

Outputs:
- fig1B_counts.log
- fig1B_bliss_hist.png
- fig1B_bliss_hist.log
- fig1B_class_balance.png
- fig1B_class_balance.log
- fig1B_pairs_per_strain.log
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from halo.paths import PROCESSED, FIGURE_PIPELINE

CSV_PATH = PROCESSED  / "halo_training_dataset.csv"
FIG_OUT_BASE = FIGURE_PIPELINE / "fig1_feats_data" / "fig1_panels" / "fig1"
FIG_OUT_BASE.parent.mkdir(parents=True, exist_ok=True)

from halo.shared_utils.data_io import classify_interaction


DRUG_A_COL = "Drug A"
DRUG_B_COL = "Drug B"
STRAIN_COL = "Strain"
SPECIES_COL = "Specie"
BLISS_COL = "Bliss Score"

CLASS_COL = "Interaction_3class"

DPI = 600
ADDITIVITY_CUTOFF = 0.05  


STRAIN_RENAME = {
    "escherichia coli bw25113": "Escherichia coli BW25113",
    "escherichia coli iai1": "Escherichia coli IAI1",
    "salmonella typhimurium 14028": "Salmonella Typhimurium 14028",
    "salmonella typhimurium lt2": "Salmonella Typhimurium LT2",
    "pseudomonas aeruginosa pa14": "Pseudomonas aeruginosa PA14",
    "pseudomonas aeruginosa pao1": "Pseudomonas aeruginosa PAO1",
    "staphylococcus aureus dsm 20231": "Staphylococcus aureus DSM 20231",
    "staphylococcus aureus newman": "Staphylococcus aureus Newman",
    "streptococcus pneumoniae": "Streptococcus pneumoniae",
    "bacillus subtilis": "Bacillus subtilis"
}


def italicize(text: str) -> str:
    """Return text wrapped in mathtext italics for matplotlib tables."""
    return rf"$\it{{{text}}}$"


def main():
    df = pd.read_csv(CSV_PATH).copy()

    df[CLASS_COL] = df[BLISS_COL].apply(
        lambda x: classify_interaction(x, additivity_cutoff=ADDITIVITY_CUTOFF)
    )

    # --- basic counts ---
    n_samples = len(df)

    # unique drug pairs (order-insensitive)
    pairs = df[[DRUG_A_COL, DRUG_B_COL]].apply(
        lambda row: tuple(sorted(row.values)), axis=1
    )
    n_unique_pairs = pairs.nunique()

    n_unique_drugs = pd.unique(pd.concat([df['Drug A Inchikey'], df['Drug B Inchikey']])).size
    n_strains = df[STRAIN_COL].nunique()
    n_species = df[SPECIES_COL].nunique() if SPECIES_COL in df.columns else None

    # --- class balance (3-class, after the new cutoff) ---
    class_counts = df[CLASS_COL].value_counts().sort_index()
    class_props = class_counts / n_samples * 100.0

    # --- Bliss score stats ---
    bliss = df[BLISS_COL].dropna()
    bliss_mean = bliss.mean()
    bliss_std = bliss.std()
    bliss_min = bliss.min()
    bliss_max = bliss.max()

    # --- unique drug pairs per strain ---
    df_pairs = df.copy()
    df_pairs["pair"] = pairs
    strain_pair_counts = (
        df_pairs.groupby(STRAIN_COL)["pair"].nunique().sort_values(ascending=False)
    )

    pretty_index = (
        strain_pair_counts.index.to_series().replace(STRAIN_RENAME).tolist()
    )
    # labels for table: italicized strain names
    row_labels = [italicize(name) for name in pretty_index]

    # ----------------------------
    # Global style
    # ----------------------------
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
            "font.size": 8,
        }
    )

    # ============================
    # Part 1: dataset + Bliss summary block (LOG ONLY)
    # ============================
    fig_counts, ax_counts = plt.subplots(figsize=(3.0, 3.0))
    ax_counts.axis("off")

    lines = [
        f"Total assays (pair–strain samples): {n_samples}",
        f"Unique drug pairs: {n_unique_pairs}",
        f"Unique antibacterials: {n_unique_drugs}",
        f"Unique strains: {n_strains}",
    ]
    if n_species is not None:
        lines.append(f"Species: {n_species}")
    lines.extend(
        [
            "",
            f"Bliss mean: {bliss_mean:.3f}",
            f"Bliss s.d.: {bliss_std:.3f}",
            f"Bliss min/max: {bliss_min:.2f} / {bliss_max:.2f}",
        ]
    )

    # --- log file for Part 1 ---
    counts_log_path = Path(f"{FIG_OUT_BASE}_counts.log")
    with open(counts_log_path, "w") as f:
        f.write("Dataset statistics after curation\n")
        f.write("---------------------------------\n")
        f.write(f"CSV_PATH: {CSV_PATH}\n")
        f.write(f"ADDITIVITY_CUTOFF: {ADDITIVITY_CUTOFF}\n\n")
        for text in lines:
            f.write(text + "\n")
    print(f"Logged: {counts_log_path}")

    y0 = 0.95
    dy = 0.10
    for i, text in enumerate(lines):
        ax_counts.text(
            0.0,
            y0 - i * dy,
            text,
            transform=ax_counts.transAxes,
            fontsize=9,
            ha="left",
            va="top",
        )

    ax_counts.set_title("Dataset statistics after curation", fontsize=10, pad=6)
    fig_counts.tight_layout()
    plt.close(fig_counts)

    # ============================
    # Part 2: Bliss histogram (PNG + LOG)
    # ============================
    # Precompute histogram so we can log bin edges + counts
    hist_counts, bin_edges = np.histogram(bliss, bins=25)

    fig_hist, ax_hist = plt.subplots(figsize=(3.0, 3.0))
    # Use precomputed bins to match what we log
    ax_hist.hist(bliss, bins=bin_edges, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax_hist.set_xlabel("Bliss score", fontsize=12)
    ax_hist.set_ylabel("Frequency", fontsize=12)
    ax_hist.set_title("Bliss score distribution", fontsize=12)

    # --- log file for Part 2 ---
    hist_log_path = Path(f"{FIG_OUT_BASE}_bliss_hist.log")
    with open(hist_log_path, "w") as f:
        f.write("Bliss score histogram\n")
        f.write("----------------------\n")
        f.write(f"CSV_PATH: {CSV_PATH}\n")
        f.write(f"ADDITIVITY_CUTOFF: {ADDITIVITY_CUTOFF}\n")
        f.write(f"Number of samples (non-NaN Bliss): {len(bliss)}\n")
        f.write("Bins: 25\n\n")
        f.write("bin_start\tbin_end\tcount\n")
        for start, end, count in zip(bin_edges[:-1], bin_edges[1:], hist_counts):
            f.write(f"{start:.6f}\t{end:.6f}\t{int(count)}\n")
    print(f"Logged: {hist_log_path}")

    fig_hist.tight_layout()
    # High-quality PNG only
    out_hist_png = Path(f"{FIG_OUT_BASE}_bliss_hist.png")
    fig_hist.savefig(out_hist_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_hist_png}")
    plt.close(fig_hist)

    # ============================
    # Part 3: class balance (PNG + LOG)
    # ============================
    fig_bar, ax_bar = plt.subplots(figsize=(4.0, 3.0))
    bars = ax_bar.bar(class_counts.index, class_counts.values, color="#1f77b4", edgecolor="black", linewidth=0.4)

    # labels: count + percentage
    ax_bar.bar_label(
        bars,
        labels=[
            f"{cnt}\n({prop:.1f}%)"
            for cnt, prop in zip(class_counts.values, class_props.values)
        ],
        label_type="edge",
        padding=6,
        fontsize=12
    )

    ax_bar.set_ylim(0, max(class_counts.values) * 1.25)
    ax_bar.set_ylabel("Count", fontsize=12)
    ax_bar.set_title(
        (
            "Underlying 3-class Bliss labels before binarization\n"
            f"(additivity cutoff ±{ADDITIVITY_CUTOFF})"
        ),
        fontsize=12,
    )

    ax_bar.set_xticks(np.arange(len(class_counts)))
    ax_bar.set_xticklabels(class_counts.index, rotation=0, fontsize=12)

    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # --- log file for Part 3 ---
    class_log_path = Path(f"{FIG_OUT_BASE}_class_balance.log")
    with open(class_log_path, "w") as f:
        f.write("3-class Bliss label distribution (before binarization)\n")
        f.write("------------------------------------------------------\n")
        f.write(f"CSV_PATH: {CSV_PATH}\n")
        f.write(f"ADDITIVITY_CUTOFF: {ADDITIVITY_CUTOFF}\n")
        f.write(f"Total samples: {n_samples}\n\n")
        f.write("class_label\tcount\tproportion_percent\n")
        for label, cnt in class_counts.items():
            prop = (cnt / n_samples) * 100.0
            f.write(f"{label}\t{int(cnt)}\t{prop:.4f}\n")
    print(f"Logged: {class_log_path}")

    fig_bar.tight_layout()
    out_bar_png = Path(f"{FIG_OUT_BASE}_class_balance.png")
    fig_bar.savefig(out_bar_png, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_bar_png}")
    plt.close(fig_bar)

    # ============================
    # Part 4: table of drug pairs per strain
    # ============================
    fig_strain, ax_strain = plt.subplots(figsize=(3.0, 3.0))
    ax_strain.axis("off")

    col_labels = ["Drug pairs"]
    cell_text = [[int(v)] for v in strain_pair_counts.values]

    table = ax_strain.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    ax_strain.set_title("Unique drug pairs per strain", fontsize=12, pad=6)
    fig_strain.tight_layout()

    # --- log file for Part 4 ---
    pairs_log_path = Path(f"{FIG_OUT_BASE}_pairs_per_strain.log")
    with open(pairs_log_path, "w") as f:
        f.write("Unique drug pairs per strain\n")
        f.write("----------------------------\n")
        f.write(f"CSV_PATH: {CSV_PATH}\n\n")
        f.write("strain_pretty\tunique_pairs\n")
        for strain_pretty, count in zip(pretty_index, strain_pair_counts.values):
            f.write(f"{strain_pretty}\t{int(count)}\n")
    print(f"Logged: {pairs_log_path}")

    plt.close(fig_strain)


if __name__ == "__main__":
    main()
