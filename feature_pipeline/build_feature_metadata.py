#!/usr/bin/env python3
"""
Build feature metadata for CC + strain-space base features.

For each column (dim_i or s_j), we annotate:
- space: 'CC' or 'S'
- dimension: integer index within that space
- cc_level: A..E or 'Strain'
- cc_sublevel: A1..E5 or 'Strain'
- cc_level_name: e.g. 'Chemistry', 'Targets', ...
- cc_sublevel_name: e.g. '2D fingerprints', 'Mechanisms of action', ...
- group_label: e.g. 'Chemistry / A1: 2D fingerprints' or 'Strain-space'
"""
import pandas as pd
from halo.paths import FEATURES, CC_FEATURES, SS_FEATURES


CC_FEATURES_PATH = CC_FEATURES / "cc_features_concat_25x128.csv"
S_FEATURES_PATH  = SS_FEATURES / "sspace.csv"
OUTPUT_PATH      = FEATURES / "feature_metadata_cc_s_full.csv"

# ==========================
# CC hierarchy definitions
# ==========================

CC_SUBLEVELS_ORDERED = [
    "A1","A2","A3","A4","A5",
    "B1","B2","B3","B4","B5",
    "C1","C2","C3","C4","C5",
    "D1","D2","D3","D4","D5",
    "E1","E2","E3","E4","E5",
]

CC_LEVEL_NAME = {
    "A": "Chemistry",
    "B": "Targets",
    "C": "Networks",
    "D": "Cells",
    "E": "Clinics",
}

CC_SUBLEVEL_NAME = {
    "A1": "2D fingerprints",
    "A2": "3D fingerprints",
    "A3": "Scaffolds",
    "A4": "Structural keys",
    "A5": "Physicochemistry",
    "B1": "Mechanisms of action",
    "B2": "Metabolic genes",
    "B3": "Crystals",
    "B4": "Binding",
    "B5": "HTS bioassays",
    "C1": "Small molecule roles",
    "C2": "Small molecule pathways",
    "C3": "Signaling pathways",
    "C4": "Biological processes",
    "C5": "Interactome",
    "D1": "Transcription",
    "D2": "Cancer cell lines",
    "D3": "Chemical genetics",
    "D4": "Morphology",
    "D5": "Cell bioassays",
    "E1": "Therapeutic areas",
    "E2": "Indications",
    "E3": "Side effects",
    "E4": "Diseases & toxicology",
    "E5": "Drug–drug interactions",
}

CC_DIM_PER_SUBLEVEL = 128
CC_TOTAL_SUBLEVELS = len(CC_SUBLEVELS_ORDERED)               # 25
CC_TOTAL_CC_DIMS = CC_DIM_PER_SUBLEVEL * CC_TOTAL_SUBLEVELS  # 3200


def main():
    # Load CC + S tables (to get columns)
    cc_df = pd.read_csv(CC_FEATURES_PATH).copy()
    ss_df = pd.read_csv(S_FEATURES_PATH).copy()

    # Normalize Inchikey if needed 
    if "inchikey" in cc_df.columns:
        cc_df["inchikey"] = cc_df["inchikey"].astype(str).str.strip().str.upper()
    if "inchikey" in ss_df.columns:
        ss_df["inchikey"] = ss_df["inchikey"].astype(str).str.strip().str.upper()

    # Merge to replicate the features_cc_s used in modeling
    # (we care about the combined columns, not the rows)
    features_cc_s = cc_df.merge(ss_df, on="inchikey", how="inner", suffixes=("", "_s"))

    ignore = {"drug", "inchikey", "level"}
    base_cols = [c for c in features_cc_s.columns if c not in ignore]

    meta_rows = []

    for c in base_cols:
        if c.startswith("s_"):
            # Strain-space dimension
            space = "S"
            # assume s_<index>
            try:
                dim_idx = int(c.split("_")[1])
            except Exception:
                dim_idx = None

            cc_level = "Strain"
            cc_sublevel = "Strain"
            cc_level_name = "Strain-space"
            cc_sublevel_name = "Strain-space"
            group_label = "Strain-space"

        else:
            # CC dimension: dim_<index>
            space = "CC"
            try:
                dim_idx = int(c.split("_")[1])
            except Exception:
                dim_idx = None

            if dim_idx is not None and dim_idx < CC_TOTAL_CC_DIMS:
                sublevel_idx = dim_idx // CC_DIM_PER_SUBLEVEL  # 0..24
                cc_sublevel = CC_SUBLEVELS_ORDERED[sublevel_idx]
                cc_level = cc_sublevel[0]
                cc_level_name = CC_LEVEL_NAME.get(cc_level, cc_level)
                cc_sublevel_name = CC_SUBLEVEL_NAME.get(cc_sublevel, cc_sublevel)
                group_label = f"{cc_level_name} / {cc_sublevel}: {cc_sublevel_name}"
            else:
                cc_level = None
                cc_sublevel = None
                cc_level_name = "Unknown"
                cc_sublevel_name = "Unknown"
                group_label = "Unknown"

        row = {
            "original_name": c,
            "space": space,                         # 'CC' or 'S'
            "space_name": "Chemical Checker" if space == "CC" else "Strain-space",
            "dimension": dim_idx,
            "cc_level": cc_level,                   # 'A'..'E' or 'Strain' or None
            "cc_sublevel": cc_sublevel,             # 'A1'..'E5' or 'Strain' or None
            "cc_level_name": cc_level_name,
            "cc_sublevel_name": cc_sublevel_name,
            "group_label": group_label,
        }
        meta_rows.append(row)

    feature_meta = pd.DataFrame(meta_rows)
    feature_meta.to_csv(OUTPUT_PATH, index=False)

    print(f"Loaded {len(cc_df)} CC rows, {len(ss_df)} S rows.")
    print(f"After merge: {len(features_cc_s)} overlapping inchikeys.")
    print(f"Found {len(base_cols)} feature columns.")
    print(f"Saved enriched feature meta to: {OUTPUT_PATH}")
    print(feature_meta.head())


if __name__ == "__main__":
    main()
