#!/usr/bin/env python3
"""
Experiment: exp06_novel_pairs 

- Trains the final HALO-CV1 model (exp06d config) on the full labeled dataset.
- Generates novel drug-pair predictions by:
    * taking all possible pairs among the unique compounds (by Inchikey),
    * excluding pairs that appear in the labeled training set,
    * scoring the remaining "novel" pairs with the final model.
- Prints and saves:
    * top 40 novel pairs ranked by P(synergy)
    * top 40 novel pairs ranked by P(antagonism)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import dump
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
from halo.paths import MODEL_RESULTS, CC_FEATURES
from halo.mappers.feature_mapper import FeatureMapper

FILTERED_PATH = MODEL_RESULTS / "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005" / "elementwise_features_filtered_cv1_cc_only.csv"
ALLPAIRS_PATH = MODEL_RESULTS / "exp05d_lgbm_bin_nosspace_elementwise_featselect_bliss005" / "elementwise_features_filtered_all_possible_pairs_cv1_cc_only.csv"

BEST_PARAMS_PATH = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005" / "best_params_cv1.json"
CC_FEATURES_PATH = CC_FEATURES / "cc_features_concat_25x128.csv"

OUT_DIR = MODEL_RESULTS / "e_validation" / "novel_pairs"

# ==========================
# 1) Load and prep training data
# ==========================

def _make_pair_id(ik_a, ik_b):
    """Order-invariant pair ID from two Inchikeys."""
    if pd.isna(ik_a) or pd.isna(ik_b):
        return np.nan
    a, b = str(ik_a), str(ik_b)
    return "||".join(sorted([a, b]))


def load_training_data(filtered_path: Path):
    if not filtered_path.exists():
        raise FileNotFoundError(f"Labeled reduced CSV not found at: {filtered_path}")

    df = pd.read_csv(filtered_path).copy()
    print("Loaded labeled reduced df shape:", df.shape)
    print(df["Interaction Type"].value_counts())

    # Keep only synergy vs antagonism
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())

    # Build Pair_ID from Inchikeys (order-invariant)
    df["Pair_ID"] = df.apply(
        lambda r: _make_pair_id(r["Drug A Inchikey"], r["Drug B Inchikey"]),
        axis=1,
    )

    drop_cols = [
        "Drug A",
        "Drug B",
        "Drug A Inchikey",
        "Drug B Inchikey",
        "Strain",
        "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source",
        "Drug Pair",
        "Pair_ID",
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nTotal labeled samples: {len(df)}")
    print(f"Number of reduced feature columns: {len(feat_cols)}")

    return df, X, y_enc, le, feat_cols

def load_cc_features(cc_features_path: Path) -> pd.DataFrame:
    if not cc_features_path.exists():
        raise FileNotFoundError(f"CC features file not found at: {cc_features_path}")

    df_cc = pd.read_csv(cc_features_path).copy()

    # Basic sanity check for what FeatureMapper expects
    if "inchikey" not in df_cc.columns:
        raise KeyError(
            "CC features file must contain an 'inchikey' column "
            "(one row per compound)."
        )

    return df_cc

# ==========================
# 2) Train final model on full data + save artifacts
# ==========================

def train_final_model(X, y_enc, le, feat_cols, best_params_path: Path):
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best params JSON not found at: {best_params_path}")

    with open(best_params_path, "r") as f:
        cfg = json.load(f)

    if "best_params" not in cfg:
        raise KeyError(
            f"'best_params' key not found in {best_params_path}. "
            "Make sure this is the JSON written by exp06d."
        )

    best_params = cfg["best_params"]
    print("\nLoaded best params from nested CV:")
    print(json.dumps(best_params, indent=2))

    # Final model: same LightGBM config as exp06d, but trained on ALL data
    m_final = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X, y_enc)
    print("\nTrained final HALO-CV1 model on full labeled dataset.")

    # Determine which class index corresponds to "synergy"
    synergy_code = le.transform(["synergy"])[0]
    pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]
    print("Synergy class index in predict_proba:", pos_idx)

    return m_final, pos_idx


def save_artifacts(model, le, feat_cols, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save LightGBM Booster as text model
    booster = model.booster_
    booster_path = out_dir / "model_final_halo_cv1.txt"
    booster.save_model(str(booster_path))
    print("Saved LightGBM booster to:", booster_path)

    # Save sklearn-wrapper model as joblib
    model_path = out_dir / "model_final_halo_cv1.joblib"
    dump(model, model_path)
    print("Saved sklearn model to:", model_path)

    # Save label encoder
    le_path = out_dir / "label_encoder_halo_cv1.joblib"
    dump(le, le_path)
    print("Saved label encoder to:", le_path)

    # Save feature column list
    feat_path = out_dir / "feat_cols_cv1.json"
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print("Saved feature list to:", feat_path)


# ==========================
# 3) Load all-pairs features & isolate novel pairs
# ==========================

def load_allpairs_features(allpairs_path: Path):
    if not allpairs_path.exists():
        raise FileNotFoundError(f"All-pairs reduced CSV not found at: {allpairs_path}")

    df_all = pd.read_csv(allpairs_path).copy()
    print("\nLoaded all-pairs feature df shape:", df_all.shape)

    required_cols = {
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Drug Pair"
    }
    missing = required_cols - set(df_all.columns)
    if missing:
        raise KeyError(
            f"All-pairs feature file is missing required columns: {missing}.\n"
            "It must contain 'Drug A', 'Drug B', 'Drug A Inchikey', "
            "'Drug B Inchikey', 'Drug Pair' and the same reduced "
            "feature columns as the labeled file."
        )

    # Ensure Pair_ID exists (if not already written by exp05d)
    if "Pair_ID" not in df_all.columns:
        df_all["Pair_ID"] = df_all.apply(
            lambda r: _make_pair_id(r["Drug A Inchikey"], r["Drug B Inchikey"]),
            axis=1,
        )

    return df_all


def get_novel_pairs(df_train, df_allpairs):
    # Pairs that appear in the labeled training df (by Inchikey identity)
    train_pairs = set(df_train["Pair_ID"].dropna().astype(str).unique())
    print(f"\nNumber of unique labeled pairs (train set, by Pair_ID): {len(train_pairs)}")

    # All possible pairs present in the all-pairs feature df
    all_pairs = set(df_allpairs["Pair_ID"].dropna().astype(str).unique())
    print(f"Number of pairs in all-pairs feature file (by Pair_ID): {len(all_pairs)}")

    # Novel = in all_pairs but not in train_pairs
    novel_pair_ids = sorted(all_pairs - train_pairs)
    print(f"Number of 'novel' pairs (no labels in training df, by Pair_ID): {len(novel_pair_ids)}")

    df_novel = df_allpairs[df_allpairs["Pair_ID"].astype(str).isin(novel_pair_ids)].copy()
    print("Novel pairs feature df shape:", df_novel.shape)

    return df_novel


# ==========================
# 4) Predict on novel pairs & report top 40 synergy + top 40 antagonism
# ==========================

def predict_novel_pairs(
    model,
    pos_idx,
    feat_cols,
    df_novel,
    out_dir: Path,
    top_k: int = 40,
    fmap: FeatureMapper | None = None,
    cc_features_df: pd.DataFrame | None = None
):
    if df_novel.empty:
        print("\nNo novel pairs found. Nothing to predict.")
        return

    # ---- Compute compact CC similarities for novel pairs ----
    if fmap is not None and cc_features_df is not None:
        print("Computing compact CC distances (cos_block_*, euc_block_*) for novel pairs...")
        df_novel = fmap.compact_similarity(
            combinations_df=df_novel,
            features_df=cc_features_df,
            block_size=128,  # adjust if your CC embedding block size differs
        )
    else:
        print("WARNING: fmap/cc_features_df not provided, skipping CC distance computation.")

    # ---- Predict probabilities ----
    missing = set(feat_cols) - set(df_novel.columns)
    if missing:
        raise KeyError(
            f"Novel pairs df is missing feature columns: {missing}.\n"
            "Ensure the all-pairs feature file uses the same feature engineering "
            "and column names as the labeled file."
        )

    X_novel = df_novel[feat_cols].copy()
    proba = model.predict_proba(X_novel)

    p_synergy = proba[:, pos_idx]
    p_antagonism = 1.0 - p_synergy

    df_novel = df_novel.copy()
    df_novel["p_synergy"] = p_synergy
    df_novel["p_antagonism"] = p_antagonism

    # ---- CC block columns (if present) ----
    cc_cols = [
        c for c in df_novel.columns
        if c.startswith("cos_block_") or c.startswith("euc_block_")
    ]

    agg_dict = {
        "Drug A": "first",
        "Drug B": "first",
        "Drug Pair": "first",
        "p_synergy": "max",
        "p_antagonism": "max",
    }
    # CC distances should be identical for a given pair, so "first" is fine
    for c in cc_cols:
        agg_dict[c] = "first"

    agg = (
        df_novel.groupby("Pair_ID", as_index=False)
        .agg(agg_dict)
    )

    # ---- Collapse 50 CC block features -> 2 summary features ----
    cos_cols_agg = [c for c in agg.columns if c.startswith("cos_block_")]
    euc_cols_agg = [c for c in agg.columns if c.startswith("euc_block_")]

    if cos_cols_agg and euc_cols_agg:
        agg["cc_cosine_mean"] = agg[cos_cols_agg].mean(axis=1)
        agg["cc_euclidean_mean"] = agg[euc_cols_agg].mean(axis=1)

        # If you *only* want the 2 summary features in the CSVs, drop blocks:
        agg = agg.drop(columns=cos_cols_agg + euc_cols_agg)

    # ---------- Top K synergy ----------
    agg_syn = agg.sort_values("p_synergy", ascending=False).reset_index(drop=True)
    top_syn = agg_syn.head(top_k).copy()

    top_syn_path = out_dir / f"novel_pairs_top{top_k}_synergy_halo_cv1.csv"
    top_syn.to_csv(top_syn_path, index=False)
    print("Saved top", top_k, "novel synergistic pairs to:", top_syn_path)

    print(f"\n=== Top {top_k} novel pairs by P(synergy) ===")
    for i, row in top_syn.iterrows():
        rank = i + 1
        pair_name = row["Drug Pair"]
        da = row["Drug A"]
        db = row["Drug B"]
        p = row["p_synergy"]
        print(f"{rank:2d}. {pair_name}  (Drug A: {da}, Drug B: {db})  ->  P(synergy) = {p:.3f}")

    # ---------- Top K antagonism ----------
    agg_ant = agg.sort_values("p_antagonism", ascending=False).reset_index(drop=True)
    top_ant = agg_ant.head(top_k).copy()

    top_ant_path = out_dir / f"novel_pairs_top{top_k}_antagonism_halo_cv1.csv"
    top_ant.to_csv(top_ant_path, index=False)
    print("Saved top", top_k, "novel antagonistic pairs to:", top_ant_path)

    print(f"\n=== Top {top_k} novel pairs by P(antagonism) ===")
    for i, row in top_ant.iterrows():
        rank = i + 1
        pair_name = row["Drug Pair"]
        da = row["Drug A"]
        db = row["Drug B"]
        p = row["p_antagonism"]
        print(f"{rank:2d}. {pair_name}  (Drug A: {da}, Drug B: {db})  ->  P(antagonism) = {p:.3f}")


# ==========================
# 5) Main
# ==========================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train, X, y_enc, le, feat_cols = load_training_data(FILTERED_PATH)

    m_final, pos_idx = train_final_model(X, y_enc, le, feat_cols, BEST_PARAMS_PATH)

    save_artifacts(m_final, le, feat_cols, OUT_DIR)

    df_allpairs = load_allpairs_features(ALLPAIRS_PATH)
    df_novel = get_novel_pairs(df_train, df_allpairs)

    fmap = FeatureMapper()
    cc_features_df = load_cc_features(CC_FEATURES_PATH)

    predict_novel_pairs(
        model=m_final,
        pos_idx=pos_idx,
        feat_cols=feat_cols,
        df_novel=df_novel,
        out_dir=OUT_DIR,
        top_k=40,
        fmap=fmap,
        cc_features_df=cc_features_df
    )


if __name__ == "__main__":
    main()


