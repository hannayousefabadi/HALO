#!/usr/bin/env python3
"""
Experiment: exp06_novel_pairs

- Trains the final HALO model (exp06d config) on the full labeled dataset.
- Performs feature selection once on the full labeled training set only.
- Generates novel drug-pair predictions by:
    * taking all possible pairs among the unique compounds (by Inchikey),
    * excluding pairs that appear in the labeled training set,
    * scoring the remaining novel pairs with the final model.

Prints and saves:
- top 40 novel pairs ranked by P(synergy)
- top 40 novel pairs ranked by P(antagonism)
"""

import json
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import dump
from sklearn.preprocessing import LabelEncoder

from halo.paths import MODEL_RESULTS, CC_FEATURES, PROCESSED
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import classify_interaction


BEST_PARAMS_PATH = MODEL_RESULTS / "exp06d_lgbm_bin_nosspace_elementwise_reduced_nestedcv_bliss005" / "best_params_cv1.json"
CC_FEATURES_PATH = CC_FEATURES / "cc_features_concat_25x128.csv"
COMBOS_PATH = PROCESSED / "halo_training_dataset.csv"

OUT_DIR = MODEL_RESULTS / "e_validation" / "novel_pairs"


def _make_pair_id(ik_a, ik_b):
    if pd.isna(ik_a) or pd.isna(ik_b):
        return np.nan
    a, b = str(ik_a), str(ik_b)
    return "||".join(sorted([a, b]))


def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    var_series = X_train.var()
    kept_after_var = [c for c in feat_cols if var_series[c] > 0.0]

    if len(kept_after_var) == 0:
        raise ValueError("No features remained after variance filtering.")

    kept_after_corr = []
    y_train_s = pd.Series(y_train, index=X_train.index)

    for col in kept_after_var:
        corr = X_train[col].corr(y_train_s)
        if corr is not None and np.isfinite(corr) and abs(corr) >= corr_min:
            kept_after_corr.append(col)

    if not kept_after_corr:
        kept_after_corr = kept_after_var.copy()

    fs_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        random_state=777,
        n_jobs=1,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        min_data_in_leaf=200,
        feature_fraction=0.4,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=50.0,
        lambda_l1=0.0,
        max_bin=127,
        min_gain_to_split=0.05,
    )

    fs_model.fit(X_train[kept_after_corr], y_train)

    feat_imp = pd.Series(
        fs_model.feature_importances_,
        index=kept_after_corr,
    ).sort_values(ascending=False)

    n_keep = max(1, int(len(feat_imp) * 0.30))
    return feat_imp.index[:n_keep].tolist()


def load_cc_features(cc_features_path: Path) -> pd.DataFrame:
    if not cc_features_path.exists():
        raise FileNotFoundError(f"CC features file not found at: {cc_features_path}")

    df_cc = pd.read_csv(cc_features_path).copy()

    if "inchikey" not in df_cc.columns:
        raise KeyError("CC features file must contain an 'inchikey' column.")

    return df_cc


def similarity_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate row-wise CC Cosine and Euclidean similarity means.
    return dataframe with two new columns:
     - cc_cosine_mean
      - cc_euclidean_mean
    """
    cos_cols = [c for c in df.columns if c.startswith("cos_elem_")]
    euc_cols = [c for c in df.columns if c.startswith("euc_elem_")]

    df["cc_cosine_mean"] = df[cos_cols].mean()
    df["cc_euclidean_mean"] = df[euc_cols].mean()

    return df


def load_training_data_from_raw(combos_path: Path, cc_features_path: Path):
    if not combos_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {combos_path}")

    cc_df = load_cc_features(cc_features_path)
    combos_df = pd.read_csv(combos_path).copy()

    fmap = FeatureMapper()
    df = fmap.elementwise_similarity(combos_df, cc_df)

    print("Loaded labeled full df shape:", df.shape)

    df["Interaction Type"] = df["Bliss Score"].apply(
        lambda x: classify_interaction(x, additivity_cutoff=0.05)
    )
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()

    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())

    df["Pair_ID"] = df.apply(
        lambda r: _make_pair_id(r["Drug A Inchikey"], r["Drug B Inchikey"]),
        axis=1,
    )
    df = similarity_calculation(df)
    cc_cos_mean_anta = df[df["Interaction Type"] == "antagonism"]["cc_cosine_mean"].mean()
    cc_euc_mean_syn = df[df["Interaction Type"] == "synergy"]["cc_euclidean_mean"].mean()
    cc_cos_mean_syn = df[df["Interaction Type"] == "synergy"]["cc_cosine_mean"].mean()
    cc_euc_mean_anta = df[df["Interaction Type"] == "antagonism"]["cc_euclidean_mean"].mean()

    print(f"Mean of Cosine similarity in antagonistic pairs of HALO training dataset: {cc_cos_mean_anta}")    
    print(f"Mean of Euclidean similarity in synergistic pairs of HALO training dataset: {cc_euc_mean_syn}")
    print(f"Mean of Cosine similarity in synergistic pairs of HALO training dataset: {cc_cos_mean_syn}")
    print(f"Mean of Euclidean similarity in antagonistic pairs of HALO training dataset: {cc_euc_mean_anta}")

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
        "cc_cosine_mean",
        "cc_euclidean_mean"
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\nTotal labeled samples: {len(df)}")
    print(f"Number of full feature columns: {len(feat_cols)}")

    return df, X, y_enc, le, feat_cols, cc_df


def train_final_model(X, y_enc, le, feat_cols, best_params_path: Path):
    if not best_params_path.exists():
        raise FileNotFoundError(f"Best params JSON not found at: {best_params_path}")

    with open(best_params_path, "r") as f:
        cfg = json.load(f)

    if "best_params" not in cfg:
        raise KeyError(f"'best_params' key not found in {best_params_path}.")

    best_params = cfg["best_params"]
    print("\nLoaded best params from nested CV:")
    print(json.dumps(best_params, indent=2))

    selected_features = select_features_lgbm(
        X_train=X,
        y_train=y_enc,
        feat_cols=feat_cols,
        corr_min=0.01,
        keep_top_frac=0.30,
    )
    X_sel = X[selected_features].copy()
    print(f"Selected features for final model: {len(selected_features)}")

    m_final = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_sel, y_enc)
    print("\nTrained final HALO model on full labeled dataset.")

    synergy_code = le.transform(["synergy"])[0]
    pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]
    print("Synergy class index in predict_proba:", pos_idx)

    return m_final, pos_idx, selected_features


def save_artifacts(model, le, feat_cols, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = model.booster_
    booster_path = out_dir / "model_final_halo.txt"
    booster.save_model(str(booster_path))
    print("Saved LightGBM booster to:", booster_path)

    model_path = out_dir / "model_final_halo.joblib"
    dump(model, model_path)
    print("Saved sklearn model to:", model_path)

    le_path = out_dir / "label_encoder_halo.joblib"
    dump(le, le_path)
    print("Saved label encoder to:", le_path)

    feat_path = out_dir / "feat_cols.json"
    with open(feat_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print("Saved feature list to:", feat_path)


def build_all_possible_pairs(cc_df: pd.DataFrame) -> pd.DataFrame:
    required = {"drug", "inchikey"}
    missing = required - set(cc_df.columns)
    if missing:
        raise KeyError(f"CC features file is missing required columns for pair generation: {missing}")

    compounds = (
        cc_df[["drug", "inchikey"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    rows = []
    for (_, a), (_, b) in itertools.combinations(compounds.iterrows(), 2):
        ik_a = str(a["inchikey"]).strip().upper()
        ik_b = str(b["inchikey"]).strip().upper()
        rows.append(
            {
                "Drug A": a["drug"],
                "Drug B": b["drug"],
                "Drug A Inchikey": ik_a,
                "Drug B Inchikey": ik_b,
                "Drug Pair": "::".join(sorted([ik_a, ik_b])),
                "Pair_ID": _make_pair_id(ik_a, ik_b),
            }
        )

    df_allpairs = pd.DataFrame(rows)
    print("\nBuilt all possible pairs shape:", df_allpairs.shape)
    return df_allpairs


def get_novel_pairs(df_train, df_allpairs):
    train_pairs = set(df_train["Pair_ID"].dropna().astype(str).unique())
    print(f"\nNumber of unique labeled pairs (train set, by Pair_ID): {len(train_pairs)}")

    all_pairs = set(df_allpairs["Pair_ID"].dropna().astype(str).unique())
    print(f"Number of pairs in all-pairs table (by Pair_ID): {len(all_pairs)}")

    novel_pair_ids = sorted(all_pairs - train_pairs)
    print(f"Number of novel pairs (not in labeled training set): {len(novel_pair_ids)}")

    df_novel = df_allpairs[df_allpairs["Pair_ID"].astype(str).isin(novel_pair_ids)].copy()
    print("Novel pairs df shape:", df_novel.shape)

    return df_novel


def predict_novel_pairs(
    model,
    pos_idx,
    feat_cols,
    df_novel,
    out_dir: Path,
    top_k: int = 40,
    fmap: FeatureMapper | None = None,
    cc_features_df: pd.DataFrame | None = None,
):
    """
    Predict probability of synergy and antagonism for novel pairs
    
    model: 
    pos_idx: 
    feat_cols: 
    df_novel: 
    out_dir: 
    out_dir: Path
    top_k: 
    top_k: int
    fmap: FeatureMapper | None
    cc_features_df: pd.DataFrame | None
    """
    if df_novel.empty:
        print("\nNo novel pairs found. Nothing to predict.")
        return

    if fmap is None or cc_features_df is None:
        raise ValueError("fmap and cc_features_df are required.")

    print("Building full elementwise CC-only features for novel pairs...")
    df_novel = fmap.elementwise_similarity(df_novel, cc_features_df)
    df_novel = similarity_calculation(df_novel)

    missing = set(feat_cols) - set(df_novel.columns)
    if missing:
        raise KeyError(
            f"Novel pairs df is missing feature columns: {missing}."
        )

    X_novel = df_novel[feat_cols].copy()
    proba = model.predict_proba(X_novel)

    p_synergy = proba[:, pos_idx]
    p_antagonism = 1.0 - p_synergy

    df_novel = df_novel.copy()
    df_novel["p_synergy"] = p_synergy
    df_novel["p_antagonism"] = p_antagonism

    agg = (
        df_novel.groupby("Pair_ID", as_index=False)
        .agg(
            {
                "Drug A": "first",
                "Drug B": "first",
                "Drug Pair": "first",
                "p_synergy": "max",
                "p_antagonism": "max",
                "cc_cosine_mean": "first",
                "cc_euclidean_mean": "first",
            }
        )
    )

    agg_syn = agg.sort_values("p_synergy", ascending=False).reset_index(drop=True)
    top_syn = agg_syn.head(top_k).copy()

    top_syn_path = out_dir / f"novel_pairs_top{top_k}_synergy_halo.csv"
    top_syn.to_csv(top_syn_path, index=False)
    print("Saved top", top_k, "novel synergistic pairs to:", top_syn_path)

    print(f"\n=== Top {top_k} novel pairs by P(synergy) ===")
    for i, row in top_syn.iterrows():
        rank = i + 1
        print(
            f"{rank:2d}. {row['Drug Pair']}  "
            f"(Drug A: {row['Drug A']}, Drug B: {row['Drug B']})  "
            f"->  P(synergy) = {row['p_synergy']:.3f}"
        )

    agg_ant = agg.sort_values("p_antagonism", ascending=False).reset_index(drop=True)
    top_ant = agg_ant.head(top_k).copy()

    top_ant_path = out_dir / f"novel_pairs_top{top_k}_antagonism_halo.csv"
    top_ant.to_csv(top_ant_path, index=False)
    print("Saved top", top_k, "novel antagonistic pairs to:", top_ant_path)

    print(f"\n=== Top {top_k} novel pairs by P(antagonism) ===")
    for i, row in top_ant.iterrows():
        rank = i + 1
        print(
            f"{rank:2d}. {row['Drug Pair']}  "
            f"(Drug A: {row['Drug A']}, Drug B: {row['Drug B']})  "
            f"->  P(antagonism) = {row['p_antagonism']:.3f}"
        )

    cc_cos_mean_anta = top_ant["cc_cosine_mean"].mean()
    cc_euc_mean_syn = top_syn["cc_euclidean_mean"].mean()
    cc_cos_mean_syn = top_syn["cc_cosine_mean"].mean()
    cc_euc_mean_anta = top_ant["cc_euclidean_mean"].mean()

    print(f"Mean of Cosine similarity in predicted novel antagonistic pairs: {cc_cos_mean_anta}")    
    print(f"Mean of Euclidean similarity in predicted novel synergistic pairs: {cc_euc_mean_syn}")
    print(f"Mean of Cosine similarity in predicted novel synergistic pairs: {cc_cos_mean_syn}")
    print(f"Mean of Euclidean similarity in predicted novel antagonistic pairs: {cc_euc_mean_anta}")
    

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_train, X, y_enc, le, feat_cols, cc_df = load_training_data_from_raw(
        COMBOS_PATH,
        CC_FEATURES_PATH,
    )

    m_final, pos_idx, selected_feat_cols = train_final_model(
        X, y_enc, le, feat_cols, BEST_PARAMS_PATH
    )

    save_artifacts(m_final, le, selected_feat_cols, OUT_DIR)

    df_allpairs = build_all_possible_pairs(cc_df)
    df_novel = get_novel_pairs(df_train, df_allpairs)

    fmap = FeatureMapper()

    predict_novel_pairs(
        model=m_final,
        pos_idx=pos_idx,
        feat_cols=selected_feat_cols,
        df_novel=df_novel,
        out_dir=OUT_DIR,
        top_k=40,
        fmap=fmap,
        cc_features_df=cc_df,
    )


if __name__ == "__main__":
    main()