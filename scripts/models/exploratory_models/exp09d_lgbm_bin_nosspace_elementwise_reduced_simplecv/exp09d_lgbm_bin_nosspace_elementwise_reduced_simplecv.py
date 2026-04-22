#!/usr/bin/env python3
"""
Experiment: exp09d_lgbm_bin_nosspace_elementwise_reduced_simplecv (M4)

Config
- model: LightGBM 
- task: binary classification
- feature_design: elementwise similarity 
- sspace: disabled 
- feature_selection: enabled (training set only)

- CV:
  - nested_cv: disabled
  - stratified train/test split (80/20)

  
Data integrity note
All preprocessing (missing values, dtypes, column validation, etc.) is performed upstream in preprocessing 
notebooks/scripts. This script assumes the processed inputs are clean and consistent.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import LabelEncoder

from halo.paths import CC_FEATURES, PROCESSED
from halo.mappers.feature_mapper import FeatureMapper
from halo.shared_utils.data_io import classify_interaction
from halo.shared_utils.metrics import classification_metrics, overfitting_report


def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    """
    Feature selection performed once on the outer training split only.

    Steps
    1) drop zero-variance features
    2) keep features with |corr(feature, y)| >= corr_min
       fallback: keep all variance-filtered features if none pass
    3) rank by LightGBM feature_importances_ and keep top fraction
    """
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

    n_keep = max(1, int(len(feat_imp) * keep_top_frac))
    selected_features = feat_imp.index[:n_keep].tolist()
    return selected_features


def main():
    print(
        "\n=== EXP09d ===\n"
    )

    corr_min = 0.01
    keep_top_frac = 0.30

    # ==========================
    # 1) Load raw inputs and build elementwise feature table
    # ==========================
    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    cc_df = pd.read_csv(cc_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    features_cc = cc_df.copy()
    df = FeatureMapper().elementwise_similarity(combinations_df, features_cc)

    print("Full df shape:", df.shape)

    # ==========================
    # 2) Keep binary classes only
    # ==========================
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()

    print("\nAfter filter to binary:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 3) Feature columns
    # ==========================
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]

    feat_cols = [c for c in df.columns if c not in drop_cols]
    # print("Feature columns:", len(feat_cols))

    X = df[feat_cols].copy()
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # ==========================
    # 4) Simple CV train/test split (LEAKY BY PAIR ON PURPOSE)
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    print("\nTrain shape before feature selection:", X_train.shape)
    print("Test shape before feature selection :", X_test.shape)

    # ==========================
    # 5) Feature selection on TRAIN ONLY
    # ==========================
    selected_features = select_features_lgbm(
        X_train=X_train,
        y_train=y_train,
        feat_cols=feat_cols,
        corr_min=corr_min,
        keep_top_frac=keep_top_frac,
    )

    X_train_sel = X_train[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()

    print("Selected feature count:", len(selected_features))
    print("Train shape after feature selection:", X_train_sel.shape)
    print("Test shape after feature selection :", X_test_sel.shape)

    # ==========================
    # 6) LightGBM baseline + randomized search
    # ==========================
    base_clf = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_jobs=1,
        random_state=42,
    )

    param_dist = {
        "learning_rate": [0.02, 0.05],
        "n_estimators": [300, 600],
        "max_depth": [3, 4, 5, 6],
        "num_leaves": [7, 15, 31],
        "min_child_samples": [50, 100, 200],
        "feature_fraction": [0.3, 0.4, 0.6, 0.8],
        "subsample": [0.6, 0.8],
        "subsample_freq": [1],
        "lambda_l1": [0.0, 0.1, 1.0, 5.0],
        "lambda_l2": [0.0, 0.1, 1.0, 5.0],
        "max_bin": [63, 127, 255],
        "min_split_gain": [0.0, 0.05, 0.1],
    }

    print("\n--- Running RandomizedSearchCV (simple CV) ---\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring="f1",
        verbose=1,
        n_jobs=1,
        random_state=42,
    )

    search.fit(X_train_sel, y_train)
    best_model = search.best_estimator_

    print("\nBest parameters:", search.best_params_)

    # ==========================
    # 7) Final evaluation
    # ==========================
    y_pred = best_model.predict(X_test_sel)
    y_score = best_model.predict_proba(X_test_sel)

    print("\n=== TEST METRICS (LEAKY SIMPLE CV) ===")
    classification_metrics(
        y_test,
        y_pred,
        y_score=y_score,
        class_names=le.classes_,
    )

    # ==========================
    # 8) Overfitting report
    # ==========================
    print("\n=== Overfitting Report ===")
    overfitting_report(
        best_model,
        X_train_sel, y_train,
        X_test_sel, y_test,
        task="classification",
        average="macro"
    )

    print("\n=== EXP09d DONE ===\n")


if __name__ == "__main__":
    main()