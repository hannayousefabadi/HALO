#!/usr/bin/env python3
"""
Experiment: exp08_lgbm_regr_sspace_elementwise_reduced_nestedcv

Config
- model: LightGBM 
- task: regression
- feature_design: elementwise similarity 
- sspace: enabled (strain-space features)
- feature_selection: enabled (within CV folds)
- bliss neutrality cutoff: ±0.1 (applied during preprocessing; this script assumes labels are already finalized)

- CV:
  - nested_cv: enabled
  - Outer split:
    - CV1 scheme: drug pair held-out
  - Inner split:
    - GroupKFold, groups = Drug Pair
    - random search over 32 sampled hyperparameter configs
    - selection metric: mean validation RMSE
  - Final fit: refit best model on full outer-train, evaluate once on outer-test

Data integrity note
All preprocessing (missing values, dtypes, column validation, etc.) is performed upstream in preprocessing 
notebooks/scripts. This script assumes the processed inputs are clean and consistent.
"""


import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error, 
    r2_score
)

from halo.paths import CC_FEATURES, SS_FEATURES, PROCESSED, MODEL_RESULTS
from halo.mappers.feature_mapper import FeatureMapper


def select_features_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feat_cols: list[str],
    corr_min: float = 0.01,
    keep_top_frac: float = 0.30,
) -> list[str]:
    """
    Feature selection performed using TRAINING DATA ONLY.

    Steps
    1) drop zero-variance features
    2) keep features with |corr(feature, y)| >= corr_min
       fallback: keep all variance-filtered features if none pass
    3) rank by LightGBM feature_importances_ and keep top fraction
    """
    # Step 1: variance filter
    var_series = X_train.var()
    kept_after_var = [c for c in feat_cols if var_series[c] > 0.0]

    if len(kept_after_var) == 0:
        raise ValueError("No features remained after variance filtering.")

    # Step 2: correlation prefilter
    kept_after_corr = []
    y_train_s = pd.Series(y_train, index=X_train.index)

    for col in kept_after_var:
        corr = X_train[col].corr(y_train_s)
        if corr is not None and np.isfinite(corr) and abs(corr) >= corr_min:
            kept_after_corr.append(col)

    if not kept_after_corr:
        kept_after_corr = kept_after_var.copy()

    # Step 3: model-based importance ranking
    fs_model = lgb.LGBMRegressor(
        objective="regression",
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
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"

    corr_min = 0.01
    keep_top_frac = 0.30

    cc_path = CC_FEATURES / "cc_features_concat_25x128.csv"
    ss_path = SS_FEATURES / "sspace.csv"
    combos_path = PROCESSED / "halo_training_dataset.csv"

    out_dir = MODEL_RESULTS / "exp08_lgbm_regr_sspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP08 ===\n"
    )
    # print("Using scheme:", SCHEME)
    print("Output dir :", out_dir)

    rng = np.random.default_rng(42)

    # ==========================
    # 1) Load and build FULL elementwise feature table
    # ==========================
    cc_df = pd.read_csv(cc_path).copy()
    ss_df = pd.read_csv(ss_path).copy()
    combinations_df = pd.read_csv(combos_path).copy()

    features_cc_s = cc_df.merge(ss_df, on="inchikey", how="inner", suffixes=("", "_s"))
    df = FeatureMapper().elementwise_similarity(combinations_df, features_cc_s)

    print("Full df shape:", df.shape)

    # ==========================
    # 2) Feature columns
    # ==========================
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
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feat_cols].copy()
    y = df["Bliss Score"].astype(float).values

    pairs = df["Drug Pair"].astype(str).values
    n = len(df)

    # ==========================
    # 3) Outer splits (CV1 only)
    # ==========================
    def make_splits_cv1(n_splits=5, verbose=True):
        """
        Split data into outer test and outer train under Drug Pair held-out scheme (CV1)
        over 5 folds.
        
        Returns:
            tr_idx (np.ndarray of positions),
            te_idx (np.ndarray of positions)
        """
        outer_cv = GroupKFold(n_splits=n_splits)
        split_gen = outer_cv.split(X, y, groups=pairs)


        splits = []
        for fold_idx, (tr_idx, te_idx) in enumerate(split_gen, 1):
            splits.append((tr_idx, te_idx))
            # if verbose:
            #     print("=" * 72)
            #     print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
            #     print(f"Train size: {len(tr_idx)}")
            #     print(f"Train size fraction: {len(tr_idx) / len(df) * 100:.2f}%")
            #     print(f"Test size: {len(te_idx)}")
            #     print(f"Test size fraction: {len(te_idx) / len(df) * 100:.2f}%")
            #     print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            #     print("-" * 72)
        return splits

    if SCHEME == "CV1":
        outer_splits = make_splits_cv1(n_splits=5, verbose=True)
    else:
        raise ValueError("SCHEME must be 'CV1' in this script.")

    # ==========================
    # 4) LightGBM logger silence
    # ==========================
    class SilentLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    lgb.register_logger(SilentLogger())

    # ==========================
    # 5) Containers for results
    # ==========================
    fold_results = []

    selected_counts = []
    fi_per_fold = []

    # ==========================
    # 6) Outer loop
    # ==========================
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print("\n" + "#" * 72)
        print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
        print("#" * 72 + "\n")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_te = y[te_idx]
        grp_tr = pairs[tr_idx]

        print(f"Outer-train shape: {X_tr.shape}")
        print(f"Outer-test shape : {X_te.shape}")

        # The inner CV split, 3 folds, grouped by drug-pair
        inner_cv = GroupKFold(n_splits=3)
        inner_splitter = lambda: inner_cv.split(X_tr, y_tr, groups=grp_tr)


        # Hyperparameter sampling
        def sample_one_params():
            max_depth = 3
            leaves_map = {3: [7, 9, 15]}

            return dict(
                boosting_type="gbdt",
                learning_rate=float(rng.choice([0.02, 0.03])),
                max_depth=max_depth,
                num_leaves=int(rng.choice(leaves_map[max_depth])),
                min_data_in_leaf=int(rng.choice([200, 300])),
                feature_fraction=float(rng.choice([0.30, 0.40, 0.50])),
                bagging_fraction=float(rng.choice([0.60, 0.70, 0.80])),
                bagging_freq=1,
                lambda_l2=float(10 ** rng.uniform(1.4, 1.9)),
                lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
                max_bin=int(rng.choice([63, 127])),
                min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
            )

        param_samples = [sample_one_params() for _ in range(32)]

        # inner-CV hyperparameters scores and feature selection on INNER-TRAIN-ONLY
        def cv_score_for_params(params):
            """
            Perform a full inner cross‑validation loop for a given hyperparameter
            configuration.

            For each inner fold:
            - Fit the feature selector only on the inner‑train split.
            - Select features for that fold.
            - Train a LightGBM model with the provided hyperparameters.
            - Evaluate RMSE on the inner‑validation split.

            Returns the mean validation RMSE across inner folds, used for
            comparing and ranking hyperparameter configurations in the nested CV.

            It is trying to answer: “If we used this hyperparameter configuration, 
            how well would the pipeline generalize?”
            """    
            scores = []

            for inner_fold_idx, (tr_f, val_f) in enumerate(inner_splitter(), 1):
                Xf_tr = X_tr.iloc[tr_f].reset_index(drop=True)
                Xf_val = X_tr.iloc[val_f].reset_index(drop=True)
                yf_tr = y_tr[tr_f]
                yf_val = y_tr[val_f]

                selected_inner = select_features_lgbm(
                    X_train=Xf_tr,
                    y_train=yf_tr,
                    feat_cols=feat_cols,
                    corr_min=corr_min,
                    keep_top_frac=keep_top_frac,
                )

                Xf_tr_sel = Xf_tr[selected_inner]
                Xf_val_sel = Xf_val[selected_inner]

                m = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=4000,
                    random_state=777,
                    n_jobs=4,
                    **params,
                )

                m.fit(
                    Xf_tr_sel,
                    yf_tr,
                    eval_set=[(Xf_val_sel, yf_val)],
                    eval_metric="rmse",
                    callbacks=[
                        lgb.early_stopping(200, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred_inner = m.predict(Xf_val_sel)
                rmse = root_mean_squared_error(yf_val, y_pred_inner)
                scores.append(rmse)

            rmse_mean = np.mean(scores)
            return rmse_mean

        print(
            "\n--- Nested CV (outer fold "
            f"{fold_idx}): inner search over {len(param_samples)} configs ---"
        )

        scores = [(cv_score_for_params(ps), ps) for ps in param_samples]
        scores.sort(key=lambda t: t[0])
        best_score, best_params = scores[0]

        print(
            f"Best inner-CV RMSE (fold {fold_idx}): {best_score:.3f}\n"
            f"Best params: {best_params}"
        )

        # feature selection on FULL outer-train only
        selected_outer = select_features_lgbm(
            X_train=X_tr,
            y_train=y_tr,
            feat_cols=feat_cols,
            corr_min=corr_min,
            keep_top_frac=keep_top_frac,
        )

        print(f"Selected features on outer-train: {len(selected_outer)}")

        selected_counts.append(
            {
                "outer_fold": fold_idx,
                "n_selected_features": len(selected_outer),
            }
        )

        selected_df = pd.DataFrame({"feature": selected_outer})
        selected_df.to_csv(
            out_dir / f"selected_features_outerfold{fold_idx}.csv",
            index=False,
        )

        X_tr_sel = X_tr[selected_outer]
        X_te_sel = X_te[selected_outer]

        # final refit on full outer-train 
        m_final = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=4000,
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr_sel, y_tr)

        # store feature importance for this outer fold 
        fi_gain_fold = m_final.booster_.feature_importance(importance_type="gain")
        fi_per_fold.append(dict(zip(selected_outer, fi_gain_fold)))

        # final evaluation on untouched outer-test
        y_pred = m_final.predict(X_te_sel)

        rmse = root_mean_squared_error(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)

        fold_results.append(
            dict(
                rmse=rmse,
                mae=mae,
                r2=r2,
            )
        )

        print(f"Finished outer fold {fold_idx}")

    # ==========================
    # 7) Summary metrics
    # ==========================
    if len(fold_results) > 0:
        print("\n" + "=" * 72)
        print(f"=== Summary over {len(fold_results)} CV1 outer folds ===")

        metric_names = list(fold_results[0].keys())
        summary_rows = []

        for metric in metric_names:
            vals = np.array([fr[metric] for fr in fold_results], dtype=float)
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"{metric}_mean={mean_val:.4f}  {metric}_std={std_val:.4f}")
            summary_rows.append(
                dict(metric=metric, mean=mean_val, std=std_val)
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"cv_metrics_summary_{SCHEME.lower()}.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nSaved CV metrics summary to:", summary_path)
        print("=" * 72)

        # save selected feature counts per fold
        selected_counts_df = pd.DataFrame(selected_counts)
        selected_counts_path = out_dir / "selected_feature_counts_per_fold.csv"
        selected_counts_df.to_csv(selected_counts_path, index=False)
        print("Saved selected feature counts to:", selected_counts_path)
        

    print("\n=== EXP08 DONE ===\n")


if __name__ == "__main__":
    main()