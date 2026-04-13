#!/usr/bin/env python3
"""
Experiment: exp07_lgbm_multi_sspace_elementwise_reduced_nestedcv

Config
- model: LightGBM 
- task: multiclass classification
- feature_design: elementwise similarity 
- sspace: enabled (strain-space features)
- feature_selection: enabled (within CV folds)
- bliss neutrality cutoff: ±0.1 (applied during preprocessing; this script assumes labels are already finalized)

- CV:
  - nested_cv: enabled
  - Outer split:
    - CV1 scheme: drug pair held-out
  - Inner split:
    - StratifiedGroupKFold, groups = Drug Pair
    - random search over 32 sampled hyperparameter configs
    - selection metric: mean validation accuracy across inner folds
  - Final fit: refit best model on full outer-train, evaluate once on outer-test

Data integrity note
All preprocessing (missing values, dtypes, column validation, and label construction from Bliss using the
±0.1 cutoff) is performed upstream in preprocessing notebooks/scripts. This script assumes the processed
inputs are clean and consistent and that `Interaction Type` already reflects that cutoff.

Class encoding
The binary target is encoded as {0,1} where 1 corresponds to synergy and is treated as the
positive class for F1 and ROC-AUC.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
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
    fs_model = lgb.LGBMClassifier(
        objective="multiclass",
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

    out_dir = MODEL_RESULTS / "exp07_lgbm_multi_sspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP07 ===\n"
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

    # keep 3 classes
    df = df[df["Interaction Type"].isin(["synergy", "antagonism", "neutral"])].copy()
    print("\nAfter filtering 3 classes:", df.shape)

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
    y = df["Interaction Type"].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pairs = df["Drug Pair"].astype(str).values
    n = len(df)

    print("Classes:", list(le.classes_))

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
        try:
            outer_cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42,
            )
            split_gen = outer_cv.split(X, y_enc, groups=pairs)
        except TypeError:
            outer_cv = GroupKFold(n_splits=n_splits)
            split_gen = outer_cv.split(X, y_enc, groups=pairs)

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
    cm_total = None

    selected_counts = []
    fi_per_fold = []

    synergy_code = le.transform(["synergy"])[0]
    ant_code = le.transform(["antagonism"])[0]

    # ==========================
    # 6) Outer loop
    # ==========================
    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print("\n" + "#" * 72)
        print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
        print("#" * 72 + "\n")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_tr = y_enc[tr_idx]
        y_te = y_enc[te_idx]
        grp_tr = pairs[tr_idx]

        print(f"Outer-train shape: {X_tr.shape}")
        print(f"Outer-test shape : {X_te.shape}")

        # The inner CV split, 3 folds, grouped by drug-pair
        try:
            inner_cv = StratifiedGroupKFold(
                n_splits=3,
                shuffle=True,
                random_state=111,
            )

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        except Exception:
            inner_cv = GroupKFold(n_splits=3)

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

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
        def cv_acc_for_params(params):
            """
            Perform a full inner cross‑validation loop for a given hyperparameter
            configuration.

            For each inner fold:
            - Fit the feature selector only on the inner‑train split.
            - Select features for that fold.
            - Train a LightGBM model with the provided hyperparameters.
            - Evaluate accuracy on the inner‑validation split.

            Returns the mean validation accuracy across inner folds, used for
            comparing and ranking hyperparameter configurations in the nested CV.

            It is trying to answer: “If we used this hyperparameter configuration, 
            how well would the pipeline generalize?”
            """    
            accs = []

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

                m = lgb.LGBMClassifier(
                    objective="multiclass",
                    n_estimators=4000,
                    random_state=777,
                    n_jobs=4,
                    **params,
                )

                m.fit(
                    Xf_tr_sel,
                    yf_tr,
                    eval_set=[(Xf_val_sel, yf_val)],
                    eval_metric="multi_logloss",
                    callbacks=[
                        lgb.early_stopping(200, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred_inner = m.predict(Xf_val_sel)
                accs.append(accuracy_score(yf_val, y_pred_inner))

            return float(np.mean(accs))

        print(
            "\n--- Nested CV (outer fold "
            f"{fold_idx}): inner search over {len(param_samples)} configs ---"
        )

        scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
        scores.sort(reverse=True, key=lambda t: t[0])
        best_acc, best_params = scores[0]

        print(
            f"Best inner-CV ACC (fold {fold_idx}): {best_acc:.3f}\n"
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
        m_final = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=4000,
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr_sel, y_tr)

        # store feature importance for this outer fold 
        fi_gain_fold = m_final.booster_.feature_importance(importance_type="gain")
        fi_per_fold.append(dict(zip(selected_outer, fi_gain_fold)))

        pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

        # final evaluation on untouched outer-test
        p_te = m_final.predict_proba(X_te_sel)[:, pos_idx]
        y_pred = (p_te >= 0.5).astype(int)

        y_te_bin = (y_te == synergy_code).astype(int)

        accuracy_test = accuracy_score(y_te, y_pred)
        f1_macro_test = f1_score(y_te, y_pred, average="macro")
        f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
        roc_auc_test = roc_auc_score(y_te_bin, p_te)

        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_te,
            y_pred,
            labels=[ant_code, synergy_code],
        )
        precision_antag, precision_syn = prec
        recall_antag, recall_syn = rec
        f1_antag, f1_syn = f1s

        order = ["antagonism", "synergy"]
        order_idx = le.transform(order)
        cm = confusion_matrix(y_te, y_pred, labels=order_idx)

        if cm_total is None:
            cm_total = cm.astype(float)
        else:
            cm_total += cm.astype(float)

        fold_results.append(
            dict(
                roc_auc=roc_auc_test,
                accuracy=accuracy_test,
                f1_weighted=f1_weighted_test,
                f1_macro=f1_macro_test,
                precision_antag=precision_antag,
                recall_antag=recall_antag,
                f1_antag=f1_antag,
                precision_syn=precision_syn,
                recall_syn=recall_syn,
                f1_syn=f1_syn,
            )
        )

        print(f"Finished outer fold {fold_idx}")

    # ==========================
    # 7) Summary metrics
    # ==========================
    if len(fold_results) > 0:
        print("\n" + "=" * 72)
        print(f"=== Summary over {len(fold_results)} outer folds ===")

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
        
        # ==========================
        # 8) Confusion matrix plot → SAVE
        # ==========================
        if cm_total is not None:
            cm_mean = cm_total / len(fold_results)

            order = ["antagonism", "synergy"]

            cm_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_mean.csv"
            cm_df = pd.DataFrame(
                cm_mean,
                index=pd.Index(order, name="true"),
                columns=pd.Index(order, name="pred"),
            )
            cm_df.to_csv(cm_path)
            print("Saved mean confusion matrix to:", cm_path)

            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_mean,
                display_labels=order,
            )
            disp.plot(cmap="Blues", ax=ax, values_format=".2f")
            ax.set_title(
                "Confusion Matrix — Mean over outer folds",
                fontsize=13,
                pad=15,
            )
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label", fontsize=11)
            plt.tight_layout()

            fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_mean.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print("Saved mean confusion matrix plot to:", fig_path)

    # ==========================
    # 8) Aggregated feature importances across outer folds
    # ==========================
    if len(fi_per_fold) > 0:
        all_features = sorted(set().union(*[d.keys() for d in fi_per_fold]))

        rows = []
        for feat in all_features:
            vals = [d.get(feat, 0.0) for d in fi_per_fold]
            rows.append(
                {
                    "feature": feat,
                    "importance_gain_mean": float(np.mean(vals)),
                    "importance_gain_std": float(np.std(vals)),
                    "selected_in_n_folds": int(sum(feat in d for d in fi_per_fold)),
                }
            )

        fi_df = pd.DataFrame(rows).sort_values(
            "importance_gain_mean",
            ascending=False,
        )

        fi_path = out_dir / "feature_importances_cv1.csv"
        fi_df.to_csv(fi_path, index=False)
        print("\nSaved aggregated feature importances to:", fi_path)

    print("\n=== EXP07 DONE ===\n")


if __name__ == "__main__":
    main()



















    # ==========================
    # 3) Outer CV split (CV1 / CV2)
    # ==========================
    def make_split_cv1(verbose=True):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y_enc, groups=pairs))
        if verbose:
            print("=" * 72)
            print("CV1 split by Drug Pair")
            print("Train:", len(tr_idx), "Test:", len(te_idx))
            print("=" * 72)
        return tr_idx, te_idx

    def make_split_cv2(
        strain_col="Strain",
        pair_col="Drug Pair",
        min_frac=0.16,
        max_frac=0.20,
        lambda_penalty=1.0,
        top_k_print=5,
        verbose=True
    ):
        # exact copy of exp03 logic
        S_all = df[strain_col].astype(str).values
        P_all = df[pair_col].astype(str).values
        strains_uni = sorted(np.unique(S_all).tolist())
        n_total = len(df)
        min_target = int(round(min_frac * n_total))
        max_target = int(round(max_frac * n_total))

        if verbose:
            print("=" * 72)
            print("CV2: Strain + Drug Pair split")
            print(f"Rows: {n_total}  Strains: {len(strains_uni)}")

        def eval_subset(S_test_set):
            if not S_test_set:
                return 0, 0, n_total, set(), -np.inf
            mask_s = np.isin(S_all, list(S_test_set))
            P_test_set = set(P_all[mask_s])
            test_mask = mask_s & np.isin(P_all, list(P_test_set))
            train_mask = (~mask_s) & (~np.isin(P_all, list(P_test_set)))
            kept = int(test_mask.sum())
            train = int(train_mask.sum())
            dropped = n_total - (kept + train)
            score = kept - lambda_penalty * dropped
            return kept, train, dropped, P_test_set, score

        candidates = []
        for r in range(1, len(strains_uni) + 1):
            for subset in itertools.combinations(strains_uni, r):
                S_test = set(subset)
                kept, train, dropped, P_test, score = eval_subset(S_test)
                if min_target <= kept <= max_target:
                    candidates.append(dict(
                        S_test=S_test,
                        P_test=P_test,
                        kept=kept,
                        train=train,
                        dropped=dropped,
                        score=score
                    ))

        if not candidates:
            print("\nCV2 could not find a valid split.")
            return np.arange(n_total), np.array([], dtype=int), None

        candidates.sort(key=lambda c: (c["dropped"], -c["kept"], -c["score"]))
        best = candidates[0]

        S_test_best = best["S_test"]
        P_test_best = best["P_test"]

        test_mask = np.isin(S_all, list(S_test_best)) & np.isin(P_all, list(P_test_best))
        train_mask = (~np.isin(S_all, list(S_test_best))) & (~np.isin(P_all, list(P_test_best)))

        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]

        if verbose:
            print("=" * 72)
            print("Chosen CV2 split:")
            print("#Train:", len(tr_idx), "#Test:", len(te_idx))
            print("=" * 72)

        return tr_idx, te_idx, best

    if SCHEME == "CV1":
        tr_idx, te_idx = make_split_cv1()
        info = None
    else:
        tr_idx, te_idx, info = make_split_cv2()

    # ==========================
    # 4) Prepare nested CV folds
    # ==========================
    class SilentLogger:
        def info(self, msg): pass
        def warning(self, msg): pass

    lgb.register_logger(SilentLogger())

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y_enc[tr_idx]
    y_te = y_enc[te_idx]
    grp_tr = pairs[tr_idx]

    # ==========================
    # 5) Hyperparameter search
    # ==========================
    def sample_one_params():
        max_depth = 3
        leaves_map = {3: [7, 15]}
        return dict(
            boosting_type=rng.choice(["gbdt", "dart"], p=[0.6, 0.4]),
            learning_rate=float(rng.choice([0.02, 0.03, 0.04, 0.05])),
            max_depth=max_depth,
            num_leaves=int(rng.choice(leaves_map[max_depth])),
            min_data_in_leaf=200,
            feature_fraction=float(rng.choice([0.30, 0.40])),
            bagging_fraction=float(rng.choice([0.60, 0.80])),
            bagging_freq=1,
            lambda_l2=float(10 ** rng.uniform(1.2, 1.7)),
            lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
            max_bin=int(rng.choice([63, 127])),
            min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
        )

    param_samples = [sample_one_params() for _ in range(32)]

    try:
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=111)
        def inner_splitter():
            return inner_cv.split(X_tr, y_tr, groups=grp_tr)
    except Exception:
        inner_cv = GroupKFold(n_splits=3)
        def inner_splitter():
            return inner_cv.split(X_tr, y_tr, groups=grp_tr)

    def cv_acc_for_params(params):
        scores = []
        for tr_f, val_f in inner_splitter():
            Xf_tr, Xf_val = X_tr.iloc[tr_f], X_tr.iloc[val_f]
            yf_tr, yf_val = y_tr[tr_f], y_tr[val_f]

            m = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=3,
                n_estimators=4000,
                random_state=777,
                n_jobs=4,
                **params,
            )
            m.fit(
                Xf_tr,
                yf_tr,
                eval_set=[(Xf_val, yf_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(200, False)],
            )

            probs = m.predict_proba(Xf_val)
            y_pred = np.argmax(probs, axis=1)
            scores.append(f1_score(yf_val, y_pred, average="macro"))

        return float(np.mean(scores))
    

    print("\n--- Inner CV hyperparameter search ---")
    scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
    scores.sort(reverse=True)
    best_acc, best_params = scores[0]
    print("Best inner-CV ACC:", round(best_acc, 3))
    print("Best params:", best_params)

    # ==========================
    # 6) Final refit
    # ==========================
    m_final = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr, y_tr)

    # ==========================
    # 7) Final evaluation
    # ==========================
    probs_te = m_final.predict_proba(X_te)
    y_pred = np.argmax(probs_te, axis=1)

    # ---- Global metrics (compute once) ----
    roc_auc_test = roc_auc_score(
        y_te,
        probs_te,
        multi_class="ovr",
        average="macro",
        labels=m_final.classes_,
    )
    accuracy_test = accuracy_score(y_te, y_pred)
    f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
    f1_macro_test = f1_score(y_te, y_pred, average="macro")

    print("\n=== Held-out Test Metrics ===")
    print(f"Macro ROC-AUC: {roc_auc_test:.3f}")
    print(f"Acc          : {accuracy_test:.3f}")
    print(f"F1 (weighted): {f1_weighted_test:.3f}")
    print(f"F1 (macro)   : {f1_macro_test:.3f}")

    print("\nConfusion matrix:\n", confusion_matrix(y_te, y_pred))
    print(
        "\nClassification Report:\n",
        classification_report(y_te, y_pred, target_names=le.classes_),
    )

    # ---- Per-class metrics (antagonism, neutral, synergy) ----
    ant_code = le.transform(["antagonism"])[0]
    neu_code = le.transform(["neutral"])[0]
    syn_code = le.transform(["synergy"])[0]

    prec, rec, f1s, _ = precision_recall_fscore_support(
        y_te,
        y_pred,
        labels=[ant_code, neu_code, syn_code],
    )

    precision_antag, precision_neutral, precision_syn = prec
    recall_antag, recall_neutral, recall_syn = rec
    f1_antag, f1_neutral, f1_syn = f1s

    # ---- Log-friendly lines (for grep / parsing) ----
    print("\n--- Metrics for Log ---")
    print(f"accuracy_test={accuracy_test:.4f}")
    print(f"f1_macro_test={f1_macro_test:.4f}")
    print(f"f1_weighted_test={f1_weighted_test:.4f}")
    print(f"roc_auc_test={roc_auc_test:.4f}")

    print(f"precision_antag={precision_antag:.4f}")
    print(f"recall_antag={recall_antag:.4f}")
    print(f"f1_antag={f1_antag:.4f}")

    print(f"precision_neutral={precision_neutral:.4f}")
    print(f"recall_neutral={recall_neutral:.4f}")
    print(f"f1_neutral={f1_neutral:.4f}")

    print(f"precision_syn={precision_syn:.4f}")
    print(f"recall_syn={recall_syn:.4f}")
    print(f"f1_syn={f1_syn:.4f}")


    # ==========================
    # 8) Overfitting check
    # ==========================
    probs_tr = m_final.predict_proba(X_tr)
    y_tr_pred = np.argmax(probs_tr, axis=1)

    auc_tr = roc_auc_score(
        y_tr,
        probs_tr,
        multi_class="ovr",
        average="macro",
        labels=m_final.classes_,
    )
    acc_tr = accuracy_score(y_tr, y_tr_pred)
    f1_macro_tr = f1_score(y_tr, y_tr_pred, average="macro")

    print("\n=== Overfitting check ===")
    print(f"Train AUC: {auc_tr:.3f} | Test AUC: {roc_auc_test:.3f}")
    print(f"Train Acc: {acc_tr:.3f} | Test Acc: {accuracy_test:.3f}")
    print(f"Train F1 macro: {f1_macro_tr:.3f} | Test F1 macro: {f1_macro_test:.3f}")


    # ==========================
    # 9) Confusion matrix → SAVE
    # ==========================
    order = ["antagonism", "neutral", "synergy"]
    order_idx = le.transform(order)
    cm = confusion_matrix(y_te, y_pred, labels=order_idx)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=order)
    disp.plot(cmap="Blues", ax=ax)

    plt.title("Confusion Matrix — Multi-class (reduced elementwise)")
    plt.tight_layout()

    fig_path = out_dir / f"confusion_matrix_{SCHEME.lower()}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("\nSaved confusion matrix:", fig_path)
    print("\n=== EXP07 DONE ===")


if __name__ == "__main__":
    main()
