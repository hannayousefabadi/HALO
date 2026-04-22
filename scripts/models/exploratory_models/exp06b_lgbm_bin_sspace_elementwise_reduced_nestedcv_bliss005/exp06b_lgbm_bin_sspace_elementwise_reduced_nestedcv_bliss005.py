#!/usr/bin/env python3
"""
Experiment: exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005 (HALO-S-CV1)

- task: binary classification (synergy vs antagonism) using a Bliss additivity cutoff of ±0.05
- feature_design: reduced elementwise similarity features (selected in exp05b; S-space included)
- use_sspace: true (already baked into the reduced feature table; no feature recomputation here)
- outer_cv: CV1-style evaluation (drug-pair held-out)
    - 5 outer folds generated with StratifiedGroupKFold grouped by Drug Pair
      (falls back to GroupKFold if StratifiedGroupKFold is unavailable)
- inner_cv: nested hyperparameter search inside each outer fold
    - 3-fold StratifiedGroupKFold (group = Drug Pair; fallback to GroupKFold)
    - 32 LightGBM configurations sampled from a predefined search space
    - best configuration selected by mean inner-CV accuracy
- model: LightGBM binary classifier; “synergy” treated as the positive class for ROC AUC
- evaluation: per outer fold, refit on the full outer-train set and evaluate once on the outer-test set

Outputs (MODEL_RESULTS/exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005):
- cv_metrics_summary_cv1.csv          : mean±std over folds for AUC/accuracy/F1 and per-class metrics
- confusion_matrix_cv1_mean.csv/.png  : mean confusion matrix over outer folds
- feature_importances_cv1.csv         : mean±std feature importance (gain) over outer folds

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.) was completed upstream.
This script assumes clean, validated input data.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from halo.paths import MODEL_RESULTS
from halo.shared_utils.data_io import classify_interaction


def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1" 

    filtered_path = MODEL_RESULTS / "exp05b_lgbm_bin_sspace_elementwise_featselect_bliss005" / "elementwise_features_filtered_cv1.csv"

    out_dir = MODEL_RESULTS / "exp06b_lgbm_bin_sspace_elementwise_reduced_nestedcv_bliss005"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\n=== EXP06b: LGBM bin + reduced elementwise (from exp05), "
        "±0.05 bliss cutoff + nested CV ===\n"
    )
    print("Using scheme:", SCHEME)
    print("Input  file:", filtered_path)
    print("Output dir :", out_dir)

    # ==========================
    # 1) Load reduced dataset
    # ==========================
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered CSV not found at: {filtered_path}")

    df = pd.read_csv(filtered_path).copy()
    print("Loaded reduced df shape:", df.shape)
    print(df["Interaction Type"].value_counts())

    # Re-label using ±0.05 cutoff and keep only synergy/antagonism
    df["Interaction Type"] = df["Bliss Score"].apply(
        lambda x: classify_interaction(x, additivity_cutoff=0.05)
    )
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering to synergy/antagonism:", df.shape)
    print(df["Interaction Type"].value_counts())

    # ==========================
    # 2) Feature columns (reduced elementwise only)
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

    rng = np.random.default_rng(42)

    print(f"\nTotal samples: {n}")
    print(f"Reduced feature columns: {len(feat_cols)}")

    # ==========================
    # 3) Outer splits (CV1 / CV2)
    # ==========================
    def make_splits_cv1(n_splits=5, verbose=True):
        """
        5-fold outer CV over Drug Pair groups (CV1).
        Uses StratifiedGroupKFold if available, falls back to GroupKFold.
        """
        try:
            outer_cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=42
            )
            split_gen = outer_cv.split(X, y_enc, groups=pairs)
        except TypeError:
            # Older sklearn: StratifiedGroupKFold may not support shuffle / seed
            outer_cv = GroupKFold(n_splits=n_splits)
            split_gen = outer_cv.split(X, y_enc, groups=pairs)

        splits = []
        for fold_idx, (tr_idx, te_idx) in enumerate(split_gen, 1):
            splits.append((tr_idx, te_idx))
            if verbose:
                print("=" * 72)
                print(f"CV1 outer fold {fold_idx}/{n_splits} (Drug Pair grouping):")
                print(f"Train size: {len(tr_idx)}")
                print(
                    f"Train size fraction: {len(tr_idx) / len(df) * 100:.2f}%"
                )
                print(f"Test size: {len(te_idx)}")
                print(
                    f"Test size fraction: {len(te_idx) / len(df) * 100:.2f}%"
                )
                print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
                print("-" * 72)
        return splits

    # Decide outer splits
    if SCHEME == "CV1":
        outer_splits = make_splits_cv1(n_splits=5, verbose=True)
        info = None
    else:
        raise ValueError("SCHEME must be 'CV1' in this simplified script")

    # ==========================
    # 4–8) Nested CV per outer fold
    # ==========================
    class SilentLogger:
        def info(self, msg):
            pass

        def warning(self, msg):
            pass

    lgb.register_logger(SilentLogger())

    # store metrics across folds
    fold_results = []
    cm_total = None
    fi_per_fold = []  # store feature importances per outer fold

    synergy_code = le.transform(["synergy"])[0]
    ant_code = le.transform(["antagonism"])[0]

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, 1):
        print("\n" + "#" * 72)
        print(f"########## OUTER FOLD {fold_idx}/{len(outer_splits)} ##########")
        print("#" * 72 + "\n")

        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        X_te = X.iloc[te_idx].reset_index(drop=True)
        y_tr = y_enc[tr_idx]
        y_te = y_enc[te_idx]
        grp_tr = pairs[tr_idx]

        # ---- Inner CV (nested) ----
        def sample_one_params():
            max_depth = 3
            leaves_map = {3: [7, 9, 15]}

            return dict(
                boosting_type="gbdt",
                learning_rate=float(rng.choice([0.02, 0.03])),
                max_depth=max_depth,
                num_leaves=int(rng.choice(leaves_map[max_depth])),
                # moderate regularization 
                min_data_in_leaf=int(rng.choice([200, 300])),
                feature_fraction=float(rng.choice([0.30, 0.40, 0.50])),
                bagging_fraction=float(rng.choice([0.60, 0.70, 0.80])),
                bagging_freq=1,
                lambda_l2=float(10 ** rng.uniform(1.4, 1.9)),  # ~25–80
                lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
                max_bin=int(rng.choice([63, 127])),
                min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
            )

        param_samples = [sample_one_params() for _ in range(32)]

        try:
            inner_cv = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=111
            )

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        except Exception:
            inner_cv = GroupKFold(n_splits=3)

            def inner_splitter():
                return inner_cv.split(X_tr, y_tr, groups=grp_tr)

        def cv_acc_for_params(params):
            accs = []
            for tr_f, val_f in inner_splitter():
                Xf_tr, Xf_val = X_tr.iloc[tr_f], X_tr.iloc[val_f]
                yf_tr, yf_val = y_tr[tr_f], y_tr[val_f]

                m = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=4000,
                    random_state=777,
                    n_jobs=4,
                    **params,
                )
                m.fit(
                    Xf_tr,
                    yf_tr,
                    eval_set=[(Xf_val, yf_val)],
                    eval_metric="binary_logloss",
                    callbacks=[
                        lgb.early_stopping(200, False),
                        lgb.log_evaluation(0),
                    ],
                )

                y_pred_inner = m.predict(Xf_val)
                accs.append(accuracy_score(yf_val, y_pred_inner))
            return float(np.mean(accs))

        print(
            "\n--- Nested CV (outer fold "
            f"{fold_idx}) : inner search over {len(param_samples)} configs ---"
        )
        scores = [(cv_acc_for_params(ps), ps) for ps in param_samples]
        scores.sort(reverse=True, key=lambda t: t[0])
        best_acc, best_params = scores[0]
        print(
            f"Best inner-CV ACC (fold {fold_idx}): {best_acc:.3f}\n"
            f"Best params: {best_params}"
        )

        # ---- Final refit on outer-train ----
        m_final = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4000,
            random_state=777,
            n_jobs=4,
            **best_params,
        )
        m_final.fit(X_tr, y_tr)

        # --- store feature importance for this fold ---
        fi_gain_fold = m_final.booster_.feature_importance(importance_type="gain")
        fi_per_fold.append(fi_gain_fold)

        pos_idx = np.flatnonzero(m_final.classes_ == synergy_code)[0]

        # ---- Final evaluation on held-out test ----
        p_te = m_final.predict_proba(X_te)[:, pos_idx]
        y_pred = (p_te >= 0.5).astype(int)

        # make sure "synergy" is the positive class for AUC
        y_te_bin = (y_te == synergy_code).astype(int)

        # Global metrics for this fold (kept internal only)
        accuracy_test = accuracy_score(y_te, y_pred)
        f1_macro_test = f1_score(y_te, y_pred, average="macro")
        f1_weighted_test = f1_score(y_te, y_pred, average="weighted")
        roc_auc_test = roc_auc_score(y_te_bin, p_te)

        # Per-class metrics (antagonism, synergy)
        syn_code = synergy_code
        prec, rec, f1s, _ = precision_recall_fscore_support(
            y_te,
            y_pred,
            labels=[ant_code, syn_code],
        )
        precision_antag, precision_syn = prec
        recall_antag, recall_syn = rec
        f1_antag, f1_syn = f1s

        # Confusion matrix for this fold (only used to build aggregate)
        order = ["antagonism", "synergy"]
        order_idx = le.transform(order)
        cm = confusion_matrix(y_te, y_pred, labels=order_idx)

        if cm_total is None:
            cm_total = cm.astype(float)
        else:
            cm_total += cm.astype(float)

        # store fold metrics internally for later averaging
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

    # ---- Summary across outer folds ----
    if len(fold_results) > 0:
        print("\n" + "=" * 72)
        print(f"=== Summary over {len(fold_results)} outer folds ===")

        # aggregate metrics: mean and std for all stored metrics
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

        # Save metrics summary to CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"cv_metrics_summary_{SCHEME.lower()}.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\nSaved CV metrics summary to:", summary_path)
        print("=" * 72)

        # Save averaged confusion matrix
        if cm_total is not None:
            # average across folds
            cm_mean = cm_total / len(fold_results)

            order = ["antagonism", "synergy"]

            # Save numeric matrix to CSV
            cm_path = out_dir / f"confusion_matrix_{SCHEME.lower()}_mean.csv"
            cm_df = pd.DataFrame(
                cm_mean,
                index=pd.Index(order, name="true"),
                columns=pd.Index(order, name="pred"),
            )
            cm_df.to_csv(cm_path)
            print("Saved mean confusion matrix to:", cm_path)

            # Plot mean confusion matrix
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm_mean, display_labels=order
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

    # ---- Aggregated feature importances across outer folds ----
    if len(fi_per_fold) > 0:
        fi_array = np.vstack(fi_per_fold)
        mean_importance = np.mean(fi_array, axis=0)
        std_importance = np.std(fi_array, axis=0)

        fi_df = pd.DataFrame({
            "feature": feat_cols,
            "importance_gain_mean": mean_importance,
            "importance_gain_std": std_importance,
        }).sort_values("importance_gain_mean", ascending=False)

        fi_path = out_dir / "feature_importances_cv1.csv"
        fi_df.to_csv(fi_path, index=False)
        print("\nSaved aggregated feature importances to:", fi_path)

    print("\n=== EXP06b DONE ===\n")


if __name__ == "__main__":
    main()
