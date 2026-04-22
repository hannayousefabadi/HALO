#!/usr/bin/env python3
"""
Experiment: exp08_lgbm_regr_sspace_elementwise_reduced_nestedcv (HALO-S-CV1, regression)

Config
- task: regression on Bliss Score (continuous outcome)
- feature_design: reduced elementwise similarity features (selected in exp05; includes CC + S-space dimensions)
- use_sspace: true (S-space is already included in the reduced feature table; no feature recomputation here)
- cv_scheme: CV1 by default (drug-pair held-out via GroupShuffleSplit, 80/20); optional CV2 (strain + pair disjoint)
- nested_cv: true
    - inner_cv: 3-fold grouped CV by Drug Pair (GroupKFold) for hyperparameter search
    - model selection: 32 sampled LightGBM configurations ranked by mean inner-CV RMSE (lower is better)
- model: LightGBM regressor (objective="regression") with early stopping on RMSE
- evaluation: held-out outer test set; reports RMSE, MAE, R², Spearman ρ, and Pearson r
- outputs: saved predicted-vs-true scatter plot (pred_vs_true_{scheme}.png) plus console-logged metrics

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.)
was completed in the preprocessing scripts.
This notebook assumes clean, validated input data.
"""

import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import spearmanr, pearsonr
from halo.paths import MODEL_RESULTS


def main():
    # ==========================
    # 0) Basic config
    # ==========================
    SCHEME = "CV1"  # or "CV2"

    # Path to reduced elementwise dataset from exp05
    filtered_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1_full.csv"

    out_dir = MODEL_RESULTS / "exp08_lgbm_regr_sspace_elementwise_reduced_nestedcv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== EXP08: LGBM regression + reduced elementwise (from exp05) + nested CV ===\n")
    print("Using scheme:", SCHEME)
    print("Input  file:", filtered_path)
    print("Output dir :", out_dir)

    rng = np.random.default_rng(42)

    # ==========================
    # 1) Load reduced dataset
    # ==========================
    if not filtered_path.exists():
        raise FileNotFoundError(f"Filtered CSV not found at: {filtered_path}")

    df = pd.read_csv(filtered_path).copy()
    print("Loaded df shape:", df.shape)

    # Making sure Bliss Score is numeric and drop rows without it
    df["Bliss Score"] = pd.to_numeric(df["Bliss Score"], errors="coerce")
    df = df[~df["Bliss Score"].isna()].copy()
    print("After dropping NaN Bliss Score:", df.shape)

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

    print(f"\nReduced feature columns: {len(feat_cols)}")

    X = df[feat_cols].copy()
    y = df["Bliss Score"].astype(float).values

    pairs = df["Drug Pair"].astype(str).values
    strains = df["Strain"].astype(str).values  # only for CV2 if used
    n = len(df)

    print(f"Total samples: {n}")

    # ==========================
    # 3) Outer splits (CV1 / CV2)
    # ==========================
    def make_split_cv1(verbose=True):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        tr_idx, te_idx = next(gss.split(X, y, groups=pairs))

        if verbose:
            print("=" * 72)
            print("CV1: Drug Pair grouping:")
            print(f"Train size: {len(tr_idx)}")
            print(f"Train size fraction: {round((len(tr_idx) / len(df)), 2) * 100}")
            print(f"Test size: {len(te_idx)}")
            print(f"Test size fraction: {round((len(te_idx) / len(df)), 2) * 100}")
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print("-" * 72)
        return tr_idx, te_idx

    def make_split_cv2(
        strain_col: str = "Strain",
        pair_col: str = "Drug Pair",
        min_frac: float = 0.16,
        max_frac: float = 0.20,
        lambda_penalty: float = 1.0,
        top_k_print: int = 5,
        verbose: bool = True,
    ):
        """
        CV2: held out by Strain + Drug Pair, copied from exp03 logic.
        """
        S_all = df[strain_col].astype(str).values
        P_all = df[pair_col].astype(str).values
        strains_uni = sorted(np.unique(S_all).tolist())
        n_total = len(df)
        min_target = int(round(min_frac * n_total))
        max_target = int(round(max_frac * n_total))

        if verbose:
            print("=" * 72)
            print("CV2 Strain + Drug Pair grouping:")
            print(f"Total rows: {n_total} | Strain levels: {len(strains_uni)}")
            print(
                f"Target kept test rows ∈ "
                f"[{min_target} ({min_frac:.0%}), {max_target} ({max_frac:.0%})]"
            )

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
                    candidates.append(
                        {
                            "S_test": S_test,
                            "P_test": P_test,
                            "kept": kept,
                            "train": train,
                            "dropped": dropped,
                            "score": score,
                        }
                    )

        if not candidates:
            if verbose:
                print("-" * 72)
                print(
                    "No subsets hit the target kept-test band. "
                    "You can widen [min_frac, max_frac] or allow manual S_test."
                )
                print("=" * 72)
            return np.arange(n_total), np.array([], dtype=int), {
                "reason": "no_candidate",
                "test_strains": set(),
                "test_pairs": set(),
                "train_strains": set(strains_uni),
                "train_pairs": set(np.unique(P_all).tolist()),
                "kept_test_rows": 0,
                "kept_train_rows": n_total,
                "dropped_rows": 0,
                "params": dict(
                    min_frac=min_frac,
                    max_frac=max_frac,
                    lambda_penalty=lambda_penalty,
                ),
            }

        candidates.sort(key=lambda c: (c["dropped"], -c["kept"], -c["score"]))

        if verbose:
            print("-" * 72)
            print(
                f"Valid candidates in band: {len(candidates)} | "
                f"Showing top {min(top_k_print, len(candidates))}"
            )
            print(
                "rank | #S | #P | kept(%) | dropped(%) | score | S_test (truncated)"
            )
            for i, c in enumerate(candidates[:top_k_print], 1):
                kept_pct = 100.0 * c["kept"] / n_total
                drop_pct = 100.0 * c["dropped"] / n_total
                s_preview = ", ".join(list(sorted(c["S_test"]))[:3])
                if len(c["S_test"]) > 3:
                    s_preview += ", …"
                print(
                    f"{i:>4} | {len(c['S_test']):>2} | {len(c['P_test']):>3} | "
                    f"{kept_pct:6.2f} | {drop_pct:9.2f} | {c['score']:>7.1f} | {s_preview}"
                )

        best = candidates[0]
        S_test_best = best["S_test"]
        P_test_best = best["P_test"]

        test_mask = np.isin(S_all, list(S_test_best)) & np.isin(
            P_all, list(P_test_best)
        )
        train_mask = (~np.isin(S_all, list(S_test_best))) & (
            ~np.isin(P_all, list(P_test_best))
        )

        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]
        dropped_rows = n_total - (te_idx.size + tr_idx.size)

        S_train_best = set(strains_uni) - set(S_test_best)
        P_train_best = set(np.unique(P_all).tolist()) - set(P_test_best)

        if verbose:
            print("-" * 72)
            print("Chosen subset (BEST):")
            print(f"#Test Strains: {len(S_test_best)} | #Test Pairs: {len(P_test_best)}")
            print(
                f"Train rows: {tr_idx.size} ({tr_idx.size/n_total*100:.2f}%)"
            )
            print(f"Test  rows: {te_idx.size} ({te_idx.size/n_total*100:.2f}%)")
            print(
                f"Dropped rows: {dropped_rows} "
                f"({dropped_rows/n_total*100:.2f}%)"
            )
            print(f"Test + Train set: {len(tr_idx) + len(te_idx)}")
            print(
                f"Overlap Strains? {len(S_train_best & S_test_best)} (expect 0)"
            )
            print(f"Overlap Pairs?   {len(P_train_best & P_test_best)} (expect 0)")
            print("=" * 72)

        info = dict(
            mode="bruteforce_strains",
            test_strains=set(S_test_best),
            train_strains=S_train_best,
            test_pairs=set(P_test_best),
            train_pairs=P_train_best,
            kept_test_rows=int(te_idx.size),
            kept_train_rows=int(tr_idx.size),
            dropped_rows=int(dropped_rows),
            params=dict(
                min_frac=min_frac,
                max_frac=max_frac,
                lambda_penalty=lambda_penalty,
            ),
            top_candidates=candidates[:top_k_print],
        )
        return tr_idx, te_idx, info

    if SCHEME == "CV1":
        tr_idx, te_idx = make_split_cv1()
        info = None
    elif SCHEME == "CV2":
        tr_idx, te_idx, info = make_split_cv2()
        print("test strains:", info["test_strains"])
    else:
        raise ValueError("SCHEME must be 'CV1' or 'CV2'")

    # ==========================
    # 4) Inner CV (nested)
    # ==========================
    class SilentLogger:
        def info(self, msg): pass
        def warning(self, msg): pass

    lgb.register_logger(SilentLogger())

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_te = y[te_idx]
    grp_tr = pairs[tr_idx]

    # random parameter sampler
    def sample_one_params():
        max_depth = 3  # shallow to fight overfit
        leaves_map = {3: [7, 15]}
        return dict(
            boosting_type=rng.choice(["gbdt", "dart"], p=[0.6, 0.4]),
            learning_rate=float(rng.choice([0.02, 0.03, 0.04, 0.05])),
            max_depth=max_depth,
            num_leaves=int(rng.choice(leaves_map[max_depth])),
            min_data_in_leaf=int(rng.choice([100, 200])),
            feature_fraction=float(rng.choice([0.30, 0.40])),
            bagging_fraction=float(rng.choice([0.60, 0.80])),
            bagging_freq=1,
            lambda_l2=float(10 ** rng.uniform(1.2, 1.7)),
            lambda_l1=float(rng.choice([0.0, 0.1, 0.5])),
            max_bin=int(rng.choice([63, 127])),
            min_gain_to_split=float(rng.choice([0.05, 0.10, 0.20])),
        )

    param_samples = [sample_one_params() for _ in range(32)]

    inner_cv = GroupKFold(n_splits=3)

    def inner_splitter():
        return inner_cv.split(X_tr, y_tr, groups=grp_tr)

    def cv_rmse_for_params(params):
        """
        Inner CV for regression: early-stop on RMSE, score = mean RMSE across folds.
        Lower is better.
        """
        rmses = []
        for tr_f, val_f in inner_splitter():
            Xf_tr, Xf_val = X_tr.iloc[tr_f], X_tr.iloc[val_f]
            yf_tr, yf_val = y_tr[tr_f], y_tr[val_f]

            m = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=4000,
                random_state=777,
                n_jobs=4,
                **params,
            )
            m.fit(
                Xf_tr,
                yf_tr,
                eval_set=[(Xf_val, yf_val)],
                eval_metric="rmse",  # early stopping on RMSE
                callbacks=[
                    lgb.early_stopping(200, False),
                    lgb.log_evaluation(0),
                ],
            )

            y_pred = m.predict(Xf_val)
            rmse = np.sqrt(mean_squared_error(yf_val, y_pred))
            rmses.append(rmse)

        return float(np.mean(rmses))

    print("\n--- Nested CV: inner search over", len(param_samples), "configs ---")
    scores = [(cv_rmse_for_params(ps), ps) for ps in param_samples]
    scores.sort(key=lambda t: t[0])  # lower RMSE is better
    best_rmse, best_params = scores[0]
    print("Best inner-CV RMSE:", round(best_rmse, 4), "\nBest params:", best_params)

    # ==========================
    # 5) Final refit on outer-train
    # ==========================
    m_final = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=4000,
        random_state=777,
        n_jobs=4,
        **best_params,
    )
    m_final.fit(X_tr, y_tr)

    # ==========================
    # 6) Final evaluation on held-out test
    # ==========================
    y_te_pred = m_final.predict(X_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_te_pred))
    mae_te = mean_absolute_error(y_te, y_te_pred)
    r2_te = r2_score(y_te, y_te_pred)
    spr_te = spearmanr(y_te, y_te_pred).statistic
    pr_te = pearsonr(y_te, y_te_pred).statistic

    print("\n=== Held-out Test (Regression) ===")
    print("RMSE      :", round(rmse_te, 4))
    print("MAE       :", round(mae_te, 4))
    print("R^2       :", round(r2_te, 4))
    print("Spearman ρ:", round(spr_te, 4))
    print("Pearson r :", round(pr_te, 4))

    # ==========================
    # 7) Overfitting check
    # ==========================
    y_tr_pred = m_final.predict(X_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
    mae_tr = mean_absolute_error(y_tr, y_tr_pred)
    r2_tr = r2_score(y_tr, y_tr_pred)
    spr_tr = spearmanr(y_tr, y_tr_pred).statistic
    pr_tr = pearsonr(y_tr, y_tr_pred).statistic

    print("\n=== Overfitting check (Regression) ===")
    print("Train RMSE:", round(rmse_tr, 4), "| Test RMSE:", round(rmse_te, 4))
    print("Train MAE :", round(mae_tr, 4),  "| Test MAE :", round(mae_te, 4))
    print("Train R^2 :", round(r2_tr, 4),   "| Test R^2 :", round(r2_te, 4))
    print("Train ρ   :", round(spr_tr, 4),  "| Test ρ   :", round(spr_te, 4))
    print("Train r   :", round(pr_tr, 4),   "| Test r   :", round(pr_te, 4))

    # ==========================
    # 8) Pred vs True scatter → SAVE
    # ==========================
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_te, y_te_pred, alpha=0.4)
    ax.set_xlabel("True Bliss Score")
    ax.set_ylabel("Predicted Bliss Score")
    ax.set_title(f"Pred vs True (R²={r2_te:.2f})")
    # 45-degree line
    min_val = min(np.min(y_te), np.min(y_te_pred))
    max_val = max(np.max(y_te), np.max(y_te_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

    plt.tight_layout()
    fig_path = out_dir / f"pred_vs_true_{SCHEME.lower()}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print("\nSaved Pred vs True plot to:", fig_path)
    print("\n=== EXP08 DONE ===\n")


if __name__ == "__main__":
    main()
