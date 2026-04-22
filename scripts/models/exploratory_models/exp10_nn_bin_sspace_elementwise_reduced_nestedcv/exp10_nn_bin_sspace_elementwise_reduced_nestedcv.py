#!/usr/bin/env python3
"""
Experiment: exp10_nn_bin_sspace_elementwise_reduced_nestedcv (NN, HALO-S-CV1)

Config
- task: binary classification (synergy vs antagonism) using Bliss cutoff ±0.10
- feature_design: reduced elementwise similarity features (selected in exp05)
- use_sspace: true (S-space features are already included in the reduced feature table; no feature recomputation here)
- cv_scheme: CV1 (drug-pair held-out)
- outer_split: single 80/20 GroupShuffleSplit grouped by Drug Pair (pair-disjoint held-out test)
- nested_cv: true (hyperparameter selection performed only on the outer-train split)
- inner_cv: 3-fold StratifiedGroupKFold grouped by Drug Pair to select the best DNN configuration
- model: feed-forward neural network (MLP) with BatchNorm + ReLU + Dropout; trained with BCEWithLogitsLoss and Adam
- selection metric: mean inner-fold ROC AUC with “synergy” treated as the positive class (ties broken by accuracy)

Training / evaluation
1) Load reduced elementwise feature matrix (exp05) and keep binary labels.
2) Create a CV1 outer split by drug-pair grouping (unseen pairs in outer-test).
3) Run inner grouped CV over a small set of pre-defined NN hyperparameter configs (hidden sizes, dropout, lr, weight decay).
4) Retrain the best config on the full outer-train set using early stopping (with an internal train/val split).
5) Evaluate once on the untouched outer-test split and report AUC/accuracy/F1 plus per-class precision/recall/F1.
6) Report an overfitting check by comparing outer-train vs outer-test metrics.

Outputs
- console: chosen best config, outer-test metrics (ROC AUC, accuracy, F1 macro/weighted), confusion matrix,
  classification report, per-class metrics, and an overfitting check
- note: this script does not write per-fold artifacts because the outer evaluation is a single CV1 split

**Data integrity note:**
All preprocessing (NA handling, dtype enforcement, column validation, etc.)
was completed in the preprocessing scripts.
This notebook assumes clean, validated input data.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from halo.paths import MODEL_RESULTS

# ==========================
#  Dataset wrapper 
# ==========================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float32)
        else:
            X = X.astype(np.float32)
        self.X = X
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================
# DNN model 
# ==========================

class CCNet(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ==========================
# Training helpers
# ==========================

def train_dnn(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    device,
    hidden,
    dropout,
    lr,
    weight_decay,
    batch_size=64,
    max_epochs=120,
    patience=10,
):
    model = CCNet(in_dim=input_dim, hidden=hidden, dropout=dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = np.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---- val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())

        mean_tr = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        print(f"    epoch {epoch:03d} | train={mean_tr:.4f} | val={mean_val:.4f}")

        if mean_val + 1e-6 < best_val_loss:
            best_val_loss = mean_val
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("    early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_proba(model, X, device, batch_size=256):
    ds = TabularDataset(X, np.zeros(len(X)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


# ==========================
# Main 
# ==========================

def main():
    print("\n=== EXP10: DNN + reduced elementwise + nested CV (CV1) ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==========================
    # 1) Load reduced feature matrix (from exp05)
    # ==========================
    reduced_path = MODEL_RESULTS / "exp05_lgbm_bin_sspace_elementwise_featselect" / "elementwise_features_filtered_cv1.csv"

    if not reduced_path.exists():
        raise FileNotFoundError(f"Reduced feature file not found: {reduced_path}")

    df = pd.read_csv(reduced_path).copy()
    print("Loaded df:", df.shape)
    print(df["Interaction Type"].value_counts())

    # binary only
    df = df[df["Interaction Type"].isin(["synergy", "antagonism"])].copy()
    print("\nAfter filtering (binary):", df.shape)
    print(df["Interaction Type"].value_counts())

    # features / target
    drop_cols = [
        "Drug A", "Drug B",
        "Drug A Inchikey", "Drug B Inchikey",
        "Strain", "Specie",
        "Bliss Score",
        "Interaction Type",
        "Source", "Drug Pair",
    ]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    X_all = df[feat_cols].copy()
    y_all = df["Interaction Type"].copy()
    pairs = df["Drug Pair"].astype(str).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    print("Classes:", list(le.classes_))
    print("Total samples:", len(df))
    print("Num features:", len(feat_cols))
    input_dim = len(feat_cols)

    # make mapping explicit
    synergy_code = le.transform(["synergy"])[0]
    antag_code = le.transform(["antagonism"])[0]


    # ==========================
    # 2) Outer split: CV1 = GroupShuffle by Drug Pair
    # ==========================
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    outer_tr_idx, outer_te_idx = next(gss.split(X_all, y_enc, groups=pairs))

    X_outer_tr = X_all.iloc[outer_tr_idx].reset_index(drop=True)
    y_outer_tr = y_enc[outer_tr_idx]
    X_outer_te = X_all.iloc[outer_te_idx].reset_index(drop=True)
    y_outer_te = y_enc[outer_te_idx]
    pairs_outer_tr = pairs[outer_tr_idx]

    print("\nCV1 outer split (by Drug Pair)")
    print("Train:", len(outer_tr_idx), "Test:", len(outer_te_idx))

    # ==========================
    # 3) Inner nested CV: hyperparam search over configs
    # ==========================
    rng = np.random.default_rng(42)

    # A small, manual hyperparam grid
    hyper_configs = [
        dict(hidden=(512, 256, 128), dropout=0.3, lr=1e-3,  weight_decay=1e-4),
        dict(hidden=(256, 128),      dropout=0.3, lr=1e-3,  weight_decay=3e-4),
        dict(hidden=(512, 256),      dropout=0.4, lr=7e-4,  weight_decay=1e-4),
        dict(hidden=(256, 256, 128), dropout=0.3, lr=1e-3,  weight_decay=1e-4),
    ]

    print("\n--- Nested CV: inner hyperparameter search ---")
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=111)

    def eval_config(cfg):
        print(f"\nConfig: {cfg}")
        fold_scores = []

        for fold_idx, (tr_idx, val_idx) in enumerate(
            inner_cv.split(X_outer_tr, y_outer_tr, groups=pairs_outer_tr),
            start=1,
        ):
            print(f"  Fold {fold_idx}...")
            X_tr = X_outer_tr.iloc[tr_idx].reset_index(drop=True)
            y_tr = y_outer_tr[tr_idx]
            X_val = X_outer_tr.iloc[val_idx].reset_index(drop=True)
            y_val = y_outer_tr[val_idx]

            model = train_dnn(
                X_tr,
                y_tr,
                X_val,
                y_val,
                input_dim=input_dim,
                device=device,
                hidden=cfg["hidden"],
                dropout=cfg["dropout"],
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                batch_size=64,
                max_epochs=80,   # shorter for inner CV
                patience=10,
            )

            probs_val = predict_proba(model, X_val, device=device)

            # synergy-positive AUC
            y_val_bin = (y_val == synergy_code).astype(int)
            auc_val = roc_auc_score(y_val_bin, probs_val)

            y_pred_val = (probs_val >= 0.5).astype(int)
            acc_val = accuracy_score(y_val, y_pred_val)

            fold_scores.append((auc_val, acc_val))
            print(f"    Fold AUC={auc_val:.3f}, Acc={acc_val:.3f}")


        # use mean AUC as primary score
        mean_auc = float(np.mean([s[0] for s in fold_scores]))
        mean_acc = float(np.mean([s[1] for s in fold_scores]))
        print(f"  --> Mean inner AUC={mean_auc:.3f}, Acc={mean_acc:.3f}")
        return mean_auc, mean_acc

    config_scores = []
    for cfg in hyper_configs:
        mean_auc, mean_acc = eval_config(cfg)
        config_scores.append((mean_auc, mean_acc, cfg))

    # pick best by mean AUC, then Acc
    config_scores.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_auc, best_acc, best_cfg = config_scores[0]

    print("\nBest config from nested CV:")
    print("Mean inner AUC:", round(best_auc, 3), "Mean inner Acc:", round(best_acc, 3))
    print("Config:", best_cfg)

    # ==========================
    # 4) Final training on outer-train using best config
    # ==========================
    print("\n--- Final training on outer-train with best config ---\n")

    # Split outer-train into train/val for early stopping (not grouped, just stratified)
    # This is only inside the train side of the outer split.
    from sklearn.model_selection import train_test_split

    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_outer_tr,
        y_outer_tr,
        test_size=0.20,
        random_state=999,
        stratify=y_outer_tr,
    )

    final_model = train_dnn(
        X_tr_final,
        y_tr_final,
        X_val_final,
        y_val_final,
        input_dim=input_dim,
        device=device,
        hidden=best_cfg["hidden"],
        dropout=best_cfg["dropout"],
        lr=best_cfg["lr"],
        weight_decay=best_cfg["weight_decay"],
        batch_size=64,
        max_epochs=120,
        patience=15,
    )

    # ==========================
    # 5) Evaluation on outer-test (unseen pairs)
    # ==========================
    print("\n=== Evaluation on outer held-out test (pair-disjoint) ===\n")

    probs_te = predict_proba(final_model, X_outer_te, device=device)
    y_pred_te = (probs_te >= 0.5).astype(int)

    # synergy-positive AUC
    y_outer_te_bin = (y_outer_te == synergy_code).astype(int)

    # ---- Global metrics ----
    roc_auc_test = roc_auc_score(y_outer_te_bin, probs_te)
    accuracy_test = accuracy_score(y_outer_te, y_pred_te)
    f1_weighted_test = f1_score(y_outer_te, y_pred_te, average="weighted")
    f1_macro_test = f1_score(y_outer_te, y_pred_te, average="macro")

    print("ROC AUC :", round(roc_auc_test, 3))
    print("Acc     :", round(accuracy_test, 3))
    print("F1 (w)  :", round(f1_weighted_test, 3))
    print("F1 (mac):", round(f1_macro_test, 3))

    print("\nConfusion matrix:\n", confusion_matrix(y_outer_te, y_pred_te))
    print(
        "\nClassification report:\n",
        classification_report(y_outer_te, y_pred_te, target_names=le.classes_),
    )

    # ---- Per-class metrics: antagonism (0), synergy (1) ----
    prec, rec, f1s, _ = precision_recall_fscore_support(
        y_outer_te,
        y_pred_te,
        labels=[antag_code, synergy_code],
    )

    precision_antag, precision_syn = prec
    recall_antag, recall_syn = rec
    f1_antag, f1_syn = f1s

    # ---- Log-friendly block ----
    print("\n--- Metrics for Log ---")
    print(f"accuracy_test={accuracy_test:.4f}")
    print(f"f1_macro_test={f1_macro_test:.4f}")
    print(f"f1_weighted_test={f1_weighted_test:.4f}")
    print(f"roc_auc_test={roc_auc_test:.4f}")

    print(f"precision_antag={precision_antag:.4f}")
    print(f"recall_antag={recall_antag:.4f}")
    print(f"f1_antag={f1_antag:.4f}")
    print(f"precision_syn={precision_syn:.4f}")
    print(f"recall_syn={recall_syn:.4f}")
    print(f"f1_syn={f1_syn:.4f}")


    # ==========================
    # 6) Overfitting check on outer-train vs outer-test
    # ==========================
    print("\n=== Overfitting check ===\n")

    probs_tr = predict_proba(final_model, X_outer_tr, device=device)
    y_pred_tr = (probs_tr >= 0.5).astype(int)

    y_outer_tr_bin = (y_outer_tr == synergy_code).astype(int)

    auc_tr = roc_auc_score(y_outer_tr_bin, probs_tr)
    acc_tr = accuracy_score(y_outer_tr, y_pred_tr)
    f1w_tr = f1_score(y_outer_tr, y_pred_tr, average="weighted")
    f1_macro_tr = f1_score(y_outer_tr, y_pred_tr, average="macro")

    print("Train AUC:", round(auc_tr, 3), "| Test AUC:", round(roc_auc_test, 3))
    print("Train Acc:", round(acc_tr, 3), "| Test Acc:", round(accuracy_test, 3))
    print("Train F1w:", round(f1w_tr, 3), "| Test F1w:", round(f1_weighted_test, 3))
    print("Train F1m:", round(f1_macro_tr, 3), "| Test F1m:", round(f1_macro_test, 3))

    print("\n=== EXP10 DONE ===\n")


if __name__ == "__main__":
    main()


