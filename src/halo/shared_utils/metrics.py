import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
    roc_auc_score, root_mean_squared_error, r2_score, accuracy_score, f1_score,
    mean_absolute_error, precision_recall_fscore_support)


def classification_metrics(
    y_test,
    y_pred,
    y_score=None,
    class_names=None,
    positive_class_name: str = "synergy",
):
    """
    Prints classification report + confusion matrix + ROC AUC +
    additional experiment-log-friendly metrics:
        - accuracy_test
        - f1_macro_test
        - f1_weighted_test
        - roc_auc_test
        - per-class: precision, recall, f1

    For binary tasks, if class_names contains `positive_class_name`
    (default: "synergy"), AUC is computed treating that class as positive.
    """

    # Make class_names indexable if provided
    if class_names is not None:
        class_names = list(class_names)

    print("=" * 75)
    print("Classification report:\n")

    if class_names is not None:
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=class_names,
                digits=3,
            )
        )
    else:
        print(classification_report(y_test, y_pred, digits=3))

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------
    print("-" * 75)
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # Basic global metrics
    # ---------------------------------------------------------
    print("-" * 75)
    print("Global Metrics:\n")

    accuracy_test = accuracy_score(y_test, y_pred)
    f1_macro_test = f1_score(y_test, y_pred, average="macro")
    f1_weighted_test = f1_score(y_test, y_pred, average="weighted")

    print(f"accuracy_test      = {accuracy_test:.4f}")
    print(f"f1_macro_test      = {f1_macro_test:.4f}")
    print(f"f1_weighted_test   = {f1_weighted_test:.4f}")

    # ---------------------------------------------------------
    # ROC AUC
    # ---------------------------------------------------------
    print("-" * 75)
    print("AUC Scores:\n")

    if y_score is None:
        print("No y_score provided; skipping AUC.")
        roc_auc_test = None
    else:
        y_score = np.asarray(y_score)
        unique_labels = np.unique(y_test)
        n_classes = len(unique_labels)

        # ----------------- Binary case -----------------
        if n_classes == 2:
            # Decide which label is "positive"
            if class_names is not None and positive_class_name in class_names:
                pos_cls_idx = class_names.index(positive_class_name)
            else:
                # fallback: assume label "1" is positive
                pos_cls_idx = 1

            # Extract positive-class scores
            if y_score.ndim == 1:
                # assume already P(positive_class)
                y_pos = y_score
            elif y_score.ndim == 2:
                # columns correspond to label indices
                y_pos = y_score[:, pos_cls_idx]
            else:
                raise ValueError(
                    "For binary, y_score must be shape (n,) or (n, n_classes)."
                )

            # Build binary ground truth: 1 = positive class, 0 = others
            y_bin = (y_test == pos_cls_idx).astype(int)

            auc_pos = roc_auc_score(y_bin, y_pos)
            auc_neg = roc_auc_score(1 - y_bin, 1 - y_pos)
            roc_auc_test = float(np.mean([auc_pos, auc_neg]))

            print(f"roc_auc_test        = {roc_auc_test:.4f}")
            print(f"AUC (neg class)     = {auc_neg:.4f}")
            print(f"AUC (pos class)     = {auc_pos:.4f}")

        # ----------------- Multiclass case -----------------
        else:
            if y_score.ndim != 2 or y_score.shape[1] != n_classes:
                raise ValueError(
                    "For multiclass, y_score must be shape (n_samples, n_classes)."
                )

            auc_macro = roc_auc_score(
                y_test,
                y_score,
                multi_class="ovr",
                average="macro",
            )
            roc_auc_test = auc_macro
            print(f"roc_auc_test        = {roc_auc_test:.4f}")

            for c in range(n_classes):
                auc_c = roc_auc_score(
                    (y_test == c).astype(int),
                    y_score[:, c],
                )
                name = (
                    class_names[c] if class_names is not None else f"class_{c}"
                )
                print(f"AUC[{name}]         = {auc_c:.4f}")

    # ---------------------------------------------------------
    # Per-class metrics (precision, recall, f1)
    # ---------------------------------------------------------
    print("-" * 75)
    print("Per-class Metrics:\n")

    prec, rec, f1s, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=np.unique(y_test),
    )

    for idx, cls in enumerate(np.unique(y_test)):
        name = class_names[cls] if class_names is not None else f"class_{cls}"
        print(
            f"{name}: precision={prec[idx]:.4f}, "
            f"recall={rec[idx]:.4f}, f1={f1s[idx]:.4f}"
        )

    print("=" * 75)



def regression_metrics(y_test, y_pred):
    """
    Print standard regression evaluation metrics.
    """
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = rmse ** 2
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=" * 75)
    print("Regression Results:")
    print("-" * 75)
    print(f"MSE  (Mean Squared Error)     : {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAE  (Mean Absolute Error)    : {mae:.4f}")
    print(f"R²   (R-squared)              : {r2:.4f}")
    print("=" * 75)


def overfitting_report(model, X_train, y_train, X_test, y_test, 
                       task: str,
                       average: str ='macro' # or "weighted"
                        ):
    """
    Prints a simple overfitting report depending on the task type. 

    Args:
        model: fitted model
        X_train, y_train, X_test, y_test: array-like data splits
        task (str): 'classification' or 'regression'
        average (str): averaging mode for f1 score (classification only)

    Returns:
        train vs. test accuracy, and f1 for classification
    """

    if task == 'classification':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, average=average)
        test_f1 = f1_score(y_test, y_test_pred, average=average)

        print("=" * 50)
        print("Overfitting Report")
        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Accuracy Gap   : {train_acc - test_acc:+.4f}")
        print("-" * 30)
        print(f"Train F1       : {train_f1:.4f}")
        print(f"Test F1        : {test_f1:.4f}")
        print(f"F1 Gap         : {train_f1 - test_f1:+.4f}")
        print("=" * 50)

    elif task == 'regression':
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print("=" * 50)
        print("Overfitting Report (Regression)")
        print(f"Train RMSE     : {train_rmse:.4f}")
        print(f"Test RMSE      : {test_rmse:.4f}")
        print(f"RMSE Gap       : {train_rmse - test_rmse:+.4f}")
        print("-" * 30)
        print(f"Train R²       : {train_r2:.4f}")
        print(f"Test R²        : {test_r2:.4f}")
        print(f"R² Gap         : {train_r2 - test_r2:+.4f}")
        print("=" * 50)

    else:
        print("Warning: Task must be 'classification' or 'regression'.")





