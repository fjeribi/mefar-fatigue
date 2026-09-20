import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def binary_metrics(y_true, y_score, threshold=0.5, y_pred=None):
    """Accuracy, precision, recall, F1 (positive class = fatigued) and ROC-AUC.

    y_score may be a probability or a decision-function value. If y_pred is
    not supplied it is obtained by thresholding y_score at `threshold`.
    """
    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def summarise_folds(fold_df, group_col,
                    metrics=("accuracy", "precision", "recall", "f1", "roc_auc",
                             "training_time_sec", "inference_time_sec")):
    """Mean and sample SD (ddof=1) across folds for each group."""
    rows = []
    for name, g in fold_df.groupby(group_col, sort=False):
        row = {group_col: name}
        for m in metrics:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_sd"] = g[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test.

    The variance of the mean difference is inflated by (1/n + n_test/n_train)
    to account for overlapping training sets across resamples.
    Returns (mean_diff, t, two-sided p).
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean_diff = diffs.mean()
    var = diffs.var(ddof=1) if n > 1 else 0.0
    denom = np.sqrt((1.0 / n + n_test / n_train) * var) if var > 0 else 1e-12
    t = mean_diff / denom
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if n > 1 else np.nan
    return mean_diff, t, p
