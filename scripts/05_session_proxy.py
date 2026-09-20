"""Session-proxy control (Table 11): predict session (evening = 1) instead of
fatigue, with the same participant-grouped protocol as script 03.
Results are cached per (model, fold) and the script can be resumed.

Usage:
    python scripts/05_session_proxy.py --out-dir results [--models SVM --folds 1 2]
"""
import argparse
import glob
import os

import _common  # noqa: F401
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ALL_FEATURES, GROUP, N_SPLITS, SEED, SUBJECTWISE_CSV
from src.data import assert_no_overlap, load_subjectwise
from src.metrics import binary_metrics, summarise_folds
from src.models_ml import p2_models, score
from src.utils import ensure_dir

MODEL_ORDER = ["XGBoost", "Gradient Boosting", "Random Forest", "SVM", "KNN",
               "Gaussian Naive Bayes", "Logistic Regression"]


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    cache = ensure_dir(os.path.join(out, "cache", "session_proxy"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    X, groups = df[ALL_FEATURES], df[GROUP]
    y = (df.session.str.lower() == "evening").astype(int)
    folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y, groups=groups))
    models = p2_models(include_rf=True)          # RF: 300 trees, as in the original run
    for name in (args.models or MODEL_ORDER):
        for k in (args.folds or range(1, N_SPLITS + 1)):
            path = os.path.join(cache, f"{name.lower().replace(' ', '_')}_fold{k}.csv")
            if os.path.exists(path):
                continue
            tr, te = folds[k - 1]
            assert_no_overlap(groups.iloc[tr], groups.iloc[te], f"fold {k}")
            m = clone(models[name]).fit(X.iloc[tr], y.iloc[tr])
            met = binary_metrics(y.iloc[te], score(m, X.iloc[te]), y_pred=m.predict(X.iloc[te]))
            pd.DataFrame([{"model": name, "fold": k, **met}]).to_csv(path, index=False)
            print(f"{name:22s} fold {k}: session AUC={met['roc_auc']:.3f}")
    files = glob.glob(os.path.join(cache, "*_fold[0-9].csv"))
    if files:
        fold_df = pd.concat(pd.read_csv(f) for f in files).sort_values(["model", "fold"])
        fold_df.to_csv(os.path.join(out, "session_proxy_fold_results.csv"), index=False)
        s = summarise_folds(fold_df, "model").sort_values("roc_auc_mean", ascending=False)
        s.to_csv(os.path.join(out, "session_proxy_summary.csv"), index=False)
        print("Completed folds:", fold_df.groupby("model").size().to_dict())
        print(s[["model", "accuracy_mean", "accuracy_sd", "roc_auc_mean", "roc_auc_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    ap.add_argument("--models", nargs="*", default=None, choices=MODEL_ORDER)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    main(ap.parse_args())
