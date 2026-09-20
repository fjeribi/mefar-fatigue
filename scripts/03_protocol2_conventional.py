"""Protocol 2 (P2), conventional classifiers and Random Forest (Table 8).

Five-fold StratifiedGroupKFold (shuffled, seed 42) over participants.
Random Forest: configuration selected by ROC-AUC on fold 1 from three
candidates, then evaluated on all five folds (as in the paper; fold 1 is
therefore included in its reported mean).

Results are cached per (model, fold) under <out-dir>/p2/cache/, so the script
can be interrupted and resumed, or run piecewise with --models / --folds.

Usage:
    python scripts/03_protocol2_conventional.py --out-dir results
    python scripts/03_protocol2_conventional.py --out-dir results --models SVM --folds 1 2
"""
import argparse
import glob
import os
import time

import _common  # noqa: F401
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import ALL_FEATURES, GROUP, N_SPLITS, SEED, SUBJECTWISE_CSV, TARGET
from src.data import assert_no_overlap, load_subjectwise
from src.metrics import binary_metrics, summarise_folds
from src.models_ml import p2_models, rf_candidates, score
from src.utils import ensure_dir

MODEL_ORDER = ["XGBoost", "Gradient Boosting", "SVM", "KNN", "Gaussian Naive Bayes",
               "Logistic Regression", "Random Forest"]


def slug(name):
    return name.lower().replace(" ", "_")


def select_rf(X, y, folds, cache):
    path = os.path.join(cache, "rf_selection.csv")
    if os.path.exists(path):
        row = pd.read_csv(path).iloc[0]
        return eval(row.selected_rf), row.fold1_auc
    tr, va = folds[0]
    best, best_auc = None, -1
    for params in rf_candidates():
        m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params).fit(X.iloc[tr], y.iloc[tr])
        auc = roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1])
        print(f"RF candidate {params}: fold-1 AUC={auc:.4f}")
        if auc > best_auc:
            best, best_auc = params, auc
    pd.DataFrame([{"selected_rf": repr(best), "fold1_auc": best_auc}]).to_csv(path, index=False)
    return best, best_auc


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    cache = ensure_dir(os.path.join(out, "cache", "conventional"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    X, y, groups = df[ALL_FEATURES], df[TARGET], df[GROUP]
    folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y, groups=groups))

    models = p2_models()
    wanted = args.models or MODEL_ORDER
    if "Random Forest" in wanted:
        best, _ = select_rf(X, y, folds, cache)
        print(f"Selected RF configuration: {best}")
        models["Random Forest"] = RandomForestClassifier(random_state=SEED, n_jobs=-1, **best)

    for name in wanted:
        for k in (args.folds or range(1, N_SPLITS + 1)):
            f_res = os.path.join(cache, f"{slug(name)}_fold{k}.csv")
            if os.path.exists(f_res):
                continue
            tr, te = folds[k - 1]
            assert_no_overlap(groups.iloc[tr], groups.iloc[te], f"fold {k}")
            m = clone(models[name])
            t0 = time.perf_counter(); m.fit(X.iloc[tr], y.iloc[tr]); t_fit = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = m.predict(X.iloc[te]); t_inf = time.perf_counter() - t0
            s = score(m, X.iloc[te])
            met = binary_metrics(y.iloc[te], s, y_pred=pred)
            pd.DataFrame([{"model": name, "fold": k, **met, "training_time_sec": t_fit,
                           "inference_time_sec": t_inf, "test_subjects": groups.iloc[te].nunique()}]).to_csv(f_res, index=False)
            pd.DataFrame({"model": name, "fold": k, "subject_id": groups.iloc[te].values,
                          "session_id": df.session_id.iloc[te].values, "y_true": y.iloc[te].values,
                          "score": s}).to_csv(f_res.replace(".csv", "_oof.csv.gz"), index=False)
            print(f"{name:22s} fold {k}: acc={met['accuracy']:.3f} auc={met['roc_auc']:.3f} ({t_fit:.0f}s)")

    files = sorted(glob.glob(os.path.join(cache, "*_fold[0-9].csv")))
    if files:
        fold_df = pd.concat(pd.read_csv(f) for f in files)
        fold_df["model"] = pd.Categorical(fold_df.model, MODEL_ORDER, ordered=True)
        fold_df = fold_df.sort_values(["model", "fold"])
        fold_df.to_csv(os.path.join(out, "p2_conventional_fold_results.csv"), index=False)
        summary = summarise_folds(fold_df.assign(model=fold_df.model.astype(str)), "model")
        summary.to_csv(os.path.join(out, "p2_conventional_summary.csv"), index=False)
        oofs = sorted(glob.glob(os.path.join(cache, "*_oof.csv.gz")))
        pd.concat(pd.read_csv(f) for f in oofs).to_csv(os.path.join(out, "p2_conventional_oof_scores.csv.gz"), index=False)
        done = fold_df.groupby("model", observed=True).size()
        print("\nCompleted folds per model:", done.to_dict())
        print(summary[["model", "accuracy_mean", "accuracy_sd", "roc_auc_mean", "roc_auc_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None, help="Reconstructed CSV (default: <out-dir>/" + SUBJECTWISE_CSV + ")")
    ap.add_argument("--models", nargs="*", default=None, choices=MODEL_ORDER)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    main(ap.parse_args())
