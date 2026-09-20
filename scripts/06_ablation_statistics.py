"""Seven-configuration ablation over five seeds x five GroupKFold folds, with
the corrected resampled t-test and McNemar's test (Table 12).

Outer folds are the same for every seed (GroupKFold is deterministic); the
inner validation split (random_state = seed + fold), weight initialisation
and training stochasticity vary with the seed.

Usage:
    python scripts/06_ablation_statistics.py --out-dir results            # train + test
    python scripts/06_ablation_statistics.py --out-dir results --stats-only
"""
import argparse
import os
import time

import _common  # noqa: F401
import numpy as np
import pandas as pd

from src.config import (ABLATION_SEEDS, ALL_FEATURES, BATCH_SIZE, EPOCHS, GROUP, INNER_VAL_FRACTION,
                        N_SPLITS, NEURO_FEATURES, PHYS_FEATURES, SUBJECTWISE_CSV, TARGET)
from src.data import assert_no_overlap, load_subjectwise
from src.metrics import binary_metrics, corrected_resampled_ttest
from src.utils import banner, ensure_dir, set_seeds

FULL, MATCHED = "7_full_mf_afen", "3_concat_matched"


def train(df, out):
    import tensorflow as tf
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from src.models_dl import ABLATION_CONFIGS, build_ablation_config, compile_model, get_callbacks

    pi = [ALL_FEATURES.index(f) for f in PHYS_FEATURES]
    ni = [ALL_FEATURES.index(f) for f in NEURO_FEATURES]
    rows, preds = [], []
    for seed in ABLATION_SEEDS:
        set_seeds(seed)
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(df, df[TARGET], groups=df[GROUP]), 1):
            banner(f"seed {seed} | fold {k}")
            train_df, test_df = df.iloc[tr], df.iloc[te]
            assert_no_overlap(train_df[GROUP], test_df[GROUP])
            gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=seed + k)
            itr, iva = next(gss.split(train_df, train_df[TARGET], groups=train_df[GROUP]))
            inner_df, val_df = train_df.iloc[itr], train_df.iloc[iva]
            sc = StandardScaler().fit(inner_df[ALL_FEATURES])
            Xs = {n: sc.transform(d[ALL_FEATURES]) for n, d in [("tr", inner_df), ("va", val_df), ("te", test_df)]}
            ys = {n: d[TARGET].values.astype(np.float32) for n, d in [("tr", inner_df), ("va", val_df), ("te", test_df)]}
            views = {n: {"concat": x, "phys": x[:, pi], "neuro": x[:, ni], "dual": [x[:, pi], x[:, ni]]}
                     for n, x in Xs.items()}
            for cfg in ABLATION_CONFIGS:
                tf.keras.backend.clear_session()
                model, mode = build_ablation_config(cfg)
                model = compile_model(model)
                t0 = time.perf_counter()
                model.fit(views["tr"][mode], ys["tr"], validation_data=(views["va"][mode], ys["va"]),
                          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=get_callbacks(), verbose=0)
                t_fit = time.perf_counter() - t0
                prob = model.predict(views["te"][mode], verbose=0).ravel()
                rows.append({"seed": seed, "fold": k, "config": cfg, **binary_metrics(ys["te"], prob),
                             "training_time_sec": t_fit, "n_params": model.count_params()})
                preds.append(pd.DataFrame({"seed": seed, "fold": k, "config": cfg,
                                           "y_true": ys["te"].astype(int), "pred": (prob >= 0.5).astype(int)}))
                print(f"{cfg:30s} auc={rows[-1]['roc_auc']:.3f}")
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out, "ablation_fold_results.csv"), index=False)
    pd.concat(preds).to_csv(os.path.join(out, "ablation_predictions.csv.gz"), index=False)
    return fold_df


def statistics(df, fold_df, out):
    from statsmodels.stats.contingency_tables import mcnemar

    summary = []
    for cfg, g in fold_df.groupby("config"):
        row = {"config": cfg, "n_params": int(g.n_params.iloc[0])}
        for m in ["accuracy", "f1", "roc_auc"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(out, "ablation_summary.csv"), index=False)
    print(summary.round(3).to_string(index=False))

    # As in the original analysis: n_train ~ inner-training size (0.8 * 0.8 * N), n_test ~ 0.2 * N
    n_train, n_test = int(len(df) * 0.8 * 0.8), int(len(df) * 0.2)
    full = fold_df[fold_df.config == FULL].sort_values(["seed", "fold"])
    tests = []
    for cfg in sorted(set(fold_df.config) - {FULL}):
        other = fold_df[fold_df.config == cfg].sort_values(["seed", "fold"])
        for m in ["accuracy", "roc_auc"]:
            diff = full[m].values - other[m].values
            diff = diff[np.isfinite(diff)]          # a fold with one class yields NaN ROC-AUC
            d, t, p = corrected_resampled_ttest(diff, n_train, n_test)
            tests.append({"comparison": f"full vs {cfg}", "metric": m, "mean_diff": d, "t": t, "p": p})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(out, "ablation_corrected_ttests.csv"), index=False)
    banner("Corrected resampled t-tests (full model vs each configuration)")
    print(tests.round(4).to_string(index=False))

    pred_path = os.path.join(out, "ablation_predictions.csv.gz")
    if os.path.exists(pred_path):
        pr = pd.read_csv(pred_path)
        res = []
        for seed, g in pr.groupby("seed"):
            a = g[g.config == FULL].sort_values("fold", kind="stable")
            b = g[g.config == MATCHED].sort_values("fold", kind="stable")
            ca, cb = a.pred.values == a.y_true.values, b.pred.values == b.y_true.values
            n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            r = mcnemar([[0, n01], [n10, 0]], exact=(n01 + n10) < 25)
            res.append({"seed": seed, "full_right_matched_wrong": n01, "full_wrong_matched_right": n10,
                        "statistic": r.statistic, "p": r.pvalue})
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(out, "ablation_mcnemar.csv"), index=False)
        banner("McNemar: full model vs matched control (per seed, pooled over folds)")
        print(res.to_string(index=False))
        print("NOTE: observations within a session are highly correlated, so McNemar p-values "
              "computed on 1-Hz samples are anti-conservative; interpret with caution.")


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    path = os.path.join(out, "ablation_fold_results.csv")
    fold_df = pd.read_csv(path) if args.stats_only else train(df, out)
    statistics(df, fold_df, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    ap.add_argument("--stats-only", action="store_true", help="Recompute statistics from saved fold results")
    main(ap.parse_args())
