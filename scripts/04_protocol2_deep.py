"""Protocol 2 (P2), deep-learning models (Table 8) and computational cost (Table 14).

Five-fold GroupKFold over participants. Inside each training partition,
20% of the training participants form a grouped validation set
(GroupShuffleSplit, random_state = 42 + fold) for early stopping.
LSTM windows are built chronologically within each session.

Usage:
    python scripts/04_protocol2_deep.py --out-dir results
"""
import argparse
import os
import time

import _common  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import (ALL_FEATURES, BATCH_SIZE, EPOCHS, GROUP, INNER_VAL_FRACTION, N_SPLITS,
                        NEURO_FEATURES, PHYS_FEATURES, SEED, SEQ_LEN, SUBJECTWISE_CSV, TARGET)
from src.data import assert_no_overlap, load_subjectwise
from src.metrics import binary_metrics, summarise_folds
from src.models_dl import build_ann, build_cnn, build_lstm, build_mf_afen, compile_model, get_callbacks
from src.sequences import session_windows
from src.utils import banner, ensure_dir, set_seeds

PI = [ALL_FEATURES.index(f) for f in PHYS_FEATURES]
NI = [ALL_FEATURES.index(f) for f in NEURO_FEATURES]


def train_eval(model, xtr, ytr, xva, yva, xte):
    model = compile_model(model)
    t0 = time.perf_counter()
    model.fit(xtr, ytr, validation_data=(xva, yva), epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=get_callbacks(), verbose=0)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); prob = model.predict(xte, verbose=0).ravel(); t_inf = time.perf_counter() - t0
    return prob, t_fit, t_inf, model.count_params()


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    set_seeds(SEED)
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    rows, oof = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(df, df[TARGET], groups=df[GROUP]), 1):
        banner(f"Fold {k}")
        train_df, test_df = df.iloc[tr], df.iloc[te]
        assert_no_overlap(train_df[GROUP], test_df[GROUP], f"fold {k}")
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=SEED + k)
        itr, iva = next(gss.split(train_df, train_df[TARGET], groups=train_df[GROUP]))
        inner_df, val_df = train_df.iloc[itr], train_df.iloc[iva]
        assert_no_overlap(inner_df[GROUP], val_df[GROUP], f"fold {k} (validation)")

        sc = StandardScaler().fit(inner_df[ALL_FEATURES])
        Xtr, Xva, Xte = (sc.transform(d[ALL_FEATURES]).astype(np.float32) for d in (inner_df, val_df, test_df))
        ytr, yva, yte = (d[TARGET].values.astype(np.float32) for d in (inner_df, val_df, test_df))

        # Same order as the original experiment: ANN, 1D-CNN, MF-AFEN, then LSTM
        runs = [
            ("ANN", lambda: build_ann(len(ALL_FEATURES)), Xtr, Xva, Xte),
            ("1D_CNN", lambda: build_cnn(len(ALL_FEATURES)), Xtr[..., None], Xva[..., None], Xte[..., None]),
            ("MF-AFEN", build_mf_afen, [Xtr[:, PI], Xtr[:, NI]], [Xva[:, PI], Xva[:, NI]],
             [Xte[:, PI], Xte[:, NI]]),
        ]
        subj_te, sess_te = test_df[GROUP].values, test_df.session_id.values
        for name, builder, a, b, c in runs:
            tf.keras.backend.clear_session()
            prob, t_fit, t_inf, n_par = train_eval(builder(), a, ytr, b, yva, c)
            rows.append({"model": name, "fold": k, **binary_metrics(yte, prob), "training_time_sec": t_fit,
                         "inference_time_sec": t_inf, "n_params": n_par})
            oof.append(pd.DataFrame({"model": name, "fold": k, "subject_id": subj_te,
                                     "session_id": sess_te, "y_true": yte, "score": prob}))
            print(f"{name:8s} acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['roc_auc']:.3f}")

        # LSTM on session-bounded chronological windows
        def scaled(d):
            d = d.copy(); d[ALL_FEATURES] = sc.transform(d[ALL_FEATURES]); return d
        (Xs_tr, ys_tr, _, _), (Xs_va, ys_va, _, _), (Xs_te, ys_te, s_te, ss_te) = (
            session_windows(scaled(d), ALL_FEATURES, TARGET, SEQ_LEN) for d in (inner_df, val_df, test_df))
        tf.keras.backend.clear_session()
        prob, t_fit, t_inf, n_par = train_eval(build_lstm(len(ALL_FEATURES)), Xs_tr, ys_tr, Xs_va, ys_va, Xs_te)
        rows.append({"model": "LSTM", "fold": k, **binary_metrics(ys_te, prob), "training_time_sec": t_fit,
                     "inference_time_sec": t_inf, "n_params": n_par})
        oof.append(pd.DataFrame({"model": "LSTM", "fold": k, "subject_id": s_te, "session_id": ss_te,
                                 "y_true": ys_te, "score": prob}))
        print(f"LSTM     acc={rows[-1]['accuracy']:.3f} auc={rows[-1]['roc_auc']:.3f}")

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out, "p2_deep_fold_results.csv"), index=False)
    summary = summarise_folds(fold_df, "model")
    summary["n_params"] = fold_df.groupby("model", sort=False)["n_params"].first().values
    summary.to_csv(os.path.join(out, "p2_deep_summary.csv"), index=False)
    pd.concat(oof).to_csv(os.path.join(out, "p2_deep_oof_scores.csv.gz"), index=False)
    print("\n", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    main(ap.parse_args())
