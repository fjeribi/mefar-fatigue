"""Self-gated vs cross-modal-gated MF-AFEN with test-time modality masking
(Table 13). Masking sets one modality's standardised inputs to zero, i.e. to
the training mean, without retraining. Single seed (42), as in the paper.

Usage:
    python scripts/07_gating_and_masking.py --out-dir results
"""
import argparse
import os

import _common  # noqa: F401
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import (ALL_FEATURES, BATCH_SIZE, EPOCHS, GROUP, INNER_VAL_FRACTION, N_SPLITS,
                        NEURO_FEATURES, PHYS_FEATURES, SEED, SUBJECTWISE_CSV, TARGET)
from src.data import assert_no_overlap, load_subjectwise
from src.metrics import binary_metrics
from src.models_dl import build_gating_variant, compile_model, get_callbacks
from src.utils import ensure_dir, set_seeds


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    pi = [ALL_FEATURES.index(f) for f in PHYS_FEATURES]
    ni = [ALL_FEATURES.index(f) for f in NEURO_FEATURES]
    set_seeds(SEED)
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(df, df[TARGET], groups=df[GROUP]), 1):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        assert_no_overlap(train_df[GROUP], test_df[GROUP])
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=SEED + k)
        itr, iva = next(gss.split(train_df, train_df[TARGET], groups=train_df[GROUP]))
        inner_df, val_df = train_df.iloc[itr], train_df.iloc[iva]
        sc = StandardScaler().fit(inner_df[ALL_FEATURES])
        (Xtr, Xva, Xte) = (sc.transform(d[ALL_FEATURES]) for d in (inner_df, val_df, test_df))
        ytr, yva, yte = (d[TARGET].values.astype(np.float32) for d in (inner_df, val_df, test_df))
        for variant in ["self_gate_mf_afen", "cross_modal_gate_mf_afen"]:
            tf.keras.backend.clear_session()
            m = compile_model(build_gating_variant(variant))
            m.fit([Xtr[:, pi], Xtr[:, ni]], ytr, validation_data=([Xva[:, pi], Xva[:, ni]], yva),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=get_callbacks(), verbose=0)
            p, n = Xte[:, pi], Xte[:, ni]
            for mask, inputs in [("none", [p, n]), ("phys_masked", [np.zeros_like(p), n]),
                                 ("neuro_masked", [p, np.zeros_like(n)])]:
                prob = m.predict(inputs, verbose=0).ravel()
                rows.append({"seed": SEED, "fold": k, "variant": variant, "masking": mask,
                             **binary_metrics(yte, prob)})
            print(f"fold {k} {variant}: " + ", ".join(
                f"{r['masking']}={r['roc_auc']:.3f}" for r in rows[-3:]))
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(os.path.join(out, "gating_masking_fold_results.csv"), index=False)
    s = fold_df.groupby(["variant", "masking"]).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std")).reset_index()
    s.to_csv(os.path.join(out, "gating_masking_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    main(ap.parse_args())
