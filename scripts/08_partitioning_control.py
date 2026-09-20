"""Partitioning control (Table 10): stratified random 80:20 sample-level split
applied to the reconstructed (P2) table, i.e. P1-style partitioning on P2 data.

Usage:
    python scripts/08_partitioning_control.py --out-dir results
"""
import argparse
import os

import _common  # noqa: F401
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.config import ALL_FEATURES, SEED, SUBJECTWISE_CSV, TARGET
from src.data import load_subjectwise
from src.metrics import binary_metrics
from src.utils import ensure_dir


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))
    X, y = df[ALL_FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = {
        "XGBoost (P1 settings)": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss",
                                               random_state=SEED, n_jobs=-1),
        "Random Forest (300 trees)": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    rows = []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        rows.append({"model": name, "split": "random 80:20 (sample-level)",
                     **binary_metrics(yte, m.predict_proba(Xte)[:, 1])})
        print(f"{name:28s} acc={rows[-1]['accuracy']:.4f} auc={rows[-1]['roc_auc']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "partitioning_control.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    main(ap.parse_args())
