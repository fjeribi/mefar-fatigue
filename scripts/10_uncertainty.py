"""Uncertainty for the P2 results (Table 9): is performance on unseen participants
above chance?

Uses the out-of-fold scores saved by scripts 03 and 04 (no retraining).
For each model:
  * participant-level bootstrap 95% CI of the mean per-fold ROC-AUC is not
    well defined, so we report the pooled out-of-fold ROC-AUC with a
    participant-level (cluster) bootstrap CI;
  * a session-level label permutation test: session labels are shuffled
    across the 46 sessions (every observation keeps its session's label),
    and the pooled ROC-AUC is recomputed. p = share of permutations with
    AUC >= observed.
Pooled out-of-fold AUC differs slightly from the mean of per-fold AUCs.
Decision-function scores (SVM) are rank-normalised within each fold first.

Usage:
    python scripts/10_uncertainty.py --out-dir results [--n-boot 2000 --n-perm 2000]
"""
import argparse
import os

import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import SEED
from src.utils import ensure_dir


def pooled_auc(d):
    return roc_auc_score(d.y_true, d.rank_score)


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    frames = [pd.read_csv(os.path.join(out, f)) for f in
              ["p2_conventional_oof_scores.csv.gz", "p2_deep_oof_scores.csv.gz"]
              if os.path.exists(os.path.join(out, f))]
    oof = pd.concat(frames, ignore_index=True)
    oof["rank_score"] = oof.groupby(["model", "fold"]).score.rank(pct=True)
    rng = np.random.default_rng(SEED)
    if args.models:
        oof = oof[oof.model.isin(args.models)]
    out_csv = os.path.join(out, "p2_uncertainty.csv")
    prev = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["model"])
    rows = []
    for model, d in oof.groupby("model"):
        d = d.reset_index(drop=True)
        obs = pooled_auc(d)
        subjects = d.subject_id.unique()
        by_subj = {s: np.flatnonzero(d.subject_id.values == s) for s in subjects}
        boots = []
        for _ in range(args.n_boot):
            idx = np.concatenate([by_subj[s] for s in rng.choice(subjects, len(subjects), replace=True)])
            b = d.iloc[idx]
            if b.y_true.nunique() == 2:
                boots.append(pooled_auc(b))
        sess = d.groupby("session_id").y_true.first()
        perm = []
        for _ in range(args.n_perm):
            mapping = pd.Series(rng.permutation(sess.values), index=sess.index)
            perm.append(roc_auc_score(d.session_id.map(mapping), d.rank_score))
        p = (1 + np.sum(np.asarray(perm) >= obs)) / (1 + len(perm))
        rows.append({"model": model, "pooled_oof_auc": obs, "ci_low": np.percentile(boots, 2.5),
                     "ci_high": np.percentile(boots, 97.5), "perm_p_session_level": p})
        print(f"{model:22s} AUC={obs:.3f} 95% CI [{rows[-1]['ci_low']:.3f}, {rows[-1]['ci_high']:.3f}] "
              f"permutation p={p:.3f}")
    new = pd.DataFrame(rows)
    merged = pd.concat([prev[~prev.model.isin(new.model)], new], ignore_index=True)
    merged.to_csv(out_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--models", nargs="*", default=None, help="subset of models (results are merged)")
    main(ap.parse_args())
