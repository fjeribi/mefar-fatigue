"""Windowed-feature baseline (Table 15): are instantaneous 1-Hz samples the
bottleneck for generalisation?

Builds physiologically meaningful features over sliding windows directly from
the raw files, then evaluates them with participant-grouped validation.

Features per window (default 60 s, step 30 s), all from native-rate signals:
  HR     mean, SD, slope
  BVP    pulse-derived rate, SDNN, RMSSD (peak detection at 64 Hz), amplitude SD
  EDA    tonic mean, SD, slope, skin-conductance-response count per minute
  TEMP   mean, slope
  ACC    magnitude mean and SD (motion)
  EEG    mean log band power (8 bands), theta/alpha, (theta+alpha)/beta,
         Attention and Meditation means
Each window inherits its session label.

Two feature variants are evaluated:
  raw          features as computed
  subject_norm features z-scored within each participant (label-free, uses
               only that participant's own windows; a standard baseline
               correction for between-person differences)

Protocols: five-fold StratifiedGroupKFold (seed 42), leave-one-subject-out
(LOSO), and a random 80:20 window-level split as a leakage reference.
For LOSO, pooled out-of-fold scores give a window-level and a session-level
ROC-AUC (mean window score per session, 46 sessions), with a participant
bootstrap 95% CI and a session-level label-permutation p-value.

Usage:
    python scripts/11_windowed_features.py --raw-root /path/to/MEFAR --out-dir results
"""
import argparse
import os
import warnings

import _common  # noqa: F401
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import N_SPLITS, NEURO_FEATURES, SEED
from src.data import assert_no_overlap
from src.metrics import binary_metrics
from src.raw_io import iter_sessions, load_labels, read_e4, read_eeg
from src.utils import banner, ensure_dir

warnings.filterwarnings("ignore")
BANDS = [b for b in NEURO_FEATURES if b not in ("Attention", "Meditation")]


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _slope(t, v):
    return np.polyfit(t, v, 1)[0] if len(v) > 2 and np.ptp(t) > 0 else np.nan


def _segment(values, rate, t0, t1):
    i0, i1 = int(t0 * rate), int(t1 * rate)
    seg = np.asarray(values[i0:i1], float)
    return seg, np.arange(len(seg)) / rate


def bvp_features(seg, rate):
    """Pulse peaks at 64 Hz -> inter-beat intervals -> rate, SDNN, RMSSD."""
    out = {"bvp_amp_sd": np.std(seg)}
    peaks, _ = find_peaks(seg, distance=int(0.33 * rate), prominence=np.std(seg) * 0.5)
    ibi = np.diff(peaks) / rate
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]                       # 40-180 bpm
    if len(ibi) >= 5:
        out.update(bvp_rate=60 / ibi.mean(), bvp_sdnn=ibi.std(ddof=1) * 1000,
                   bvp_rmssd=np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000)
    else:
        out.update(bvp_rate=np.nan, bvp_sdnn=np.nan, bvp_rmssd=np.nan)
    return out


def eda_features(seg, rate, win):
    t = np.arange(len(seg)) / rate
    peaks, _ = find_peaks(seg, prominence=0.01, distance=max(int(rate), 1))   # SCR >= 0.01 uS
    return {"eda_mean": seg.mean(), "eda_sd": seg.std(), "eda_slope": _slope(t, seg),
            "eda_scr_per_min": len(peaks) / (win / 60)}


def session_windows(folder, win, step):
    sig = {n: read_e4(os.path.join(folder, f"{n}.csv")) for n in ["BVP", "EDA", "TEMP", "HR"]}
    _, acc_rate, acc = read_e4(os.path.join(folder, "ACC.csv"), n_cols=3)
    eeg = read_eeg(os.path.join(folder, "EEG.csv"))
    duration = min([len(v) / r for (_, r, v) in sig.values()] + [len(acc) / acc_rate, eeg.time.max()])
    acc_mag = np.sqrt((acc.values.astype(float) ** 2).sum(axis=1))

    rows = []
    for t0 in np.arange(0, duration - win + 1e-9, step):
        t1 = t0 + win
        f = {"t_start": t0}
        _, r, v = sig["HR"]; seg, t = _segment(v.iloc[:, 0].values, r, t0, t1)
        f.update(hr_mean=seg.mean(), hr_sd=seg.std(), hr_slope=_slope(t, seg))
        _, r, v = sig["BVP"]; seg, _ = _segment(v.iloc[:, 0].values, r, t0, t1)
        f.update(bvp_features(seg, r))
        _, r, v = sig["EDA"]; seg, _ = _segment(v.iloc[:, 0].values, r, t0, t1)
        f.update(eda_features(seg, r, win))
        _, r, v = sig["TEMP"]; seg, t = _segment(v.iloc[:, 0].values, r, t0, t1)
        f.update(temp_mean=seg.mean(), temp_slope=_slope(t, seg))
        seg, _ = _segment(acc_mag, acc_rate, t0, t1)
        f.update(acc_mag_mean=seg.mean(), acc_mag_sd=seg.std())
        e = eeg[(eeg.time >= t0) & (eeg.time < t1)]
        if len(e) >= 5:
            for b in BANDS:
                f[f"log_{b}"] = np.log1p(e[b].clip(lower=0)).mean()
            alpha = e.Alpha1 + e.Alpha2; beta = e.Beta1 + e.Beta2
            f["theta_alpha"] = np.log1p(e.Theta / alpha.replace(0, np.nan)).mean()
            f["theta_alpha_beta"] = np.log1p((e.Theta + alpha) / beta.replace(0, np.nan)).mean()
            f["attention"] = e.Attention.mean(); f["meditation"] = e.Meditation.mean()
        rows.append(f)
    return pd.DataFrame(rows)


def build_table(raw_root, win, step):
    labels, _ = load_labels(os.path.join(raw_root, "general_info.xlsx"))
    frames = []
    for sid, session, folder in iter_sessions(raw_root):
        if (sid, session) not in labels:
            continue
        w = session_windows(folder, win, step)
        w["subject_id"], w["session"], w["session_id"] = sid, session, f"{sid}_{session}"
        w["class"] = labels[(sid, session)]
        frames.append(w)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def models():
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }


def impute_train(Xtr, Xte):
    med = Xtr.median()
    return Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)


def run_cv(model, X, y, groups, splitter, name):
    oof = np.full(len(y), np.nan); per_fold = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups=groups), 1):
        assert_no_overlap(groups.iloc[tr], groups.iloc[te], f"{name} fold {k}")
        Xtr, Xte = impute_train(X.iloc[tr], X.iloc[te])
        m = clone(model).fit(Xtr, y.iloc[tr])
        s = m.predict_proba(Xte)[:, 1]
        oof[te] = s
        if y.iloc[te].nunique() == 2:
            per_fold.append(binary_metrics(y.iloc[te], s))
    return oof, pd.DataFrame(per_fold)


def session_auc(df, scores):
    s = pd.DataFrame({"session_id": df.session_id, "y": df["class"], "s": scores}).groupby("session_id").mean()
    return roc_auc_score(s.y, s.s), s


def uncertainty(df, scores, rng, n_boot, n_perm):
    obs = roc_auc_score(df["class"], scores)
    subjects = df.subject_id.unique()
    idx_by = {s: np.flatnonzero(df.subject_id.values == s) for s in subjects}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[s] for s in rng.choice(subjects, len(subjects))])
        yy = df["class"].values[idx]
        if len(np.unique(yy)) == 2:
            boots.append(roc_auc_score(yy, scores[idx]))
    sess = df.groupby("session_id")["class"].first()
    perm = [roc_auc_score(df.session_id.map(pd.Series(rng.permutation(sess.values), index=sess.index)), scores)
            for _ in range(n_perm)]
    return obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5), (1 + np.sum(np.array(perm) >= obs)) / (1 + n_perm)


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "windowed"))
    table_path = os.path.join(out, f"windowed_features_{args.window}s.csv")
    if os.path.exists(table_path) and not args.rebuild:
        df = pd.read_csv(table_path)
    else:
        banner("Extracting windowed features")
        df = build_table(args.raw_root, args.window, args.step)
        df.to_csv(table_path, index=False)
    feats = [c for c in df.columns if c not in ("t_start", "subject_id", "session", "session_id", "class")]
    print(f"{len(df)} windows | {df.subject_id.nunique()} participants | {df.session_id.nunique()} sessions "
          f"| {len(feats)} features")

    variants = {"raw": df[feats]}
    zs = df.groupby("subject_id")[feats].transform(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-9))
    variants["subject_norm"] = zs
    y, groups = df["class"].astype(int), df.subject_id.astype(str)
    rng = np.random.default_rng(SEED)
    rows = []
    for vname, X in variants.items():
        for mname, model in models().items():
            banner(f"{vname} | {mname}")
            _, f5 = run_cv(model, X, y, groups, StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED), "SGKF")
            oof, _ = run_cv(model, X, y, groups, LeaveOneGroupOut(), "LOSO")
            auc, lo, hi, p = uncertainty(df, oof, rng, args.n_boot, args.n_perm)
            s_auc, _ = session_auc(df, oof)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
            Xtr, Xte = impute_train(Xtr, Xte)
            rand_auc = roc_auc_score(yte, clone(model).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
            rows.append({"features": vname, "model": mname,
                         "sgkf5_auc_mean": f5.roc_auc.mean(), "sgkf5_auc_sd": f5.roc_auc.std(ddof=1),
                         "loso_window_auc": auc, "loso_ci_low": lo, "loso_ci_high": hi, "loso_perm_p": p,
                         "loso_session_auc": s_auc, "random_split_auc": rand_auc})
            r = rows[-1]
            print(f"5-fold AUC {r['sgkf5_auc_mean']:.3f}±{r['sgkf5_auc_sd']:.3f} | LOSO {auc:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] p={p:.3f} | session-level {s_auc:.3f} | random split {rand_auc:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, f"windowed_results_{args.window}s.csv"), index=False)
    banner("Summary")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--window", type=int, default=60, help="window length in seconds")
    ap.add_argument("--step", type=int, default=30, help="window step in seconds")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true", help="recompute the feature table")
    main(ap.parse_args())
