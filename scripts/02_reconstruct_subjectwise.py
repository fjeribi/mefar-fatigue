"""Rebuild a participant-level 1-Hz feature table from the raw MEFAR files
(Section 4.6.1). Output: <out-dir>/MEFAR_subjectwise_1Hz_FINAL.csv

Expected raw layout (as distributed on Mendeley Data):
    <raw-root>/general_info.xlsx            (sheet "Subject List")
    <raw-root>/subject_<k>/1.morning/{BVP,EDA,TEMP,ACC,HR,EEG}.csv
    <raw-root>/subject_<k>/2.evening/{BVP,EDA,TEMP,ACC,HR,EEG}.csv

Empatica E4 CSV files start with two header rows (start timestamp, sample
rate) followed by the samples.

--legacy-e4-parsing
    Reproduces an earlier development version of this script that treated
    the two header rows of BVP/EDA/TEMP/HR as samples, so the second-0 row of
    every session contained the Unix start timestamp. It is kept only for
    transparency and is NOT used for any result in the paper.

Usage:
    python scripts/02_reconstruct_subjectwise.py --raw-root /path/to/MEFAR --out-dir results
"""
import argparse
import os
import warnings

import _common  # noqa: F401
import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, NEURO_FEATURES, SUBJECTWISE_CSV
from src.raw_io import iter_sessions, load_labels, read_e4, read_eeg
from src.utils import banner, ensure_dir

warnings.filterwarnings("ignore")


def to_1hz(values, rate, duration, name):
    """Linearly interpolate a signal (time = index / rate) onto a 1-Hz grid [0, duration)."""
    t = np.arange(len(values)) / rate
    keep = t <= duration
    s = pd.Series(np.asarray(values, float)[keep], index=t[keep])
    s = s[~s.index.duplicated(keep="first")]
    grid = np.arange(0, duration, 1.0)
    s = s.reindex(np.union1d(s.index.values, grid)).interpolate(method="index", limit_direction="both")
    return s.reindex(grid).rename(name)


def legacy_values(path):
    """Replicates the original parsing: every row of column 0, header rows included."""
    raw = pd.read_csv(path, header=None).dropna(how="all").reset_index(drop=True)
    return raw.iloc[:, 0].astype(float).values


def eeg_to_1hz(path, duration):
    eeg = read_eeg(path)
    grid = np.arange(0, int(np.floor(duration)) + 1, 1, dtype=float)
    out = pd.DataFrame({"second": grid})
    for c in NEURO_FEATURES:
        out[c] = np.interp(grid, eeg["time"].values, eeg[c].interpolate().bfill().ffill().values)
    return out, float(eeg["time"].max())


def process_session(subject_id, folder, session, label, legacy):
    sig = {}
    for name in ["BVP", "EDA", "TEMP", "HR"]:
        sig[name] = read_e4(os.path.join(folder, f"{name}.csv"))
    _, acc_rate, acc = read_e4(os.path.join(folder, "ACC.csv"), n_cols=3)
    eeg_path = os.path.join(folder, "EEG.csv")
    eeg_time_max = pd.to_numeric(pd.read_csv(eeg_path).rename(columns=str.strip)["time"], errors="coerce").max()

    durations = [len(v) / r for (_, r, v) in sig.values()] + [len(acc) / acc_rate, eeg_time_max]
    duration = int(np.floor(min(durations)))

    cols = []
    for name in ["BVP", "EDA", "TEMP"]:
        _, rate, v = sig[name]
        vals = legacy_values(os.path.join(folder, f"{name}.csv")) if legacy else v.iloc[:, 0].values
        cols.append(to_1hz(vals, rate, duration, name))
    acc.columns = ["AccX", "AccY", "AccZ"]
    acc.index = np.arange(len(acc)) / acc_rate
    grid = np.arange(duration)
    acc = acc.reindex(np.union1d(acc.index.values, grid)).interpolate(method="index", limit_direction="both").reindex(grid)
    cols.append(acc)
    _, rate, v = sig["HR"]
    vals = legacy_values(os.path.join(folder, "HR.csv")) if legacy else v.iloc[:, 0].values
    cols.append(to_1hz(vals, rate, duration, "HR"))
    eeg, _ = eeg_to_1hz(eeg_path, duration)
    cols.append(eeg[NEURO_FEATURES])

    data = pd.concat(cols, axis=1)
    data["subject_id"], data["session"] = subject_id, session
    data["session_id"] = f"{subject_id}_{session}"
    data["class"] = int(label)
    data["second"] = np.arange(len(data))
    return data[["subject_id", "session", "session_id", "second"] + ALL_FEATURES + ["class"]]


def main(args):
    out = ensure_dir(args.out_dir)
    labels, scores = load_labels(os.path.join(args.raw_root, "general_info.xlsx"))
    scores.to_csv(os.path.join(out, "chalder_scores.csv"), index=False)
    if args.legacy_e4_parsing:
        print("WARNING: --legacy-e4-parsing reproduces the header-row issue described in the docstring.")

    sessions = []
    for sid, session, folder in iter_sessions(args.raw_root):
        if (sid, session) in labels:
            sessions.append(process_session(sid, folder, session, labels[(sid, session)],
                                            args.legacy_e4_parsing))
    df = pd.concat(sessions, ignore_index=True)
    df[ALL_FEATURES] = df[ALL_FEATURES].apply(pd.to_numeric, errors="coerce")
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=ALL_FEATURES + ["class"]).reset_index(drop=True)

    banner("Reconstructed table")
    print(f"Rows: {len(df)} (removed {n0 - len(df)}) | participants: {df.subject_id.nunique()} "
          f"| sessions: {df.session_id.nunique()}")
    print(pd.crosstab(df.session, df["class"], margins=True))
    hdr = df.loc[df.second == 0, ["BVP", "EDA", "TEMP", "HR"]].abs().max()
    print("Max |value| at second 0 (should be physiological, not ~1e9):\n", hdr)
    path = os.path.join(out, SUBJECTWISE_CSV)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--legacy-e4-parsing", action="store_true")
    main(ap.parse_args())
