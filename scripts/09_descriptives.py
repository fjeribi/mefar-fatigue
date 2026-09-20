"""Descriptive statistics and checks on the reconstructed table (Tables 3 and 4,
Sections 4.4 and 4.6.2), the EEG structure check (Section 4.4) and exact
parameter counts of the deep-learning models (Table 14).

Also reports whether E4 header rows leaked into the second-0 rows
(see scripts/02_reconstruct_subjectwise.py, --legacy-e4-parsing).

Usage:
    python scripts/09_descriptives.py --out-dir results [--release MEFAR_DOWN.csv]
"""
import argparse
import os

import _common  # noqa: F401
import pandas as pd

from src.config import ALL_FEATURES, SUBJECTWISE_CSV, TARGET
from src.data import load_subjectwise
from src.utils import banner, ensure_dir


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p2"))
    df = load_subjectwise(args.data or os.path.join(args.out_dir, SUBJECTWISE_CSV))

    banner("Label structure (Table 4)")
    print(f"Rows {len(df)} | participants {df.subject_id.nunique()} | sessions {df.session_id.nunique()}")
    sess = df.groupby(["subject_id", "session"])[TARGET].first().reset_index()
    t_sessions = pd.crosstab(sess.session, sess[TARGET], margins=True)
    t_rows = pd.crosstab(df.session, df[TARGET], margins=True)
    print("Sessions by label:\n", t_sessions, "\n\nObservations by label:\n", t_rows)
    wide = sess.pivot(index="subject_id", columns="session", values=TARGET)
    changed = wide[wide.morning != wide.evening]
    print(f"\nParticipants whose label changes between sessions: {len(changed)} of {len(wide)}")
    print(f"Session agrees with label (evening=fatigued) for "
          f"{((sess.session == 'evening').astype(int) == sess[TARGET]).mean():.1%} of sessions")
    t_sessions.to_csv(os.path.join(out, "label_structure_sessions.csv"))
    t_rows.to_csv(os.path.join(out, "label_structure_rows.csv"))
    wide.to_csv(os.path.join(out, "labels_per_participant.csv"))

    banner("Class proportions")
    print("Reconstructed:", df[TARGET].value_counts(normalize=True).round(4).to_dict())
    if args.release:
        rel = pd.read_csv(args.release); rel.columns = rel.columns.str.strip()
        print("Release:      ", rel[TARGET].value_counts(normalize=True).round(4).to_dict())

    banner("Descriptive statistics (Table 3)")
    desc = df[ALL_FEATURES].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(2).to_string())
    desc.to_csv(os.path.join(out, "descriptives_p2_table.csv"))

    if args.raw_root:
        banner("EEG structure check on the raw files (Section 4.4)")
        import glob
        import numpy as np
        bands = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]
        frames = []
        for f in glob.glob(os.path.join(args.raw_root, "subject_*", "*", "EEG.csv")):
            e = pd.read_csv(f); e.columns = e.columns.str.strip(); frames.append(e)
        eeg = pd.concat(frames)
        for c in bands + ["Attention"]:
            eeg[c] = pd.to_numeric(eeg[c], errors="coerce")
        rho = eeg[bands].corr(method="spearman").values[np.triu_indices(len(bands), 1)].mean()
        print("Median band power:", eeg[bands].median().round(0).to_dict())
        print(f"Mean inter-band Spearman correlation: {rho:.3f}")
        print(f"P(Delta > Gamma2): {(eeg.Delta > eeg.Gamma2).mean():.3f}")
        print("Attention:", eeg.Attention.describe().round(1).to_dict())

    banner("E4 header-row check (second-0 rows)")
    cols = ["BVP", "EDA", "TEMP", "HR"]
    at0 = df.loc[df.second == 0, cols]
    print(at0.describe().T[["min", "max"]])
    if (at0.abs() > 1e8).any().any():
        print("WARNING: second-0 values look like Unix timestamps; the table was built with legacy parsing.")
    else:
        print("OK: no timestamp values at second 0.")

    if not args.skip_params:
        banner("Parameter counts (Table 14)")
        from src.models_dl import (ABLATION_CONFIGS, build_ablation_config, build_ann, build_cnn,
                                   build_lstm, build_mf_afen)
        counts = {"ANN / MLP": build_ann(17).count_params(), "1D-CNN": build_cnn(17).count_params(),
                  "LSTM": build_lstm(17).count_params(), "MF-AFEN": build_mf_afen().count_params()}
        counts.update({f"ablation {c}": build_ablation_config(c)[0].count_params() for c in ABLATION_CONFIGS})
        for k, v in counts.items():
            print(f"{k:40s} {v:>8,}")
        pd.Series(counts, name="n_params").to_csv(os.path.join(out, "parameter_counts.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--data", default=None)
    ap.add_argument("--release", default=None)
    ap.add_argument("--raw-root", default=None, help="raw MEFAR folder (enables the EEG structure check)")
    ap.add_argument("--skip-params", action="store_true")
    main(ap.parse_args())
