"""Loading the preprocessed release (P1) and the reconstructed table (P2)."""
import numpy as np
import pandas as pd

from .config import ALL_FEATURES, GROUP, TARGET


def load_release(path):
    """Preprocessed MEFAR release (e.g. MEFAR_DOWN.csv), cleaned as in the paper:
    +/-inf -> NaN, drop rows with any NaN, drop exact duplicate rows."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates().copy()
    print(f"Release: {n0} rows -> {len(df)} after cleaning ({n0 - len(df)} removed)")
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return df, X, y


def load_subjectwise(path, extra_required=("session", "session_id", "second")):
    """Reconstructed participant-level table produced by 02_reconstruct_subjectwise.py."""
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    required = list(ALL_FEATURES) + [TARGET, GROUP] + [c for c in extra_required if c in df.columns]
    df = df.dropna(subset=required).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    df[GROUP] = df[GROUP].astype(str)
    return df


def assert_no_overlap(train_groups, test_groups, label="fold"):
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise RuntimeError(f"PARTICIPANT LEAKAGE in {label}: {sorted(overlap)}")
