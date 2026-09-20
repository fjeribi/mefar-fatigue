"""Reading the raw MEFAR files (Empatica E4 + NeuroSky EEG) with correct
handling of the two E4 header rows (start timestamp, sample rate)."""
import glob
import os

import numpy as np
import pandas as pd

from .config import CHALDER_THRESHOLD, NEURO_FEATURES

SESSIONS = [("1.morning", "morning"), ("2.evening", "evening")]


def load_labels(general_info_path):
    """Session labels: fatigued (1) if Chalder score >= CHALDER_THRESHOLD."""
    gi = pd.read_excel(general_info_path, sheet_name="Subject List")
    gi.columns = gi.columns.str.strip().str.lower()
    labels, scores = {}, []
    for _, row in gi.iterrows():
        subj = str(row["subjects"]).strip()
        folder = f"subject_{subj[1:]}" if subj.upper().startswith("S") else subj
        for session, col in [("morning", "morning-mental fatigue state"),
                             ("evening", "evening-mental fatigue state")]:
            s = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(s):
                labels[(folder, session)] = int(s >= CHALDER_THRESHOLD)
                scores.append({"subject_id": folder, "session": session, "chalder_score": s})
    return labels, pd.DataFrame(scores)


def read_e4(path, n_cols=1):
    """Return (start_time, sample_rate, samples DataFrame) for an E4 CSV file."""
    raw = pd.read_csv(path, header=None).dropna(how="all").reset_index(drop=True)
    start, rate = float(raw.iloc[0, 0]), float(raw.iloc[1, 0])
    samples = raw.iloc[2:, :n_cols].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return start, rate, samples


def read_eeg(path):
    """NeuroSky EEG features with numeric time column; duplicate timestamps averaged."""
    eeg = pd.read_csv(path)
    eeg.columns = eeg.columns.astype(str).str.strip()
    tcol = next(c for c in ["Time", "time", "timestamp", "Timestamp"] if c in eeg.columns)
    eeg[tcol] = pd.to_numeric(eeg[tcol], errors="coerce")
    for c in NEURO_FEATURES:
        eeg[c] = pd.to_numeric(eeg[c], errors="coerce")
    eeg = eeg.dropna(subset=[tcol]).sort_values(tcol)
    eeg = eeg.groupby(tcol, as_index=False)[NEURO_FEATURES].mean().rename(columns={tcol: "time"})
    return eeg


def iter_sessions(raw_root):
    """Yield (subject_id, session_name, folder) for every available session."""
    dirs = sorted(glob.glob(os.path.join(raw_root, "subject_*")),
                  key=lambda p: int(os.path.basename(p).split("_")[1]))
    for sdir in dirs:
        sid = os.path.basename(sdir)
        for sub, session in SESSIONS:
            folder = os.path.join(sdir, sub)
            if os.path.isdir(folder):
                yield sid, session, folder
