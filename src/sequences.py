import numpy as np


def rolling_windows(X, y, seq_len):
    """Windows over consecutive ROWS of X in their given order (used under P1).

    Under P1 the rows have already been shuffled by the random 80:20 split,
    so these windows are NOT temporal sequences (see Section 4.9.3)."""
    Xs = np.stack([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)]).astype(np.float32)
    ys = np.asarray([y[i + seq_len - 1] for i in range(len(X) - seq_len + 1)], dtype=np.float32)
    return Xs, ys


def session_windows(df, features, target, seq_len):
    """Chronological windows built separately within each session (used under P2).
    No window spans two sessions or two participants."""
    Xs, ys, subj, sess = [], [], [], []
    ordered = df.sort_values(["subject_id", "session_id", "second"])
    for (s, sid), g in ordered.groupby(["subject_id", "session_id"], sort=False):
        if len(g) < seq_len:
            continue
        xv = g[features].values.astype(np.float32)
        yv = g[target].values.astype(np.float32)
        for i in range(len(g) - seq_len + 1):
            Xs.append(xv[i:i + seq_len]); ys.append(yv[i + seq_len - 1]); subj.append(s); sess.append(sid)
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32), np.asarray(subj), np.asarray(sess)
