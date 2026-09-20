"""Protocol 1 (P1): stratified random 80:20 sample-level holdout on the
preprocessed MEFAR release. Reproduces Table 7 and Figure 3 (plus supplementary
confusion matrices, ROC and training curves).

Usage:
    python scripts/01_protocol1_sample_level.py --release MEFAR_DOWN.csv --out-dir results
"""
import argparse
import os

import _common  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import ALL_FEATURES, BATCH_SIZE, EPOCHS, NEURO_FEATURES, PHYS_FEATURES, SEED, SEQ_LEN
from src.data import load_release
from src.metrics import binary_metrics
from src.models_dl import (build_ann, build_cnn, build_lstm, build_mf_afen,
                           compile_model, get_callbacks)
from src.models_ml import P1_SCALED, p1_models
from src.sequences import rolling_windows
from src.utils import banner, ensure_dir, set_seeds


def main(args):
    out = ensure_dir(os.path.join(args.out_dir, "p1"))
    set_seeds(SEED)
    _, X, y = load_release(args.release)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"Train {len(X_tr)} | Test {len(X_te)} (fatigued {int(y_te.sum())})")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    results, probs, preds, histories = [], {}, {}, {}

    banner("Conventional models (P1)")
    for name, model in p1_models().items():
        m = clone(model)
        Xa, Xb = (X_tr_s, X_te_s) if name in P1_SCALED else (X_tr, X_te)
        m.fit(Xa, y_tr)
        prob = m.predict_proba(Xb)[:, 1]
        pred = m.predict(Xb)
        probs[name], preds[name] = prob, pred
        results.append({"model": name, "type": "ML", **binary_metrics(y_te, prob, y_pred=pred)})
        print(f"{name:22s} acc={results[-1]['accuracy']:.4f} auc={results[-1]['roc_auc']:.4f}")

    banner("Deep-learning models (P1)")
    set_seeds(SEED)
    Xtr = np.asarray(X_tr_s, np.float32); Xte = np.asarray(X_te_s, np.float32)
    ytr = np.asarray(y_tr, np.float32); yte = np.asarray(y_te, np.float32)
    fit_kw = dict(validation_split=0.20, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=args.verbose)

    def run(name, model, xtr, ytr_, xte, yte_):
        model = compile_model(model)
        h = model.fit(xtr, ytr_, callbacks=get_callbacks(), **fit_kw)
        prob = model.predict(xte, verbose=0).ravel()
        histories[name] = h.history
        probs[name], preds[name] = prob, (prob >= 0.5).astype(int)
        results.append({"model": name, "type": "DL", "n_params": model.count_params(),
                        **binary_metrics(yte_, prob)})
        print(f"{name:22s} acc={results[-1]['accuracy']:.4f} auc={results[-1]['roc_auc']:.4f}")

    run("ANN / MLP", build_ann(Xtr.shape[1]), Xtr, ytr, Xte, yte)
    run("1D-CNN", build_cnn(Xtr.shape[1]), Xtr[..., None], ytr, Xte[..., None], yte)
    # LSTM: windows over rows that were already shuffled by the random split,
    # exactly as in the original P1 experiment (Section 4.9.3).
    Xs_tr, ys_tr = rolling_windows(Xtr, ytr, SEQ_LEN)
    Xs_te, ys_te = rolling_windows(Xte, yte, SEQ_LEN)
    run("LSTM", build_lstm(Xtr.shape[1], SEQ_LEN), Xs_tr, ys_tr, Xs_te, ys_te)
    pi = [ALL_FEATURES.index(f) for f in PHYS_FEATURES]
    ni = [ALL_FEATURES.index(f) for f in NEURO_FEATURES]
    run("MF-AFEN", build_mf_afen(), [Xtr[:, pi], Xtr[:, ni]], ytr, [Xte[:, pi], Xte[:, ni]], yte)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(out, "p1_results.csv"), index=False)
    print("\n", res.round(4).to_string(index=False))

    # Confusion matrices for the nine models with a common test set (LSTM excluded)
    cm_rows, names = [], [m for m in res.model if m != "LSTM"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_te, preds[name])
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append({"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                        "missed_fatigue_pct": 100 * fn / (fn + tp)})
        ConfusionMatrixDisplay(cm, display_labels=["Non-fatigued", "Fatigued"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_confusion_matrices.png"), dpi=300); plt.close(fig)
    pd.DataFrame(cm_rows).to_csv(os.path.join(out, "p1_confusion_counts.csv"), index=False)

    # ROC curves of the conventional models
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in [r["model"] for r in results if r["type"] == "ML"]:
        fpr, tpr, _ = roc_curve(y_te, probs[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, probs[name]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_roc_conventional.png"), dpi=300); plt.close(fig)

    # ANN training curves
    for key, fname in [("accuracy", "fig_ann_accuracy.png"), ("loss", "fig_ann_loss.png")]:
        h = histories["ANN / MLP"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(h[key], label=f"Training {key}"); ax.plot(h[f"val_{key}"], label=f"Validation {key}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(key.capitalize()); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=300); plt.close(fig)

    # Metric comparison
    plot = res.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ax = plot.sort_values("roc_auc", ascending=False).plot(kind="bar", figsize=(11, 5), ylim=(0.4, 1.02))
    ax.set_ylabel("Score"); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_model_comparison.png"), dpi=300); plt.close()
    print(f"\nSaved results and figures to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="Path to MEFAR_DOWN.csv")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--verbose", type=int, default=0)
    main(ap.parse_args())
