"""Shared constants for all experiments.

Paths are passed on the command line (see README); nothing here is
machine-specific.
"""
import os

PHYS_FEATURES = ["BVP", "EDA", "TEMP", "AccX", "AccY", "AccZ", "HR"]
NEURO_FEATURES = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2",
                  "Gamma1", "Gamma2", "Attention", "Meditation"]
ALL_FEATURES = PHYS_FEATURES + NEURO_FEATURES      # 17 predictors, fixed order

TARGET = "class"
GROUP = "subject_id"

SEED = 42
ABLATION_SEEDS = [int(x) for x in os.environ.get("MEFAR_ABLATION_SEEDS", "42,7,123,2024,99").split(",")]
N_SPLITS = 5

# Deep-learning training settings (identical for P1 and P2)
EPOCHS = int(os.environ.get("MEFAR_EPOCHS", 100))   # override only for smoke tests
BATCH_SIZE = 64
SEQ_LEN = 10               # LSTM window length
INNER_VAL_FRACTION = 0.20  # P2: share of training participants used for validation

# Label rule for the reconstructed table (Chalder Fatigue Scale, Likert 0-33)
CHALDER_THRESHOLD = 12

# Default file name of the reconstructed table (written inside --out-dir)
SUBJECTWISE_CSV = "MEFAR_subjectwise_1Hz_FINAL.csv"
