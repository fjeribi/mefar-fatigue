"""Keras architectures used in the paper.

build_ann / build_cnn / build_lstm / build_mf_afen: Sections 4.9-4.10
build_ablation_config: seven ablation configurations (Section 4.11.1)
build_gating_variant: self-gated and cross-modal-gated MF-AFEN (Section 4.11.3)
"""
import tensorflow as tf

from .config import NEURO_FEATURES, PHYS_FEATURES

L = tf.keras.layers
N_PHYS, N_NEURO = len(PHYS_FEATURES), len(NEURO_FEATURES)


def compile_model(model):
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]


# --------------------------------------------------------------------------
# Baselines and MF-AFEN (identical under P1 and P2)
# --------------------------------------------------------------------------
def build_ann(input_dim):
    i = tf.keras.Input(shape=(input_dim,))
    x = L.Dense(128, activation="relu")(i); x = L.BatchNormalization()(x); x = L.Dropout(0.30)(x)
    x = L.Dense(64, activation="relu")(x); x = L.BatchNormalization()(x); x = L.Dropout(0.25)(x)
    x = L.Dense(32, activation="relu")(x); x = L.Dropout(0.20)(x)
    return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="ANN")


def build_cnn(input_dim):
    i = tf.keras.Input(shape=(input_dim, 1))
    x = L.Conv1D(64, 3, activation="relu", padding="same")(i); x = L.BatchNormalization()(x)
    x = L.MaxPooling1D(pool_size=2)(x); x = L.Dropout(0.25)(x)
    x = L.Conv1D(128, 3, activation="relu", padding="same")(x); x = L.BatchNormalization()(x)
    x = L.MaxPooling1D(pool_size=2)(x); x = L.Dropout(0.25)(x)
    x = L.Flatten()(x); x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.30)(x)
    return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="1D_CNN")


def build_lstm(input_dim, seq_len=None):
    i = tf.keras.Input(shape=(seq_len, input_dim))
    x = L.LSTM(64, return_sequences=True)(i); x = L.Dropout(0.30)(x)
    x = L.LSTM(32)(x); x = L.Dropout(0.25)(x)
    x = L.Dense(32, activation="relu")(x); x = L.Dropout(0.20)(x)
    return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="LSTM")


def build_mf_afen():
    """MF-AFEN as reported in Tables 9, 10 and 15 (23,745 parameters)."""
    pi = tf.keras.Input(shape=(N_PHYS,), name="physiological_input")
    p = L.Dense(64, activation="relu")(pi); p = L.BatchNormalization()(p); p = L.Dropout(0.30)(p)
    p = L.Dense(32, activation="relu")(p)
    p = L.Multiply()([p, L.Dense(32, activation="sigmoid", name="phys_gate")(p)])

    ni = tf.keras.Input(shape=(N_NEURO,), name="neurophysiological_input")
    n = L.Dense(128, activation="relu")(ni); n = L.BatchNormalization()(n); n = L.Dropout(0.30)(n)
    n = L.Dense(64, activation="relu")(n); n = L.Dropout(0.20)(n)
    n = L.Dense(32, activation="relu")(n)
    n = L.Multiply()([n, L.Dense(32, activation="sigmoid", name="neuro_gate")(n)])

    x = L.Concatenate()([p, n])
    x = L.Dense(64, activation="relu")(x); x = L.BatchNormalization()(x); x = L.Dropout(0.30)(x)
    x = L.Dense(32, activation="relu")(x); x = L.Dropout(0.20)(x)
    return tf.keras.Model([pi, ni], L.Dense(1, activation="sigmoid")(x), name="MF_AFEN")


# --------------------------------------------------------------------------
# Ablation (M3/M4) and gating/masking (M6) implementations
# NOTE: these use dense_block(), which adds batch normalisation after the
# 64-unit neurophysiological layer; the full model therefore has 24,001
# parameters here rather than 23,745 (see Section 4.11.1 of the paper).
# --------------------------------------------------------------------------
def dense_block(x, units, dropout, batchnorm=True):
    x = L.Dense(units, activation="relu")(x)
    if batchnorm:
        x = L.BatchNormalization()(x)
    return L.Dropout(dropout)(x)


def _branches():
    pi = tf.keras.Input(shape=(N_PHYS,), name="phys_input")
    p = dense_block(pi, 64, 0.30)
    p = dense_block(p, 32, 0.0, batchnorm=False)
    ni = tf.keras.Input(shape=(N_NEURO,), name="neuro_input")
    n = dense_block(ni, 128, 0.30)
    n = dense_block(n, 64, 0.20)
    n = dense_block(n, 32, 0.0, batchnorm=False)
    return pi, p, ni, n


def _head(x):
    x = dense_block(x, 64, 0.30)
    x = dense_block(x, 32, 0.20, batchnorm=False)
    return L.Dense(1, activation="sigmoid")(x)


ABLATION_CONFIGS = [
    "1_phys_only", "2_neuro_only", "3_concat_matched", "4_two_branch_no_gate",
    "5_gate_neuro_only", "6_both_gates_no_fusion_head", "7_full_mf_afen",
]


def build_ablation_config(name):
    """Returns (model, input_mode); input_mode in {'phys','neuro','concat','dual'}."""
    if name == "1_phys_only":
        i = tf.keras.Input(shape=(N_PHYS,))
        x = dense_block(i, 64, 0.30); x = dense_block(x, 32, 0.20)
        return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="Phys_Only"), "phys"
    if name == "2_neuro_only":
        i = tf.keras.Input(shape=(N_NEURO,))
        x = dense_block(i, 128, 0.30); x = dense_block(x, 64, 0.20)
        x = dense_block(x, 32, 0.0, batchnorm=False)
        return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="Neuro_Only"), "neuro"
    if name == "3_concat_matched":
        i = tf.keras.Input(shape=(N_PHYS + N_NEURO,))
        x = dense_block(i, 160, 0.30); x = dense_block(x, 64, 0.25)
        x = dense_block(x, 32, 0.0, batchnorm=False)
        x = dense_block(x, 64, 0.30); x = dense_block(x, 32, 0.20, batchnorm=False)
        return tf.keras.Model(i, L.Dense(1, activation="sigmoid")(x), name="Concat_Matched"), "concat"

    pi, p, ni, n = _branches()
    if name == "4_two_branch_no_gate":
        return tf.keras.Model([pi, ni], _head(L.Concatenate()([p, n])), name="TwoBranch_NoGate"), "dual"
    if name == "5_gate_neuro_only":
        n = L.Multiply()([n, L.Dense(32, activation="sigmoid", name="neuro_gate")(n)])
        return tf.keras.Model([pi, ni], _head(L.Concatenate()([p, n])), name="Gate_NeuroOnly"), "dual"
    p = L.Multiply()([p, L.Dense(32, activation="sigmoid", name="phys_gate")(p)])
    n = L.Multiply()([n, L.Dense(32, activation="sigmoid", name="neuro_gate")(n)])
    fused = L.Concatenate()([p, n])
    if name == "6_both_gates_no_fusion_head":
        return tf.keras.Model([pi, ni], L.Dense(1, activation="sigmoid")(fused),
                              name="BothGates_NoFusionHead"), "dual"
    if name == "7_full_mf_afen":
        return tf.keras.Model([pi, ni], _head(fused), name="MF_AFEN_Full"), "dual"
    raise ValueError(f"Unknown ablation config: {name}")


def build_gating_variant(variant):
    """'self_gate_mf_afen': each branch gates itself.
    'cross_modal_gate_mf_afen': each branch's gate is computed from the other branch."""
    pi, p, ni, n = _branches()
    if variant == "self_gate_mf_afen":
        pg = L.Dense(32, activation="sigmoid", name="phys_self_gate")(p)
        ng = L.Dense(32, activation="sigmoid", name="neuro_self_gate")(n)
    elif variant == "cross_modal_gate_mf_afen":
        pg = L.Dense(32, activation="sigmoid", name="phys_cross_gate")(n)
        ng = L.Dense(32, activation="sigmoid", name="neuro_cross_gate")(p)
    else:
        raise ValueError(variant)
    fused = L.Concatenate()([L.Multiply()([p, pg]), L.Multiply()([n, ng])])
    return tf.keras.Model([pi, ni], _head(fused), name=variant)
