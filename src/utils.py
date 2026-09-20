import os
import random

import numpy as np


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def banner(text: str) -> None:
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)
