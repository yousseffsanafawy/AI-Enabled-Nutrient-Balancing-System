"""
utils/seed.py — Global random seed setter for reproducibility.

Call set_all_seeds() at the very start of every script/notebook.
"""

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """
    Set deterministic seeds across Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed. Default matches config.RANDOM_SEED = 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow — import lazily so the module works even without TF installed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Force deterministic ops on GPU where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass

    print(f"[seed] All random seeds set to {seed}.")
