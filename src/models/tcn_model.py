"""
models/tcn_model.py — Temporal Convolutional Network Baseline (Model B5).

Dilated causal 1D convolutions with residual connections for temporal sequence modeling.
Used for:
  - Sprint 4: Baseline in benchmark table (B5)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, SpatialDropout1D, Add, Dense, GlobalAveragePooling1D, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import List, Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    DROPOUT, LEARNING_RATE, LOSS,
)


def _residual_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float, block_idx: int):
    """A single dilated causal residual block."""
    res = x
    # First dilated conv
    conv1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"tcn_b{block_idx}_conv1",
    )(x)
    drop1 = SpatialDropout1D(dropout, name=f"tcn_b{block_idx}_drop1")(conv1)

    # Second dilated conv
    conv2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"tcn_b{block_idx}_conv2",
    )(drop1)
    drop2 = SpatialDropout1D(dropout, name=f"tcn_b{block_idx}_drop2")(conv2)

    # Match dimensions for skip connection if needed
    if x.shape[-1] != filters:
        res = Conv1D(filters=filters, kernel_size=1, padding="same", name=f"tcn_b{block_idx}_match")(x)

    out = Add(name=f"tcn_b{block_idx}_add")([res, drop2])
    return Activation("relu", name=f"tcn_b{block_idx}_relu")(out)


def build_tcn(
    time_steps:      int             = TIME_STEPS,
    n_features:      int             = N_FEATURES,
    n_outputs:       int             = N_FEATURES,
    filters:         int             = 64,
    kernel_size:     int             = 3,
    dilations:       List[int]       = [1, 2, 4],
    dropout:         float           = DROPOUT,
    learning_rate:   float           = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build Temporal Convolutional Network Model (B5).

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)
    """
    inputs = Input(shape=(time_steps, n_features), name="input_tcn")
    x = inputs

    for i, d in enumerate(dilations):
        x = _residual_block(x, filters=filters, kernel_size=kernel_size, dilation_rate=d, dropout=dropout, block_idx=i)

    # Pooling over temporal dimension
    x = GlobalAveragePooling1D(name="tcn_gap")(x)
    x = Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = Dense(n_outputs, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TCN_Baseline")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )
    return model
