"""
models/cnn_gru_model.py — CNN + GRU Hybrid Baseline (Model B7).

Combines 1D-CNN temporal feature extraction with a Gated Recurrent Unit (GRU).
Used for:
  - Sprint 4: Baseline in benchmark table (B7)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from typing import Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    CNN_FILTERS, CNN_KERNEL_SIZE, POOL_SIZE,
    DROPOUT, LEARNING_RATE, LOSS,
)


def build_cnn_gru(
    time_steps:      int   = TIME_STEPS,
    n_features:      int   = N_FEATURES,
    n_outputs:       int   = N_FEATURES,
    cnn_filters:     int   = CNN_FILTERS,
    kernel_size:     int   = CNN_KERNEL_SIZE,
    pool_size:       int   = POOL_SIZE,
    gru_units:       int   = 64,
    dropout:         float = DROPOUT,
    learning_rate:   float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build CNN-GRU Hybrid Model (B7).

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)
    """
    model = Sequential(name="CNN_GRU_Baseline")
    model.add(tf.keras.layers.Input(shape=(time_steps, n_features)))
    model.add(Conv1D(
        filters=cnn_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        name="conv1d",
    ))
    model.add(MaxPooling1D(pool_size=pool_size, name="maxpool"))
    model.add(GRU(units=gru_units, return_sequences=False, name="gru"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(32, activation="relu", name="dense_hidden"))
    model.add(Dense(n_outputs, activation="linear", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )
    return model
