"""
models/lstm_model.py — Vanilla Unidirectional LSTM Baseline (Model B2).

Standard single/multi-layer unidirectional LSTM architecture.
Used for:
  - Sprint 4: Baseline in benchmark table (B2)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from typing import Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    DROPOUT, LEARNING_RATE, LOSS,
)


def build_lstm(
    time_steps:    int   = TIME_STEPS,
    n_features:    int   = N_FEATURES,
    n_outputs:     int   = N_FEATURES,
    lstm_units:    int   = 64,
    dropout:       float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build Vanilla Unidirectional LSTM Model (B2).

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)
    """
    model = Sequential(name="LSTM_Baseline")
    model.add(tf.keras.layers.Input(shape=(time_steps, n_features)))
    model.add(LSTM(units=lstm_units, return_sequences=False, name="lstm"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(32, activation="relu", name="dense_hidden"))
    model.add(Dense(n_outputs, activation="linear", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )
    return model
