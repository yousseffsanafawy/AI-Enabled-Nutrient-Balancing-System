"""
models/bilstm_model.py — Bidirectional LSTM Baseline (Model B4).

Pure Bidirectional LSTM architecture (without CNN feature extraction).
Used for:
  - Sprint 4: Baseline in benchmark table (B4)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from typing import Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    LSTM_UNITS, DROPOUT, LEARNING_RATE, LOSS,
)


def build_bilstm(
    time_steps:    int   = TIME_STEPS,
    n_features:    int   = N_FEATURES,
    n_outputs:     int   = N_FEATURES,
    lstm_units:    int   = LSTM_UNITS,
    dropout:       float = DROPOUT,
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build Bidirectional LSTM Model without CNN front-end (B4).

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)
    """
    model = Sequential(name="BiLSTM_Baseline")
    model.add(tf.keras.layers.Input(shape=(time_steps, n_features)))
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False), name="bilstm"))
    model.add(Dropout(dropout, name="dropout"))
    model.add(Dense(32, activation="relu", name="dense_hidden"))
    model.add(Dense(n_outputs, activation="linear", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )
    return model
