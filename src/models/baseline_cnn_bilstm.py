"""
models/baseline_cnn_bilstm.py — The original CNN-BiLSTM model.

Faithfully reproduces the architecture from AI_PBL (1).ipynb.
Used for:
  - Sprint 2: Reproduction of existing results
  - Sprint 4: Baseline in benchmark table (B1)

Architecture (confirmed from source code):
  Conv1D(128, k=3, ReLU) → MaxPool1D(2) → Bidirectional(LSTM(50)) → Dropout(0.4) → Dense(5, linear)

Training config (confirmed from source code):
  Adam(lr=0.0001), loss=MSE, epochs=35, batch_size=32
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense,
)
from tensorflow.keras.optimizers import Adam
from typing import Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    CNN_FILTERS, CNN_KERNEL_SIZE, POOL_SIZE, LSTM_UNITS,
    DROPOUT, LEARNING_RATE, EPOCHS, BATCH_SIZE, LOSS,
)


def build_cnn_bilstm(
    time_steps:     int   = TIME_STEPS,
    n_features:     int   = N_FEATURES,
    n_outputs:      int   = N_FEATURES,
    cnn_filters:    int   = CNN_FILTERS,
    kernel_size:    int   = CNN_KERNEL_SIZE,
    pool_size:      int   = POOL_SIZE,
    lstm_units:     int   = LSTM_UNITS,
    dropout:        float = DROPOUT,
    learning_rate:  float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build the original CNN-BiLSTM model.

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)  — multi-output regression

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential(name="CNN_BiLSTM_Baseline")

    # Local temporal pattern extraction
    model.add(Conv1D(
        filters=cnn_filters,
        kernel_size=kernel_size,
        activation="relu",
        input_shape=(time_steps, n_features),
        name="conv1d",
    ))

    # Downsample temporal dimension
    model.add(MaxPooling1D(pool_size=pool_size, name="maxpool"))

    # Bidirectional LSTM — reads the (pooled) temporal sequence forward & backward
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=False), name="bilstm"))

    # Regularisation
    model.add(Dropout(dropout, name="dropout"))

    # Multi-output regression head
    model.add(Dense(n_outputs, activation="linear", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )

    return model


def train(
    model: tf.keras.Model,
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val:   "np.ndarray",
    y_val:   "np.ndarray",
    epochs:     int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    callbacks:  Optional[list] = None,
    verbose:    int = 1,
) -> tf.keras.callbacks.History:
    """
    Train the model with a proper val set (not the test set).

    Note: The original code used X_test as validation_data during training.
    Here we pass a genuine validation set to avoid contaminating the test set.

    Args:
        model      : Compiled Keras model.
        X_train    : Training sequences (N_train, time_steps, n_features).
        y_train    : Training targets   (N_train, n_outputs).
        X_val      : Validation sequences.
        y_val      : Validation targets.
        epochs     : Number of training epochs.
        batch_size : Mini-batch size.
        callbacks  : Optional list of Keras callbacks.
        verbose    : Keras verbosity level.

    Returns:
        Keras History object.
    """
    if callbacks is None:
        callbacks = _default_callbacks()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),   # ← genuine val set, NOT test
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


def _default_callbacks() -> list:
    """Early stopping + model checkpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
