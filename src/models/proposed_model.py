"""
models/proposed_model.py — Multi-Task Temporal Forecasting + Stress Detection (Proposed Architecture).

Sprint 5 Implementation:
  - Architecture: Multi-Task Dilated Temporal Convolutional LSTM Network (MT-TCN-LSTM)
  - Motivated by Sprint 4 findings:
    * TCN (B5) excelled at local gradient smoothing & dynamic chemical tracking (lowest TDS MAE).
    * LSTM (B2) excelled at global sequence memory & pH stability (highest average R²).
    * Baselines suffered from lack of domain-aware regularization and zero stress head.
  - Structure:
    1. Input: (batch, time_steps, n_features)
    2. Shared Temporal Feature Extractor:
       - Multi-scale Dilated Causal Conv1D residual blocks (d=1, 2)
       - Unidirectional LSTM sequence aggregator (64 units)
    3. Dual Output Heads:
       - 'forecast_output': Dense(5, linear) for continuous next-step predictions (MSE)
       - 'stress_output':   Dense(1, sigmoid) for out-of-bounds stress classification (BCE)
    4. Combined Multi-Task Loss:
       L_total = (1 - λ) * MSE_forecast + λ * BCE_stress (default λ = 0.2)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, SpatialDropout1D, Add, Activation,
    LSTM, Dropout, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional, Dict

from src.config import (
    N_FEATURES, TIME_STEPS,
    DROPOUT, LEARNING_RATE, LAMBDA_STRESS,
)


def _dilated_conv_block(x, filters: int, kernel_size: int, dilation_rate: int, dropout: float, name_prefix: str):
    """Causal dilated convolutional block with residual connection."""
    res = x
    c1 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"{name_prefix}_conv1",
    )(x)
    d1 = SpatialDropout1D(dropout, name=f"{name_prefix}_drop1")(c1)

    c2 = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        activation="relu",
        name=f"{name_prefix}_conv2",
    )(d1)
    d2 = SpatialDropout1D(dropout, name=f"{name_prefix}_drop2")(c2)

    if x.shape[-1] != filters:
        res = Conv1D(filters=filters, kernel_size=1, padding="same", name=f"{name_prefix}_res_match")(x)

    out = Add(name=f"{name_prefix}_add")([res, d2])
    return Activation("relu", name=f"{name_prefix}_relu")(out)


def build_proposed_model(
    time_steps:      int   = TIME_STEPS,
    n_features:      int   = N_FEATURES,
    n_outputs:       int   = N_FEATURES,
    conv_filters:    int   = 64,
    kernel_size:     int   = 3,
    lstm_units:      int   = 64,
    dropout:         float = DROPOUT,
    lambda_stress:   float = LAMBDA_STRESS,
    learning_rate:   float = LEARNING_RATE,
    class_weight_pos: float = 1.0,
) -> tf.keras.Model:
    """
    Build the Proposed Multi-Task MT-TCN-LSTM Model.

    Inputs:
        X: (batch, time_steps, n_features)

    Outputs:
        'forecast_output': (batch, n_outputs) — continuous forecast
        'stress_output':   (batch, 1)         — stress probability [0, 1]
    """
    inputs = Input(shape=(time_steps, n_features), name="sensor_input")

    # 1. Multi-Scale Dilated Causal Convolutional Front-End
    x_conv = _dilated_conv_block(inputs, filters=conv_filters, kernel_size=kernel_size, dilation_rate=1, dropout=0.2, name_prefix="tcn_b1")
    x_conv = _dilated_conv_block(x_conv, filters=conv_filters, kernel_size=kernel_size, dilation_rate=2, dropout=0.2, name_prefix="tcn_b2")

    # 2. Recurrent Sequence Aggregator
    x_lstm = LSTM(units=lstm_units, return_sequences=False, name="lstm_aggregator")(x_conv)
    shared_repr = Dropout(dropout, name="shared_dropout")(x_lstm)

    # 3. Head 1: Continuous Sensor Forecasting (Regression)
    h_fc = Dense(32, activation="relu", name="fc_forecast")(shared_repr)
    forecast_out = Dense(n_outputs, activation="linear", name="forecast_output")(h_fc)

    # 4. Head 2: Agronomic Stress Detection (Classification)
    s_fc = Dense(32, activation="relu", name="fc_stress")(shared_repr)
    s_drop = Dropout(dropout, name="stress_dropout")(s_fc)
    stress_out = Dense(1, activation="sigmoid", name="stress_output")(s_drop)

    model = Model(
        inputs=inputs,
        outputs={"forecast_output": forecast_out, "stress_output": stress_out},
        name="Proposed_MT_TCN_LSTM",
    )

    # Multi-task loss weights: (1 - λ) on regression MSE, λ on stress BCE
    weight_reg = float(1.0 - lambda_stress)
    weight_cls = float(lambda_stress)

    # Weighted Binary Cross-Entropy if class weighting is supplied
    if class_weight_pos != 1.0:
        def weighted_bce(y_true, y_pred):
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            weight_vector = y_true * class_weight_pos + (1.0 - y_true) * 1.0
            return tf.reduce_mean(bce * weight_vector)
        loss_cls = weighted_bce
    else:
        loss_cls = "binary_crossentropy"

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            "forecast_output": "mse",
            "stress_output": loss_cls,
        },
        loss_weights={
            "forecast_output": weight_reg,
            "stress_output": weight_cls,
        },
        metrics={
            "forecast_output": ["mae"],
            "stress_output": ["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
        },
    )

    return model
