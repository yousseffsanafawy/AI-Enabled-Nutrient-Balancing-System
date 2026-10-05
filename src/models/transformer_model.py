"""
models/transformer_model.py — Temporal Self-Attention / Transformer Baseline (Model B6).

Multi-Head Self-Attention encoder for temporal sequences.
Used for:
  - Sprint 4: Baseline in benchmark table (B6)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from typing import Optional

from src.config import (
    N_FEATURES, TIME_STEPS,
    DROPOUT, LEARNING_RATE, LOSS,
)


def _transformer_encoder(inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float, block_idx: int):
    """Single Transformer Encoder block."""
    # Multi-head attention
    attn_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout, name=f"mha_{block_idx}"
    )(inputs, inputs)
    attn_output = Dropout(dropout, name=f"attn_drop_{block_idx}")(attn_output)
    x1 = Add(name=f"attn_add_{block_idx}")([inputs, attn_output])
    x1 = LayerNormalization(epsilon=1e-6, name=f"attn_ln_{block_idx}")(x1)

    # Feed-forward network
    ffn_output = Dense(ff_dim, activation="relu", name=f"ffn_1_{block_idx}")(x1)
    ffn_output = Dropout(dropout, name=f"ffn_drop_{block_idx}")(ffn_output)
    ffn_output = Dense(inputs.shape[-1], name=f"ffn_2_{block_idx}")(ffn_output)
    x2 = Add(name=f"ffn_add_{block_idx}")([x1, ffn_output])
    return LayerNormalization(epsilon=1e-6, name=f"ffn_ln_{block_idx}")(x2)


def build_transformer(
    time_steps:      int   = TIME_STEPS,
    n_features:      int   = N_FEATURES,
    n_outputs:       int   = N_FEATURES,
    head_size:       int   = 32,
    num_heads:       int   = 2,
    ff_dim:          int   = 64,
    num_blocks:      int   = 2,
    dropout:         float = DROPOUT,
    learning_rate:   float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Build Temporal Transformer Encoder Model (B6).

    Input shape : (batch, time_steps, n_features)
    Output shape: (batch, n_outputs)
    """
    inputs = Input(shape=(time_steps, n_features), name="input_transformer")
    
    # Project input features to embedding dimension if needed
    x = Dense(head_size * num_heads, name="feature_projection")(inputs)

    # Stack encoder blocks
    for i in range(num_blocks):
        x = _transformer_encoder(
            x, head_size=head_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, block_idx=i
        )

    # Global average pooling over temporal sequence
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = Dense(n_outputs, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer_Baseline")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS,
        metrics=["mae"],
    )
    return model
