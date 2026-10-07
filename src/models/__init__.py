"""
src/models package — all baseline and candidate architectures.

Exposes:
  - B1: build_cnn_bilstm (Original Baseline)
  - B2: build_lstm       (Vanilla LSTM)
  - B3: build_gru        (Gated Recurrent Unit)
  - B4: build_bilstm     (Bidirectional LSTM)
  - B5: build_tcn        (Temporal Convolutional Network)
  - B6: build_transformer (Temporal Self-Attention Transformer)
  - B7: build_cnn_gru    (CNN + GRU Hybrid)
"""

from src.models.baseline_cnn_bilstm import build_cnn_bilstm
from src.models.lstm_model import build_lstm
from src.models.gru_model import build_gru
from src.models.bilstm_model import build_bilstm
from src.models.tcn_model import build_tcn
from src.models.transformer_model import build_transformer
from src.models.cnn_gru_model import build_cnn_gru
from src.models.proposed_model import build_proposed_model

__all__ = [
    "build_cnn_bilstm",
    "build_lstm",
    "build_gru",
    "build_bilstm",
    "build_tcn",
    "build_transformer",
    "build_cnn_gru",
    "build_proposed_model",
]

