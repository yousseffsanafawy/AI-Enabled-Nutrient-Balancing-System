"""
preprocessing.py — Leakage-free preprocessing pipeline.

Key design decisions (from code audit of AI_PBL (1).ipynb):
  - CONFIRMED ISSUE: Original code fits MinMaxScaler on the FULL dataset before split.
  - CONFIRMED ISSUE: Original code uses test set as validation_data during training.
  - FIX: This module enforces chronological split FIRST, then fits scalers on
    training data only, then transforms val/test using the train-fitted scaler.

Pipeline:
  1. Interpolate (linear) — fills NaN values
  2. Chronological split (raw time-series rows)
  3. Fit MinMaxScaler on train rows only
  4. Transform train / val / test
  5. Create sliding-window sequences per partition separately
  6. Return X_train, y_train, X_val, y_val, X_test, y_test + fitted scalers
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

from src.config import (
    FEATURES,
    TARGET_COLS,
    TIME_STEPS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SCALE_RANGE,
    PROCESSED_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    interpolate: bool = True,
    scaler_save_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Full leakage-free preprocessing pipeline.

    Args:
        df            : Raw DataFrame (chronologically ordered).
        features      : Input feature column names.
        target_cols   : Target column names (same as features for multi-output).
        time_steps    : Lookback window length.
        train_ratio   : Fraction of raw rows for training.
        val_ratio     : Fraction of raw rows for validation.
        interpolate   : If True, apply linear interpolation before splitting.
        scaler_save_path : If given, save fitted scalers to this directory.
        verbose       : Print progress messages.

    Returns:
        dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test  (np.ndarray)
          scaler_X, scaler_y                               (fitted MinMaxScaler)
          split_info                                       (dict with row counts)
    """
    assert abs(train_ratio + val_ratio + (1 - train_ratio - val_ratio) - 1.0) < 1e-6

    df_work = df[features].copy()

    # ── 1. Interpolation (on the full series, before split) ──────────────────
    # Note: interpolation before split is a pragmatic choice for a single
    # continuous run. If multiple independent runs are present, interpolation
    # should be done per-run. This is investigated in Sprint 1 (EDA).
    if interpolate:
        n_before = df_work.isna().sum().sum()
        df_work = df_work.interpolate(method="linear").dropna()
        n_after = df_work.isna().sum().sum()
        if verbose:
            print(f"  Interpolation: {n_before} NaNs filled, {n_after} remaining.")

    n_rows = len(df_work)

    # ── 2. Chronological split (on raw rows, BEFORE scaling/sequencing) ───────
    train_end = int(n_rows * train_ratio)
    val_end   = int(n_rows * (train_ratio + val_ratio))

    train_raw = df_work.iloc[:train_end].values
    val_raw   = df_work.iloc[train_end:val_end].values
    test_raw  = df_work.iloc[val_end:].values

    if verbose:
        print(f"\n  Split (rows):")
        print(f"    Train : {len(train_raw):,} ({100*len(train_raw)/n_rows:.1f}%)")
        print(f"    Val   : {len(val_raw):,} ({100*len(val_raw)/n_rows:.1f}%)")
        print(f"    Test  : {len(test_raw):,} ({100*len(test_raw)/n_rows:.1f}%)")

    # ── 3. Fit scalers on TRAINING data ONLY ─────────────────────────────────
    # This is the critical fix vs the original code.
    scaler_X = MinMaxScaler(feature_range=SCALE_RANGE)
    scaler_y = MinMaxScaler(feature_range=SCALE_RANGE)

    # Features and targets are the same columns; fit independently to allow
    # independent inverse-transforming of predictions later.
    train_X_scaled = scaler_X.fit_transform(train_raw)
    train_y_scaled = scaler_y.fit_transform(train_raw)

    # ── 4. Transform val/test using the train-fitted scaler ──────────────────
    val_X_scaled  = scaler_X.transform(val_raw)
    val_y_scaled  = scaler_y.transform(val_raw)
    test_X_scaled = scaler_X.transform(test_raw)
    test_y_scaled = scaler_y.transform(test_raw)

    # ── 5. Create sliding-window sequences per partition ─────────────────────
    X_train, y_train = _create_sequences(train_X_scaled, train_y_scaled, time_steps)
    X_val,   y_val   = _create_sequences(val_X_scaled,   val_y_scaled,   time_steps)
    X_test,  y_test  = _create_sequences(test_X_scaled,  test_y_scaled,  time_steps)

    if verbose:
        print(f"\n  Sequence shapes:")
        print(f"    X_train: {X_train.shape}  y_train: {y_train.shape}")
        print(f"    X_val  : {X_val.shape}  y_val  : {y_val.shape}")
        print(f"    X_test : {X_test.shape}  y_test : {y_test.shape}")

    # ── 6. Optionally save scalers ────────────────────────────────────────────
    if scaler_save_path:
        os.makedirs(scaler_save_path, exist_ok=True)
        joblib.dump(scaler_X, os.path.join(scaler_save_path, "scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(scaler_save_path, "scaler_y.pkl"))
        if verbose:
            print(f"\n  Scalers saved to: {scaler_save_path}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "split_info": {
            "n_rows_total":     n_rows,
            "train_rows":       len(train_raw),
            "val_rows":         len(val_raw),
            "test_rows":        len(test_raw),
            "train_sequences":  len(X_train),
            "val_sequences":    len(X_val),
            "test_sequences":   len(X_test),
            "time_steps":       time_steps,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL PIPELINE (preserved for reproduction / Sprint 2 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_original(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    train_ratio: float = 0.80,   # original used 80/20
    verbose: bool = True,
) -> dict:
    """
    Reproduce the ORIGINAL preprocessing from AI_PBL (1).ipynb.

    KNOWN ISSUES (confirmed from source code audit):
      - Scaler is fit on the full dataset before splitting (scaling leakage).
      - There is no separate validation set (test set used as val during training).
      - Sequences are created before the split, then the split is applied to sequences.

    This function is preserved ONLY for reproduction/comparison in Sprint 2.
    Do NOT use this for new model training.
    """
    df_work = df[features].copy()

    # 1. Interpolate on full dataset
    df_work = df_work.interpolate(method="linear").dropna()

    # 2. Fit scaler on FULL dataset (the confirmed leakage point)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(df_work.values)
    scaled_y = scaler_y.fit_transform(df_work.values)

    # 3. Generate sequences from the full scaled dataset
    X_seq, y_seq = _create_sequences(scaled_X, scaled_y, time_steps)

    # 4. Chronological 80/20 split (applied to sequences, not raw rows)
    split_idx = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    if verbose:
        print(f"  [ORIGINAL pipeline] X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  WARNING: This pipeline has known evaluation issues (scaling leakage, no val set).")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _create_sequences(
    data_X: np.ndarray,
    data_y: np.ndarray,
    time_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences (one-step-ahead forecasting).

    Window: data_X[i : i+time_steps]  →  target: data_y[i + time_steps]

    Args:
        data_X    : Scaled feature array of shape (N, n_features).
        data_y    : Scaled target array of shape (N, n_targets).
        time_steps: Number of historical steps per input window.

    Returns:
        X : np.ndarray of shape (N - time_steps, time_steps, n_features)
        y : np.ndarray of shape (N - time_steps, n_targets)
    """
    X, y = [], []
    for i in range(len(data_X) - time_steps):
        X.append(data_X[i : i + time_steps])
        y.append(data_y[i + time_steps])
    return np.array(X), np.array(y)
