"""
preprocessing.py — Leakage-free preprocessing pipeline and splitting strategies.

Key design principles (from code audit & Sprint 3 protocol):
  1. LEAKAGE PREVENTION:
     - Scaler (MinMaxScaler) is fit strictly on TRAINING data only.
     - Val and Test are transformed using parameters learned strictly from Train.
     - Validation set is strictly separate from Test set (70% train / 10% val / 20% test).
  2. BOUNDARY-AWARE SEQUENCE SLICING:
     - The physical IoT dataset contains multi-hour shutdowns (up to 24.7h).
     - Sequences must NEVER bridge across system shutdowns or split boundaries.
     - Sliding windows are generated strictly within contiguous operational segments.
  3. SPLIT STRATEGIES FOR COMPARATIVE AUDIT:
     - "chronological" : Strict 70/10/20 chronological split with boundary awareness.
     - "random"        : Negative control showing severe temporal data leakage.
     - "group"         : Session-based split holding out entire operational runs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES,
    TARGET_COLS,
    TIME_STEPS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SCALE_RANGE,
    RANDOM_SEED,
    PROCESSED_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. GAP DETECTION & SEGMENT IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_session_segments(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    max_gap_seconds: float = 60.0,
) -> List[Tuple[int, int]]:
    """
    Identify contiguous operational segments separated by gaps > max_gap_seconds.

    Args:
        df              : DataFrame containing a timestamp column.
        timestamp_col   : Name of the timestamp column.
        max_gap_seconds : Threshold in seconds beyond which a gap is declared.

    Returns:
        List of (start_row_idx, end_row_idx) tuples (half-open: [start, end)).
    """
    if timestamp_col not in df.columns:
        return [(0, len(df))]

    dt = pd.to_datetime(df[timestamp_col])
    dt_diff = dt.diff()
    gap_locs = df.index[dt_diff > pd.Timedelta(seconds=max_gap_seconds)].tolist()

    # Convert to relative 0-indexed positions if index is non-standard
    if isinstance(gap_locs[0] if gap_locs else 0, int) and (df.index == pd.RangeIndex(len(df))).all():
        gap_indices = gap_locs
    else:
        gap_indices = [df.index.get_loc(idx) for idx in gap_locs]

    cut_points = [0] + gap_indices + [len(df)]
    segments = []
    for i in range(len(cut_points) - 1):
        start = cut_points[i]
        end = cut_points[i + 1]
        if end > start:
            segments.append((start, end))

    return segments


def create_boundary_aware_sequences(
    data_X: np.ndarray,
    data_y: np.ndarray,
    time_steps: int,
    segments: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding-window sequences strictly within contiguous operational segments.
    Guarantees no sequence bridges across system shutdowns or split boundaries.

    Args:
        data_X     : Feature array of shape (N, n_features).
        data_y     : Target array of shape (N, n_targets).
        time_steps : Lookback window size.
        segments   : List of (start_idx, end_idx) tuples. If None, treats data as 1 segment.

    Returns:
        X : np.ndarray of shape (num_valid_seqs, time_steps, n_features)
        y : np.ndarray of shape (num_valid_seqs, n_targets)
    """
    if segments is None:
        segments = [(0, len(data_X))]

    X_list, y_list = [], []

    for start, end in segments:
        seg_len = end - start
        if seg_len > time_steps:
            for i in range(start, end - time_steps):
                X_list.append(data_X[i : i + time_steps])
                y_list.append(data_y[i + time_steps])

    if len(X_list) == 0:
        n_feat = data_X.shape[1] if data_X.ndim > 1 else 1
        n_targ = data_y.shape[1] if data_y.ndim > 1 else 1
        return np.empty((0, time_steps, n_feat)), np.empty((0, n_targ))

    return np.array(X_list), np.array(y_list)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT STRATEGY IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def split_chronological(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    max_gap_seconds: float = 60.0,
    interpolate: bool = True,
    scale_range: Tuple[float, float] = SCALE_RANGE,
    scaler_type: str = "minmax",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Strict Chronological Split (70% train / 10% val / 20% test).
    
    Guarantees:
      - Split is performed FIRST on raw chronologically ordered rows.
      - Scaler is fitted ONLY on the train partition.
      - Val and Test are transformed with train scaler.
      - Sequences are created within partition segments, never crossing boundaries or gaps.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "Sum of train and val ratio must be < 1.0"

    df_clean = df.copy()
    if interpolate:
        df_clean[features] = df_clean[features].interpolate(method="linear").bfill().ffill()

    n_rows = len(df_clean)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df_clean.iloc[:train_end].copy()
    val_df = df_clean.iloc[train_end:val_end].copy()
    test_df = df_clean.iloc[val_end:].copy()

    # Fit scalers STRICTLY on training data
    if scaler_type.lower() in ("standard", "zscore", "standardscaler"):
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    else:
        scaler_X = MinMaxScaler(feature_range=scale_range)
        scaler_y = MinMaxScaler(feature_range=scale_range)

    train_X_raw = train_df[features].values
    train_y_raw = train_df[target_cols].values
    val_X_raw = val_df[features].values
    val_y_raw = val_df[target_cols].values
    test_X_raw = test_df[features].values
    test_y_raw = test_df[target_cols].values

    train_X_scaled = scaler_X.fit_transform(train_X_raw)
    train_y_scaled = scaler_y.fit_transform(train_y_raw)

    val_X_scaled = scaler_X.transform(val_X_raw)
    val_y_scaled = scaler_y.transform(val_y_raw)

    test_X_scaled = scaler_X.transform(test_X_raw)
    test_y_scaled = scaler_y.transform(test_y_raw)

    # Detect gaps within each partition so sequences don't bridge internal shutdowns
    train_segs = detect_session_segments(train_df, max_gap_seconds=max_gap_seconds)
    val_segs = detect_session_segments(val_df, max_gap_seconds=max_gap_seconds)
    test_segs = detect_session_segments(test_df, max_gap_seconds=max_gap_seconds)

    X_train, y_train = create_boundary_aware_sequences(train_X_scaled, train_y_scaled, time_steps, train_segs)
    X_val, y_val = create_boundary_aware_sequences(val_X_scaled, val_y_scaled, time_steps, val_segs)
    X_test, y_test = create_boundary_aware_sequences(test_X_scaled, test_y_scaled, time_steps, test_segs)

    if verbose:
        print("\n  [Chronological Split]")
        print(f"    Train rows: {len(train_df):,} ({100*len(train_df)/n_rows:.1f}%) -> {len(X_train):,} sequences")
        print(f"    Val rows:   {len(val_df):,} ({100*len(val_df)/n_rows:.1f}%) -> {len(X_val):,} sequences")
        print(f"    Test rows:  {len(test_df):,} ({100*len(test_df)/n_rows:.1f}%) -> {len(X_test):,} sequences")
        print(f"    Total sequences: {len(X_train) + len(X_val) + len(X_test):,}")

    return {
        "strategy": "chronological",
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "split_info": {
            "n_rows_total": n_rows,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
            "time_steps": time_steps,
        },
    }


def split_random(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_SEED,
    interpolate: bool = True,
    scale_range: Tuple[float, float] = SCALE_RANGE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Random Split (Negative Control / Audit Baseline).

    Demonstrates temporal data leakage:
      - Sequences are generated from the continuous series, then shuffled randomly.
      - Training sequences and test sequences share 14/15 timesteps of overlap.
      - Artificially inflates performance because the model memorizes local transitions.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    df_clean = df.copy()
    if interpolate:
        df_clean[features] = df_clean[features].interpolate(method="linear").bfill().ffill()

    # Fit scaler on full data to mirror common naive leakage or fit on train
    scaler_X = MinMaxScaler(feature_range=scale_range)
    scaler_y = MinMaxScaler(feature_range=scale_range)

    raw_X = df_clean[features].values
    raw_y = df_clean[target_cols].values

    scaled_X = scaler_X.fit_transform(raw_X)
    scaled_y = scaler_y.fit_transform(raw_y)

    # Generate all sequences
    X_all, y_all = create_boundary_aware_sequences(scaled_X, scaled_y, time_steps)
    n_seq = len(X_all)

    # Randomly shuffle sequences
    indices = np.arange(n_seq)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_end = int(n_seq * train_ratio)
    val_end = int(n_seq * (train_ratio + val_ratio))

    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_test, y_test = X_all[idx_test], y_all[idx_test]

    if verbose:
        print("\n  [Random Split (Negative Control)]")
        print(f"    Train sequences: {len(X_train):,} ({100*len(X_train)/n_seq:.1f}%)")
        print(f"    Val sequences:   {len(X_val):,} ({100*len(X_val)/n_seq:.1f}%)")
        print(f"    Test sequences:  {len(X_test):,} ({100*len(X_test)/n_seq:.1f}%)")
        print(f"    NOTE: Severe temporal leakage expected due to sequence overlap!")

    return {
        "strategy": "random",
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "split_info": {
            "n_rows_total": len(df_clean),
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
            "time_steps": time_steps,
        },
    }


def split_group_sessions(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    session_gap_seconds: float = 3600.0,
    interpolate: bool = True,
    scale_range: Tuple[float, float] = SCALE_RANGE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Session-Based / Group Split (Physical Run Holdout).

    Divides the dataset into distinct continuous operational sessions based on
    system shutdown gaps (> 1 hour).
    
    Session Partitioning:
      - Train Sessions (Sessions 1-4): Rows 0..12884 (12,885 rows, 50.4%)
        Covers initial filling, Day 1-2 operations, Day 3 manual dosing jump (row 4244).
      - Val Session (Session 5): Rows 12885..13469 (585 rows, 2.3%)
        Morning operational settling on Dec 25.
      - Test Session (Session 6): Rows 13470..25569 (12,100 rows, 47.3%)
        Completely held-out operational run under altered water level and stabilized regime!
    """
    df_clean = df.copy()
    if interpolate:
        df_clean[features] = df_clean[features].interpolate(method="linear").bfill().ffill()

    # Identify major sessions (> 1h gap)
    sessions = detect_session_segments(df_clean, max_gap_seconds=session_gap_seconds)
    if verbose:
        print(f"\n  [Group/Session Split] Identified {len(sessions)} physical sessions (> 1h gaps):")
        for i, (s, e) in enumerate(sessions):
            print(f"    Session {i+1}: rows {s}..{e-1} ({e-s} rows)")

    # Assign sessions
    # Sessions 1-4 for train, Session 5 for val, Session 6 for test
    train_slices = sessions[:4]
    val_slices = [sessions[4]] if len(sessions) > 4 else [sessions[-1]]
    test_slices = sessions[5:] if len(sessions) > 5 else [sessions[-1]]

    train_dfs = [df_clean.iloc[s:e] for s, e in train_slices]
    val_dfs = [df_clean.iloc[s:e] for s, e in val_slices]
    test_dfs = [df_clean.iloc[s:e] for s, e in test_slices]

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    # Fit scalers STRICTLY on training sessions
    scaler_X = MinMaxScaler(feature_range=scale_range)
    scaler_y = MinMaxScaler(feature_range=scale_range)

    scaler_X.fit(train_df[features].values)
    scaler_y.fit(train_df[target_cols].values)

    # Generate boundary-aware sequences session by session
    def _seqs_from_session_slices(session_list):
        X_parts, y_parts = [], []
        for s, e in session_list:
            sdf = df_clean.iloc[s:e]
            s_scaled_X = scaler_X.transform(sdf[features].values)
            s_scaled_y = scaler_y.transform(sdf[target_cols].values)
            # Within this session, check for any smaller gaps (> 60s)
            inner_segs = detect_session_segments(sdf, max_gap_seconds=60.0)
            sx, sy = create_boundary_aware_sequences(s_scaled_X, s_scaled_y, time_steps, inner_segs)
            if len(sx) > 0:
                X_parts.append(sx)
                y_parts.append(sy)
        return (np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, time_steps, len(features))),
                np.concatenate(y_parts, axis=0) if y_parts else np.empty((0, len(target_cols))))

    X_train, y_train = _seqs_from_session_slices(train_slices)
    X_val, y_val = _seqs_from_session_slices(val_slices)
    X_test, y_test = _seqs_from_session_slices(test_slices)

    if verbose:
        print(f"    Train sequences: {len(X_train):,} ({len(train_df):,} rows)")
        print(f"    Val sequences:   {len(X_val):,} ({len(val_df):,} rows)")
        print(f"    Test sequences:  {len(X_test):,} ({len(test_df):,} rows)")

    return {
        "strategy": "group_session",
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "split_info": {
            "n_rows_total": len(df_clean),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
            "time_steps": time_steps,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    split_strategy: str = "chronological",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    max_gap_seconds: float = 60.0,
    interpolate: bool = True,
    scaler_type: str = "minmax",
    scaler_save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point for preprocessing with explicit split strategy.

    Args:
        df               : Raw DataFrame.
        features         : Feature columns.
        target_cols      : Target columns.
        time_steps       : Lookback window length.
        split_strategy   : "chronological" (recommended), "random" (negative control), or "group".
        train_ratio      : Train ratio (for chronological and random).
        val_ratio        : Val ratio (for chronological and random).
        max_gap_seconds  : Maximum gap in seconds before segment reset.
        interpolate      : Whether to interpolate missing values.
        scaler_type      : "minmax" or "standard" (Z-score normalization).
        scaler_save_path : Optional directory to save fitted scalers.
        verbose          : Whether to print logging.

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test,
        scaler_X, scaler_y, and split_info.
    """
    if split_strategy == "chronological":
        res = split_chronological(
            df=df, features=features, target_cols=target_cols,
            time_steps=time_steps, train_ratio=train_ratio, val_ratio=val_ratio,
            max_gap_seconds=max_gap_seconds, interpolate=interpolate,
            scaler_type=scaler_type, verbose=verbose,
        )
    elif split_strategy == "random":
        res = split_random(
            df=df, features=features, target_cols=target_cols,
            time_steps=time_steps, train_ratio=train_ratio, val_ratio=val_ratio,
            interpolate=interpolate, verbose=verbose,
        )
    elif split_strategy in ("group", "group_session", "session"):
        res = split_group_sessions(
            df=df, features=features, target_cols=target_cols,
            time_steps=time_steps, session_gap_seconds=3600.0,
            interpolate=interpolate, verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}. Expected 'chronological', 'random', or 'group'.")

    if scaler_save_path:
        os.makedirs(scaler_save_path, exist_ok=True)
        joblib.dump(res["scaler_X"], os.path.join(scaler_save_path, "scaler_X.pkl"))
        joblib.dump(res["scaler_y"], os.path.join(scaler_save_path, "scaler_y.pkl"))
        if verbose:
            print(f"  Scalers saved to: {scaler_save_path}")

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORIGINAL PIPELINE (Preserved for reproduction / comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_original(
    df: pd.DataFrame,
    features: list = FEATURES,
    target_cols: list = TARGET_COLS,
    time_steps: int = TIME_STEPS,
    train_ratio: float = 0.80,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Reproduces the original naive preprocessing from AI_PBL (1).ipynb.
    Used exclusively for baseline audit comparison.
    """
    df_work = df[features].copy()
    df_work = df_work.interpolate(method="linear").dropna()

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(df_work.values)
    scaled_y = scaler_y.fit_transform(df_work.values)

    X_seq, y_seq = [], []
    for i in range(len(scaled_X) - time_steps):
        X_seq.append(scaled_X[i : i + time_steps])
        y_seq.append(scaled_y[i + time_steps])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    split_idx = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    if verbose:
        print(f"  [ORIGINAL naive pipeline] X_train: {X_train.shape}, X_test: {X_test.shape}")
        print("  WARNING: Has known scaling leakage and evaluation leakage.")

    return {
        "strategy": "original_naive",
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
    }
