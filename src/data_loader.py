"""
data_loader.py — Dataset loading utilities.

Responsibilities:
  - Load raw CSV(s) into a DataFrame
  - Validate columns and dtypes
  - Detect timestamps (if present) and sort chronologically
  - Report basic shape/column info
"""

import os
import pandas as pd
from src.config import RAW_CSV, FEATURES


def load_raw(
    filepath: str = RAW_CSV,
    expected_features: list = FEATURES,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the raw hydroponics CSV dataset.

    Args:
        filepath         : Absolute path to the CSV file.
        expected_features: List of feature column names that must be present.
        verbose          : If True, print a brief summary after loading.

    Returns:
        pd.DataFrame with at least the expected feature columns.

    Raises:
        FileNotFoundError : If the CSV does not exist.
        ValueError        : If one or more expected feature columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            f"Make sure data/raw/ contains the CSV."
        )

    df = pd.read_csv(filepath)

    # ── Validate columns ──────────────────────────────────────────────────────
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Actual columns: {list(df.columns)}"
        )

    # ── Detect and handle timestamp column ────────────────────────────────────
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        ts_col = ts_candidates[0]
        ts_parsed = False
        # Try standard datetime string parsing
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
            df = df.sort_values(ts_col).reset_index(drop=True)
            if verbose:
                print(f"  Timestamp column detected: '{ts_col}'")
                print(f"  Range: {df[ts_col].min()} --> {df[ts_col].max()}")
            ts_parsed = True
        except Exception:
            pass
        # Try Unix epoch (seconds or milliseconds)
        if not ts_parsed:
            try:
                sample = float(str(df[ts_col].dropna().iloc[0]).replace(",", ""))
                unit = "ms" if sample > 1e10 else "s"
                df[ts_col] = pd.to_datetime(df[ts_col], unit=unit, errors="coerce")
                if df[ts_col].notna().sum() > 0:
                    df = df.sort_values(ts_col).reset_index(drop=True)
                    if verbose:
                        print(f"  Timestamp column '{ts_col}' parsed as Unix epoch ({unit}).")
                    ts_parsed = True
            except Exception:
                pass
        if not ts_parsed and verbose:
            print(f"  Column '{ts_col}' found but could not be parsed as datetime.")
            print("  Assuming rows are already in chronological order.")
    else:
        if verbose:
            print("  No timestamp column detected. Assuming rows are in chronological order.")


    if verbose:
        _print_summary(df, expected_features)

    return df


def _print_summary(df: pd.DataFrame, features: list) -> None:
    """Print a brief summary of the loaded DataFrame."""
    sep = "=" * 50
    print("\n" + sep)
    print("  DATA LOADER SUMMARY")
    print(sep)
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {len(df.columns)}")
    print(f"  Feature cols  : {features}")
    print(f"  Memory usage  : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("\n  Missing values per feature column:")
    for col in features:
        n_miss = df[col].isna().sum()
        pct = 100 * n_miss / len(df)
        print(f"    {col:<20} {n_miss:>6,} ({pct:.2f}%)")
    print(sep + "\n")
