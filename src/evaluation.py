"""
evaluation.py — Unified evaluation metrics for all models.

Covers:
  - Regression forecasting metrics per sensor (MAE, RMSE, R², MAPE)
  - Tolerance-based within-tolerance rate (the custom metric from the original code)
  - Stress classification metrics (Precision, Recall, F1, confusion matrix)
  - A summary report printer
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import Optional, Dict, Any

from src.config import FEATURES, TOLERANCES


# -----------------------------------------------------------------------------
# REGRESSION METRICS
# -----------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list = FEATURES,
    tolerances: dict = TOLERANCES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute per-sensor and aggregate regression metrics.

    Args:
        y_true        : Ground truth array of shape (N, n_sensors). Real-world units.
        y_pred        : Predicted array of shape (N, n_sensors). Real-world units.
        feature_names : Ordered list of sensor names.
        tolerances    : Dict mapping sensor name → acceptable absolute error threshold.
        verbose       : If True, print the results table.

    Returns:
        Dict mapping sensor name (and 'aggregate') → metric dict.
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
    )

    results = {}

    for i, name in enumerate(feature_names):
        actual = y_true[:, i]
        pred   = y_pred[:, i]

        mae  = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2   = r2_score(actual, pred)

        # MAPE — guarded against division by zero
        nonzero_mask = actual != 0
        mape = (
            float(np.mean(np.abs((actual[nonzero_mask] - pred[nonzero_mask]) / actual[nonzero_mask])) * 100)
            if nonzero_mask.any() else float("nan")
        )

        # Within-tolerance rate (the original project's "accuracy" metric)
        tol = tolerances.get(name, None)
        if tol is not None:
            within_tol = float(np.mean(np.abs(actual - pred) <= tol) * 100)
        else:
            within_tol = None

        results[name] = {
            "MAE":   round(mae, 6),
            "RMSE":  round(rmse, 6),
            "R2":    round(r2, 6),
            "MAPE":  round(mape, 4) if not np.isnan(mape) else None,
            "within_tolerance_pct": round(within_tol, 4) if within_tol is not None else None,
            "tolerance_used": tol,
        }

    # Aggregate across all sensors
    results["aggregate"] = {
        "mean_MAE":  round(float(np.mean([results[n]["MAE"]  for n in feature_names])), 6),
        "mean_RMSE": round(float(np.mean([results[n]["RMSE"] for n in feature_names])), 6),
        "mean_R2":   round(float(np.mean([results[n]["R2"]   for n in feature_names])), 6),
    }

    if verbose:
        _print_regression_report(results, feature_names)

    return results


def _print_regression_report(results: dict, feature_names: list) -> None:
    """Pretty-print regression metrics table."""
    print("\n" + "=" * 75)
    print("  REGRESSION METRICS (real-world sensor units)")
    print("=" * 75)
    header = f"  {'Sensor':<18} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8} {'Within-Tol%':>13}"
    print(header)
    print("  " + "-" * 73)
    for name in feature_names:
        r = results[name]
        mape_str = f"{r['MAPE']:>8.4f}" if r["MAPE"] is not None else "     N/A"
        tol_str  = f"{r['within_tolerance_pct']:>12.2f}%" if r["within_tolerance_pct"] is not None else "          N/A"
        print(f"  {name:<18} {r['MAE']:>10.6f} {r['RMSE']:>10.6f} {r['R2']:>8.4f} {mape_str} {tol_str}")
    print("  " + "-" * 73)
    agg = results["aggregate"]
    print(f"  {'AGGREGATE':<18} {agg['mean_MAE']:>10.6f} {agg['mean_RMSE']:>10.6f} {agg['mean_R2']:>8.4f}")
    print("=" * 75 + "\n")


# -----------------------------------------------------------------------------
# STRESS CLASSIFICATION METRICS
# -----------------------------------------------------------------------------

def compute_stress_metrics(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    class_names: Optional[list] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute classification metrics for the stress detection head.

    Args:
        y_true_labels : Ground truth integer class labels (N,).
        y_pred_labels : Predicted integer class labels (N,).
        class_names   : Optional list of class name strings.
        verbose       : If True, print the report.

    Returns:
        Dict with precision, recall, F1 per class and macro-averaged.
    """
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    results = {
        "per_class": {
            (class_names[i] if class_names else str(i)): {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "F1":        round(float(f1[i]),   4),
                "support":   int(support[i]),
            }
            for i in range(len(prec))
        },
        "macro_F1":        round(float(macro_f1),   4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall":    round(float(macro_rec),  4),
        "confusion_matrix": cm.tolist(),
    }

    if verbose:
        _print_stress_report(results, class_names)

    return results


def _print_stress_report(results: dict, class_names: Optional[list]) -> None:
    """Pretty-print stress classification report."""
    print("\n" + "=" * 60)
    print("  STRESS CLASSIFICATION METRICS")
    print("=" * 60)
    for cls, m in results["per_class"].items():
        print(f"  {cls:<20}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['F1']:.4f}  n={m['support']}")
    print("  " + "-" * 58)
    print(f"  {'MACRO':<20}  P={results['macro_precision']:.4f}  R={results['macro_recall']:.4f}  F1={results['macro_F1']:.4f}")
    print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# PERSISTENCE BASELINE
# -----------------------------------------------------------------------------

def persistence_baseline(X_test: np.ndarray, scaler_y) -> np.ndarray:
    """
    Predict the last observed value as the forecast (naive baseline).

    For each sample in X_test (shape: N, time_steps, n_features),
    the last timestep's values are used as the prediction.

    Returns predictions in real-world units (inverse-scaled).
    """
    last_step_scaled = X_test[:, -1, :]             # (N, n_features)
    return scaler_y.inverse_transform(last_step_scaled)
