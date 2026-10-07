"""
uncertainty.py — Monte Carlo Dropout Uncertainty Estimation & Prediction Intervals.

Sprint 5 & Sprint 7 Implementation:
  - Evaluates epistemic uncertainty by keeping dropout active at test time (training=True).
  - Batched stochastic inference for memory-efficient scaling over large test partitions.
  - Computes:
    * Predictive Mean (point forecast)
    * Predictive Standard Deviation (epistemic uncertainty metric σ)
    * Prediction Intervals (50%, 80%, 90%, 95% nominal confidence bounds)
    * Calibration metrics: Pearson r, Spearman rank ρ, and reliability binning
    * Calibration of operational thresholds (MED_THRESHOLD, HIGH_THRESHOLD) on validation data
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any, Optional, List
from scipy.stats import spearmanr


def predict_with_mc_dropout(
    model: tf.keras.Model,
    X: np.ndarray,
    n_samples: int = 50,
    batch_size: int = 128,
    forecast_output_key: Optional[str] = "forecast_output",
) -> Dict[str, np.ndarray]:
    """
    Perform Monte Carlo Dropout inference by running N forward passes with training=True.
    Processes X in batches to ensure robust memory management.

    Args:
        model: Trained Keras model with Dropout layers.
        X: Input array of shape (N_samples, time_steps, n_features).
        n_samples: Number of stochastic forward passes (default: 50).
        batch_size: Batch size for batched forward passes.
        forecast_output_key: For multi-task models, key for the regression output.

    Returns:
        Dict with:
          'mean': Mean prediction array (N_samples, n_outputs)
          'std': Epistemic uncertainty array (N_samples, n_outputs)
          'lower_90': 5th percentile array (N_samples, n_outputs)
          'upper_90': 95th percentile array (N_samples, n_outputs)
          'samples': Full tensor of MC samples (n_samples, N_samples, n_outputs)
    """
    N = len(X)
    n_batches = int(np.ceil(N / batch_size))
    all_batch_samples = []

    for b in range(n_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, N)
        X_batch = X[start_idx:end_idx]

        batch_mc_passes = []
        for _ in range(n_samples):
            preds = model(X_batch, training=True)
            if isinstance(preds, dict):
                preds = preds[forecast_output_key]
            elif isinstance(preds, (list, tuple)):
                preds = preds[0]
            batch_mc_passes.append(preds.numpy())

        # Shape: (n_samples, batch_len, n_outputs)
        batch_mc_tensor = np.array(batch_mc_passes)
        all_batch_samples.append(batch_mc_tensor)

    # Concatenate across batch dimension: (n_samples, N, n_outputs)
    mc_tensor = np.concatenate(all_batch_samples, axis=1)

    # Compute summary statistics across MC forward passes
    pred_mean = np.mean(mc_tensor, axis=0)
    pred_std  = np.std(mc_tensor, axis=0)
    lower_90  = np.percentile(mc_tensor, 5, axis=0)
    upper_90  = np.percentile(mc_tensor, 95, axis=0)

    return {
        "mean": pred_mean,
        "std": pred_std,
        "lower_90": lower_90,
        "upper_90": upper_90,
        "samples": mc_tensor,
    }


def compute_prediction_intervals(
    mc_samples: np.ndarray,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute empirical quantile prediction intervals for multiple confidence levels.

    Args:
        mc_samples: Array of shape (n_samples, N, n_outputs).
        confidence_levels: List of nominal confidence levels, e.g. [0.50, 0.80, 0.90, 0.95].

    Returns:
        Dict mapping level string (e.g. '90%') to {'lower': np.ndarray, 'upper': np.ndarray}.
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.80, 0.90, 0.95]

    intervals = {}
    for cl in confidence_levels:
        alpha = 1.0 - cl
        q_lower = (alpha / 2.0) * 100.0
        q_upper = (1.0 - alpha / 2.0) * 100.0

        lower_bound = np.percentile(mc_samples, q_lower, axis=0)
        upper_bound = np.percentile(mc_samples, q_upper, axis=0)

        key = f"{int(cl * 100)}%"
        intervals[key] = {
            "lower": lower_bound,
            "upper": upper_bound,
            "nominal_coverage": cl,
        }
    return intervals


def evaluate_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    feature_names: Optional[list] = None,
    mc_samples: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Assess whether predicted uncertainty σ correlates with actual forecasting error.
    Computes Pearson r, Spearman rank ρ, and coverage across multiple confidence intervals.
    """
    abs_errors = np.abs(y_true - y_pred_mean)
    n_features = y_true.shape[1] if y_true.ndim > 1 else 1

    calibration_results = {}
    pearson_corrs = []
    spearman_corrs = []

    # Optional multi-interval coverage
    intervals = compute_prediction_intervals(mc_samples) if mc_samples is not None else None

    for i in range(n_features):
        err_i = abs_errors[:, i]
        std_i = y_pred_std[:, i]
        name = feature_names[i] if feature_names else f"feature_{i}"

        # Pearson correlation
        if np.std(std_i) > 1e-8 and np.std(err_i) > 1e-8:
            p_corr = float(np.corrcoef(err_i, std_i)[0, 1])
            s_corr, _ = spearmanr(err_i, std_i)
            s_corr = float(s_corr)
        else:
            p_corr = 0.0
            s_corr = 0.0

        pearson_corrs.append(p_corr)
        spearman_corrs.append(s_corr)

        # Standard 95% interval via Gaussian approximation (±1.96 σ)
        in_bounds_95 = err_i <= (1.96 * std_i)
        gauss_cov_95 = float(np.mean(in_bounds_95) * 100.0)

        # Empirical quantile coverages if MC samples provided
        empirical_coverages = {}
        if intervals is not None:
            for cl_name, bnds in intervals.items():
                low = bnds["lower"][:, i]
                upp = bnds["upper"][:, i]
                true_vals = y_true[:, i]
                covered = (true_vals >= low) & (true_vals <= upp)
                empirical_coverages[cl_name] = round(float(np.mean(covered) * 100.0), 2)

        # Decile Error-Uncertainty Binned Statistics (Reliability Bins)
        n_bins = 5
        bins = np.linspace(0, 100, n_bins + 1)
        bin_stats = []
        for b_idx in range(n_bins):
            p_low, p_high = bins[b_idx], bins[b_idx + 1]
            q_low = np.percentile(std_i, p_low)
            q_high = np.percentile(std_i, p_high)
            mask = (std_i >= q_low) & (std_i <= q_high)
            if np.any(mask):
                bin_stats.append({
                    "bin_percentile": f"{int(p_low)}-{int(p_high)}%",
                    "mean_uncertainty_sigma": round(float(np.mean(std_i[mask])), 4),
                    "mean_absolute_error": round(float(np.mean(err_i[mask])), 4),
                })

        calibration_results[name] = {
            "pearson_corr": round(p_corr, 4),
            "spearman_corr": round(s_corr, 4),
            "mean_uncertainty_sigma": round(float(np.mean(std_i)), 4),
            "gaussian_coverage_95_pct": round(gauss_cov_95, 2),
            "empirical_coverages": empirical_coverages,
            "reliability_bins": bin_stats,
        }

    calibration_results["mean_pearson_corr"] = round(float(np.mean(pearson_corrs)), 4)
    calibration_results["mean_spearman_corr"] = round(float(np.mean(spearman_corrs)), 4)
    return calibration_results


def calibrate_uncertainty_thresholds(
    val_pred_std: np.ndarray,
    feature_names: List[str],
    med_percentile: float = 75.0,
    high_percentile: float = 95.0,
) -> Dict[str, Dict[str, float]]:
    """
    Empirically calibrate MED_THRESHOLD and HIGH_THRESHOLD per sensor using the Validation Set.
    Strictly prevents data leakage by establishing operational gates before observing test data.

    Args:
        val_pred_std: Array of predictive std deviations on validation partition (N_val, n_features).
        feature_names: List of target sensor names.
        med_percentile: Percentile for human oversight alert (default 75th percentile).
        high_percentile: Percentile for automated shutdown / abort dosing (default 95th percentile).

    Returns:
        Dict mapping sensor name to {'med_threshold': float, 'high_threshold': float}.
    """
    thresholds = {}
    for i, name in enumerate(feature_names):
        std_i = val_pred_std[:, i]
        med_val = float(np.percentile(std_i, med_percentile))
        high_val = float(np.percentile(std_i, high_percentile))
        thresholds[name] = {
            "med_threshold": round(med_val, 4),
            "high_threshold": round(high_val, 4),
            "calibrated_on_percentiles": f"P{int(med_percentile)}/P{int(high_percentile)}",
        }
    return thresholds
