"""
src/explainability.py — Interpretable Machine Learning & Agronomic Reasoning Engine.

Sprint 9 Implementation:
  - Permutation Feature Importance: Quantifies sensitivity across 5 sensor modalities.
  - Temporal Sensitivity Analysis: Measures timestep attributions across the 15-step lookback window (t-150s to t).
  - Integrated Gradients & Gradient Saliency: Deep neural network attribution satisfying completeness & implementation invariance.
  - Agronomic Explanation Generator: Translates mathematical attributions into actionable biological & operational diagnostics.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score

from src.config import FEATURES, N_FEATURES, TIME_STEPS, RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# 1. PERMUTATION FEATURE IMPORTANCE (S9-T1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_permutation_importance(
    model: tf.keras.Model,
    X: np.ndarray,
    y_forecast: np.ndarray,
    y_stress: Optional[np.ndarray] = None,
    feature_names: List[str] = FEATURES,
    n_repeats: int = 5,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Compute permutation feature importance across sensor modalities.

    For each feature f:
      Shuffles feature f across all samples (preserving temporal structure within sample),
      then evaluates the performance drop.

    Returns:
      Dictionary containing:
        - 'baseline_mse': Baseline forecasting MSE
        - 'baseline_stress_auc': Baseline stress ROC-AUC (if y_stress provided)
        - 'feature_importance': Dict mapping feature_name -> {
             'mean_forecast_mse_increase': float,
             'std_forecast_mse_increase': float,
             'per_sensor_mse_increase': Dict[str, float],
             'stress_auc_drop': float,
             'normalized_importance': float,
          }
    """
    rng = np.random.RandomState(random_seed)
    N, T, F = X.shape

    # 1. Compute baseline performance
    base_preds = model.predict(X, batch_size=256, verbose=0)
    if isinstance(base_preds, dict):
        base_f_pred = base_preds["forecast_output"]
        base_s_pred = base_preds.get("stress_output", None)
    elif isinstance(base_preds, list):
        base_f_pred = base_preds[0]
        base_s_pred = base_preds[1] if len(base_preds) > 1 else None
    else:
        base_f_pred = base_preds
        base_s_pred = None

    base_mse = float(mean_squared_error(y_forecast, base_f_pred))
    base_per_sensor_mse = {
        feature_names[i]: float(mean_squared_error(y_forecast[:, i], base_f_pred[:, i]))
        for i in range(F)
    }

    base_auc = None
    if y_stress is not None and base_s_pred is not None and len(np.unique(y_stress)) > 1:
        base_auc = float(roc_auc_score(y_stress, base_s_pred))

    results = {
        "baseline_mse": base_mse,
        "baseline_per_sensor_mse": base_per_sensor_mse,
        "baseline_stress_auc": base_auc,
        "feature_importance": {},
    }

    raw_increases = []

    # 2. Permutation loop
    for feat_idx, feat_name in enumerate(feature_names):
        repeat_mse_diffs = []
        repeat_auc_drops = []
        repeat_per_sensor_diffs = {fn: [] for fn in feature_names}

        for r in range(n_repeats):
            X_perm = X.copy()
            # Shuffle feature across samples
            perm_indices = rng.permutation(N)
            X_perm[:, :, feat_idx] = X_perm[perm_indices, :, feat_idx]

            perm_preds = model.predict(X_perm, batch_size=256, verbose=0)
            if isinstance(perm_preds, dict):
                perm_f_pred = perm_preds["forecast_output"]
                perm_s_pred = perm_preds.get("stress_output", None)
            elif isinstance(perm_preds, list):
                perm_f_pred = perm_preds[0]
                perm_s_pred = perm_preds[1] if len(perm_preds) > 1 else None
            else:
                perm_f_pred = perm_preds
                perm_s_pred = None

            perm_mse = float(mean_squared_error(y_forecast, perm_f_pred))
            repeat_mse_diffs.append(max(0.0, perm_mse - base_mse))

            for i, fn in enumerate(feature_names):
                s_mse = float(mean_squared_error(y_forecast[:, i], perm_f_pred[:, i]))
                repeat_per_sensor_diffs[fn].append(max(0.0, s_mse - base_per_sensor_mse[fn]))

            if base_auc is not None and perm_s_pred is not None:
                perm_auc = float(roc_auc_score(y_stress, perm_s_pred))
                repeat_auc_drops.append(max(0.0, base_auc - perm_auc))

        mean_diff = float(np.mean(repeat_mse_diffs))
        std_diff = float(np.std(repeat_mse_diffs))
        mean_auc_drop = float(np.mean(repeat_auc_drops)) if repeat_auc_drops else 0.0

        raw_increases.append(mean_diff)
        results["feature_importance"][feat_name] = {
            "mean_mse_increase": mean_diff,
            "std_mse_increase": std_diff,
            "per_sensor_mse_increase": {
                fn: float(np.mean(repeat_per_sensor_diffs[fn])) for fn in feature_names
            },
            "stress_auc_drop": mean_auc_drop,
        }

    # Normalize relative importance
    total_increase = sum(raw_increases) if sum(raw_increases) > 0 else 1.0
    for feat_idx, feat_name in enumerate(feature_names):
        results["feature_importance"][feat_name]["normalized_importance"] = float(
            raw_increases[feat_idx] / total_increase
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEMPORAL SENSITIVITY & TIMESTEP ATTRIBUTION (S9-T2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_sensitivity(
    model: tf.keras.Model,
    X: np.ndarray,
    y_forecast: np.ndarray,
    feature_names: List[str] = FEATURES,
    n_repeats: int = 3,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Compute sensitivity across the temporal lookback axis (15 timesteps: t-150s to t).

    Evaluates:
      1. Timestep Permutation Importance: Shuffles specific timestep t across samples.
      2. Temporal Gradient Saliency: Differentiates outputs with respect to each (t, f) input.
    """
    rng = np.random.RandomState(random_seed)
    N, T, F = X.shape

    base_preds = model.predict(X, batch_size=256, verbose=0)
    if isinstance(base_preds, dict):
        base_f_pred = base_preds["forecast_output"]
    elif isinstance(base_preds, list):
        base_f_pred = base_preds[0]
    else:
        base_f_pred = base_preds
    base_mse = float(mean_squared_error(y_forecast, base_f_pred))

    timestep_importance = []
    # Loop over each timestep in lookback window
    for t in range(T):
        diffs = []
        for r in range(n_repeats):
            X_perm = X.copy()
            perm_idx = rng.permutation(N)
            X_perm[:, t, :] = X_perm[perm_idx, t, :]

            perm_preds = model.predict(X_perm, batch_size=256, verbose=0)
            if isinstance(perm_preds, dict):
                perm_f = perm_preds["forecast_output"]
            elif isinstance(perm_preds, list):
                perm_f = perm_preds[0]
            else:
                perm_f = perm_preds

            diffs.append(max(0.0, float(mean_squared_error(y_forecast, perm_f)) - base_mse))

        timestep_importance.append(float(np.mean(diffs)))

    # Compute Gradient Saliency Map across a representative subset
    subset_size = min(200, N)
    X_sub = tf.convert_to_tensor(X[:subset_size], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_sub)
        preds = model(X_sub, training=False)
        if isinstance(preds, dict):
            target_f = preds["forecast_output"]
        elif isinstance(preds, list):
            target_f = preds[0]
        else:
            target_f = preds
        loss = tf.reduce_mean(tf.square(target_f))

    grads = tape.gradient(loss, X_sub)
    # Saliency: Mean absolute gradient over samples
    saliency_matrix = tf.reduce_mean(tf.abs(grads), axis=0).numpy()  # Shape: (T, F)

    # Normalize temporal importance
    tot_time = sum(timestep_importance) if sum(timestep_importance) > 0 else 1.0
    norm_time_imp = [val / tot_time for val in timestep_importance]

    return {
        "timestep_importance_raw": timestep_importance,
        "timestep_importance_normalized": norm_time_imp,
        "temporal_saliency_heatmap": saliency_matrix.tolist(),  # (15, 5)
        "timesteps_seconds_lag": [-(T - 1 - t) * 10 for t in range(T)],  # [-140, -130, ..., 0]
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. INTEGRATED GRADIENTS (S9-T3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_integrated_gradients(
    model: tf.keras.Model,
    x_input: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    target_head: str = "forecast_output",
    target_idx: int = 3,  # Default: pH
    steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients (Sundararajan et al., 2017) for a single sequence.

    Args:
      model: Trained MT-TCN-LSTM Keras model.
      x_input: Input sequence of shape (1, TIME_STEPS, N_FEATURES) or (TIME_STEPS, N_FEATURES).
      baseline: Reference sequence. If None, zeros baseline is used.
      target_head: 'forecast_output' or 'stress_output'.
      target_idx: Index of target feature in forecast_output (e.g. 3 for pH, 2 for TDS).
      steps: Number of Riemann interpolation steps along path.

    Returns:
      Attribution matrix of shape (TIME_STEPS, N_FEATURES).
    """
    if x_input.ndim == 2:
        x_input = np.expand_dims(x_input, axis=0)

    if baseline is None:
        baseline = np.zeros_like(x_input)
    elif baseline.ndim == 2:
        baseline = np.expand_dims(baseline, axis=0)

    # Generate alphas along linear path
    alphas = tf.linspace(0.0, 1.0, steps + 1)  # shape (steps + 1,)
    x_input_tf = tf.convert_to_tensor(x_input, dtype=tf.float32)
    baseline_tf = tf.convert_to_tensor(baseline, dtype=tf.float32)

    # Interpolate: x' + alpha * (x - x')
    delta = x_input_tf - baseline_tf
    interpolated = [baseline_tf + alpha * delta for alpha in alphas]
    interpolated = tf.concat(interpolated, axis=0)  # (steps+1, T, F)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        preds = model(interpolated, training=False)
        if isinstance(preds, dict):
            output = preds[target_head]
        elif isinstance(preds, list):
            output = preds[0] if target_head == "forecast_output" else preds[1]
        else:
            output = preds

        if target_head == "forecast_output":
            target_val = output[:, target_idx]
        else:
            target_val = tf.squeeze(output)

    grads = tape.gradient(target_val, interpolated)  # (steps+1, T, F)

    # Approximate Riemann integral using trapezoidal rule
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated_grads = tf.reduce_mean(avg_grads, axis=0) * tf.squeeze(delta, axis=0)

    return integrated_grads.numpy()  # (TIME_STEPS, N_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGRONOMIC EXPLANATION ENGINE (S9-T5 & S9-T6)
# ─────────────────────────────────────────────────────────────────────────────

class AgronomicReasoningEngine:
    """
    Translates neural network feature attributions, temporal dynamics,
    and uncertainty estimates into human-readable agronomic diagnostics.
    """

    def __init__(self, feature_names: List[str] = FEATURES):
        self.feature_names = feature_names

    def explain_prediction(
        self,
        current_readings: Dict[str, float],
        predicted_readings: Dict[str, float],
        attribution_matrix: np.ndarray,
        stress_probability: float,
        uncertainty_sigma: Dict[str, float],
        target_sensor: str = "pH",
    ) -> Dict[str, Any]:
        """
        Generate structured and natural-language agronomic explanation for a forecast.
        """
        target_idx = self.feature_names.index(target_sensor)
        curr_val = current_readings[target_sensor]
        pred_val = predicted_readings[target_sensor]
        delta = pred_val - curr_val
        sigma = uncertainty_sigma.get(target_sensor, 0.02)

        # 1. Feature attribution shares
        feat_attr_raw = np.sum(np.abs(attribution_matrix), axis=0)  # Shape: (F,)
        tot_attr = np.sum(feat_attr_raw) if np.sum(feat_attr_raw) > 0 else 1.0
        feat_shares = {
            self.feature_names[i]: float(feat_attr_raw[i] / tot_attr)
            for i in range(len(self.feature_names))
        }

        # 2. Temporal attribution shares (Immediate vs Historical)
        T = attribution_matrix.shape[0]
        time_attr_raw = np.sum(np.abs(attribution_matrix), axis=1)  # Shape: (T,)
        tot_time = np.sum(time_attr_raw) if np.sum(time_attr_raw) > 0 else 1.0

        # Recent (last 5 steps = 50s) vs Older history (first 10 steps = 100s)
        recent_share = float(np.sum(time_attr_raw[-5:]) / tot_time)
        historical_share = float(np.sum(time_attr_raw[:-5]) / tot_time)

        # 3. Determine primary driving features
        sorted_features = sorted(feat_shares.items(), key=lambda x: x[1], reverse=True)
        top_driver, top_driver_pct = sorted_features[0]
        secondary_driver, secondary_driver_pct = sorted_features[1]

        # 4. Agronomic diagnosis logic based on plant chemistry & hydroponics
        diagnosis_reasons = []
        action_recommendation = ""
        risk_level = "LOW"

        if stress_probability > 0.5:
            risk_level = "HIGH (STRESS DETECTED)"
        elif stress_probability > 0.2:
            risk_level = "ELEVATED"

        if target_sensor == "pH":
            if delta < -0.15:
                # Rapid acidification
                if current_readings.get("TDS", 0) > 800:
                    diagnosis_reasons.append(
                        f"Rapid acidification (Delta={delta:+.2f}): Plant uptake of ammonium (NH4+) or basic cations "
                        f"coupled with high nutrient concentration (TDS={current_readings.get('TDS', 0):.0f} ppm)."
                    )
                else:
                    diagnosis_reasons.append(
                        f"Acidic drift (Delta={delta:+.2f}): Dilution or acidic nutrient consumption without sufficient buffering capacity."
                    )
                action_recommendation = "Dose 0.5 - 1.0 mL pH Up (0.1M KOH); hold nutrient dosing until pH > 5.8."

            elif delta > +0.15:
                # Alkalinization
                diagnosis_reasons.append(
                    f"Alkaline drift (Delta={delta:+.2f}): Nitrate (NO3-) uptake dominance in active vegetative growth "
                    f"displacing OH- ions into solution."
                )
                action_recommendation = "Dose 0.5 - 1.0 mL pH Down (0.1M HNO3); verify root oxygenation."

            else:
                diagnosis_reasons.append(
                    f"Stable chemical equilibrium (Delta={delta:+.2f}): Solution within active biological buffering deadband."
                )
                action_recommendation = "Maintain current standby equilibrium; no chemical intervention required."

        elif target_sensor == "TDS":
            if delta > 30.0:
                diagnosis_reasons.append(
                    f"Salinity concentration surge (Delta={delta:+.0f} ppm): Evaporative water consumption exceeds ion absorption, "
                    f"concentrating dissolved mineral salts."
                )
                action_recommendation = "Top up fresh RO water to prevent osmotic root shock."
            elif delta < -30.0:
                diagnosis_reasons.append(
                    f"Nutrient depletion (Delta={delta:+.0f} ppm): Active macro-element absorption (N-P-K) reducing conductivity."
                )
                action_recommendation = "Schedule automated A+B concentrate replenishment (5 - 10 mL)."
            else:
                diagnosis_reasons.append("Nutrient concentration stable within crop target range.")
                action_recommendation = "Maintain recirculation pump flow."

        # Temporal nuance
        if recent_share > 0.60:
            temporal_insight = f"Dynamic transient: {recent_share*100:.1f}% of predictive attribution originates from the last 50 seconds."
        else:
            temporal_insight = f"Sustained trend: {historical_share*100:.1f}% of predictive attribution spans the preceding 100-150s lookback period."

        # Format natural language summary
        explanation_text = (
            f"=== Agronomic Diagnostic Report [{target_sensor}] ===\n"
            f"Current Reading: {curr_val:.2f} | Forecast (t+10s): {pred_val:.2f} (Delta={delta:+.2f})\n"
            f"System Risk Level: {risk_level} (Stress Probability: {stress_probability:.1%})\n"
            f"Uncertainty: sigma = {sigma:.4f} (High confidence: {sigma < 0.015})\n"
            f"\n"
            f"Primary Drivers:\n"
            f"  1. {top_driver}: {top_driver_pct*100:.1f}% contribution\n"
            f"  2. {secondary_driver}: {secondary_driver_pct*100:.1f}% contribution\n"
            f"\n"
            f"Temporal Dynamics:\n"
            f"  {temporal_insight}\n"
            f"\n"
            f"Agronomic Diagnosis:\n"
            f"  " + "\n  ".join(diagnosis_reasons) + "\n"
            f"\n"
            f"Recommended Action:\n"
            f"  {action_recommendation}\n"
            f"======================================================"
        )

        return {
            "target_sensor": target_sensor,
            "current_value": curr_val,
            "predicted_value": pred_val,
            "delta": delta,
            "uncertainty_sigma": sigma,
            "stress_probability": stress_probability,
            "risk_level": risk_level,
            "feature_attribution_shares": feat_shares,
            "top_driver": top_driver,
            "top_driver_share": top_driver_pct,
            "secondary_driver": secondary_driver,
            "secondary_driver_share": secondary_driver_pct,
            "temporal_recent_share": recent_share,
            "temporal_historical_share": historical_share,
            "temporal_insight": temporal_insight,
            "diagnosis_reasons": diagnosis_reasons,
            "recommended_action": action_recommendation,
            "formatted_report": explanation_text,
        }
