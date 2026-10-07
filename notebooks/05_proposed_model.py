"""
05_proposed_model.py — Training & Evaluation of Proposed Multi-Task Model (Sprint 5).

Proposed Architecture:
  - Multi-Task Dilated Temporal Convolutional LSTM Network (MT-TCN-LSTM)
  - Joint Training:
    1. 5-Sensor Continuous Next-Step Forecasting (MSE loss, weight = 0.8)
    2. Agronomic Stress / Anomaly Detection (Weighted BCE loss, weight = 0.2)
  - Rigorous Methodological Protocol:
    * Standardized Chronological 70/10/20 partition
    * Scalers fitted strictly on training partition
    * Boundary-aware sequence isolation (17 gaps > 60s isolated)
    * Holdout test set touched strictly once
  - Uncertainty Quantification:
    * Monte Carlo (MC) Dropout with N=50 stochastic forward passes
    * Epistemic uncertainty σ per prediction
    * Empirical 90% confidence bands & calibration check
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Ensure workspace root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    RAW_CSV,
    FEATURES,
    TARGET_COLS,
    TIME_STEPS,
    RANDOM_SEED,
    TOLERANCES,
    SAFE_RANGES,
    LAMBDA_STRESS,
    BATCH_SIZE,
    EPOCHS,
    MODELS_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
)
from src.utils.seed import set_all_seeds
from src.preprocessing import build_pipeline
from src.models import build_proposed_model
from src.evaluation import (
    compute_regression_metrics,
    evaluate_tds_sensitivity,
    compute_stress_metrics,
)
from src.uncertainty import (
    predict_with_mc_dropout,
    evaluate_uncertainty_calibration,
)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def compute_stress_ground_truth(y_real: np.ndarray, feature_names: list = FEATURES) -> np.ndarray:
    """
    Determine binary stress ground truth (1 = stress, 0 = nominal)
    based on whether any continuous sensor violates safe agronomic thresholds.
    """
    wl_idx   = feature_names.index("water_level")
    temp_idx = feature_names.index("DHT_temp")
    tds_idx  = feature_names.index("TDS")
    ph_idx   = feature_names.index("pH")
    hum_idx  = feature_names.index("DHT_humidity")

    ph   = y_real[:, ph_idx]
    tds  = y_real[:, tds_idx]
    temp = y_real[:, temp_idx]
    hum  = y_real[:, hum_idx]

    stress = (
        (ph < SAFE_RANGES["pH"][0]) | (ph > SAFE_RANGES["pH"][1]) |
        (tds < SAFE_RANGES["TDS"][0]) | (tds > SAFE_RANGES["TDS"][1]) |
        (temp < SAFE_RANGES["DHT_temp"][0]) | (temp > SAFE_RANGES["DHT_temp"][1]) |
        (hum < SAFE_RANGES["DHT_humidity"][0]) | (hum > SAFE_RANGES["DHT_humidity"][1])
    ).astype(np.float32)

    return stress.reshape(-1, 1)


def run_sprint5_proposed():
    print("=" * 80)
    print("  SPRINT 5: PROPOSED MULTI-TASK ARCHITECTURE (MT-TCN-LSTM)")
    print("=" * 80)

    set_all_seeds(RANDOM_SEED)

    # 1. Load Dataset
    print(f"\n[1/6] Loading raw sensor dataset from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"  Dataset Shape: {df.shape}")

    # 2. Preprocess using Standardized Chronological 70/10/20 Pipeline
    print("\n[2/6] Building standardized chronological 70/10/20 pipeline...")
    pipe = build_pipeline(
        df=df,
        features=FEATURES,
        target_cols=TARGET_COLS,
        time_steps=TIME_STEPS,
        split_strategy="chronological",
        train_ratio=0.70,
        val_ratio=0.10,
        max_gap_seconds=60.0,
        interpolate=True,
        verbose=True,
    )

    X_train, y_train_scaled = pipe["X_train"], pipe["y_train"]
    X_val, y_val_scaled     = pipe["X_val"], pipe["y_val"]
    X_test, y_test_scaled   = pipe["X_test"], pipe["y_test"]
    scaler_y                = pipe["scaler_y"]

    # Invert scaling to obtain real physical ground truths for stress labeling & evaluation
    y_train_real = scaler_y.inverse_transform(y_train_scaled)
    y_val_real   = scaler_y.inverse_transform(y_val_scaled)
    y_test_real  = scaler_y.inverse_transform(y_test_scaled)

    # 3. Generate Multi-Task Stress Labels
    print("\n[3/6] Generating binary agronomic stress ground truth labels...")
    stress_train = compute_stress_ground_truth(y_train_real)
    stress_val   = compute_stress_ground_truth(y_val_real)
    stress_test  = compute_stress_ground_truth(y_test_real)

    n_pos_train = float(np.sum(stress_train == 1.0))
    n_neg_train = float(np.sum(stress_train == 0.0))
    pos_weight  = n_neg_train / max(n_pos_train, 1.0)

    print(f"  Train: {int(n_pos_train):,} Stressed / {len(stress_train):,} Total ({100*n_pos_train/len(stress_train):.2f}%) | Pos Weight: {pos_weight:.2f}")
    print(f"  Val:   {int(np.sum(stress_val == 1.0)):,} Stressed / {len(stress_val):,} Total ({100*np.mean(stress_val)*100:.2f}%)")
    print(f"  Test:  {int(np.sum(stress_test == 1.0)):,} Stressed / {len(stress_test):,} Total ({100*np.mean(stress_test)*100:.2f}%)")

    # Format multi-task training targets
    train_targets = {"forecast_output": y_train_scaled, "stress_output": stress_train}
    val_targets   = {"forecast_output": y_val_scaled,   "stress_output": stress_val}

    # 4. Build Proposed Multi-Task Model
    print("\n[4/6] Building Proposed MT-TCN-LSTM Architecture...")
    model = build_proposed_model(
        time_steps=TIME_STEPS,
        n_features=len(FEATURES),
        n_outputs=len(TARGET_COLS),
        conv_filters=64,
        kernel_size=3,
        lstm_units=64,
        dropout=0.3,
        lambda_stress=LAMBDA_STRESS,
        learning_rate=0.0001,
        class_weight_pos=pos_weight,
    )
    model.summary()

    # Callbacks
    ckpt_path = os.path.join(MODELS_DIR, "proposed_model_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    # Train Model
    print(f"\n  Training Proposed Model for up to {EPOCHS} epochs (EarlyStopping patience=8)...")
    t0 = time.time()
    history = model.fit(
        X_train, train_targets,
        validation_data=(X_val, val_targets),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    t_train = time.time() - t0
    print(f"  Training finished in {t_train:.1f}s ({t_train/60:.2f} min).")

    # 5. Model Inference & Evaluation
    print("\n[5/6] Evaluating on Holdout Test Set...")
    # Single-sample latency measurement
    print("  Measuring single-sample inference latency...")
    single_x = X_test[:1]
    for _ in range(10): _ = model(single_x, training=False)
    t_lat0 = time.perf_counter()
    n_trials = 100
    for _ in range(n_trials): _ = model(single_x, training=False)
    t_lat1 = time.perf_counter()
    latency_ms = round(((t_lat1 - t_lat0) / n_trials) * 1000.0, 3)
    print(f"  Inference Latency: {latency_ms:.3f} ms")

    # Predict test set
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    pred_forecast_scaled = preds["forecast_output"]
    pred_stress_prob    = preds["stress_output"].ravel()
    pred_stress_binary  = (pred_stress_prob >= 0.5).astype(int)

    # Invert forecast scaling to real units
    pred_forecast_real = scaler_y.inverse_transform(pred_forecast_scaled)

    # Compute Continuous Regression Metrics
    print("\n--- CONTINUOUS FORECASTING METRICS ---")
    reg_metrics = compute_regression_metrics(
        y_true=y_test_real,
        y_pred=pred_forecast_real,
        feature_names=FEATURES,
        tolerances=TOLERANCES,
        verbose=True,
    )

    tds_sens = evaluate_tds_sensitivity(
        actual_tds=y_test_real[:, FEATURES.index("TDS")],
        pred_tds=pred_forecast_real[:, FEATURES.index("TDS")],
        verbose=True,
    )

    # Compute Stress Classification Metrics
    print("\n--- AGRONOMIC STRESS DETECTION HEAD METRICS ---")
    stress_true_binary = stress_test.ravel().astype(int)
    cls_metrics = compute_stress_metrics(
        y_true_labels=stress_true_binary,
        y_pred_labels=pred_stress_binary,
        class_names=["Nominal", "Stress"],
        verbose=True,
    )

    # ROC AUC
    try:
        roc_auc = float(roc_auc_score(stress_true_binary, pred_stress_prob))
    except Exception:
        roc_auc = 0.5
    print(f"  Stress ROC-AUC Score: {roc_auc:.4f}")

    # 6. Monte Carlo Dropout Uncertainty Estimation
    print("\n[6/6] Computing Monte Carlo Dropout Uncertainty (N=50 passes)...")
    t_mc0 = time.time()
    mc_results = predict_with_mc_dropout(
        model=model,
        X=X_test,
        n_samples=50,
        batch_size=64,
        forecast_output_key="forecast_output",
    )
    t_mc_duration = time.time() - t_mc0
    print(f"  MC Dropout completed in {t_mc_duration:.1f}s.")

    # Uncertainty in real physical units
    # Standard deviation scale in real units: σ_real = σ_scaled * (max - min)
    scale_span = (scaler_y.data_max_ - scaler_y.data_min_)
    pred_uncertainty_real = mc_results["std"] * scale_span
    pred_lower_90_real    = scaler_y.inverse_transform(mc_results["lower_90"])
    pred_upper_90_real    = scaler_y.inverse_transform(mc_results["upper_90"])

    # Uncertainty Calibration Evaluation
    calib_results = evaluate_uncertainty_calibration(
        y_true=y_test_real,
        y_pred_mean=pred_forecast_real,
        y_pred_std=pred_uncertainty_real,
        feature_names=FEATURES,
    )
    print("  Uncertainty Calibration Results:")
    for feat, cres in calib_results.items():
        if isinstance(cres, dict):
            print(f"    {feat:<15}: Error-Uncertainty Correlation = {cres['error_uncertainty_corr']:+.4f} | 95% Coverage = {cres['coverage_95_pct']:.1f}%")
    print(f"    Mean Calibration Correlation: {calib_results.get('mean_correlation', 0.0):+.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Generate Publication Figures
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating publication figures...")

    # Load baseline master benchmark results for comparison
    master_bench_path = os.path.join(EXPERIMENTS_DIR, "exp_004_baseline_benchmark_master.json")
    if os.path.exists(master_bench_path):
        with open(master_bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        b_summary = bench_data["summary_table"]
    else:
        b_summary = []

    # Figure 16: Proposed vs Baselines (MAE & R²)
    comp_models = ["B0", "B1", "B2", "B5", "Proposed"]
    # Extract baseline data
    b_dict = {row["ID"]: row for row in b_summary}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    model_labels = ["Persistence (B0)", "CNN-BiLSTM (B1)", "Vanilla LSTM (B2)", "TCN (B5)", "Proposed (MT-TCN-LSTM)"]
    c_list = ["#64748b", "#2563eb", "#0284c7", "#d97706", "#10b981"]

    avg_r2_comp = [
        float(b_dict.get("B0", {}).get("Avg R²", 0.741)),
        float(b_dict.get("B1", {}).get("Avg R²", 0.106)),
        float(b_dict.get("B2", {}).get("Avg R²", 0.532)),
        float(b_dict.get("B5", {}).get("Avg R²", 0.473)),
        reg_metrics["aggregate"]["mean_R2"],
    ]

    tds_mae_comp = [
        float(b_dict.get("B0", {}).get("TDS MAE (ppm)", 0.48)),
        float(b_dict.get("B1", {}).get("TDS MAE (ppm)", 53.44)),
        float(b_dict.get("B2", {}).get("TDS MAE (ppm)", 22.03)),
        float(b_dict.get("B5", {}).get("TDS MAE (ppm)", 20.15)),
        reg_metrics["TDS"]["MAE"],
    ]

    bars1 = axes[0].bar(model_labels, avg_r2_comp, color=c_list, alpha=0.85, edgecolor="#0f172a", linewidth=1.2)
    axes[0].set_ylabel("Average Coefficient of Determination ($R^2$)", fontsize=11, fontweight="bold")
    axes[0].set_title("Average $R^2$ Across Sensors: Proposed vs Top Baselines", fontsize=12, fontweight="bold")
    axes[0].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        axes[0].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                         ha="center", va="bottom" if h >= 0 else "top", fontweight="bold")

    bars2 = axes[1].bar(model_labels, tds_mae_comp, color=c_list, alpha=0.85, edgecolor="#0f172a", linewidth=1.2)
    axes[1].set_ylabel("TDS MAE (ppm) — Lower is Better", fontsize=11, fontweight="bold")
    axes[1].set_title("TDS Forecasting Error: Proposed vs Top Baselines", fontsize=12, fontweight="bold")
    axes[1].set_xticklabels(model_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        axes[1].annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    fig16_path = os.path.join(FIG_DIR, "fig16_proposed_vs_baselines_mae_r2.png")
    fig.savefig(fig16_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig16_path}")

    # Figure 17: Stress Head Confusion Matrix & Probability Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(stress_true_binary, pred_stress_binary)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Nominal (0)", "Stress (1)"],
                yticklabels=["Nominal (0)", "Stress (1)"])
    axes[0].set_title("Stress Detection Head Confusion Matrix (Test Set)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Ground Truth Label", fontsize=10, fontweight="bold")

    # Right: Predicted Probability Histogram by True Class
    axes[1].hist(pred_stress_prob[stress_true_binary == 0], bins=30, alpha=0.6, label="Actual Nominal", color="#2563eb", density=True)
    axes[1].hist(pred_stress_prob[stress_true_binary == 1], bins=30, alpha=0.6, label="Actual Stress", color="#ef4444", density=True)
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    axes[1].set_title("Predicted Stress Probability Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Stress Probability", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Density", fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig17_path = os.path.join(FIG_DIR, "fig17_stress_confusion_matrix.png")
    fig.savefig(fig17_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig17_path}")

    # Figure 18: MC Dropout Uncertainty Bands on Test Set (First 150 Steps)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    zoom = 150
    t_idx = np.arange(zoom)

    # pH Tracking with 90% Confidence Interval
    ph_i = FEATURES.index("pH")
    axes[0].plot(t_idx, y_test_real[:zoom, ph_i], label="Actual pH", color="#0f172a", linewidth=2.0)
    axes[0].plot(t_idx, pred_forecast_real[:zoom, ph_i], label="MC Mean Prediction", color="#2563eb", linestyle="--", linewidth=1.5)
    axes[0].fill_between(t_idx, pred_lower_90_real[:zoom, ph_i], pred_upper_90_real[:zoom, ph_i],
                         color="#3b82f6", alpha=0.25, label="90% Prediction Interval [q0.05, q0.95]")
    axes[0].set_ylabel("pH Units", fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # TDS Tracking with 90% Confidence Interval
    tds_i = FEATURES.index("TDS")
    axes[1].plot(t_idx, y_test_real[:zoom, tds_i], label="Actual TDS", color="#0f172a", linewidth=2.0)
    axes[1].plot(t_idx, pred_forecast_real[:zoom, tds_i], label="MC Mean Prediction", color="#10b981", linestyle="--", linewidth=1.5)
    axes[1].fill_between(t_idx, pred_lower_90_real[:zoom, tds_i], pred_upper_90_real[:zoom, tds_i],
                         color="#10b981", alpha=0.25, label="90% Prediction Interval [q0.05, q0.95]")
    axes[1].set_ylabel("TDS (ppm)", fontsize=10, fontweight="bold")
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Epistemic Uncertainty σ (pH and TDS)
    axes[2].plot(t_idx, pred_uncertainty_real[:zoom, ph_i], label="pH Uncertainty (σ)", color="#2563eb", linewidth=1.5)
    axes[2].plot(t_idx, pred_uncertainty_real[:zoom, tds_i] / 50.0, label="TDS Uncertainty (σ / 50)", color="#10b981", linewidth=1.5, linestyle=":")
    axes[2].set_ylabel("Uncertainty (σ)", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Test Steps (10s intervals)", fontsize=10, fontweight="bold")
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Proposed Model: One-Step-Ahead Forecasts with Monte Carlo Dropout Uncertainty Bounds", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig18_path = os.path.join(FIG_DIR, "fig18_mc_dropout_uncertainty_bands.png")
    fig.savefig(fig18_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig18_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Log Master Experiment Record
    # ─────────────────────────────────────────────────────────────────────────
    exp_record = {
        "experiment_id": "exp_005_proposed_model",
        "description": "Multi-Task Dilated Temporal Convolutional LSTM Network (MT-TCN-LSTM) with MC Dropout",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "chronological_70_10_20_boundary_aware",
        "architecture": {
            "name": "MT-TCN-LSTM",
            "params": model.count_params(),
            "shared_encoder": "Causal Dilated Conv1D (d=1, 2) + LSTM(64)",
            "heads": ["forecast_output (Dense 5)", "stress_output (Dense 1 sigmoid)"],
            "lambda_stress": LAMBDA_STRESS,
            "latency_ms": latency_ms,
            "train_duration_sec": round(t_train, 2),
        },
        "regression_metrics": reg_metrics,
        "tds_sensitivity": tds_sens,
        "classification_metrics": cls_metrics,
        "stress_roc_auc": round(roc_auc, 4),
        "uncertainty_calibration": calib_results,
    }

    exp_json_path = os.path.join(EXPERIMENTS_DIR, "exp_005_proposed_model.json")
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)
    print(f"  Saved experiment record: {exp_json_path}")

    # Update benchmark_table.md to add Proposed Model
    _update_benchmark_table_with_proposed(reg_metrics, latency_ms, t_train, model.count_params())

    print("\n" + "=" * 80)
    print("  SPRINT 5 PROPOSED MODEL TRAINING & BENCHMARKING COMPLETED!")
    print("=" * 80)
    return exp_record


def _update_benchmark_table_with_proposed(reg_metrics, latency_ms, train_time, params):
    """Append the proposed model to the master benchmark table in reports/benchmark_table.md."""
    bench_file = os.path.join(REPORTS_DIR, "benchmark_table.md")
    if not os.path.exists(bench_file):
        return

    with open(bench_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create proposed model row
    agg = reg_metrics["aggregate"]
    prop_row = (
        f"| **P1** | **Proposed MT-TCN-LSTM** | **{reg_metrics['pH']['MAE']:.4f}** | "
        f"**{reg_metrics['TDS']['MAE']:.2f}** | **{reg_metrics['DHT_temp']['MAE']:.3f}** | "
        f"**{reg_metrics['water_level']['MAE']:.3f}** | **{reg_metrics['DHT_humidity']['MAE']:.3f}** | "
        f"**{agg['mean_R2']:.3f}** | **{agg.get('mean_within_tolerance_pct', 0.0):.2f}%** | "
        f"**{agg.get('mean_within_tolerance_no_tds', 0.0):.2f}%** | **{params:,}** | "
        f"**{latency_ms:.2f}** | **{train_time:.1f}** |"
    )

    if "Proposed MT-TCN-LSTM" not in content:
        # Insert before section 3
        sec3_marker = "--- \n\n## 3." if "--- \n\n## 3." in content else "## 3."
        if sec3_marker in content:
            new_content = content.replace(sec3_marker, f"{prop_row}\n\n{sec3_marker}")
            with open(bench_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("  Updated reports/benchmark_table.md with Proposed Model P1.")


if __name__ == "__main__":
    run_sprint5_proposed()
