"""
tests/test_explainability.py — Unit Tests for Sprint 9 Explainability Engine.
"""

import pytest
import numpy as np
import tensorflow as tf

from src.explainability import (
    compute_permutation_importance,
    compute_temporal_sensitivity,
    compute_integrated_gradients,
    AgronomicReasoningEngine,
)
from src.config import FEATURES, N_FEATURES, TIME_STEPS


@pytest.fixture
def dummy_model():
    """Build a lightweight dummy multi-task model for testing explainability."""
    inputs = tf.keras.Input(shape=(TIME_STEPS, N_FEATURES), name="sensor_input")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    f_out = tf.keras.layers.Dense(N_FEATURES, name="forecast_output")(x)
    s_out = tf.keras.layers.Dense(1, activation="sigmoid", name="stress_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs={"forecast_output": f_out, "stress_output": s_out})
    model.compile(optimizer="adam", loss={"forecast_output": "mse", "stress_output": "binary_crossentropy"})
    return model


@pytest.fixture
def dummy_data():
    """Create synthetic sequence data for testing."""
    np.random.seed(42)
    N = 30
    X = np.random.uniform(0.1, 0.9, size=(N, TIME_STEPS, N_FEATURES)).astype(np.float32)
    y_forecast = np.random.uniform(0.1, 0.9, size=(N, N_FEATURES)).astype(np.float32)
    y_stress = (np.random.rand(N) > 0.7).astype(np.float32)
    return X, y_forecast, y_stress


def test_permutation_importance(dummy_model, dummy_data):
    X, y_forecast, y_stress = dummy_data
    results = compute_permutation_importance(
        model=dummy_model,
        X=X,
        y_forecast=y_forecast,
        y_stress=y_stress,
        feature_names=FEATURES,
        n_repeats=2,
    )

    assert "baseline_mse" in results
    assert "feature_importance" in results
    assert len(results["feature_importance"]) == N_FEATURES

    # Check normalized importance sums to ~1.0
    norm_sum = sum(v["normalized_importance"] for v in results["feature_importance"].values())
    assert np.isclose(norm_sum, 1.0, atol=1e-3)


def test_temporal_sensitivity(dummy_model, dummy_data):
    X, y_forecast, _ = dummy_data
    results = compute_temporal_sensitivity(
        model=dummy_model,
        X=X,
        y_forecast=y_forecast,
        feature_names=FEATURES,
        n_repeats=2,
    )

    assert "timestep_importance_raw" in results
    assert len(results["timestep_importance_raw"]) == TIME_STEPS
    assert len(results["timestep_importance_normalized"]) == TIME_STEPS

    saliency = np.array(results["temporal_saliency_heatmap"])
    assert saliency.shape == (TIME_STEPS, N_FEATURES)
    assert np.all(np.isfinite(saliency))


def test_integrated_gradients(dummy_model, dummy_data):
    X, _, _ = dummy_data
    sample_seq = X[0]  # (TIME_STEPS, N_FEATURES)

    ig = compute_integrated_gradients(
        model=dummy_model,
        x_input=sample_seq,
        target_head="forecast_output",
        target_idx=3,  # pH
        steps=20,
    )

    assert ig.shape == (TIME_STEPS, N_FEATURES)
    assert np.all(np.isfinite(ig))


def test_agronomic_reasoning_acidification():
    engine = AgronomicReasoningEngine(feature_names=FEATURES)

    current_readings = {"pH": 6.2, "TDS": 950.0, "DHT_temp": 24.0, "DHT_humidity": 65.0, "water_level": 2.0}
    predicted_readings = {"pH": 5.4, "TDS": 980.0, "DHT_temp": 24.2, "DHT_humidity": 64.0, "water_level": 2.0}
    attribution_matrix = np.ones((TIME_STEPS, N_FEATURES)) * 0.1
    attribution_matrix[:, 3] = 0.5  # pH attribution dominant

    explanation = engine.explain_prediction(
        current_readings=current_readings,
        predicted_readings=predicted_readings,
        attribution_matrix=attribution_matrix,
        stress_probability=0.85,
        uncertainty_sigma={"pH": 0.008},
        target_sensor="pH",
    )

    assert explanation["target_sensor"] == "pH"
    assert explanation["delta"] < -0.15
    assert "HIGH" in explanation["risk_level"]
    assert "pH Up" in explanation["recommended_action"]
    assert explanation["top_driver"] == "pH"
    assert "formatted_report" in explanation
    assert len(explanation["diagnosis_reasons"]) > 0


def test_agronomic_reasoning_alkalinization():
    engine = AgronomicReasoningEngine(feature_names=FEATURES)

    current_readings = {"pH": 5.8, "TDS": 700.0, "DHT_temp": 23.0, "DHT_humidity": 60.0, "water_level": 2.0}
    predicted_readings = {"pH": 6.4, "TDS": 700.0, "DHT_temp": 23.0, "DHT_humidity": 60.0, "water_level": 2.0}
    attribution_matrix = np.ones((TIME_STEPS, N_FEATURES)) * 0.1

    explanation = engine.explain_prediction(
        current_readings=current_readings,
        predicted_readings=predicted_readings,
        attribution_matrix=attribution_matrix,
        stress_probability=0.10,
        uncertainty_sigma={"pH": 0.005},
        target_sensor="pH",
    )

    assert explanation["delta"] > +0.15
    assert "pH Down" in explanation["recommended_action"]
    assert "LOW" in explanation["risk_level"]
