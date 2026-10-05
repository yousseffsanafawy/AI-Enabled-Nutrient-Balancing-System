"""
tests/test_inference.py — Unit Tests for Production Inference Pipeline (Sprint 10).
"""

import pytest
import numpy as np

from src.inference import ProductionInferenceEngine
from src.config import TIME_STEPS, N_FEATURES, FEATURES


@pytest.fixture(scope="module")
def inference_engine():
    """Load production inference engine once for test suite."""
    return ProductionInferenceEngine()


def test_inference_engine_initialization(inference_engine):
    assert inference_engine.model is not None
    assert inference_engine.scaler_X is not None
    assert inference_engine.scaler_y is not None
    assert len(inference_engine.feature_names) == N_FEATURES


def test_inference_execution_shape_and_keys(inference_engine):
    # Create dynamic window with realistic slope to avoid flatline check
    t = np.linspace(0, 1, TIME_STEPS)[:, None]
    base_vals = np.array([2.0, 23.5, 750.0, 6.1, 62.0])
    noise = np.sin(t * np.pi) * np.array([0.0, 0.5, 20.0, 0.1, 1.5])
    window = (base_vals + noise).astype(np.float32)

    result = inference_engine.run_inference(window, n_mc_samples=10, run_attribution=False)

    # Check top-level keys
    assert "timestamp" in result
    assert "inference_latency_ms" in result
    assert "current_telemetry" in result
    assert "forecast_next_step" in result
    assert "uncertainty_sigma" in result
    assert "stress_detection" in result
    assert "safety_layer" in result
    assert "agronomic_diagnosis" in result

    # Check telemetry and predictions
    for feat in FEATURES:
        assert feat in result["current_telemetry"]
        assert feat in result["forecast_next_step"]
        assert feat in result["uncertainty_sigma"]
        assert np.isfinite(result["forecast_next_step"][feat])
        assert result["uncertainty_sigma"][feat] >= 0.0

    # Check stress detection
    stress_info = result["stress_detection"]
    assert 0.0 <= stress_info["stress_probability"] <= 1.0
    assert isinstance(stress_info["is_stress"], bool)

    # Check safety layer triage
    safety_info = result["safety_layer"]
    assert safety_info["action"] in ("auto_dose", "alert_human", "no_action")
    assert "actuator_commands" in safety_info


def test_inference_integrated_attribution(inference_engine):
    window = np.array([
        [2.0, 22.0 + i*0.05, 650.0 + i*2.0, 6.0 - i*0.01, 65.0 - i*0.1]
        for i in range(TIME_STEPS)
    ], dtype=np.float32)

    result = inference_engine.run_inference(window, n_mc_samples=5, run_attribution=True)
    diag = result["agronomic_diagnosis"]

    assert diag["top_driver"] in FEATURES
    assert diag["secondary_driver"] in FEATURES
    assert 0.0 <= diag["top_driver_share"] <= 1.0
    assert len(diag["formatted_report"]) > 50
