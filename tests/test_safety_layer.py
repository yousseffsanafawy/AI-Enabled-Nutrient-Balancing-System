"""
test_safety_layer.py — Unit Tests for Closed-Loop Safety Layer & Dosing Triage Engine.

Sprint 7 Verification (Task S7-T7):
  - Validates all safety edge cases:
    1. Sensor out-of-physical-bounds rejection
    2. Sensor flatline / frozen reading detection
    3. Sensor rate-of-change spike rejection
    4. High epistemic uncertainty abort gate (σ > HIGH)
    5. Moderate uncertainty supervisor alert gate (MED < σ ≤ HIGH)
    6. Extreme biological hazard intervention halt
    7. Safe autonomous corrective dosing execution
    8. System nominal equilibrium standby
    9. Single-cycle dosage safety clamp enforcement
    10. Stateful streaming engine integration with fault injection
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.safety_layer import (
    validate_sensor_reading,
    within_biological_limits,
    compute_corrective_dosage,
    make_dosing_decision,
    SafetyTriageEngine,
    MAX_DOSING_LIMITS,
)


@pytest.fixture
def calibrated_thresholds():
    """Mock calibrated validation thresholds."""
    return {
        "pH": {"med_threshold": 0.060, "high_threshold": 0.120},
        "TDS": {"med_threshold": 25.0, "high_threshold": 55.0},
        "DHT_temp": {"med_threshold": 0.35, "high_threshold": 0.80},
        "DHT_humidity": {"med_threshold": 1.50, "high_threshold": 3.50},
        "water_level": {"med_threshold": 0.05, "high_threshold": 0.15},
    }


@pytest.fixture
def nominal_reading():
    return {
        "pH": 6.05,
        "TDS": 645.0,
        "DHT_temp": 22.5,
        "DHT_humidity": 63.0,
        "water_level": 2.0,
    }


@pytest.fixture
def low_uncertainty():
    return {
        "pH": 0.025,
        "TDS": 12.0,
        "DHT_temp": 0.15,
        "DHT_humidity": 0.80,
        "water_level": 0.01,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sensor Validity Checks
# ─────────────────────────────────────────────────────────────────────────────

def test_sensor_out_of_physical_bounds(nominal_reading):
    """Test that physically impossible readings trigger immediate rejection."""
    bad_reading = nominal_reading.copy()
    bad_reading["pH"] = -0.5  # Impossible pH
    report = validate_sensor_reading(bad_reading)
    assert not report.is_valid
    assert report.fault_type == "OUT_OF_PHYSICAL_BOUNDS"
    assert report.fault_sensor == "pH"

    bad_tds = nominal_reading.copy()
    bad_tds["TDS"] = 9999.0  # Impossible TDS
    report_tds = validate_sensor_reading(bad_tds)
    assert not report_tds.is_valid
    assert report_tds.fault_sensor == "TDS"


def test_sensor_flatline_detection(nominal_reading):
    """Test that a frozen/flatlined sensor with zero variance is flagged."""
    # 12 identical readings in history buffer
    frozen_history = [nominal_reading.copy() for _ in range(12)]
    current = nominal_reading.copy()

    report = validate_sensor_reading(current, history_buffer=frozen_history, min_flatline_samples=10)
    assert not report.is_valid
    assert report.fault_type == "SENSOR_FLATLINE"


def test_sensor_rate_of_change_spike(nominal_reading):
    """Test that an unphysical jump in 10s is flagged as a spike fault."""
    history = [nominal_reading.copy()]
    spiked_reading = nominal_reading.copy()
    spiked_reading["pH"] = nominal_reading["pH"] + 1.20  # Max jump allowed is 0.50

    report = validate_sensor_reading(spiked_reading, history_buffer=history)
    assert not report.is_valid
    assert report.fault_type == "RATE_OF_CHANGE_SPIKE"
    assert report.fault_sensor == "pH"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decision Triage Engine Gates
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_on_sensor_fault(nominal_reading, low_uncertainty, calibrated_thresholds):
    """If sensor validity check fails, decision must be 'no_action'."""
    prediction = {"pH": 6.3, "TDS": 590.0, "water_level": 2.0}
    decision = make_dosing_decision(
        prediction=prediction,
        uncertainty=low_uncertainty,
        sensor_validity=False,  # Fault injected
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "no_action"
    assert decision.triage_code == "SENSOR_INTEGRITY_FAULT"
    assert decision.risk_level == "critical"
    assert all(v == 0.0 for v in decision.actuator_commands.values())


def test_decision_on_high_uncertainty(nominal_reading, low_uncertainty, calibrated_thresholds):
    """If uncertainty on critical sensor exceeds HIGH_THRESHOLD, decision must be 'no_action'."""
    prediction = {"pH": 6.35, "TDS": 590.0, "water_level": 2.0}
    high_unc = low_uncertainty.copy()
    high_unc["pH"] = 0.150  # Exceeds high_threshold (0.120)

    decision = make_dosing_decision(
        prediction=prediction,
        uncertainty=high_unc,
        sensor_validity=True,
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "no_action"
    assert decision.triage_code == "HIGH_EPISTEMIC_UNCERTAINTY"
    assert decision.risk_level == "critical"
    assert all(v == 0.0 for v in decision.actuator_commands.values())


def test_decision_on_critical_biological_hazard(nominal_reading, low_uncertainty, calibrated_thresholds):
    """If prediction is in severe hazard zone, halt auto-dosing and alert human."""
    extreme_pred = {"pH": 3.8, "TDS": 650.0, "water_level": 2.0}  # pH < 4.5 is critical acid burn

    decision = make_dosing_decision(
        prediction=extreme_pred,
        uncertainty=low_uncertainty,
        sensor_validity=True,
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "alert_human"
    assert decision.triage_code == "CRITICAL_BIOLOGICAL_HAZARD"
    assert decision.risk_level == "critical"


def test_decision_on_moderate_uncertainty(nominal_reading, low_uncertainty, calibrated_thresholds):
    """If MED < σ ≤ HIGH, alert human with tentative commands for operator approval."""
    prediction = {"pH": 6.30, "TDS": 580.0, "water_level": 2.0}
    med_unc = low_uncertainty.copy()
    med_unc["TDS"] = 35.0  # med_threshold is 25.0, high_threshold is 55.0

    decision = make_dosing_decision(
        prediction=prediction,
        uncertainty=med_unc,
        sensor_validity=True,
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "alert_human"
    assert decision.triage_code == "MODERATE_UNCERTAINTY_DRIFT"
    assert decision.risk_level == "medium"
    assert decision.actuator_commands["nutrient_ml"] > 0.0  # Tentative plan generated


def test_decision_safe_auto_dose(nominal_reading, low_uncertainty, calibrated_thresholds):
    """Under low uncertainty and valid sensors, corrective dosing is executed automatically."""
    # Predicted pH is 6.35 (above target 6.00 + deadband 0.10)
    prediction = {"pH": 6.35, "TDS": 645.0, "water_level": 2.0}

    decision = make_dosing_decision(
        prediction=prediction,
        uncertainty=low_uncertainty,
        sensor_validity=True,
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "auto_dose"
    assert decision.triage_code == "SAFE_AUTONOMOUS_DOSE"
    assert decision.risk_level == "low"
    assert decision.actuator_commands["acid_ml"] > 0.0
    assert decision.actuator_commands["base_ml"] == 0.0


def test_decision_system_at_equilibrium(nominal_reading, low_uncertainty, calibrated_thresholds):
    """When sensors are perfectly centered around target setpoints, standby without dosing."""
    prediction = {"pH": 6.02, "TDS": 652.0, "water_level": 2.0}

    decision = make_dosing_decision(
        prediction=prediction,
        uncertainty=low_uncertainty,
        sensor_validity=True,
        calibrated_thresholds=calibrated_thresholds,
        current_reading=nominal_reading,
    )
    assert decision.action == "no_action"
    assert decision.triage_code == "SYSTEM_AT_EQUILIBRIUM"
    assert decision.risk_level == "nominal"
    assert all(v == 0.0 for v in decision.actuator_commands.values())


def test_safety_clamp_enforcement():
    """Verify that calculated corrective doses never exceed maximum single-cycle volume."""
    commands, clamped = compute_corrective_dosage(
        pred_pH=7.50,            # Very high pH, raw acid demand would be ~37.5 mL
        pred_TDS=250.0,          # Very low TDS, raw nutrient demand would be ~20.0 mL
        pred_water_level=2.0,
        target_setpoints={"pH": 6.0, "TDS": 650.0},
    )
    assert clamped is True
    assert commands["acid_ml"] == MAX_DOSING_LIMITS["acid_ml"]  # Clamped at 5.0 mL
    assert commands["nutrient_ml"] <= MAX_DOSING_LIMITS["nutrient_ml"]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stateful Streaming Engine Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_streaming_engine_workflow(nominal_reading, low_uncertainty, calibrated_thresholds):
    """Test full sequential workflow including transient fault injection."""
    engine = SafetyTriageEngine(calibrated_thresholds=calibrated_thresholds, buffer_size=15)

    # 1. Run 10 nominal steps with small drift (should auto_dose)
    for i in range(10):
        current = nominal_reading.copy()
        current["pH"] = 6.05 + 0.01 * (i % 3)
        pred = {"pH": 6.25, "TDS": 640.0, "water_level": 2.0}
        d = engine.process_step(current, pred, low_uncertainty)
        assert d.action == "auto_dose"

    # 2. Inject flatline sensor fault: repeat exact same reading for 12 steps
    flat_reading = {"pH": 6.1000, "TDS": 640.0, "DHT_temp": 22.0, "DHT_humidity": 60.0, "water_level": 2.0}
    for _ in range(12):
        pred = {"pH": 6.30, "TDS": 640.0, "water_level": 2.0}
        d = engine.process_step(flat_reading, pred, low_uncertainty)

    # At step 12 of identical readings, the sensor flatline MUST be caught
    assert d.action == "no_action"
    assert d.triage_code == "SENSOR_INTEGRITY_FAULT"

    # Check summary stats
    stats = engine.get_summary_statistics()
    assert stats["total_decisions"] == 22
    assert stats["actions_breakdown"]["auto_dose"]["count"] >= 10
    assert stats["actions_breakdown"]["no_action"]["count"] >= 1
