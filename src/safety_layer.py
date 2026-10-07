"""
safety_layer.py — Rule-Based Safety Layer & Closed-Loop Dosing Triage Engine.

Sprint 7 Implementation:
  - Protects physical hydroponic crop and dosing hardware from faulty predictions and sensor failures.
  - Three-tier triage decisions:
    1. 'auto_dose'    : Low uncertainty (σ ≤ MED), valid telemetry, safe corrective dosage within bounds.
    2. 'alert_human'  : Moderate uncertainty (MED < σ ≤ HIGH) OR prediction approaches biological hazard limits.
    3. 'no_action'    : High uncertainty (σ > HIGH) OR sensor validity failure (flatline, spike, disconnected).
  - Physical dosage calculation with single-shot maximum safety clamps.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np

from src.config import SAFE_RANGES, TOLERANCES


# Extreme biological hazard boundaries where automated chemical dosing is strictly inhibited
CRITICAL_BIOLOGICAL_LIMITS = {
    "pH": (4.5, 8.0),             # Outside: severe root destruction / chemical shock
    "TDS": (200.0, 1200.0),       # Outside: osmotic lysis or extreme nutrient starvation
    "DHT_temp": (10.0, 38.0),     # Outside: biological enzyme failure
    "DHT_humidity": (20.0, 95.0), # Outside: extreme desiccation or mold explosion
    "water_level": (0.5, 5.0),    # Outside: dry pump burnout or tank overflow
}

# Physical bounds for sensor integrity checks
PHYSICAL_SENSOR_BOUNDS = {
    "pH": (0.0, 14.0),
    "TDS": (0.0, 5000.0),
    "DHT_temp": (0.0, 60.0),
    "DHT_humidity": (0.0, 100.0),
    "water_level": (0.0, 10.0),
}

# Maximum allowable single-step jump (rate of change per 10s interval)
MAX_STEP_DELTA = {
    "pH": 0.50,            # pH cannot physically shift > 0.50 in 10s without massive injection
    "TDS": 250.0,          # TDS cannot shift > 250 ppm in 10s
    "DHT_temp": 3.0,       # Air temp cannot jump > 3°C in 10s
    "DHT_humidity": 15.0,  # Humidity cannot jump > 15% in 10s
    "water_level": 1.5,    # Water level cannot jump > 1.5 in 10s
}

# Maximum single-cycle actuator dosing limits (Safety Clamps)
MAX_DOSING_LIMITS = {
    "acid_ml": 5.0,        # Max acid volume per cycle (mL)
    "base_ml": 5.0,        # Max base volume per cycle (mL)
    "nutrient_ml": 25.0,   # Max A+B concentrated fertilizer volume (mL)
    "refill_sec": 30.0,    # Max fresh water solenoid valve open duration (sec)
}

# Proportional gain constants for corrective dosing
PROPORTIONAL_GAINS = {
    "acid_ml_per_pH": 2.5,       # mL acid per 0.1 pH above target
    "base_ml_per_pH": 2.5,       # mL base per 0.1 pH below target
    "nutrient_ml_per_100_tds": 5.0, # mL fertilizer per 100 ppm below target
}

# Target setpoints for optimal crop growth
DEFAULT_TARGET_SETPOINTS = {
    "pH": 6.0,
    "TDS": 650.0,
    "DHT_temp": 22.0,
    "DHT_humidity": 65.0,
    "water_level": 2.0,
}


@dataclass
class SensorValidityReport:
    """Diagnostic report for sensor health checks."""
    is_valid: bool
    fault_sensor: Optional[str] = None
    fault_type: Optional[str] = None
    details: str = "All sensors nominal."


@dataclass
class DosingDecision:
    """Structured decision output from the Safety Layer."""
    action: str                        # 'auto_dose' | 'alert_human' | 'no_action'
    risk_level: str                    # 'nominal' | 'low' | 'medium' | 'critical'
    triage_code: str                   # Diagnostic code
    reason: str                        # Human-readable explanation
    actuator_commands: Dict[str, float] = field(default_factory=dict)
    safety_clamp_applied: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "risk_level": self.risk_level,
            "triage_code": self.triage_code,
            "reason": self.reason,
            "actuator_commands": self.actuator_commands,
            "safety_clamp_applied": self.safety_clamp_applied,
            "metrics": self.metrics,
        }


def validate_sensor_reading(
    current_reading: Dict[str, float],
    history_buffer: Optional[List[Dict[str, float]]] = None,
    min_flatline_samples: int = 10,
) -> SensorValidityReport:
    """
    Validate incoming sensor telemetry against physical plausibility, rate of change, and flatlines.

    Args:
        current_reading: Dict mapping sensor names to latest values.
        history_buffer: Optional list of past readings (most recent last) for temporal checks.
        min_flatline_samples: Minimum identical readings to flag a frozen sensor.

    Returns:
        SensorValidityReport indicating whether readings are safe to trust.
    """
    # 1. Physical Bounds Check
    for sensor, val in current_reading.items():
        if sensor in PHYSICAL_SENSOR_BOUNDS:
            low, high = PHYSICAL_SENSOR_BOUNDS[sensor]
            if val < low or val > high or np.isnan(val) or np.isinf(val):
                return SensorValidityReport(
                    is_valid=False,
                    fault_sensor=sensor,
                    fault_type="OUT_OF_PHYSICAL_BOUNDS",
                    details=f"Sensor {sensor} value {val} is outside physical limits [{low}, {high}]."
                )

    # If no history provided, pass range check
    if not history_buffer or len(history_buffer) == 0:
        return SensorValidityReport(is_valid=True)

    # 2. Rate of Change (Spike) Check
    prev_reading = history_buffer[-1]
    for sensor, val in current_reading.items():
        if sensor in prev_reading and sensor in MAX_STEP_DELTA:
            delta = abs(val - prev_reading[sensor])
            max_delta = MAX_STEP_DELTA[sensor]
            if delta > max_delta:
                return SensorValidityReport(
                    is_valid=False,
                    fault_sensor=sensor,
                    fault_type="RATE_OF_CHANGE_SPIKE",
                    details=f"Sensor {sensor} experienced impossible instantaneous jump of {delta:.2f} (max allowed: {max_delta:.2f})."
                )

    # 3. Flatline / Frozen Telemetry Check
    if len(history_buffer) >= min_flatline_samples:
        recent_window = history_buffer[-min_flatline_samples:] + [current_reading]
        for sensor in ["pH", "TDS", "DHT_temp", "DHT_humidity"]:
            if all(sensor in r for r in recent_window):
                vals = [r[sensor] for r in recent_window]
                var = np.var(vals)
                if var < 1e-7:
                    return SensorValidityReport(
                        is_valid=False,
                        fault_sensor=sensor,
                        fault_type="SENSOR_FLATLINE",
                        details=f"Sensor {sensor} is frozen/flatlined (variance={var:.2e} across {len(recent_window)} readings)."
                    )

    return SensorValidityReport(is_valid=True)


def within_biological_limits(
    prediction: Dict[str, float],
    use_critical: bool = False,
) -> bool:
    """
    Check if predicted values fall within agronomic safe boundaries.

    Args:
        prediction: Dict of predicted sensor values.
        use_critical: If True, check against extreme fatal boundaries (CRITICAL_BIOLOGICAL_LIMITS).
                      If False, check against optimal agronomic ranges (SAFE_RANGES).
    """
    limits = CRITICAL_BIOLOGICAL_LIMITS if use_critical else SAFE_RANGES
    for sensor, val in prediction.items():
        if sensor in limits:
            low, high = limits[sensor]
            if val < low or val > high:
                return False
    return True


def compute_corrective_dosage(
    pred_pH: float,
    pred_TDS: float,
    pred_water_level: float,
    target_setpoints: Dict[str, float],
) -> Tuple[Dict[str, float], bool]:
    """
    Compute required physical dosing amounts with single-cycle safety clamps.

    Returns:
        (actuator_commands, safety_clamp_applied)
    """
    target_pH = target_setpoints.get("pH", 6.0)
    target_TDS = target_setpoints.get("TDS", 650.0)
    clamp_applied = False

    acid_ml = 0.0
    base_ml = 0.0
    nutrient_ml = 0.0
    refill_sec = 0.0

    # pH correction (deadband ± 0.1 to avoid chatter)
    if pred_pH > target_pH + 0.10:
        raw_acid = (pred_pH - target_pH) * 10.0 * PROPORTIONAL_GAINS["acid_ml_per_pH"]
        acid_ml = min(raw_acid, MAX_DOSING_LIMITS["acid_ml"])
        if raw_acid > MAX_DOSING_LIMITS["acid_ml"]:
            clamp_applied = True
    elif pred_pH < target_pH - 0.10:
        raw_base = (target_pH - pred_pH) * 10.0 * PROPORTIONAL_GAINS["base_ml_per_pH"]
        base_ml = min(raw_base, MAX_DOSING_LIMITS["base_ml"])
        if raw_base > MAX_DOSING_LIMITS["base_ml"]:
            clamp_applied = True

    # TDS correction (deadband ± 20 ppm)
    if pred_TDS < target_TDS - 20.0:
        raw_nut = ((target_TDS - pred_TDS) / 100.0) * PROPORTIONAL_GAINS["nutrient_ml_per_100_tds"]
        nutrient_ml = min(raw_nut, MAX_DOSING_LIMITS["nutrient_ml"])
        if raw_nut > MAX_DOSING_LIMITS["nutrient_ml"]:
            clamp_applied = True
    elif pred_TDS > target_TDS + 100.0:
        # High TDS: trigger fresh water dilution
        refill_sec = min(15.0, MAX_DOSING_LIMITS["refill_sec"])

    # Water Level Low check
    if pred_water_level < 1.0:
        refill_sec = max(refill_sec, 20.0)

    commands = {
        "acid_ml": round(acid_ml, 2),
        "base_ml": round(base_ml, 2),
        "nutrient_ml": round(nutrient_ml, 2),
        "refill_sec": round(refill_sec, 2),
    }
    return commands, clamp_applied


def make_dosing_decision(
    prediction: Dict[str, float],
    uncertainty: Dict[str, float],
    sensor_validity: Union[bool, SensorValidityReport],
    calibrated_thresholds: Dict[str, Dict[str, float]],
    current_reading: Optional[Dict[str, float]] = None,
    target_setpoints: Optional[Dict[str, float]] = None,
) -> DosingDecision:
    """
    Master closed-loop decision engine for physical nutrient dosing.

    Evaluation Hierarchy:
      1. Sensor Integrity Gate: If any sensor is invalid/flatlined/spiked -> 'no_action'
      2. High Epistemic Uncertainty Gate: If σ > HIGH_THRESHOLD -> 'no_action' (model is untrusted)
      3. Critical Biological Hazard Gate: If predicted or current reading is extreme -> 'alert_human'
      4. Moderate Uncertainty Oversight Gate: If σ > MED_THRESHOLD -> 'alert_human' (operator confirmation)
      5. Safe Autonomous Dosing: Low uncertainty, valid sensors, safe bounds -> 'auto_dose' (or 'no_action' if already nominal)

    Args:
        prediction: Dict of predicted sensor values (e.g. {'pH': 6.2, 'TDS': 580.0, ...}).
        uncertainty: Dict of predictive standard deviations σ.
        sensor_validity: bool or SensorValidityReport object.
        calibrated_thresholds: Dict of thresholds per sensor containing 'med_threshold' and 'high_threshold'.
        current_reading: Optional dict of actual current sensor readings.
        target_setpoints: Optional dict of agronomic setpoints.

    Returns:
        DosingDecision dataclass.
    """
    if target_setpoints is None:
        target_setpoints = DEFAULT_TARGET_SETPOINTS

    # Normalize sensor validity
    if isinstance(sensor_validity, bool):
        is_sensor_valid = sensor_validity
        fault_info = "Sensor validity flag false."
    else:
        is_sensor_valid = sensor_validity.is_valid
        fault_info = sensor_validity.details

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 1: Sensor Integrity Gate (Hardware / Streaming Health)
    # ─────────────────────────────────────────────────────────────────────────
    if not is_sensor_valid:
        return DosingDecision(
            action="no_action",
            risk_level="critical",
            triage_code="SENSOR_INTEGRITY_FAULT",
            reason=f"Telemetry rejected: {fault_info}. Actuators held in safe quiescent state.",
            actuator_commands={"acid_ml": 0.0, "base_ml": 0.0, "nutrient_ml": 0.0, "refill_sec": 0.0},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 2: High Epistemic Uncertainty Gate (Model Disorientation Check)
    # ─────────────────────────────────────────────────────────────────────────
    for sensor in ["pH", "TDS"]:
        if sensor in uncertainty and sensor in calibrated_thresholds:
            sigma = uncertainty[sensor]
            high_thresh = calibrated_thresholds[sensor]["high_threshold"]
            if sigma > high_thresh:
                return DosingDecision(
                    action="no_action",
                    risk_level="critical",
                    triage_code="HIGH_EPISTEMIC_UNCERTAINTY",
                    reason=f"Model uncertainty for {sensor} (sigma={sigma:.3f}) exceeds critical threshold ({high_thresh:.3f}). Autonomous dosing prohibited.",
                    actuator_commands={"acid_ml": 0.0, "base_ml": 0.0, "nutrient_ml": 0.0, "refill_sec": 0.0},
                    metrics={"trigger_sensor": sensor, "sigma": sigma, "threshold": high_thresh},
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 3: Critical Biological Hazard Gate (Extreme Value Emergency)
    # ─────────────────────────────────────────────────────────────────────────
    if not within_biological_limits(prediction, use_critical=True):
        extreme_sensors = [
            f"{s}={v:.1f}" for s, v in prediction.items()
            if s in CRITICAL_BIOLOGICAL_LIMITS and (v < CRITICAL_BIOLOGICAL_LIMITS[s][0] or v > CRITICAL_BIOLOGICAL_LIMITS[s][1])
        ]
        return DosingDecision(
            action="alert_human",
            risk_level="critical",
            triage_code="CRITICAL_BIOLOGICAL_HAZARD",
            reason=f"Predicted severe biological violation: {', '.join(extreme_sensors)}. Automated intervention halted to prevent chemical shock.",
            actuator_commands={"acid_ml": 0.0, "base_ml": 0.0, "nutrient_ml": 0.0, "refill_sec": 0.0},
            metrics={"extreme_sensors": extreme_sensors},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 4: Moderate Uncertainty Oversight Gate (Supervised Human Approval)
    # ─────────────────────────────────────────────────────────────────────────
    for sensor in ["pH", "TDS"]:
        if sensor in uncertainty and sensor in calibrated_thresholds:
            sigma = uncertainty[sensor]
            med_thresh = calibrated_thresholds[sensor]["med_threshold"]
            if sigma > med_thresh:
                # Compute tentative commands for operator review
                tentative_cmds, clamped = compute_corrective_dosage(
                    pred_pH=prediction.get("pH", 6.0),
                    pred_TDS=prediction.get("TDS", 650.0),
                    pred_water_level=prediction.get("water_level", 2.0),
                    target_setpoints=target_setpoints,
                )
                return DosingDecision(
                    action="alert_human",
                    risk_level="medium",
                    triage_code="MODERATE_UNCERTAINTY_DRIFT",
                    reason=f"Elevated predictive uncertainty for {sensor} (sigma={sigma:.3f} > med_thresh={med_thresh:.3f}). Proposed actions require operator sign-off.",
                    actuator_commands=tentative_cmds,
                    safety_clamp_applied=clamped,
                    metrics={"trigger_sensor": sensor, "sigma": sigma, "threshold": med_thresh},
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Tier 5: Safe Autonomous Dosing or Nominal Standby
    # ─────────────────────────────────────────────────────────────────────────
    pred_pH = prediction.get("pH", 6.0)
    pred_TDS = prediction.get("TDS", 650.0)
    pred_wl = prediction.get("water_level", 2.0)

    commands, clamped = compute_corrective_dosage(
        pred_pH=pred_pH,
        pred_TDS=pred_TDS,
        pred_water_level=pred_wl,
        target_setpoints=target_setpoints,
    )

    is_actionable = any(v > 0.0 for v in commands.values())

    if is_actionable:
        action_desc = []
        if commands["acid_ml"] > 0:
            action_desc.append(f"Dose Acid {commands['acid_ml']}mL")
        if commands["base_ml"] > 0:
            action_desc.append(f"Dose Base {commands['base_ml']}mL")
        if commands["nutrient_ml"] > 0:
            action_desc.append(f"Dose Nutrient {commands['nutrient_ml']}mL")
        if commands["refill_sec"] > 0:
            action_desc.append(f"Refill Water {commands['refill_sec']}s")

        return DosingDecision(
            action="auto_dose",
            risk_level="low",
            triage_code="SAFE_AUTONOMOUS_DOSE",
            reason=f"High model confidence (all sigma ≤ MED) and valid sensors. Executing: {', '.join(action_desc)}.",
            actuator_commands=commands,
            safety_clamp_applied=clamped,
            metrics={"prediction": prediction, "uncertainty": uncertainty},
        )
    else:
        return DosingDecision(
            action="no_action",
            risk_level="nominal",
            triage_code="SYSTEM_AT_EQUILIBRIUM",
            reason="All telemetry within target tolerance deadbands. System in optimal agronomic equilibrium.",
            actuator_commands=commands,
            safety_clamp_applied=False,
            metrics={"prediction": prediction, "uncertainty": uncertainty},
        )


class SafetyTriageEngine:
    """
    Stateful streaming controller maintaining temporal history buffers,
    evaluating sequential sensor steps, and recording safety audit logs.
    """

    def __init__(
        self,
        calibrated_thresholds: Dict[str, Dict[str, float]],
        target_setpoints: Optional[Dict[str, float]] = None,
        buffer_size: int = 15,
    ):
        self.calibrated_thresholds = calibrated_thresholds
        self.target_setpoints = target_setpoints or DEFAULT_TARGET_SETPOINTS
        self.buffer_size = buffer_size
        self.history_buffer: List[Dict[str, float]] = []
        self.decision_log: List[DosingDecision] = []

    def process_step(
        self,
        current_reading: Dict[str, float],
        prediction: Dict[str, float],
        uncertainty: Dict[str, float],
    ) -> DosingDecision:
        """Process a single real-time sensor timestep."""
        # 1. Validate sensor health
        validity = validate_sensor_reading(current_reading, self.history_buffer)

        # 2. Make safety triage decision
        decision = make_dosing_decision(
            prediction=prediction,
            uncertainty=uncertainty,
            sensor_validity=validity,
            calibrated_thresholds=self.calibrated_thresholds,
            current_reading=current_reading,
            target_setpoints=self.target_setpoints,
        )

        # 3. Update temporal history and audit log
        self.history_buffer.append(current_reading.copy())
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)

        self.decision_log.append(decision)
        return decision

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics of all triage decisions made."""
        if not self.decision_log:
            return {}

        total = len(self.decision_log)
        actions = {"auto_dose": 0, "alert_human": 0, "no_action": 0}
        triage_codes = {}
        clamped_count = 0

        for d in self.decision_log:
            actions[d.action] = actions.get(d.action, 0) + 1
            triage_codes[d.triage_code] = triage_codes.get(d.triage_code, 0) + 1
            if d.safety_clamp_applied:
                clamped_count += 1

        return {
            "total_decisions": total,
            "actions_breakdown": {
                k: {"count": v, "pct": round(v / total * 100.0, 2)}
                for k, v in actions.items()
            },
            "triage_codes": triage_codes,
            "safety_clamps_triggered": clamped_count,
        }
