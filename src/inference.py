"""
src/inference.py — Real-Time Production Inference Pipeline & Dashboard Bridge.

Sprint 10 Implementation:
  - Loads trained MT-TCN-LSTM model and training-fitted scalers (scaler_X, scaler_y).
  - Accepts streaming 15-step lookback windows (150s of physical IoT telemetry).
  - Executes Monte Carlo Dropout (N=50 stochastic passes) for epistemic uncertainty.
  - Integrates 5-tier Closed-Loop Safety Layer (safety_layer.py) for physical actuation gating.
  - Integrates Agronomic Reasoning Engine (explainability.py) for real-time explanations.
  - Provides a self-contained HTTP/JSON API server for dashboard telemetry streaming.
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    FEATURES, N_FEATURES, TIME_STEPS, RANDOM_SEED,
    PROCESSED_DIR, MODELS_DIR, RAW_CSV,
)
from src.safety_layer import (
    make_dosing_decision,
    validate_sensor_reading,
    DEFAULT_TARGET_SETPOINTS,
)
from src.explainability import AgronomicReasoningEngine, compute_integrated_gradients


class ProductionInferenceEngine:
    """
    Production inference engine coordinating model prediction, epistemic uncertainty,
    hardware safety checks, and agronomic explanations.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_dir: str = PROCESSED_DIR,
    ):
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "proposed_model_best.keras")

        print(f"[InferenceEngine] Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_name = getattr(self.model, "name", "Proposed_MT_TCN_LSTM")
        self.param_count = self.model.count_params()

        # Load scalers
        scaler_x_path = os.path.join(scaler_dir, "scaler_X.pkl")
        scaler_y_path = os.path.join(scaler_dir, "scaler_y.pkl")

        if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
            raise FileNotFoundError(
                f"Scalers not found in {scaler_dir}. Run preprocessing pipeline first."
            )

        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        self.feature_names = list(FEATURES)

        # Agronomic Engine
        self.agronomic_engine = AgronomicReasoningEngine(feature_names=self.feature_names)

        # Warm up graph
        dummy = np.zeros((1, TIME_STEPS, N_FEATURES), dtype=np.float32)
        _ = self.model(dummy, training=False)
        print("[InferenceEngine] Initialization and graph warmup complete.")

    def run_inference(
        self,
        window_physical: np.ndarray,
        n_mc_samples: int = 50,
        run_attribution: bool = False,
    ) -> Dict[str, Any]:
        """
        Run end-to-end inference on a 15-step lookback window.

        Args:
          window_physical: (15, 5) or (1, 15, 5) array of raw physical sensor readings.
          n_mc_samples: Number of stochastic forward passes for MC Dropout.
          run_attribution: Whether to compute Integrated Gradients (slightly higher latency).

        Returns:
          Dictionary with forecasts, uncertainty bounds, safety triage, and agronomic diagnostics.
        """
        t_start = time.perf_counter()

        if window_physical.ndim == 2:
            window_physical = np.expand_dims(window_physical, axis=0)

        # 1. Scaling (strictly using training-fitted scaler)
        N, T, F = window_physical.shape
        window_reshaped = window_physical.reshape(-1, F)
        window_scaled = self.scaler_X.transform(window_reshaped).reshape(N, T, F).astype(np.float32)

        # 2. Monte Carlo Dropout (N stochastic passes)
        # Tile across batch for fast parallel execution
        tiled_input = tf.repeat(tf.convert_to_tensor(window_scaled), repeats=n_mc_samples, axis=0)
        stochastic_preds = self.model(tiled_input, training=True)

        if isinstance(stochastic_preds, dict):
            mc_forecasts = stochastic_preds["forecast_output"].numpy()
            mc_stress = stochastic_preds["stress_output"].numpy()
        else:
            mc_forecasts = stochastic_preds[0].numpy()
            mc_stress = stochastic_preds[1].numpy()

        # Epistemic mean and standard deviation in scaled space
        mean_forecast_scaled = np.mean(mc_forecasts, axis=0)  # (5,)
        std_forecast_scaled = np.std(mc_forecasts, axis=0)    # (5,)
        mean_stress_prob = float(np.mean(mc_stress))

        # Unscale to physical units
        mean_forecast_physical = self.scaler_y.inverse_transform(
            mean_forecast_scaled.reshape(1, -1)
        )[0]

        # Convert standard deviation to physical scale
        scale_factors = self.scaler_y.scale_  # scale_ = 1 / (max - min)
        std_forecast_physical = std_forecast_scaled / scale_factors

        curr_physical = window_physical[0, -1, :]  # Most recent reading in window

        # Pack sensor dictionaries
        curr_dict = {self.feature_names[i]: float(curr_physical[i]) for i in range(F)}
        pred_dict = {self.feature_names[i]: float(mean_forecast_physical[i]) for i in range(F)}
        sigma_dict = {self.feature_names[i]: float(std_forecast_physical[i]) for i in range(F)}

        # 3. Closed-Loop Safety Layer Triage
        calibrated_thresholds = {
            "pH": {"med_threshold": 0.0090, "high_threshold": 0.0135},
            "TDS": {"med_threshold": 0.0110, "high_threshold": 0.0150},
            "DHT_temp": {"med_threshold": 0.0080, "high_threshold": 0.0120},
            "DHT_humidity": {"med_threshold": 0.0100, "high_threshold": 0.0140},
            "water_level": {"med_threshold": 0.0050, "high_threshold": 0.0100},
        }

        # Build history buffer for rate of change check
        history_buffer = []
        for step_i in range(T - 1):
            history_buffer.append({
                self.feature_names[f]: float(window_physical[0, step_i, f])
                for f in range(F)
            })

        validity_report = validate_sensor_reading(
            current_reading=curr_dict,
            history_buffer=history_buffer,
        )

        safety_result = make_dosing_decision(
            prediction=pred_dict,
            uncertainty=sigma_dict,
            sensor_validity=validity_report,
            calibrated_thresholds=calibrated_thresholds,
            current_reading=curr_dict,
            target_setpoints=DEFAULT_TARGET_SETPOINTS,
        )

        # 4. Attribution & Agronomic Reasoning
        attribution_matrix = np.zeros((TIME_STEPS, F))
        if run_attribution:
            attribution_matrix = compute_integrated_gradients(
                model=self.model,
                x_input=window_scaled[0],
                target_head="forecast_output",
                target_idx=3,  # pH
                steps=25,
            )

        explanation = self.agronomic_engine.explain_prediction(
            current_readings=curr_dict,
            predicted_readings=pred_dict,
            attribution_matrix=attribution_matrix,
            stress_probability=mean_stress_prob,
            uncertainty_sigma=sigma_dict,
            target_sensor="pH",
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "inference_latency_ms": round(latency_ms, 2),
            "model_metadata": {
                "name": self.model_name,
                "parameters": self.param_count,
                "mc_samples": n_mc_samples,
                "lookback_steps": TIME_STEPS,
                "lookback_seconds": TIME_STEPS * 10,
            },
            "current_telemetry": curr_dict,
            "forecast_next_step": pred_dict,
            "uncertainty_sigma": sigma_dict,
            "stress_detection": {
                "stress_probability": round(mean_stress_prob, 4),
                "is_stress": bool(mean_stress_prob > 0.5),
                "risk_tier": explanation["risk_level"],
            },
            "safety_layer": {
                "action": safety_result.action,
                "risk_level": safety_result.risk_level,
                "triage_code": safety_result.triage_code,
                "reason": safety_result.reason,
                "actuator_commands": safety_result.actuator_commands,
                "safety_clamp_applied": safety_result.safety_clamp_applied,
            },
            "agronomic_diagnosis": {
                "top_driver": explanation["top_driver"],
                "top_driver_share": round(explanation["top_driver_share"], 3),
                "secondary_driver": explanation["secondary_driver"],
                "secondary_driver_share": round(explanation["secondary_driver_share"], 3),
                "temporal_recent_share": round(explanation["temporal_recent_share"], 3),
                "recommended_action": explanation["recommended_action"],
                "formatted_report": explanation["formatted_report"],
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. STANDALONE REST API SERVER FOR DASHBOARD BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

class DashboardTelemetryBridge:
    """
    Lightweight HTTP server providing streaming live telemetry, real-time predictions,
    and system status to the web dashboard.
    """

    def __init__(
        self,
        engine: ProductionInferenceEngine,
        port: int = 8085,
        dataset_csv: str = RAW_CSV,
    ):
        self.engine = engine
        self.port = port
        self.df_stream = pd.read_csv(dataset_csv)[FEATURES].dropna().values
        self.stream_idx = 0
        self.server = None
        self.server_thread = None

    def get_next_simulated_window(self) -> np.ndarray:
        """Fetch next 15-step sequential window from historical dataset."""
        if self.stream_idx + TIME_STEPS >= len(self.df_stream):
            self.stream_idx = 0  # Loop back
        window = self.df_stream[self.stream_idx : self.stream_idx + TIME_STEPS]
        self.stream_idx += 1
        return window

    def start(self):
        bridge_self = self

        class RequestHandler(BaseHTTPRequestHandler):
            def _send_cors_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self):
                self.send_response(200)
                self._send_cors_headers()
                self.end_headers()

            def do_GET(self):
                self._send_cors_headers()
                if self.path == "/api/status":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    resp = {
                        "status": "ONLINE",
                        "model": bridge_self.engine.model_name,
                        "parameters": bridge_self.engine.param_count,
                        "port": bridge_self.port,
                        "features": bridge_self.engine.feature_names,
                    }
                    self.wfile.write(json.dumps(resp).encode("utf-8"))

                elif self.path == "/api/stream/next":
                    # Advance one simulated 10-second step and run live inference
                    window = bridge_self.get_next_simulated_window()
                    result = bridge_self.engine.run_inference(window, n_mc_samples=30)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode("utf-8"))

                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                self._send_cors_headers()
                if self.path == "/api/predict":
                    content_len = int(self.headers.get("Content-Length", 0))
                    post_body = self.rfile.read(content_len)
                    try:
                        data = json.loads(post_body.decode("utf-8"))
                        window = np.array(data["window"], dtype=np.float32)
                        result = bridge_self.engine.run_inference(window)
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode("utf-8"))
                    except Exception as e:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress noisy HTTP request logging
                return

        self.server = HTTPServer(("0.0.0.0", self.port), RequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[DashboardTelemetryBridge] Server running on http://localhost:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[DashboardTelemetryBridge] Server stopped.")


if __name__ == "__main__":
    engine = ProductionInferenceEngine()
    dummy_window = np.array([
        [2.0, 24.5, 850.0, 5.85, 65.0] for _ in range(TIME_STEPS)
    ])
    result = engine.run_inference(dummy_window, run_attribution=True)
    print("\n--- Sample Live Inference Output ---")
    print(json.dumps(result, indent=2))
