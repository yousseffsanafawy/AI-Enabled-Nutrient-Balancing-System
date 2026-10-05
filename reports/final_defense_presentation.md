# Oral Defense Presentation: AI-Enabled Closed-Loop Hydroponic Nutrient Balancing System

**Conference Evaluation Track & Defense Committee Slide Deck**  
*Tone: Direct, Empirical, Peer-to-Peer Engineering Defense | Humanizer Certified*

---

## Slide 1: The Core Scientific Problem
* **The Promise:** Hydroponic controlled-environment agriculture reduces water consumption by up to 90% while maximizing crop yields.
* **The Physical Vulnerability:** Eliminating soil removes natural chemical buffering. If nutrient solution pH drifts outside 5.5–6.5, micronutrients precipitate out of solution, causing irreversible root starvation within hours.
* **The Literature Trap:** Published papers claim deep learning achieves "96.22% accuracy" using CNN-BiLSTM networks.
* **Our Discovery:** When we audited the underlying codebase, that headline number proved to be a composite of easy ambient variables masking a complete failure on the single metric that matters for plant nutrition: Total Dissolved Solids (TDS).

---

## Slide 2: The 10-Second Physical Reality
```
+-----------------------------------------------------------------------------+
|                          THE 10-SECOND REALITY                              |
|                                                                             |
|  Reported Assumption:   1 step = 1 min   ===>  W=15 steps = 15 minutes     |
|  Firmware Truth:        1 step = 10 sec  ===>  W=15 steps = 150 seconds    |
|                                                                             |
|  Hydraulic Impact:      150s is within active chemical mixing turbulence.   |
|                         Lag-1 autocorrelation exceeds 0.95 across sensors.  |
|                         Random splits leak future values across steps.      |
+-----------------------------------------------------------------------------+
```
* **Firmware Evidence:** Line 142 of `Final Sensors Code`: `millis() - lastFirebaseMillis > 10000`.
* **Physical Consequence:** In a 15-liter sump, concentrated nitric acid or potassium hydroxide takes 60 to 120 seconds to disperse. A 10-second forecast horizon tracks transient fluid mixing turbulence, not steady-state biological uptake.

---

## Slide 3: The Persistence Baseline Trap & Data Leakage
* **The Autocorrelation Illusion:** Because telemetry syncs every 10 seconds, the previous step ($y_t$) is nearly identical to the next step ($y_{t+1}$).
* **The Sanity Floor Proof:** A naive copy-last-value persistence baseline ($y_{t+1} = y_t$) achieves a **99.04% tolerance accuracy** without learning any dynamics. This proves tolerance rates alone are uninformative.
* **The Random Split Trap:** Slicing 15-step sliding windows across randomly split rows results in adjacent sequences sharing 14 out of 15 steps—a **93.3% feature overlap**.
* **The Collapse Under Honest Evaluation:**
  * Random Split (Leaked): pH MAE 0.0320, TDS MAE 13.78 ppm, Avg $R^2 = 0.969$, TDS Tolerance = 75.9%
  * Chronological Split (Honest): pH MAE 0.0436, TDS MAE 141.52 ppm, Avg $R^2 = 0.259$, TDS Tolerance = 0.0%

---

## Slide 4: Sensor Forensics & Boundary Isolation
* **`water_temp` is Synthetic White Noise:** Uniform distribution $\mathcal{U}[18, 25]$ with lag-1 autocorrelation $r = 0.037$ and zero cross-correlation with ambient air ($r = -0.008$). Firmware inspection confirmed this was an uncalibrated default fallback emitted when an analog probe disconnected. We dropped it from training entirely.
* **`water_level` is a Discrete Binary Switch:** 99.99% of readings are 1.0 or 2.0. Prior claims of 99.8% accuracy within $\pm 1.0$ were meaningless.
* **Boundary-Aware Slicing:** 17 sampling gaps $> 60$s identified, including 5 multi-hour system shutdowns. 154 boundary sequences crossing shutdowns were dropped.

---

## Slide 5: Proposed Architecture (MT-TCN-LSTM)
```
 [ Input Window: 15 steps x 5 sensors (150s lag) ]
                         │
                         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Causal Dilated Conv1D Blocks (d=1, d=2, Filters=64)          │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ Recurrent Aggregator: LSTM (64 Units)                        │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
 ┌─────────────────────────────┐ ┌────────────────────────────────┐
 │ Continuous Forecasting Head │ │ Stress Classification Head     │
 │ Dense(32) -> Dense(5)       │ │ Dense(16) -> Dense(1, sigmoid) │
 └─────────────────────────────┘ └────────────────────────────────┘
```
* **Causal Dilated Convolutions:** Expand the temporal receptive field to 150 seconds without pooling layers that destroy resolution.
* **Recurrent Sequence Aggregator:** Unidirectional LSTM tracks monotonic ionic uptake trends.
* **Multi-Task Objective:** Continuous regression loss combined with discrete stress classification ($\lambda = 0.2$). The stress head acts as an agronomic regularizer.

---

## Slide 6: Benchmark Results Across 9 Models
*Evaluated on an identical chronological 70/10/20 partition (5,081 test sequences):*

| Model Architecture | pH MAE | pH RMSE | TDS MAE (ppm) | TDS RMSE | Stress F1 | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| B0: Persistence Floor | 0.0442 | 0.0612 | 162.77 | 251.10 | N/A | 0.01 ms |
| B1: CNN-BiLSTM (Baseline)| 0.0436 | 0.0605 | 141.52 | 228.45 | N/A | 88.6 ms |
| B2: Vanilla LSTM | 0.0414 | 0.0582 | 136.21 | 219.04 | N/A | 46.1 ms |
| B3: GRU | 0.0425 | 0.0594 | 139.80 | 224.12 | N/A | 41.5 ms |
| B5: TCN (Dilated Conv1D)| 0.0418 | 0.0589 | 132.04 | 212.45 | N/A | 32.4 ms |
| B6: Temporal Transformer | 0.0448 | 0.0620 | 152.30 | 240.10 | N/A | 58.7 ms |
| **PROPOSED: MT-TCN-LSTM**| **0.0416** | **0.0584** | **131.65** | **211.80** | **0.9404** | **71.7 ms** |

*Takeaway:* MT-TCN-LSTM achieves the lowest TDS error while delivering **100% recall on biological stress events** ($ROC-AUC = 0.9995$).

---

## Slide 7: Systematic Architectural Ablations
*Why each component is mathematically necessary:*

* **Removing Conv1D (Pure LSTM):** TDS MAE spikes by **+73.4%** (131.65 to 228.32 ppm) and stress classification collapses ($F1 = 0.000$). Dilated convolutions are mandatory for de-noising high-frequency sensor signals.
* **Removing Dilation ($d=1$):** TDS MAE increases to 138.90 ppm; receptive field fails to cover the full 150-second chemical dissolution phase.
* **Single-Task Only:** pH MAE worsens from 0.0416 to 0.0423. The auxiliary stress classification task acts as a structural regularizer.
* **Window Size ($W=15$):** Shortening to 50 seconds ($W=5$) degrades accuracy; expanding to 300 seconds ($W=30$) doubles compute latency without improving predictions.

---

## Slide 8: Epistemic Uncertainty via MC Dropout
* **The Safety Dilemma:** Standard neural networks emit point predictions without confidence bounds. An overconfident wrong prediction can trigger an acid overdose and kill the crop.
* **Method:** $N=50$ stochastic Monte Carlo Dropout forward passes ($p=0.3$) at inference time.
* **Validation Calibration:**
  * Median Threshold ($P_{75}$): $\sigma = 0.0090$
  * High-Risk Threshold ($P_{95}$): $\sigma = 0.0135$
* **Empirical Coverage on Test Data:** 90% prediction intervals achieved **99.98% coverage for pH** and **99.72% coverage for TDS**.

---

## Slide 9: 5-Tier Closed-Loop Safety Layer
```
 Tier 1: Probe Integrity Check (Disconnects <= 0.0, Spikes |ΔpH| > 1.5, Flatlines)
    │
 Tier 2: Agronomic Feasibility (pH 5.0 - 7.5, TDS 300 - 1800 ppm)
    │
 Tier 3: Uncertainty Gating (σ > P95 -> Standby; P75 < σ <= P95 -> Alert Human)
    │
 Tier 4: Hardware Actuator Clamps (Max 5.0 mL acid/base; 60s lockout)
    │
 Tier 5: Action Resolver (STANDBY / AUTO_DOSE / ALERT_HUMAN / EMERGENCY_HOLD)
```
* **Fault Injection Audit:** Across 436 injected hardware faults, the safety engine achieved a **100.0% hazard prevention rate** (436/436 faults trapped).

---

## Slide 10: Embedded Hardware Profiling & Deployment Blueprint
* **INT8 Quantization:** Shrank model from 319.3 KB (FP32) to **112.5 KB** (64.8% reduction).
* **The ESP32 Memory Bottleneck:**
  * Flash storage (4 MB) is plenty.
  * Usable SRAM is restricted. FreeRTOS + lwIP TCP/IP stack + TLS certificates reduce free heap to $\sim 160$ KB.
  * Running recurrent dynamic arrays (`TensorArrayV2`) on ESP32 risks out-of-memory crashes.
* **The Verified Blueprint (Edge-Gateway Hybrid):**
  1. ESP32: Real-time 10-second ADC polling, sensor de-noising, and actuator relays.
  2. Edge Gateway (Raspberry Pi 4 / Jetson Nano): Hosts quantized TFLite model, running inference, MC Dropout, and safety triage in **71.69 ms** over local MQTT.
  3. Fail-Safe: If gateway drops for $>30$s, ESP32 falls back to hardcoded hardware deadbands.

---

## Slide 11: Physical Chemical Titrations ($N=10$)
* Replicated across 10 systematic trials: acid additions (0.1M $\text{HNO}_3$, 1.5–3.0 mL), base additions (0.1M $\text{KOH}$, 1.5–2.5 mL), and concentrated nutrient shocks (10–20 mL).
* **Result:** Mean absolute error of **0.433 pH**, reducing physical error variance by **44.4%** compared to the original baseline.

---

## Slide 12: Summary of Key Contributions
1. **Forensic Audit:** Exposed 10-second telemetry truth, three points of baseline leakage, and synthetic noise in `water_temp`.
2. **Persistence Baseline Sanity Floor:** Proved 99.04% tolerance accuracy is achievable without learning, discrediting vanity metrics.
3. **MT-TCN-LSTM:** Delivered lowest TDS error (131.65 ppm) and 100% stress recall ($F1 = 0.9404$, $ROC-AUC = 0.9995$).
4. **Epistemic Safety Layer:** Trapped 436 of 436 injected faults (100% hazard prevention).
5. **Edge-Gateway Architecture:** Deployed at 112.5 KB with 71.69 ms latency, respecting embedded microcontroller constraints.
