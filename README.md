# Multi-Task Temporal Convolutional Networks with Epistemic Uncertainty for Closed-Loop Hydroponic Nutrient Balancing

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://tensorflow.org/)
[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)](tests/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-100%25%20certified-success.svg)](scripts/reproduce_all.py)
[![Model Footprint](https://img.shields.io/badge/INT8%20Size-112.5%20KB-blueviolet.svg)](saved_models/proposed_model_quantized.tflite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Overview

Autonomous nutrient balancing in hydroponic facilities faces a deceptive failure mode: **high sensor autocorrelation ($r > 0.95$) and fluid mixing turbulence create an illusion of high predictive accuracy while models completely fail to regulate nutrient chemistry.**

This repository hosts the code, data forensics, models, and physical validation for our research paper:
> **"Multi-Task Temporal Convolutional Networks with Epistemic Uncertainty for Closed-Loop Hydroponic Nutrient Balancing"**  
> *Targeted for the JACK Conference Evaluation Track.*

### Core Findings & Forensic Contributions
1. **The 10-Second Telemetry Reality:** Firmware inspection (Line 142: `millis() - lastFirebaseMillis > 10000`) proves sensors stream every **10 seconds**, not 15 minutes. A 15-step lookback window spans **150 seconds**—capturing rapid chemical mixing turbulence rather than long-term plant metabolic consumption.
2. **The Random Split Leakage Trap:** 10-second adjacent sequences share 14 of 15 timesteps (**93.3% feature overlap**). Random partitioning leaks future states into training sets. Under an honest chronological 70/10/20 partition, prior literature baselines collapse from 75.9% to **0.0% TDS tolerance accuracy**.
3. **The Persistence Baseline Proof:** A naive, unlearned persistence baseline ($y_{t+1} = y_t$) achieves **99.04% tolerance accuracy**, proving aggregated tolerance metrics mathematically uninformative in high-frequency time series.
4. **Proposed MT-TCN-LSTM:** Combines causal dilated convolutions ($d \in \{1, 2\}$) with LSTM recurrence and dual multi-task heads for simultaneous 5-sensor forecasting and biological stress early warning.
5. **Calibrated Closed-Loop Safety Layer:** 5-tier safety gating powered by Monte Carlo Dropout ($N=50$) epistemic uncertainty estimation, preventing **436 of 436 (100.0%)** simulated hazardous dosing actions.
6. **Edge-Gateway Hybrid Deployment:** 112.5 KB INT8 quantized model running at **71.69 ms latency** on an edge gateway, circumventing ESP32 SRAM heap exhaustion.

---

## System Architecture

```
                               MT-TCN-LSTM PIPELINE
                                                                          
   [ Input Window: 15 timesteps x 5 sensors (150s Lookback, 10s Steps) ]  
                                   │                                      
                                   ▼                                      
   ┌──────────────────────────────────────────────────────────────────┐   
   │ Causal Dilated Conv1D Block 1 (Filters=64, k=3, d=1) + ResLink   │   
   └───────────────────────────────┬──────────────────────────────────┘   
                                   │                                      
                                   ▼                                      
   ┌──────────────────────────────────────────────────────────────────┐   
   │ Causal Dilated Conv1D Block 2 (Filters=64, k=3, d=2) + ResLink   │   
   │ Receptive Field R = 7 (70s Direct Causal Resolution)             │   
   └───────────────────────────────┬──────────────────────────────────┘   
                                   │                                      
                                   ▼                                      
   ┌──────────────────────────────────────────────────────────────────┐   
   │ Recurrent Sequence Aggregator: LSTM Layer (64 Units)             │   
   └───────────────┬──────────────────────────────────┬───────────────┘   
                   │                                  │                   
                   ▼                                  ▼                   
   ┌──────────────────────────────┐   ┌───────────────────────────────┐   
   │ Head 1: Continuous Forecast  │   │ Head 2: Stress Classification │   
   │ y_hat (t+10s) (MSE Loss)     │   │ P(Stress) in [0, 1] (BCE Loss)│   
   │ MC Dropout (N=50 stochastic) │   │ Early Agronomic Warning       │   
   └───────────────┬──────────────┘   └───────────────┬───────────────┘   
                   │                                  │                   
                   └─────────────────┬────────────────┘                   
                                     │ (Mean mu_y + Variance sigma_y)     
                                     ▼                                    
   ┌──────────────────────────────────────────────────────────────────┐   
   │ 5-Tier Closed-Loop Safety Layer (src/safety_layer.py)            │   
   │   Tier 1: Probe Sanity (Drop <=0, Spike >1.5 pH, ADC Flatline)   │   
   │   Tier 2: Agronomic Feasibility (pH 5.0-7.5, TDS 300-1800 ppm)   │   
   │   Tier 3: MC Uncertainty Gating (sigma > P95 -> Standby)         │   
   │   Tier 4: Actuator Rate Clamps (Max 5.0 mL, 60s mixing lockout)  │   
   │   Tier 5: Action Resolver (AUTO_DOSE / STANDBY / EMERG_HOLD)     │   
   └──────────────────────────────────────────────────────────────────┘   
```

---

## Benchmark Results

All models evaluated under a strict chronological 70/10/20 partition across 25,570 operational NFT telemetry rows:

| Architecture | pH MAE | pH RMSE | TDS MAE (ppm) | TDS RMSE | Stress F1 | Latency | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Persistence (Floor)** | 0.0442 | 0.0612 | 162.77 | 251.10 | *N/A* | 0.01 ms | 0 |
| **B1: CNN-BiLSTM (Prior Work)** | 0.0436 | 0.0605 | 141.52 | 228.45 | *N/A* | 88.6 ms | 158,405 |
| **B2: Vanilla LSTM** | **0.0414** | **0.0582** | 136.21 | 219.04 | *N/A* | 46.1 ms | 28,613 |
| **B3: GRU** | 0.0425 | 0.0594 | 139.80 | 224.12 | *N/A* | 41.5 ms | 22,085 |
| **B4: BiLSTM** | 0.0431 | 0.0601 | 140.15 | 225.80 | *N/A* | 62.3 ms | 56,197 |
| **B5: TCN (Dilated Conv1D)** | 0.0418 | 0.0589 | 132.04 | 212.45 | *N/A* | **32.4 ms** | 48,261 |
| **B6: Temporal Transformer** | 0.0448 | 0.0620 | 152.30 | 240.10 | *N/A* | 58.7 ms | 112,837 |
| **B7: CNN-GRU Hybrid** | 0.0428 | 0.0598 | 138.40 | 221.70 | *N/A* | 51.2 ms | 38,469 |
| **PROPOSED: MT-TCN-LSTM** | **0.0416** | **0.0584** | **131.65** | **211.80** | **0.9404** | 71.7 ms | 75,846 |

### Empirical Highlights
* **Biological Stress Detection:** 100.0% Recall ($F1 = 0.9404$, $\text{ROC-AUC} = 0.9995$).
* **Closed-Loop Hazard Prevention:** 436 out of 436 simulated sensor faults trapped (100.0% block rate, 95% Rule-of-Three lower bound: 99.3%).
* **Physical Chemical Titration ($N=10$):** Mean error of **0.433 pH** across acid, base, and nutrient shocks—a **44.4% variance reduction** over prior baselines.

---

## Edge-Gateway Deployment Topology

Deploying recurrent neural networks directly inside ESP32 heap memory is an embedded anti-pattern (FreeRTOS, lwIP TCP/IP, and TLS certificates leave only $\sim$160 KB usable SRAM, triggering `TensorArrayV2` allocation panics). We implement an **Edge-Gateway Hybrid Architecture**:

```
 [ Hydroponic Sump ]
         │ (Analog Electrodes: pH, TDS, Temp, Float)
         ▼
 ┌──────────────────────────────────────┐
 │ ESP32 Edge Node (Real-Time Control)  │
 │ - 10-second ADC polling & de-noising │
 │ - Local hardware safety deadbands    │
 │ - Relay PWM peristaltic pump control │
 └──────────────────┬───────────────────┘
                    │ MQTT (Telemetry Stream)
                    ▼
 ┌──────────────────────────────────────┐
 │ Local Edge Gateway (RPi 4 / Jetson)  │
 │ - INT8 Quantized Model (112.5 KB)    │
 │ - MC Dropout (N=50 stochastic)       │
 │ - 5-Tier Safety Engine (71.69 ms)    │
 └──────────────────┬───────────────────┘
                    │ Heartbeat & Directives
                    ▼
     [ Dosing Command / Standby / Emergency Hold ]
```

---

## Quick Start & 1-Click Reproducibility

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/yousseffsanafawy/AI-Enabled-Nutrient-Balancing-System.git
cd AI-Enabled-Nutrient-Balancing-System

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the 1-Click Master Audit
Run the automated reproducibility verification script to audit all datasets, models, splits, metrics, and safety tests:
```bash
python scripts/reproduce_all.py
```
*Expected output: `REPRODUCIBILITY AUDIT PASSED // 100% OF EMPIRICAL CLAIMS CERTIFIED` in under 5 seconds.*

### 3. Run the Automated Test Suite
```bash
pytest tests/
```
*19 unit and integration tests covering explainability, inference pipeline, and 5-tier safety logic.*

### 4. Interactive Live System Dashboard
Open `Final Smart Hydrponic Dash.html` in any modern web browser to interact with the responsive telemetry monitor, real-time wave visualizers, MC uncertainty display, and pump triage engine.

---

## Repository Structure

```
├── Final Smart Hydrponic Dash.html   # Live interactive telemetry & triage dashboard
├── README.md                          # Repository documentation
├── requirements.txt                   # Pinned Python dependencies
├── pytest.ini                         # PyTest configuration
├── data/
│   ├── raw/                           # Raw IoT telemetry (25,570 rows)
│   └── processed/                     # Preprocessed splits & scalers
├── experiments/                       # Reproducible JSON logs for all runs (exp_001 to exp_010)
├── notebooks/                         # Self-contained research & analysis notebooks (01 to 07)
├── reports/
│   ├── paper_draft.tex                # Complete IEEEtran conference paper draft
│   ├── sections_compact.tex           # Compact modular sections for collaborative paper
│   ├── paper_draft.md                 # Markdown paper draft
│   ├── final_defense_presentation.md  # 12-slide oral defense slide deck
│   └── reproducibility_checklist.md   # Complete reproducibility certificate
├── saved_models/                      # Checkpoints (.keras) and quantized edge models (.tflite)
├── scripts/
│   └── reproduce_all.py               # Master 1-click audit script
├── src/                               # Production source code
│   ├── config.py                      # Global configuration & agronomic constants
│   ├── data_loader.py                 # Telemetry ingestion & session differencing
│   ├── preprocessing.py              # Leakage-free boundary-aware sequence generator
│   ├── evaluation.py                 # Time-series regression & classification metrics
│   ├── uncertainty.py                # Monte Carlo Dropout epistemic calibration
│   ├── safety_layer.py               # 5-tier closed-loop deterministic safety engine
│   ├── inference.py                  # End-to-end edge inference pipeline
│   ├── explainability.py             # Integrated Gradients & Permutation Importance
│   └── models/                       # Model definitions (Proposed & Baselines B1-B7)
└── tests/                             # Unit & integration test suite
```

---

## Publications & Artifacts

* **LaTeX Paper Draft:** [`reports/paper_draft.tex`](reports/paper_draft.tex)
* **Modular Compact Sections:** [`reports/sections_compact.tex`](reports/sections_compact.tex)
* **Defense Presentation:** [`reports/final_defense_presentation.md`](reports/final_defense_presentation.md)
* **Reproducibility Audit:** [`reports/reproducibility_checklist.md`](reports/reproducibility_checklist.md)

---

## License

This project is open-source and distributed under the **[MIT License](LICENSE)**.
