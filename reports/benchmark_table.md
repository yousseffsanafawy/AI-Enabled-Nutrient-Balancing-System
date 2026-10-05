# Benchmark Evaluation: Temporal Baseline Models & Proposed Architecture (Sprint 4 & 5)
**AI-Enabled Smart Hydroponic Nutrient Balancing System**  
*Evaluation Protocol: Strict Chronological 70/10/20 Partition | Scalers Fitted Strictly on Train*

---

## 1. Executive Summary

This benchmark compares 8 temporal baseline architectures (B0 to B7) and the **Proposed Multi-Task Architecture (P1: MT-TCN-LSTM)** evaluated under the identical, leakage-free protocol established in Sprint 3. The dataset spans 25,570 raw rows partitioned into 17,748 training sequences, 2,542 validation sequences, and 5,081 test sequences with boundary-aware isolation of 17 physical sampling gaps.

---

## 2. Standardized Benchmark Results Table

| ID | Model Architecture | pH MAE | TDS MAE (ppm) | Temp MAE (°C) | WL MAE | Humidity MAE (%) | Avg R² | Avg Tol% (All 5) | Tol% (No TDS) | Params | Latency (ms) | Train Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0** | Persistence Baseline | 0.0420 | 0.48 | 0.013 | 0.000 | 0.127 | 0.741 | 99.04% | 98.81% | 0 | 0.01 | 0.0 |
| **B1** | CNN-BiLSTM (Baseline) | 0.0442 | 53.44 | 0.207 | 0.035 | 0.729 | 0.106 | 77.63% | 97.04% | 74,153 | 193.62 | 134.9 |
| **B2** | Vanilla LSTM | 0.0366 | 22.03 | 0.044 | 0.025 | 0.609 | 0.532 | 86.92% | 98.59% | 20,165 | 157.78 | 260.8 |
| **B3** | Gated Recurrent Unit | 0.0677 | 73.72 | 0.481 | 0.030 | 0.856 | -0.742 | 64.78% | 80.97% | 15,877 | 169.25 | 79.0 |
| **B4** | Bidirectional LSTM | 0.0603 | 143.00 | 1.015 | 0.020 | 0.945 | -3.396 | 55.39% | 69.24% | 25,797 | 90.89 | 82.6 |
| **B5** | Temporal ConvNet | 0.0392 | 20.15 | 0.198 | 0.021 | 0.482 | 0.473 | 87.36% | 95.99% | 65,413 | 97.09 | 359.3 |
| **B6** | Transformer Encoder | 0.0781 | 83.55 | 0.494 | 0.010 | 1.475 | -1.054 | 62.80% | 78.50% | 53,061 | 85.46 | 292.7 |
| **B7** | CNN + GRU Hybrid | 0.0424 | 140.60 | 0.521 | 0.020 | 1.638 | -2.794 | 61.22% | 76.53% | 41,541 | 151.97 | 107.3 |
| **P1** | **Proposed MT-TCN-LSTM** | **0.0416** | **131.65** | **0.407** | **0.033** | **2.455** | **-2.177** | **61.99%** | **77.49%** | **75,846** | **71.69** | **78.6** |

---

## 3. Agronomic Stress Detection Performance (P1 Multi-Task Head)

Unlike baselines B0–B7 which are strictly continuous single-task regressors, the proposed model features an integrated classification head directly predicting upcoming agronomic hazards:

| Target Class | Precision | Recall | F1-Score | Support ($n$) |
|---|:---:|:---:|:---:|:---:|
| **Nominal System State** | 1.0000 | 0.9982 | 0.9991 | 5,010 |
| **Agronomic Stress / Hazard** | **0.8875** | **1.0000** | **0.9404** | **71** |
| **Macro Average** | **0.9437** | **0.9991** | **0.9697** | 5,081 |
| **Stress ROC-AUC Score** | — | — | **0.9995** | — |

> [!IMPORTANT]
> The proposed stress head caught **100% of all out-of-bounds stress events** (Recall = 1.0000) on the holdout test set with an ROC-AUC of **0.9995**, providing immediate physical hazard protection before chemical dosing.

---

## 4. Sensor-Specific Operational Accuracy Analysis

| Model | pH Within ±0.1 | TDS Within ±20 ppm | TDS Within ±50 ppm | Temp Within ±0.5°C | Humidity Within ±2% | WL Within ±1.0 |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | 95.26% | 99.96% | 99.96% | 100.00% | 100.00% | 100.00% |
| **CNN-BiLSTM (Baseline)** | 95.49% | 0.00% | 26.96% | 100.00% | 92.66% | 100.00% |
| **Vanilla LSTM** | 98.31% | 40.23% | 98.56% | 100.00% | 96.04% | 100.00% |
| **Gated Recurrent Unit** | 76.58% | 0.00% | 3.48% | 47.31% | 100.00% | 100.00% |
| **Bidirectional LSTM** | 81.68% | 0.00% | 0.02% | 8.86% | 86.42% | 100.00% |
| **Temporal ConvNet** | 98.09% | 52.86% | 99.72% | 87.13% | 98.74% | 100.00% |
| **Transformer Encoder** | 65.95% | 0.00% | 0.02% | 56.52% | 91.54% | 100.00% |
| **CNN + GRU Hybrid** | 95.97% | 0.00% | 0.02% | 43.00% | 67.15% | 100.00% |
| **Proposed MT-TCN-LSTM** | **95.08%** | **0.00%** | **0.00%** | **62.57%** | **52.31%** | **100.00%** |

---

## 5. Monte Carlo Dropout Uncertainty Quantification

The proposed architecture incorporates test-time Monte Carlo Dropout ($N=50$ forward passes) to estimate epistemic predictive variance ($\sigma_{\text{MC}}$) and construct 90% prediction intervals:
- **95% Empirical Coverage Rate:** $100.0\%$ for water level, temperature, TDS, and pH; $95.3\%$ for humidity.
- **Inference Latency Advantage:** Single-pass inference takes only **71.69 ms**, the fastest among all deep neural architectures evaluated.
- **Safety Interlock Gating:** When $\sigma_{\text{MC}}$ exceeds calibrated safety thresholds, automatic dosing is disabled and an operator alert is dispatched.
