# Project Progress Summary: Sprints 1 to 3

**AI-Enabled Smart Hydroponic Nutrient Balancing System**  
*Document Version: 1.0 | Date: October 2026 | Status: Complete & Verified*

---

## 📌 Executive Overview

This report provides a concise, high-level review of all work completed to date across **Sprint 1 (EDA & Forensic Audit)**, **Sprint 2 (Baseline Reproduction)**, and **Sprint 3 (Leakage Audit & Rigorous Splitting)**.

### Quick Status Dashboard

| Sprint | Phase Name | Status | Key Deliverable | Primary Outcome |
| --- | --- | :---: | --- | --- |
| **Sprint 1** | EDA & Forensic Audit | ✅ Done | [`reports/audit_report.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/audit_report.md) | Discovered 10s sampling interval; uncovered 5 major shutdowns; debunked `water_temp` (noise) and `water_level` (discrete state). |
| **Sprint 2** | Baseline Exact Reproduction | ✅ Done | [`notebooks/02_reproduction_script.py`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/notebooks/02_reproduction_script.py) | Faithfully reproduced CNN-BiLSTM baseline; confirmed 3 methodological leakage points; reconciled original "96% accuracy". |
| **Sprint 3** | Leakage Audit & Rigorous Splitting | ✅ Done | [`reports/evaluation_protocol.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/evaluation_protocol.md) | Built boundary-aware pipeline; eliminated 154 corrupt sequences; proved Random Split inflation vs honest Chronological 70/10/20. |

---

## 🔍 Sprint 1: Forensic Dataset Audit & Exploratory Data Analysis

### What Was Done

1. **Dataset Discovery & Inventory:**
   - Audited all 6 raw CSV files; identified the canonical ground truth: `IoTData_25K_without_interpolation.csv` (25,570 rows × 14 columns, covering Dec 21–26, 2023).
2. **True Timebase Discovery:**
   - Established that timestamps increment by **10 seconds per step** ($\Delta t = 10\text{ s}$).
   - Corrected literature misconception: a lookback window of 15 steps represents **150 physical seconds (2.5 minutes)**, *not* 15 minutes.
3. **Multi-Session Shutdown Analysis:**
   - Detected 17 gaps $>60\text{ s}$, including **5 major shutdowns** ($1.06\text{ to }24.72\text{ hours}$), dividing data into 6 operational sessions.
   - Identified massive manual nutrient dosing at Row 4,244 (Dec 23), causing a +1,095 ppm TDS jump within a single recorded step.
4. **Two Critical Forensic Anomalies Solved:**
   - **`water_temp` is Synthetic White Noise:** Proven mathematically to follow uniform distribution $\mathcal{U}[18, 25]$ with lag-1 autocorrelation $r = 0.037$ and zero cross-correlation with any system sensor. **Permanently excluded.**
   - **`water_level` is a Discrete State Sensor:** 99.99% of values are exactly $1.0$ (low) or $2.0$ (adequate). A reported "99.8% accuracy within $\pm 1.0$" was mathematically vacuous because the total range is only $1.0$.
5. **Reconciling the "96.22% Accuracy" Metric:**
   - Proved that "accuracy" was actually a **within-tolerance regression rate**, averaging four high-tolerance sensors with a low-performing TDS sensor.

### Key Deliverables

- Comprehensive report: [`reports/audit_report.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/audit_report.md)
- Audit figures: `fig01` to `fig06` in [`reports/figures/`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/figures/)
- Audit experiment log: [`experiments/exp_001_data_audit.json`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/experiments/exp_001_data_audit.json)

---

## 🔁 Sprint 2: Baseline Code Audit & Exact Reproduction

### What Was Done

1. **Exact Reproduction Pipeline:**
   - Reproduced the original `AI_PBL (1).ipynb` training pipeline in modular code ([`notebooks/02_reproduction_script.py`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/notebooks/02_reproduction_script.py)).
   - Architecture: Conv1D(128, k=3) → MaxPool(2) → BiLSTM(50) → Dropout(0.4) → Dense(5).
   - Training: Adam(lr=1e-4), loss=MSE, batch_size=32, 35 epochs.
2. **Three Confirmed Methodological Flaws:**
   - **Scaling Leakage:** Scaler fitted on 100% of data before split.
   - **Evaluation Leakage:** Test set supplied as `validation_data` during training.
   - **Discontinuity Blindness:** Sliding window generated sequences directly across 24-hour shutdowns.
3. **Empirical Reproduction Verification:**
   - Baseline trained in 88.6 seconds.
   - Accuracies verified: Water Level (100.0%), Temperature (99.9%), pH (98.3%), Humidity (91.5%).
   - Mean 4-sensor accuracy: **97.43%**.
   - TDS at strict $\pm 20$ ppm tolerance was only **3.8%**, proving that TDS error was diluted in the original aggregated claim.

### Key Deliverables

- Model checkpoint: `saved_models/baseline_cnn_bilstm_reproduction.keras`
- Loss curve and forecast tracking: `fig07` and `fig08` in [`reports/figures/`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/figures/)
- Reproduction experiment log: [`experiments/exp_002_baseline_reproduction.json`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/experiments/exp_002_baseline_reproduction.json)

---

## ⚖️ Sprint 3: Leakage Audit & Rigorous Splitting Protocols

### What Was Done

1. **Leakage-Free Preprocessing Pipeline:**
   - Updated [`src/preprocessing.py`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/src/preprocessing.py) with **strict chronological 70% train / 10% val / 20% test** partitioning.
   - **Train-Only Scaling:** Scaler fit strictly on training rows; validation and test sets transformed using training statistics.
   - **Boundary-Aware Sequence Slicing:** Sequences generated strictly within continuous segments; dropped **154 corrupt bridging sequences**.
2. **Standardized Unified Evaluation Metrics:**
   - Updated [`src/evaluation.py`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/src/evaluation.py) to compute per-sensor MAE, RMSE, $R^2$, MAPE, operational within-tolerance rates, and multi-threshold TDS sensitivity ($\pm 10\text{ to }\pm 100\text{ ppm}$).
3. **Three-Way Split Protocol Benchmark:**
   Trained and evaluated the CNN-BiLSTM baseline under three distinct partitioning strategies:
   - **Strategy 1 (Random Split — Negative Control):** Shuffled sequences across partitions.
   - **Strategy 2 (Chronological 70/10/20):** Strict temporal ordering, train-only scaling, no future leakage.
   - **Strategy 3 (Group / Session Split):** Held out Session 6 entirely (12,100 rows) to test cross-session domain generalization.

### Empirical Split Comparison Summary Table

| Strategy | pH MAE | pH $R^2$ | pH Tol% ($\pm 0.1$) | TDS MAE (ppm) | TDS $R^2$ | TDS Tol% ($\pm 20$ ppm) | Temp MAE (°C) | Temp $R^2$ | Humidity MAE (%) | Humidity $R^2$ | Avg $R^2$ | Avg Tol% (All 5) | Tol% (No TDS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Random Split** *(Negative Control — Leakage)* | **0.032** | **0.903** | **97.8%** | **13.78** | **0.990** | **75.9%** | **0.070** | **0.967** | **0.326** | **0.983** | **0.969** | **94.6%** | **99.3%** |
| **Chronological Split** *(70/10/20 Rigorous)* | **0.044** | -0.225 | **95.1%** | **41.30** | -0.220 | **0.02%** | **0.215** | **0.815** | **0.621** | **0.924** | **0.259** | **78.1%** | **97.7%** |
| **Group / Session Split** *(Physical Run Holdout)* | **0.082** | -1.202 | **61.9%** | **89.12** | 0.367 | **0.00%** | **0.085** | **0.953** | **1.459** | **0.719** | **0.168** | **69.3%** | **86.6%** |

### Key Scientific Findings for Paper

1. **The Autocorrelation Trap:**  
   Random split yields an inflated $R^2 = 0.969$ and $75.9\%$ TDS accuracy because adjacent sequences share 14/15 timesteps ($93.3\%$ feature overlap). The model memorizes neighboring values rather than learning forecasting. Random splitting is scientifically invalid for time series.
2. **Honest Out-of-Sample Performance:**  
   Under Chronological 70/10/20, ambient variables remain highly predictable (Temp $R^2 = 0.815$, Humidity $R^2 = 0.924$, 4-sensor tolerance rate = **97.67%**), while chemical parameters (pH, TDS) reflect true physical mixing latency and sensor noise.
3. **The "Water Level" Illusion Exposed:**  
   In Group Split, water level dropped from 2 to 1 in held-out Session 6. The model predicted $\approx 1.96$ (error = $0.96$). Yet because tolerance is $\pm 1.0$, it still achieved **100.0% within-tolerance accuracy**, proving tolerance-based accuracy alone is misleading without MAE/RMSE.

### Key Deliverables

- Formal protocol document: [`reports/evaluation_protocol.md`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/evaluation_protocol.md)
- Interactive analysis notebook: [`notebooks/03_leakage_comparison.ipynb`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/notebooks/03_leakage_comparison.ipynb)
- Experiment records: [`experiments/exp_001_chronological_split.json`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/experiments/exp_001_chronological_split.json) and [`experiments/exp_003_leakage_comparison.json`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/experiments/exp_003_leakage_comparison.json)
- Comparative figures: `fig09` to `fig12` in [`reports/figures/`](file:///c:/Users/youse/Downloads/JackHom/REPO/AI-Enabled-Nutrient-Balancing-System/reports/figures/)
- Trained model checkpoints: `saved_models/model_chronological_split.keras`, `model_random_split.keras`, `model_group_split.keras`

---

## 🗺️ Master Repository Map

```
AI-Enabled-Nutrient-Balancing-System/
├── data/
│   └── raw/IoTData_25K_without_interpolation.csv     # Verified canonical dataset (25,570 rows)
├── src/
│   ├── config.py                                      # Central constants, tolerances, paths
│   ├── preprocessing.py                               # Leakage-free, boundary-aware pipeline (3 strategies)
│   ├── evaluation.py                                  # Unified metrics (MAE, RMSE, R², WTR, TDS sensitivity)
│   ├── models/baseline_cnn_bilstm.py                  # Standard baseline architecture definition
│   └── utils/seed.py                                  # Deterministic seed management
├── notebooks/
│   ├── 01_eda_script.py                               # Sprint 1 full exploratory audit script
│   ├── 02_reproduction_script.py                     # Sprint 2 exact baseline reproduction script
│   ├── 03_leakage_comparison.py                       # Sprint 3 three-way split execution script
│   └── 03_leakage_comparison.ipynb                    # Sprint 3 interactive Jupyter notebook
├── experiments/
│   ├── exp_001_data_audit.json                        # Sprint 1 statistical & gap audit log
│   ├── exp_002_baseline_reproduction.json             # Sprint 2 baseline reproduction log
│   ├── exp_001_chronological_split.json               # Sprint 3 rigorous chronological run log
│   └── exp_003_leakage_comparison.json                # Sprint 3 comparative protocol master log
├── reports/
│   ├── audit_report.md                                # Sprint 1 comprehensive forensic audit report
│   ├── evaluation_protocol.md                         # Sprint 3 formal evaluation protocol for paper
│   ├── reproducibility_checklist.md                   # Verified reproducibility checklist
│   ├── project_progress_summary.md                    # This review summary document
│   └── figures/
│       ├── fig01_sensor_timeseries_overview.png
│       ├── fig02_sampling_interval_dist.png
│       ├── fig03_sensor_distributions.png
│       ├── fig04_correlation_matrix.png
│       ├── fig05_stress_breakdown.png
│       ├── fig06_gap_and_intervention_zoom.png
│       ├── fig07_baseline_reproduction_loss.png
│       ├── fig08_baseline_predictions_vs_actual.png
│       ├── fig09_split_strategy_mae_comparison.png
│       ├── fig10_split_strategy_r2_comparison.png
│       ├── fig11_strategy_training_histories.png
│       └── fig12_leakage_residual_distribution.png
└── saved_models/
    ├── baseline_cnn_bilstm_reproduction.keras
    ├── model_chronological_split.keras
    ├── model_random_split.keras
    └── model_group_split.keras
```

**summary of 3 sprints:**

1. Audited the raw dataset.
2. Verified the actual sampling interval.
3. Identified operational sessions and shutdown gaps.
4. Removed/flagged unreliable sensor behavior.
5. Audited the original evaluation methodology.
6. Reproduced the original CNN-BiLSTM baseline.
7. Verified the reported performance independently.
8. Discovered that TDS performance was much worse than the aggregated accuracy suggested.
9. Identified scaling leakage.
10. Identified test-as-validation leakage.
11. Identified cross-shutdown sequence corruption.
12. Built a leakage-free chronological pipeline.
13. Added proper train/validation/test separation.
14. Compared random, chronological, and session-based evaluation.
15. Demonstrated that random splitting severely inflates performance.
16. Established a rigorous evaluation protocol for future experiments.

---

## 🚀 Next Step: Sprint 4 — Baseline Benchmarking

With the rigorous, leakage-free Chronological 70/10/20 protocol established and verified:

- **Sprint 4 Goal:** Train and benchmark candidate temporal architectures under this identical protocol:
  - **B0:** Persistence Baseline (sanity floor)
  - **B1:** CNN-BiLSTM (rigorous re-run)
  - **B2:** Vanilla LSTM
  - **B3:** GRU
  - **B4:** BiLSTM (without CNN)
  - **B5:** Temporal Convolutional Network (TCN)
  - **B6:** Transformer / Temporal Self-Attention
  - **B7:** CNN + GRU
- Deliverable: Unified benchmark comparison table (`reports/benchmark_table.md`) logging forecasting accuracy, inference latency (ms), and parameter counts.
