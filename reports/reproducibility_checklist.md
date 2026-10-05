# Reproducibility Checklist
## AI-Enabled Smart Nutrient Balancing System

Every experiment that produces a reported result **must** complete this checklist before being cited in the paper.

---

## ✅ Per-Experiment Items (Sprint 3, 4, 5, 6, 7, 8, 9, 10 Verified: `exp_001`, `exp_003`, `exp_004`, `exp_005`, `exp_006`, `exp_007`, `exp_008`, `exp_009`, `exp_010`)

### Data
- [x] Dataset filename and version recorded (`IoTData_25K_without_interpolation.csv` and `IoTData_Raw.csv`)
- [x] SHA-256 hash of the CSV file saved (`3b7fe00ec64b38df94be6a7b7a95aa9e8a7153a8123281c5a92a543666b6c203`)
- [x] Number of rows and columns confirmed (25,570 rows × 14 columns; real-world raw: 25,000 rows × 6 columns)
- [x] Timestamp range documented (2023-12-21 11:17:03 to 2023-12-26 21:36:40 UTC; raw hardware: 2023-11-26 to 2023-12-21)
- [x] Preprocessing configuration noted (Linear interpolation, MinMaxScaler (0,1), fitted on train partition only, zero re-fitting on real test)

### Split
- [x] Split strategy clearly named: `chronological` (70/10/20), `random` (negative control), `group_session`, and `real_world_hardware_deployment`
- [x] Train / validation / test row counts recorded (Train: 17,899, Val: 2,557, Test: 5,114; Real-World: 24,467 valid rows)
- [x] Train / validation / test sequence counts recorded (Train: 17,748, Val: 2,542, Test: 5,081; Real-World: 24,452 continuous sequences)
- [x] Row indices and boundary timestamps saved (Boundary-aware slicing isolates 17 sampling gaps > 60s)

### Scaling
- [x] `scaler_X.pkl` saved (fitted only on training partition)
- [x] `scaler_y.pkl` saved (fitted only on training partition)

### Model
- [x] Architecture clearly documented: Proposed MT-TCN-LSTM (Dilated Conv1D d=1,2 + LSTM 64 + Multi-Task Heads)
- [x] Hyperparameters fully specified: Adam(lr=1e-4), loss=(1-λ)*MSE + λ*BCE (λ=0.2), batch_size=32, max_epochs=35
- [x] `random_seed` recorded and set before training (seed=42 in TensorFlow, NumPy, and Python random)

### Training
- [x] Training history (loss/val_loss per epoch) logged for all heads
- [x] Best checkpoint saved: `saved_models/proposed_model_best.keras`
- [x] Callbacks used: EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

### Evaluation
- [x] Final metrics computed on **test set only** (test set never seen during training or model selection)
- [x] Per-sensor MAE, RMSE, R², MAPE recorded in real physical units
- [x] Within-tolerance rates recorded with tolerance values explicitly stated (±0.1 pH, ±20 ppm TDS, ±0.5°C Temp, ±2% RH, ±1.0 WL)
- [x] Stress Classification: Precision=0.8875, Recall=1.0000, F1=0.9404, ROC-AUC=0.9995
- [x] Monte Carlo Dropout Uncertainty: N=50 stochastic passes, 90% confidence intervals, empirical coverage
- [x] Systematic Ablations: 11 variants isolating Conv1D, recurrent cells, multi-task heads, window size, and scaling
- [x] Closed-Loop Safety Layer: 11 unit tests passed; 100.0% hazardous dosing prevention rate (436/436 faults trapped)
- [x] Real-World Hardware & Domain Shift Validation: Evaluated on 24,452 real-world telemetry sequences; 10 systematic titration trials logged; INT8 TFLite quantized (112.5 KB, 64.8% compression)

### Environment
- [x] `requirements.txt` committed
- [x] Python version noted (Python 3.10.11)
- [x] TensorFlow/Keras version noted (TensorFlow 2.16.1 / Keras 3.3.3)
- [x] Execution environment: CPU / GPU local workstation

### Experiment Log
- [x] `experiments/exp_001_chronological_split.json` saved
- [x] `experiments/exp_003_leakage_comparison.json` saved
- [x] `experiments/exp_004_baseline_benchmark_master.json` & individual model logs saved
- [x] `experiments/exp_005_proposed_model.json` saved
- [x] `experiments/exp_006_ablation_master.json` saved
- [x] `experiments/exp_007_uncertainty_safety.json` saved
- [x] `experiments/exp_008_realworld_validation.json` saved
- [x] `experiments/exp_009_explainability.json` saved
- [x] `experiments/exp_010_dashboard_deployment.json` saved
- [x] Comparative visualizations saved: `fig09` to `fig29` in `reports/figures/`
- [x] Benchmark table published & updated: `reports/benchmark_table.md`
- [x] Master ablation table published: `reports/ablation_table.md`
- [x] Safety layer evaluation published: `reports/safety_evaluation.md`
- [x] Real-world validation report published: `reports/realworld_validation_report.md`
- [x] Explainability & sensitivity report published: `reports/explainability_report.md`
- [x] Dashboard & deployment integration report published: `reports/dashboard_integration_report.md`

---

## 🚫 Common Anti-Patterns (Never Do These)

| Anti-Pattern | Why Forbidden |
|---|---|
| Fitting scaler on full data before split | Test statistics leak into training scale |
| Using test set as validation during training | Model implicitly optimises on test set |
| Reporting results without noting tolerance definition | "96% accuracy" is ambiguous |
| Tuning hyperparameters using test-set metrics | Overfits to test set |
| Comparing models trained on different splits | Unfair comparison |
| Omitting random seed | Results not reproducible |
| Claiming 15 steps = 15 minutes without confirming sampling interval | Physically incorrect |
