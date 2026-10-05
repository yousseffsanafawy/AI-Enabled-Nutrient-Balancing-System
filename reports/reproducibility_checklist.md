# Reproducibility Checklist
## AI-Enabled Smart Nutrient Balancing System

Every experiment that produces a reported result **must** complete this checklist before being cited in the paper.

---

## ✅ Per-Experiment Items (Sprint 3 Verified: `exp_001` & `exp_003`)

### Data
- [x] Dataset filename and version recorded (`IoTData_25K_without_interpolation.csv`)
- [x] SHA-256 hash of the CSV file saved (`3b7fe00ec64b38df94be6a7b7a95aa9e8a7153a8123281c5a92a543666b6c203`)
- [x] Number of rows and columns confirmed (25,570 rows × 14 columns)
- [x] Timestamp range documented (2023-12-21 11:17:03 to 2023-12-26 21:36:40 UTC)
- [x] Preprocessing configuration noted (Linear interpolation, MinMaxScaler (0,1), fitted on train partition only)

### Split
- [x] Split strategy clearly named: `chronological` (70/10/20), `random` (negative control), `group_session`
- [x] Train / validation / test row counts recorded (Train: 17,899, Val: 2,557, Test: 5,114)
- [x] Train / validation / test sequence counts recorded (Train: 17,748, Val: 2,542, Test: 5,081)
- [x] Row indices and boundary timestamps saved (Boundary-aware slicing isolates 17 sampling gaps > 60s)

### Scaling
- [x] `scaler_X.pkl` saved (fitted only on training partition)
- [x] `scaler_y.pkl` saved (fitted only on training partition)

### Model
- [x] Architecture clearly documented: Conv1D(128, k=3) → MaxPool(2) → BiLSTM(50) → Dropout(0.4) → Dense(5)
- [x] Hyperparameters fully specified: Adam(lr=1e-4), loss=MSE, batch_size=32, max_epochs=35
- [x] `random_seed` recorded and set before training (seed=42 in TensorFlow, NumPy, and Python random)

### Training
- [x] Training history (loss/val_loss per epoch) saved as figure (`fig11_strategy_training_histories.png`)
- [x] Best checkpoint saved: `saved_models/model_chronological_split.keras`, `model_random_split.keras`, `model_group_split.keras`
- [x] Callbacks used: EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

### Evaluation
- [x] Final metrics computed on **test set only** (test set never seen during training or model selection)
- [x] Per-sensor MAE, RMSE, R², MAPE recorded in real physical units
- [x] Within-tolerance rates recorded with tolerance values explicitly stated (±0.1 pH, ±20 ppm TDS, ±0.5°C Temp, ±2% RH, ±1.0 WL)
- [x] TDS tolerance sensitivity analyzed across ±10, ±20, ±30, ±50, ±75, ±100 ppm

### Environment
- [x] `requirements.txt` committed
- [x] Python version noted (Python 3.10.11)
- [x] TensorFlow/Keras version noted (TensorFlow 2.16.1 / Keras 3.3.3)
- [x] Execution environment: CPU / GPU local workstation

### Experiment Log
- [x] `experiments/exp_001_chronological_split.json` saved
- [x] `experiments/exp_003_leakage_comparison.json` saved
- [x] Comparative visualizations saved: `fig09`, `fig10`, `fig11`, `fig12` in `reports/figures/`

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
