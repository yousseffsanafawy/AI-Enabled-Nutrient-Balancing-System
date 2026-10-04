# Reproducibility Checklist
## AI-Enabled Smart Nutrient Balancing System

Every experiment that produces a reported result **must** complete this checklist before being cited in the paper.

---

## ✅ Per-Experiment Items

### Data
- [ ] Dataset filename and version recorded
- [ ] SHA-256 hash of the CSV file saved
- [ ] Number of rows and columns confirmed
- [ ] Timestamp range (if timestamps exist) documented
- [ ] Preprocessing configuration (interpolation: yes/no, scaler type, fit partition) noted

### Split
- [ ] Split strategy clearly named: `chronological_70_10_20` | `random_80_20` | `group`
- [ ] Train / validation / test row counts recorded
- [ ] Train / validation / test sequence counts recorded
- [ ] Row indices or boundary timestamps saved (so the exact split can be re-created)

### Scaling
- [ ] `scaler_X.pkl` saved (fitted only on training rows)
- [ ] `scaler_y.pkl` saved (fitted only on training rows)

### Model
- [ ] Architecture clearly documented (layer types, sizes, activations)
- [ ] Hyperparameters fully specified (lr, epochs, batch_size, dropout, kernel sizes, etc.)
- [ ] `random_seed` recorded and set before training

### Training
- [ ] Training history (loss/val_loss per epoch) saved as CSV or PNG
- [ ] Best checkpoint saved: `saved_models/<experiment_id>_best.keras`
- [ ] Callbacks used (EarlyStopping, ReduceLROnPlateau, etc.) documented

### Evaluation
- [ ] Final metrics computed on **test set only** (test set never seen during training or model selection)
- [ ] Per-sensor MAE, RMSE, R² recorded
- [ ] Within-tolerance rates recorded with tolerance values explicitly stated
- [ ] (If applicable) Stress classification: Precision, Recall, F1, confusion matrix saved

### Environment
- [ ] `requirements.txt` or `environment.yml` committed
- [ ] Python version noted
- [ ] TensorFlow/Keras version noted
- [ ] GPU/CPU environment noted

### Experiment Log
- [ ] `experiments/<experiment_id>.json` saved with all above metadata
- [ ] Experiment noted in `reports/benchmark_table.md`

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
