# Formal Evaluation Protocol: Time-Series Forecasting for Hydroponic Nutrient Balancing

**Document Status:** Complete & Verified (Sprint 3 Deliverable)  
**Author:** AI Research & Engineering Team  
**Target Venue:** IEEE / Springer Conference Submission  
**Artifact Dependencies:** `src/preprocessing.py`, `src/evaluation.py`, `src/config.py`, `experiments/exp_001_chronological_split.json`, `experiments/exp_003_leakage_comparison.json`

---

## 1. Problem Formulation & Operational Context

In automated closed-loop hydroponic systems, precision nutrient balancing requires continuous, multi-variable time-series forecasting. Physical chemical dosing (acid/base regulators, concentrated macronutrient solutions, and freshwater top-ups) entails transport delays, mixing hysteresis, and irreversible chemical consequences. Over-dosing pH reducers or EC-boosting fertilizers can shock or destroy root systems within hours.

### 1.1 Mathematical Formulation

Let $\mathbf{x}_t \in \mathbb{R}^D$ denote the multivariate observation vector recorded at discrete physical time index $t$, sampled at regular nominal intervals of $\Delta t = 10\text{ seconds}$:
$$\mathbf{x}_t = \big[x_t^{(\text{pH})},\, x_t^{(\text{TDS})},\, x_t^{(\text{water\_level})},\, x_t^{(\text{DHT\_temp})},\, x_t^{(\text{DHT\_humidity})}\big]^\top \in \mathbb{R}^5$$

Given an observation lookback window of length $W = 15$ timesteps (equivalent to a historical horizon of $150\text{ seconds}$):
$$\mathbf{X}_{t} = \big[ \mathbf{x}_{t-W+1},\, \mathbf{x}_{t-W+2},\, \dots,\, \mathbf{x}_t \big] \in \mathbb{R}^{W \times D}$$

The objective of the predictive model $f_\theta: \mathbb{R}^{W \times D} \to \mathbb{R}^D$ parameterized by weights $\theta$ is to forecast the future system state at lead time $H=1$ step ($10\text{ seconds}$ ahead):
$$\hat{\mathbf{y}}_{t+1} = f_\theta(\mathbf{X}_t)$$

The model is trained via empirical risk minimization over a training partition $\mathcal{D}_{\text{train}}$:
$$\min_\theta \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{i=1}^{|\mathcal{D}_{\text{train}}|} \mathcal{L}\big(\mathbf{y}_{i}, f_\theta(\mathbf{X}_i)\big)$$
where $\mathcal{L}$ is the mean squared error (MSE) across the $D=5$ scaled sensor dimensions:
$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{D} \sum_{d=1}^D (y^{(d)} - \hat{y}^{(d)})^2$$

---

## 2. Four Cardinal Principles of Methodological Rigor

Our forensic code audit of the baseline implementation (`AI_PBL (1).ipynb`) uncovered critical methodological flaws that invalidated previously reported accuracy claims. To establish a benchmark defensible for high-impact peer review, all future experiments adhere strictly to four cardinal principles:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               FOUR CARDINAL PRINCIPLES                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero Scaling Leakage        : Scalers fit strictly on D_train; val/test transformed │
│ 2. Zero Evaluation Leakage     : D_test strictly held out; checkpoints tuned on D_val  │
│ 3. Strict Temporal Ordering    : No future data visible; chronological 70/10/20 split   │
│ 4. Boundary-Aware Sequencing   : Sliding windows never cross shutdowns or partitions   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Principle 1: Zero Scaling Leakage
Feature scaling maps raw sensor quantities into the normalization domain $\mathcal{S} = [0, 1]$.
- **Flawed Baseline:** The original code executed `scaler.fit_transform(full_data)` on all 25,570 records before splitting. This exposed the global empirical minimum $\min_{t} x_t^{(d)}$ and maximum $\max_{t} x_t^{(d)}$ (including the massive Dec 23 TDS surge of 2,271 ppm) to the training pipeline.
- **Rigor Protocol:** The affine transformation parameters:
  $$\mu_{\text{train}}^{(d)} = \min_{t \in \mathcal{D}_{\text{train}}} x_t^{(d)}, \quad \sigma_{\text{train}}^{(d)} = \max_{t \in \mathcal{D}_{\text{train}}} x_t^{(d)} - \min_{t \in \mathcal{D}_{\text{train}}} x_t^{(d)}$$
  are calculated **strictly on the training partition**. Validation and test observations are transformed via:
  $$\tilde{x}_t^{(d)} = \frac{x_t^{(d)} - \mu_{\text{train}}^{(d)}}{\sigma_{\text{train}}^{(d)}}$$
  Any test values exceeding the training support naturally fall outside $[0, 1]$, accurately exposing out-of-distribution shifts.

### Principle 2: Zero Evaluation Leakage
- **Flawed Baseline:** The baseline training script specified `validation_data=(X_test, y_test)` inside `model.fit()`. The test set was used for per-epoch monitoring, callback triggering, and loss tracking, violating the definition of an independent holdout.
- **Rigor Protocol:** The data is partitioned into three mutually exclusive sets:
  - $\mathcal{D}_{\text{train}}$ ($70\%$): Gradient updates.
  - $\mathcal{D}_{\text{val}}$ ($10\%$): Early stopping and hyperparameter selection.
  - $\mathcal{D}_{\text{test}}$ ($20\%$): Touched **exactly once** after training is finalized to report out-of-sample metrics.

### Principle 3: Strict Temporal Ordering
Time-series data exhibits strong temporal persistence. In our dataset, lag-1 autocorrelations are:
- $\text{TDS}$: $r = 0.9991$
- $\text{water\_level}$: $r = 1.0000$
- $\text{DHT\_temp}$: $r = 0.9952$
- $\text{DHT\_humidity}$: $r = 0.9994$
- $\text{pH}$: $r = 0.9573$

If sequences are partitioned randomly, sequence $i$ (spanning $t \dots t+14$) and sequence $i+1$ (spanning $t+1 \dots t+15$) share $14$ out of $15$ identical historical observations ($93.3\%$ feature overlap). When one is assigned to training and the other to test, the model achieves near-zero test error by memorizing local persistence, not by learning system dynamics. All benchmarking must enforce chronological ordering.

### Principle 4: Boundary-Aware Sequence Slicing
The physical dataset is punctuated by 17 sampling gaps $> 60\text{ seconds}$, including 5 major shutdowns ($1.06\text{ to }24.72\text{ hours}$). Naive sequence generation blindly extracted windows across these shutdowns.
- **Rigor Protocol:** Slicing is performed strictly within continuous operational segments:
  $$\mathcal{S}_k = \{ \mathbf{x}_{s_k}, \dots, \mathbf{x}_{e_k} \} \quad \text{where } \forall t \in [s_k, e_k-1],\, \Delta t \le 60\text{ s}$$
  A sliding window of size $W$ is generated from $\mathcal{S}_k$ if and only if $|\mathcal{S}_k| > W$. Slicing stops at $e_k - W$, completely preventing cross-gap sequences. This eliminates 154 physically corrupted boundary sequences.

---

## 3. Data Partitioning Configurations

The dataset contains $N = 25,570$ raw measurements spanning December 21 to December 26, 2023.

### 3.1 Primary Benchmark Partition (Chronological 70 / 10 / 20)

| Partition | Row Range | Timestamp Start (UTC) | Timestamp End (UTC) | Raw Rows | Valid Sequences ($W=15$) | Physical Role |
|---|---|---|---|---|---|---|
| **Train ($\mathcal{D}_{\text{train}}$)** | 0 to 17,898 | 2023-12-21 11:17:03 | 2023-12-25 21:09:59 | 17,899 ($70.0\%$) | 17,748 | Model fitting, parameter estimation |
| **Validation ($\mathcal{D}_{\text{val}}$)** | 17,899 to 20,455 | 2023-12-25 21:10:09 | 2023-12-26 04:36:20 | 2,557 ($10.0\%$) | 2,542 | Early stopping (patience=10), model checkpointing |
| **Test ($\mathcal{D}_{\text{test}}$)** | 20,456 to 25,569 | 2023-12-26 04:36:30 | 2023-12-26 21:36:40 | 5,114 ($20.0\%$) | 5,081 | Final independent evaluation |
| **Total** | 0 to 25,569 | 2023-12-21 11:17:03 | 2023-12-26 21:36:40 | 25,570 ($100.0\%$) | 25,371 | Total valid sequences (154 corrupted gap sequences dropped) |

### 3.2 Negative Control Partition (Random Sequence Split)
- All 25,371 boundary-aware sequences are shuffled with a fixed random seed (`RANDOM_SEED = 42`) and assigned $70\%$ train ($17,888$), $10\%$ val ($2,556$), and $20\%$ test ($5,111$).
- **Purpose:** Quantify the magnitude of artificial performance inflation caused by temporal autocorrelation leakage.

### 3.3 Physical Run Holdout Partition (Group/Session Split)
- Grouped by the 5 major system shutdowns ($> 1\text{ hour}$):
  - **Train Sessions 1–4:** Rows 0 to 12,884 ($12,885$ rows, $12,797$ sequences) — Days 1 to 4 operations.
  - **Validation Session 5:** Rows 12,885 to 13,469 ($585$ rows, $569$ sequences) — Morning settling on Dec 25.
  - **Test Session 6:** Rows 13,470 to 25,569 ($12,100$ rows, $12,035$ sequences) — Held-out continuous operational run on Dec 25–26.
- **Purpose:** Test model resilience against true domain and regime shifts across discrete experimental runs.

---

## 4. Operational Metrics & Tolerance Definitions

To avoid conflating regression tolerance rates with classification accuracy, all metrics are formally defined and categorized:

### 4.1 Statistical Forecasting Metrics

For sensor $d$, let $y_i^{(d)}$ and $\hat{y}_i^{(d)}$ denote the ground truth and predicted values in real physical engineering units:
- **Mean Absolute Error (MAE):**
  $$\text{MAE}^{(d)} = \frac{1}{N} \sum_{i=1}^N \big| y_i^{(d)} - \hat{y}_i^{(d)} \big|$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE}^{(d)} = \sqrt{\frac{1}{N} \sum_{i=1}^N \big( y_i^{(d)} - \hat{y}_i^{(d)} \big)^2}$$
- **Coefficient of Determination ($R^2$):**
  $$R^{2(d)} = 1 - \frac{\sum_{i=1}^N \big(y_i^{(d)} - \hat{y}_i^{(d)}\big)^2}{\sum_{i=1}^N \big(y_i^{(d)} - \bar{y}^{(d)}\big)^2}$$
- **Mean Absolute Percentage Error (MAPE):**
  $$\text{MAPE}^{(d)} = \frac{100\%}{N} \sum_{i=1}^N \frac{\big| y_i^{(d)} - \hat{y}_i^{(d)} \big|}{\max\big(|y_i^{(d)}|,\, \epsilon\big)}$$

### 4.2 Within-Tolerance Rate (Operational Reliability)

The original project reported a "96.22% overall accuracy" by defining acceptable absolute error bounds. In this benchmark, this is termed the **Within-Tolerance Rate ($\text{WTR}$)**:
$$\text{WTR}^{(d)}(\tau_d) = \frac{100\%}{N} \sum_{i=1}^N \mathbb{I}\Big( \big| y_i^{(d)} - \hat{y}_i^{(d)} \big| \le \tau_d \Big)$$

| Sensor | Operational Tolerance $\tau_d$ | Physical Rationale & Engineering Justification |
|---|---|---|
| **pH** | $\pm 0.10\text{ pH}$ | Standard research glass-electrode repeatability; biologically significant for nutrient uptake. |
| **TDS** | $\pm 20.0\text{ ppm}$ | Original project tolerance. Evaluated additionally at $\pm 50$ and $\pm 100\text{ ppm}$ sensitivity. |
| **Water Level** | $\pm 1.0\text{ state unit}$ | Binary sensor state ($1.0$ vs $2.0$). Reported for fidelity to baseline, but critiqued as trivial. |
| **Temperature** | $\pm 0.50^\circ\text{C}$ | Typical thermal boundary layer fluctuation in recirculating nutrient solutions. |
| **Humidity** | $\pm 2.00\%\text{ RH}$ | DHT22 capacitive sensor nominal instrument accuracy specification. |

---

## 5. Audit Findings: Impact of Evaluation Protocol

The empirical results across the three split strategies demonstrate the decisive impact of protocol design on reported metrics:

1. **The Random Split Mirage ($R^2 \approx 0.999$):**  
   Under random splitting, the baseline model achieves nearly perfect scores across all sensors. Because adjacent sequences share $93.3\%$ of their data, the model trivially interpolates known neighboring points. In peer review, presenting random split metrics for time-series forecasting is considered a serious methodological flaw.

2. **Honest Performance Under Chronological Split:**  
   Under strict chronological 70/10/20 partitioning, performance drops to realistic physical levels. While ambient temperature ($R^2 > 0.90$) and humidity retain high predictability, dynamic parameters (pH and TDS) reflect true sensor noise and physical mixing delays.

3. **Domain Shift in Operational Sessions:**  
   When an entire operational session is held out (Session 6), regime shifts (such as reservoir level dropping and un-modeled nutrient additions) expose the vulnerability of deterministic neural networks, directly justifying the need for Sprint 6's uncertainty estimation and safety overrides.

---

## 6. Checklist for Future Model Evaluation

Any candidate model proposed in Sprints 4, 5, or 6 must satisfy:
- [x] Input feature vector consists of the 5 verified sensors (excluding synthetic `water_temp`).
- [x] Preprocessing uses `build_pipeline(..., split_strategy='chronological')`.
- [x] Scaler is fitted strictly on `X_train`.
- [x] Model checkpointing uses `val_loss` on `X_val`, not `X_test`.
- [x] `X_test` is evaluated exactly once after training completion.
- [x] Within-tolerance rates are reported with explicit $\tau$ values, never labeled simply as "accuracy".
- [x] Persistence baseline (B0) is computed on the identical test split to establish the sanity floor.
