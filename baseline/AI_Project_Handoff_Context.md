# AI Handoff Context --- Smart Nutrient Balancing System

## Purpose of This File

This file is a complete handoff/context document for an AI
coding/research CLI joining the project as the **AI developer**.

The goal is **not** to immediately replace the existing model.

The first goal is to:

1.  Understand the existing system and AI pipeline.
2.  Reproduce the current results.
3.  Audit the dataset, preprocessing, sequence generation, split
    strategy, training, evaluation, and deployment.
4.  Identify scientific/ML weaknesses and possible leakage.
5.  Build fair baselines.
6.  Only then decide whether a new model or a redesigned AI pipeline is
    justified.
7.  Improve the project into something strong enough for a
    conference-paper-level contribution.

**Important:** The project report is the source for the existing-system
description. Some observations in this file are research hypotheses or
concerns that must be verified from the actual code and dataset before
being presented as facts.

------------------------------------------------------------------------

# 1. Project Overview

## Project Title

**AI Enabled Smart Nutrient Balancing System for Controlled Hydroponic
Systems**

## Main Idea

The project is a smart hydroponics system combining:

-   IoT sensors
-   ESP32
-   Firebase Realtime Database
-   Deep Learning
-   Web dashboard
-   Automated dosing pumps

The intended shift is from:

> Reactive monitoring → Predictive/proactive nutrient management

Instead of waiting for pH/TDS/etc. to cross a dangerous threshold, the
AI is intended to forecast future environmental/nutrient states and
allow the system to react before plant stress occurs.

------------------------------------------------------------------------

# 2. High-Level System Architecture

Current conceptual pipeline:

``` text
Hydroponic Reservoir
        │
        ▼
     Sensors
        │
        ▼
      ESP32
        │
        ▼
Firebase Realtime Database
        │
        ▼
   AI Prediction Model
   Conv1D + BiLSTM
        │
        ▼
 Future Sensor States
        │
        ▼
 Stress / Threshold Logic
        │
        ├──────────────► Dashboard / Alerts
        │
        ▼
 Automated Dosing Logic
        │
        ▼
 Peristaltic Pumps / Valves
        │
        ▼
Hydroponic Reservoir
```

This is intended to form a closed-loop system:

``` text
Sense → Predict → Decide → Act → Sense Again
```

------------------------------------------------------------------------

# 3. Hardware / IoT Layer

The report describes an ESP32-based sensing and actuation system.

## Sensors

Reported sensors include:

-   Analog pH Meter V2.0 → pH
-   UGE TDS Sensor → TDS / nutrient concentration
-   DHT22 → ambient temperature and humidity
-   DS18B20 → water temperature
-   Analog water-level sensor → water level

The exact set of features actually fed to the AI model must be verified
from the code.

The report describes the AI as working with five sensor inputs in
several places.

------------------------------------------------------------------------

# 4. Actuation Layer

The system uses dosing/actuation hardware including:

-   12V peristaltic dosing pumps
-   Nutrient A/B dosing
-   pH Up / pH Down dosing
-   L298N / relays
-   Water/freshwater valves in some system descriptions

The AI prediction is intended to eventually support automated corrective
actions.

Important distinction:

> The current AI appears primarily to be a **predictive model**, while
> the actual dosing decision may be performed by separate rule/threshold
> logic.

This must be verified in code.

If the architecture is:

``` text
AI → predicts pH/TDS/etc.
        ↓
Rule-based thresholds
        ↓
Pump action
```

then the learned model is not itself learning the control policy.

That is not necessarily bad, but it should be described accurately in a
research paper.

------------------------------------------------------------------------

# 5. Data Pipeline

The report describes the following general pipeline:

``` text
Raw sensor data
      ↓
Missing-value handling
      ↓
Linear interpolation
      ↓
Scaling with MinMaxScaler
      ↓
Sliding-window sequence generation
      ↓
CNN + BiLSTM
      ↓
Multi-output regression
      ↓
Inverse scaling
      ↓
MAE / RMSE / operational metrics
      ↓
Stress / action logic
```

------------------------------------------------------------------------

# 6. Current Dataset

The project reportedly uses a **Kaggle Hydroponics dataset**.

The report describes the dataset as containing environmental data from
controlled setups representing multiple stress conditions, including
examples such as:

-   pH imbalance
-   TDS depletion
-   temperature spikes
-   Optimal Daily Growth (ODG)

The report also discusses imbalance between optimal-growth samples and
deficiency/stress samples.

The exact dataset identity, URL/version, number of samples, sampling
interval, plant species, experiment IDs, and exact columns are **not
sufficiently specified in the report**.

### Required action

Before modifying the model, inspect the actual dataset and document:

-   exact dataset source
-   number of rows
-   number of experiments
-   number of plants/crops if available
-   sampling interval
-   exact feature names
-   target columns
-   stress labels
-   experiment/plant identifiers
-   missing-value distribution
-   class distribution
-   temporal ordering
-   duplicated timestamps
-   whether samples from different experiments are mixed

------------------------------------------------------------------------

# 7. Existing AI Model

## Framework

The report says the model was developed using:

-   Python
-   TensorFlow
-   Keras

## Architecture

Current model:

**Conv1D + Bidirectional LSTM + Dense multi-output regression**

Conceptually:

``` text
15-step time-series input
        │
        ▼
     Conv1D
        │
        ▼
 Batch Normalization / Regularization
        │
        ▼
   Bidirectional LSTM
        │
        ▼
 Dense Multi-Output Layer
        │
        ▼
Future sensor values
```

The report describes the CNN as extracting local patterns and the BiLSTM
as modeling temporal dependencies.

------------------------------------------------------------------------

# 8. Current Sequence Generation

The report says:

> 15 continuous historical readings are used to predict the 16th
> reading.

So conceptually:

``` text
t-14
t-13
t-12
...
t-2
t-1
t
        ↓
    predict t+1
```

The report also describes this as a **15-minute historical window** and
says the model predicts approximately **15 minutes ahead**.

## Critical Verification Required

There is a potential timing inconsistency.

The report also describes ESP32/Firebase synchronization around **10
seconds** in the system.

If raw AI samples arrive every 10 seconds, then:

``` text
15 steps × 10 seconds = 150 seconds = 2.5 minutes
```

not 15 minutes.

However, this may be resolved if:

-   the AI receives data after temporal aggregation,
-   the dataset has a 1-minute sampling interval,
-   the firmware/database stream is different from the AI sampling
    interval,
-   or preprocessing resamples the data.

Therefore:

> **Do NOT call this a confirmed bug until the actual code and
> timestamps are inspected.**

The CLI must determine the true sampling frequency from the real
pipeline.

------------------------------------------------------------------------

# 9. Preprocessing

The report describes:

## Missing Values

Linear interpolation is used to fill missing values.

Conceptually:

``` text
Before:
10
NaN
14

After interpolation:
10
12
14
```

## Scaling

A `MinMaxScaler` maps features into approximately:

``` text
[0, 1]
```

The reason is that TDS may have much larger numerical values than pH.

Without scaling, a model may have optimization difficulty because
feature magnitudes differ substantially.

------------------------------------------------------------------------

# 10. Current Train/Test Split

The report says:

> 80% training / 20% testing

This is one of the first things that must be audited.

## Major Potential Problem: Time-Series Leakage

If the workflow is:

``` text
1. Create all overlapping windows
2. Randomly split windows into train/test
```

then leakage may occur.

Example:

``` text
Training:
[t1 t2 t3 ... t15] → t16

Testing:
[t2 t3 t4 ... t16] → t17
```

These two samples share almost the entire history.

The test set then is not truly independent.

This can make results look much better than real-world generalization.

## Preferred Approach

For a temporal forecasting problem, first consider:

``` text
Chronological raw data
        ↓
Train period | Validation period | Test period
        ↓
Create windows separately
```

Potentially even stronger:

``` text
Experiment A/B/C → Train
Experiment D → Validation
Experiment E → Test
```

if experiment IDs exist.

For generalization claims, a **group/experiment/plant-based split** may
be more scientifically meaningful than a random row split.

The exact strategy must depend on the dataset structure.

------------------------------------------------------------------------

# 11. Current Training Configuration

The report describes:

-   Optimizer: Adam
-   Loss: MSE
-   Training: 100 epochs
-   Dropout: 0.4
-   Regularization
-   Batch normalization
-   Weighted/cost-sensitive learning
-   Multi-output regression

The exact architecture, number of filters, kernel sizes, LSTM units,
learning rate, batch size, callbacks, early stopping, and regularization
coefficients must be extracted from the actual code.

The report mentions CNN layers with up to 128 filters.

Do not rely on the report for exact hyperparameters if the source code
is available.

------------------------------------------------------------------------

# 12. Imbalance Handling

The report says the hydroponics dataset contains significantly more
optimal-growth samples than deficiency/stress samples.

Instead of artificial resampling, the project reportedly uses:

> Cost-sensitive learning / weighted loss

The idea is:

``` text
Minority / stress mistakes
        ↓
Higher penalty
        ↓
Model pays more attention to stress events
```

This is a reasonable approach in principle.

However, it needs to be audited because the current model is described
primarily as a multi-output regression model.

Questions to answer:

-   What exactly is being weighted?
-   Are weights attached to stress labels?
-   Is there a classification head?
-   Is the weight applied to regression loss?
-   How is stress represented?
-   Is the weighting mathematically appropriate for the actual
    objective?

------------------------------------------------------------------------

# 13. Current Evaluation

The report mentions:

-   MAE
-   RMSE
-   operational accuracy
-   stress detection accuracy
-   inference latency

Reported headline results include:

-   **96.22% overall stress detection accuracy**
-   **96.25% operational accuracy**
-   **6% mean predictive error**
-   inference latency **under 2 seconds**

These numbers must NOT automatically be treated as standard ML
classification metrics.

------------------------------------------------------------------------

# 14. Critical Issue: What Does "96.22% Accuracy" Mean?

The current model is described as a regression model predicting future
sensor values.

Therefore, a statement such as:

> "96.22% accuracy"

is potentially ambiguous.

The report indicates that custom operational tolerances are used to
determine whether predictions are sufficiently close to
expected/acceptable values.

Examples mentioned include tolerances such as:

-   pH around ±0.1
-   TDS around ±20 ppm

The exact formula must be extracted from the code.

Possible interpretation:

``` text
prediction within acceptable tolerance
        ↓
count as operationally correct
```

That is different from conventional classification accuracy.

For a conference paper, use precise names such as:

-   MAE
-   RMSE
-   MAPE, if appropriate
-   R², if appropriate
-   per-target error
-   tolerance-based accuracy
-   precision/recall/F1 for actual stress classification
-   confusion matrix

Do not call a tolerance-based regression metric simply "accuracy"
without defining it mathematically.

------------------------------------------------------------------------

# 15. Real-World Validation

The report describes controlled laboratory validation using pH
titration.

A summarized procedure is:

1.  Prepare the hydroponic solution.
2.  Gradually introduce nitric acid (HNO3).
3.  Move pH toward approximately 5.0.
4.  Capture 15 consecutive readings.
5.  Feed the sequence into the model.
6.  Wait approximately 15 minutes.
7.  Measure the actual physical pH.
8.  Compare actual vs predicted pH.
9.  Repeat the experiment.

Reported trials:

### Trial 1

Absolute error:

``` text
0.2 pH
```

Relative error:

``` text
4%
```

### Trial 2

Absolute error:

``` text
0.4 pH
```

Relative error:

``` text
8%
```

Mean:

``` text
(4% + 8%) / 2 = 6%
```

## Major Research Concern

Two physical trials are not enough to strongly establish generalization
or reliability.

For a conference-quality study, stronger validation would ideally
include:

-   multiple independent trials
-   different starting conditions
-   different stress magnitudes
-   multiple days
-   multiple nutrient conditions
-   possibly different growth stages
-   comparison between predicted and actual sensor trajectories
-   test data that was never used in model selection

------------------------------------------------------------------------

# 16. Major Research/ML Weaknesses to Investigate

These are the most important areas for the AI developer.

## 16.1 Data Leakage

Highest priority.

Check:

-   scaling before/after split
-   interpolation before/after split
-   sequence creation before/after split
-   random vs chronological split
-   overlapping windows across train/test
-   duplicated experiments across splits
-   whether validation data leaks into preprocessing

Correct approach should be determined from the actual dataset structure.

------------------------------------------------------------------------

## 16.2 Temporal Horizon

Determine exactly:

``` text
What does one timestep represent?
```

Then determine:

``` text
15 timesteps = how much real time?
```

Then determine:

``` text
What is the actual forecast horizon?
```

This must match:

-   dataset timestamps
-   preprocessing
-   ESP32 sampling
-   Firebase update rate
-   deployed inference loop

------------------------------------------------------------------------

## 16.3 BiLSTM in Forecasting

The existing model uses a Bidirectional LSTM.

A BiLSTM reads a sequence in both directions.

This is potentially questionable for true causal forecasting depending
on how the sequence is defined and how inference works.

Important nuance:

If all 15 historical observations are already available at prediction
time, a BiLSTM is not automatically "seeing the future" beyond the input
window.

However, it uses information from both ends of the observed window to
encode the sequence.

The paper needs a clear justification for why bidirectionality is
beneficial for this forecasting task.

We should experimentally compare:

-   LSTM
-   GRU
-   BiLSTM
-   TCN
-   Transformer / temporal attention model
-   existing CNN-BiLSTM

Do not remove BiLSTM simply because it is bidirectional.

------------------------------------------------------------------------

## 16.4 CNN Across Sensor Features

The report describes Conv1D as scanning across the sensor inputs.

But sensor variables are not naturally spatial pixels.

We need to inspect the actual tensor layout.

Example:

``` text
(batch, time, features)
```

If Conv1D uses:

``` text
kernel_size > 1
```

it typically moves across the temporal axis.

That means the CNN may actually be learning local temporal patterns, not
necessarily "spatial correlations among sensors."

The paper should explain this accurately based on the actual
architecture.

------------------------------------------------------------------------

## 16.5 Prediction vs Diagnosis

The current system predicts sensor values.

Prediction:

``` text
Future pH = 5.4
Future TDS = 650
```

Diagnosis:

``` text
Likely nutrient deficiency
Likely pH imbalance
Likely water-level problem
```

These are not the same task.

A stronger AI system could potentially contain multiple outputs:

``` text
Shared Temporal Encoder
        │
        ├── pH forecast
        ├── TDS forecast
        ├── water temperature forecast
        ├── humidity forecast
        ├── water-level forecast
        │
        └── stress/anomaly classification
```

This is a possible **multi-task learning** direction.

------------------------------------------------------------------------

# 17. AI vs Rule-Based Control

A critical architectural question:

``` text
Does AI decide the pump action?
```

or:

``` text
AI predicts future values
        ↓
Threshold/rule engine decides action
        ↓
Pump
```

If the second is true, the system is best described as:

> Predictive AI + rule-based safety/control layer

This may actually be preferable for a physical chemical dosing system
because deterministic safety rules can provide constraints.

A future research contribution could be:

``` text
AI prediction
+
uncertainty
+
safety-constrained decision layer
```

rather than blindly allowing a neural network to control chemical
dosing.

------------------------------------------------------------------------

# 18. Uncertainty Is Important

Automatic dosing is a physical intervention.

A model can be wrong.

Example:

``` text
Prediction:
pH = 5.2

Model confidence:
LOW
```

It would be dangerous to immediately dose chemicals.

A safer architecture:

``` text
Prediction
    │
    ▼
Uncertainty / confidence estimation
    │
    ├── High confidence → automatic action allowed
    │
    ├── Medium confidence → alert / human confirmation
    │
    └── Low confidence → no automatic dosing
```

Possible future methods:

-   prediction intervals
-   Monte Carlo dropout
-   deep ensembles
-   quantile regression
-   probabilistic forecasting

This could be a strong research improvement.

------------------------------------------------------------------------

# 19. Domain Shift / Generalization

This may be one of the biggest problems.

Training data:

``` text
Kaggle hydroponics dataset
```

Deployment data:

``` text
Real ESP32 sensors
+ real reservoir
+ real electrical noise
+ real pumps
+ real calibration
+ real environmental variation
```

These distributions may differ.

A model can perform extremely well on Kaggle data and still perform
poorly on the actual hardware.

Therefore, we need to evaluate:

``` text
Train: Kaggle
Test: Kaggle
```

and then:

``` text
Train: Kaggle
Test: Real ESP32 data
```

The second test is much more valuable for deployment/generalization.

Potential strategies:

-   collect real data
-   calibration alignment
-   sensor normalization
-   domain adaptation
-   fine-tuning
-   mixed-source training
-   leave-one-experiment-out validation

------------------------------------------------------------------------

# 20. Explainability

For an agricultural/physical system, it is useful to answer:

> Why did the AI predict a nutrient problem?

Potential explanation:

``` text
TDS decreasing rapidly
+
water level decreasing
+
pH drifting
        ↓
high predicted stress risk
```

Potential methods:

-   temporal feature importance
-   permutation importance
-   SHAP where appropriate
-   attention weights if using attention
-   saliency/gradient methods
-   interpretable auxiliary features

The goal is not just a pretty explanation.

It should help researchers and operators understand whether the model is
learning physically meaningful patterns.

------------------------------------------------------------------------

# 21. Stronger Research Direction

Do NOT decide the final architecture yet.

A promising direction is a **multi-task temporal forecasting + stress
detection framework**.

Possible architecture:

``` text
Input:
15+ historical timesteps
        │
        ▼
Temporal Encoder
(TCN / Transformer / GRU / LSTM)
        │
        ▼
Shared Representation
        │
 ┌──────┼────────┬─────────┐
 ▼      ▼        ▼         ▼
pH     TDS      Temp      Water
Forecast Forecast Forecast Forecast
        │
        └──────────────┐
                       ▼
                Stress / Anomaly
                  Classification
                       │
                       ▼
             Safe Decision Layer
                       │
          ┌────────────┴───────────┐
          ▼                        ▼
    Auto Action              Human Alert
```

This is only a candidate direction.

It must be justified experimentally.

------------------------------------------------------------------------

# 22. Candidate Models for Benchmarking

Before claiming a new model is better, establish strong baselines.

Recommended benchmark set:

### Baseline 1

Current model:

``` text
Conv1D + BiLSTM
```

### Baseline 2

``` text
LSTM
```

### Baseline 3

``` text
GRU
```

### Baseline 4

``` text
TCN
```

### Baseline 5

``` text
Temporal Transformer / Attention
```

### Optional Baseline 6

``` text
CNN + Attention + LSTM/GRU
```

The benchmark must use:

-   identical data splits
-   identical preprocessing rules where appropriate
-   identical forecast horizon
-   comparable hyperparameter tuning budgets
-   same evaluation metrics

Otherwise the comparison is unfair.

------------------------------------------------------------------------

# 23. Evaluation Protocol We Should Aim For

A better experimental design:

``` text
Raw Dataset
    │
    ├── chronological / group split
    │
    ▼
Train
Validation
Test
```

Then:

### Forecasting metrics

For each target:

-   MAE
-   RMSE
-   R²
-   possibly MAPE where mathematically appropriate
-   normalized MAE
-   prediction interval coverage if uncertainty is added

### Stress metrics

If stress classification is implemented:

-   Precision
-   Recall
-   F1
-   Macro F1
-   PR-AUC
-   ROC-AUC if appropriate
-   confusion matrix

### Operational metrics

-   false alarm rate
-   missed-stress rate
-   action accuracy
-   unnecessary dosing events
-   time-to-detection
-   lead time before stress
-   safety violations

### Deployment metrics

-   inference latency
-   memory usage
-   CPU/GPU usage
-   network latency
-   Firebase latency
-   robustness to missing sensor readings

------------------------------------------------------------------------

# 24. Important Ablation Studies

A conference paper becomes stronger when we can show what actually
contributes to performance.

Possible ablations:

``` text
Full model

vs

Without CNN

vs

Without recurrent component

vs

LSTM instead of BiLSTM

vs

GRU

vs

Without attention

vs

Without uncertainty

vs

Single-task forecasting

vs

Multi-task forecasting + stress detection
```

Also test:

-   different window sizes
-   different forecast horizons
-   different sensor subsets
-   with/without interpolation
-   different scaling methods

------------------------------------------------------------------------

# 25. Window Size Study

The current window is 15 steps.

Do not assume 15 is optimal.

Test something like:

``` text
5 steps
10 steps
15 steps
30 steps
60 steps
```

if supported by the data.

The best window should be chosen based on:

-   validation performance
-   forecast horizon
-   computational cost
-   physical meaning

------------------------------------------------------------------------

# 26. Feature Importance / Sensor Contribution

A useful research question:

> Which sensors actually contribute most to predicting nutrient stress?

Potential experiments:

``` text
All sensors

pH only

TDS only

pH + TDS

Chemical + temperature

Chemical + environmental

All sensors
```

This can reveal redundancy and improve interpretability.

------------------------------------------------------------------------

# 27. Potential New Contribution

One strong paper direction could be framed around:

> **Robust, leakage-free, uncertainty-aware, multi-task temporal
> forecasting for proactive nutrient-stress detection in hydroponic
> systems.**

Possible contribution components:

1.  Correct temporal/group evaluation.
2.  Strong baseline comparison.
3.  Multi-task prediction + stress classification.
4.  Uncertainty-aware predictions.
5.  Safety-constrained action layer.
6.  Real ESP32 validation.
7.  Explainable predictions.

The exact contribution should only be finalized after auditing the
existing implementation and dataset.

------------------------------------------------------------------------

# 28. What NOT To Do Yet

Do not immediately:

-   delete the CNN-LSTM
-   replace it with a Transformer just because Transformers are newer
-   claim the existing model is bad
-   claim data leakage exists without checking code
-   claim the 15-minute horizon is wrong without checking timestamps
-   claim 96.22% is fake
-   report 96.22% as standard classification accuracy
-   tune on the test set
-   compare models using different splits
-   optimize only one metric
-   ignore real ESP32 data
-   allow uncontrolled automatic dosing without safety constraints

The correct workflow is:

``` text
Understand
→ Reproduce
→ Audit
→ Benchmark
→ Improve
→ Validate
→ Write
```

------------------------------------------------------------------------

# 29. Exact First Tasks for the AI Developer

## Phase 1 --- Reproduction

Get from the team:

1.  AI notebook/script
2.  Exact dataset file
3.  Saved model (`.keras`, `.h5`, etc.)
4.  Scaler (`.pkl` or equivalent)
5.  Label/threshold configuration
6.  Training logs
7.  Evaluation code
8.  Any real ESP32/Firebase sample data

Then reproduce the reported result.

------------------------------------------------------------------------

# 30. Phase 2 --- Code Audit

Audit every stage:

### Data loading

-   Where does the data come from?
-   What columns are selected?
-   Are rows sorted by timestamp?
-   Are experiment IDs preserved?

### Cleaning

-   How are missing values handled?
-   Is interpolation performed before or after splitting?
-   Is smoothing performed?
-   Does smoothing use future points?

### Scaling

Critical question:

``` text
Is scaler.fit() performed only on training data?
```

Correct:

``` python
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

Potential leakage:

``` python
scaler.fit(all_data)
```

before splitting.

### Sequence generation

Determine exactly where windows are generated relative to train/test
split.

### Training

Check:

-   loss
-   optimizer
-   learning rate
-   epochs
-   batch size
-   callbacks
-   early stopping
-   regularization
-   class/target weights

### Evaluation

Check exactly how:

``` text
96.22%
96.25%
6%
```

are calculated.

------------------------------------------------------------------------

# 31. Phase 3 --- Data Audit

Generate a data report containing:

``` text
Shape
Columns
Data types
Missing values
Duplicates
Timestamp range
Sampling interval
Unique experiments
Unique plants
Stress distribution
Feature statistics
Outliers
```

Visualize:

-   sensor time series
-   pH distribution
-   TDS distribution
-   temperature distribution
-   water-level distribution
-   stress periods
-   correlation matrix
-   missing-value timeline

------------------------------------------------------------------------

# 32. Phase 4 --- Leakage Audit

Explicitly test:

1.  Random row split
2.  Chronological split
3.  Group/experiment split

Compare the results.

If performance collapses under a proper split, that is scientifically
important.

Do not hide it.

Instead, turn it into part of the research story:

> The original evaluation may overestimate generalization because of
> temporal overlap, while the proposed evaluation provides a more
> realistic estimate.

Only make this claim if experimentally demonstrated.

------------------------------------------------------------------------

# 33. Phase 5 --- Baseline Benchmark

Run:

``` text
Current CNN-BiLSTM
LSTM
GRU
TCN
Transformer/Attention
```

using the same evaluation protocol.

Create a table:

  Model                  MAE   RMSE    R²   Stress F1   Lead Time   Inference
  -------------------- ----- ------ ----- ----------- ----------- -----------
  Current CNN-BiLSTM     TBD    TBD   TBD         TBD         TBD         TBD
  LSTM                   TBD    TBD   TBD         TBD         TBD         TBD
  GRU                    TBD    TBD   TBD         TBD         TBD         TBD
  TCN                    TBD    TBD   TBD         TBD         TBD         TBD
  Transformer            TBD    TBD   TBD         TBD         TBD         TBD

Do not fill numbers until experiments are actually run.

------------------------------------------------------------------------

# 34. Phase 6 --- Proposed Model

Only after the benchmark should we decide whether the proposed model
should be:

-   TCN
-   GRU
-   LSTM + Attention
-   Transformer
-   CNN + Attention + recurrent model
-   multi-task architecture
-   uncertainty-aware architecture
-   another architecture suggested by the data

The model should be selected based on:

``` text
Performance
+
Generalization
+
Computational cost
+
Interpretability
+
Physical safety
```

not novelty alone.

------------------------------------------------------------------------

# 35. Phase 7 --- Real-World Validation

The strongest practical test:

``` text
Model trained on historical data
        ↓
ESP32 real sensor stream
        ↓
Predictions
        ↓
Actual future sensor measurements
        ↓
Compare prediction vs reality
```

Record:

-   timestamp
-   input window
-   prediction
-   actual value
-   absolute error
-   relative error
-   stress prediction
-   confidence
-   action taken
-   actual outcome

This creates a proper deployment evaluation dataset.

------------------------------------------------------------------------

# 36. Safety-Constrained Control

Because the system can dose chemicals, AI should not have unrestricted
control.

Potential architecture:

``` text
AI Prediction
      ↓
Confidence Check
      ↓
Safety Constraints
      ↓
Biological / Operational Limits
      ↓
Action Decision
      ↓
Pump
```

Example:

``` text
If:
    predicted pH is unsafe
AND confidence is high
AND sensor readings are valid
AND rate-of-change is plausible
AND dosing limits are not exceeded

Then:
    allow controlled dosing
```

Otherwise:

``` text
Alert human operator
```

The exact control rules must be designed with domain knowledge and
validated experimentally.

------------------------------------------------------------------------

# 37. Paper-Level Story

A weak paper story would be:

> We used CNN-LSTM and got 96%.

A stronger story is:

> We investigated predictive nutrient management in hydroponic systems
> under realistic temporal evaluation constraints, audited
> leakage/generalization issues, benchmarked temporal architectures, and
> developed a robust predictive framework that combines multi-target
> forecasting, stress detection, uncertainty estimation, and
> safety-constrained action.

Again, this is a target direction, not a claim about completed work.

------------------------------------------------------------------------

# 38. Questions the AI CLI Must Answer Before Model Replacement

The CLI should not recommend a new model until it can answer:

### Dataset

1.  What exactly is the dataset?
2.  How many samples?
3.  What is the sampling interval?
4.  What are the target variables?
5.  What are the stress labels?
6.  How many independent experiments exist?

### Preprocessing

7.  Where is interpolation performed?
8.  Where is scaling fitted?
9.  Is any future information used during preprocessing?
10. Is smoothing causal?

### Splitting

11. Is the split random or chronological?
12. Are overlapping windows shared across train/test?
13. Are experiments/plants shared across splits?

### Model

14. What exactly is the tensor shape?
15. What exactly does Conv1D convolve over?
16. Why BiLSTM?
17. What are the exact hyperparameters?
18. Is the task regression, classification, or both?

### Evaluation

19. How is 96.22% calculated?
20. How is 96.25% calculated?
21. How is the 6% error calculated?
22. Are metrics computed on truly unseen data?

### Deployment

23. What is the actual sampling interval?
24. What is the real forecast horizon?
25. Does the deployed data distribution match Kaggle?
26. Does AI directly control pumps or does a rule engine do it?

### Scientific contribution

27. What is genuinely new compared with existing hydroponic forecasting
    systems?
28. What experiment would prove that the proposed method is better?
29. What is the strongest baseline?
30. What is the real-world validation protocol?

------------------------------------------------------------------------

# 39. Recommended Working Style for This Project

The AI developer should work incrementally.

For every modification:

``` text
Change one thing
        ↓
Run experiment
        ↓
Record metric
        ↓
Compare with baseline
        ↓
Explain why it improved/worsened
```

Avoid making many simultaneous changes.

Keep an experiment log:

``` text
Experiment ID
Date
Dataset split
Window size
Features
Model
Hyperparameters
Seed
Metrics
Notes
```

Use fixed/random seeds where appropriate so experiments are
reproducible.

------------------------------------------------------------------------

# 40. Reproducibility Checklist

Save:

-   dataset version
-   preprocessing configuration
-   scaler
-   random seed
-   train/validation/test indices
-   model architecture
-   hyperparameters
-   training history
-   best checkpoint
-   final test metrics
-   inference script
-   environment/package versions

A conference result should be reproducible.

------------------------------------------------------------------------

# 41. Final Mental Model

Think about the project as five layers:

``` text
LAYER 1 — SENSING
ESP32 + sensors
        ↓

LAYER 2 — DATA
Firebase + historical dataset
        ↓

LAYER 3 — AI
Temporal forecasting + stress detection
        ↓

LAYER 4 — DECISION
Confidence + biological thresholds + safety rules
        ↓

LAYER 5 — ACTION
Pumps / valves / dashboard / alerts
```

The current project already has the skeleton.

Our job as the AI developer is to determine:

> **Is the existing AI scientifically reliable, and if not, what is the
> smallest but strongest redesign that makes the whole system more
> accurate, generalizable, explainable, and publishable?**

------------------------------------------------------------------------

# 42. Immediate Next Step

**Do not start by writing a new model.**

Ask the team for the actual AI implementation and dataset.

Then inspect:

``` text
AI notebook / Python scripts
        +
Dataset
        +
Saved model
        +
Scaler
        +
Evaluation code
        +
Real sensor data
```

The first deliverable from the AI side should be an **AI Audit Report**,
not a new architecture.

The audit should answer:

``` text
What they built
        ↓
What is correct
        ↓
What is questionable
        ↓
What is actually wrong
        ↓
What can be improved
        ↓
What experiment proves the improvement
```

Only after that should we choose the final model.

------------------------------------------------------------------------

# 43. Current Bottom Line

The existing project is not something to throw away.

Its strongest existing concept is:

> **Predictive + closed-loop hydroponic nutrient management**

The biggest opportunities for strengthening it are likely to be:

1.  **Leakage-free temporal evaluation**
2.  **Correct forecast-horizon definition**
3.  **Real-world/domain-shift validation**
4.  **Clear separation of forecasting vs stress classification**
5.  **Better evaluation metrics**
6.  **Strong baseline comparisons**
7.  **Multi-task learning**
8.  **Uncertainty-aware prediction**
9.  **Explainability**
10. **Safety-constrained dosing**

The key principle:

> **Do not replace the model because a newer architecture looks better.
> Replace or redesign it only when experiments show a measurable and
> scientifically defensible improvement.**

---

# SOURCE CODE AUDIT CONTEXT — CURRENT AI IMPLEMENTATION

## 1. Actual source code provided by the team

The current AI training/evaluation code is reproduced below. Treat this as the authoritative implementation for the current model, alongside the project report. Do NOT assume the report's wording is more accurate than the code when they conflict.

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# ==========================================
# 1. DATA LOADING & SETUP
# ==========================================
print("Checking for dataset...")
file_name = list(uploaded.keys())[0] if 'uploaded' in locals() else 'IoTData_25K_without_interpolation.csv'

if not os.path.exists(file_name):
    print(f"File not found. Please upload:")
    from google.colab import files
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]

print("\nLoading data from CSV...")
df = pd.read_csv(file_name)

# ==========================================
# 2. DATA PREPROCESSING & SCALING
# ==========================================
print("\nStarting Preprocessing...")

features = ['water_level', 'DHT_temp', 'TDS', 'pH', 'DHT_humidity']
# MULTI-OUTPUT UPDATE: The target is now ALL features, not just pH
target_cols = features

df_interpolated = df.interpolate(method='linear').dropna()

# Scale features
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_X.fit_transform(df_interpolated[features])

# Scale targets (which is the exact same shape as features now)
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_y.fit_transform(df_interpolated[target_cols])

# ==========================================
# 3. SEQUENCE GENERATION (15-STEP WINDOW)
# ==========================================
def create_sequences(data_X, data_y, time_steps):
    X, y = [], []
    for i in range(len(data_X) - time_steps):
        X.append(data_X[i:(i + time_steps)])
        y.append(data_y[i + time_steps]) # Y now contains 5 values instead of 1
    return np.array(X), np.array(y)

time_steps = 15
X_seq, y_seq = create_sequences(scaled_features, scaled_target, time_steps)

split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# ==========================================
# 4. HYBRID CNN-LSTM ARCHITECTURE
# ==========================================
print("\nBuilding Multi-Output CNN-LSTM Model...")
model = Sequential()

model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dropout(0.4))

# MULTI-OUTPUT UPDATE: The output layer now has nodes equal to the number of features (5)
model.add(Dense(len(features), activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])

# ==========================================
# 5. MODEL TRAINING
# ==========================================
print("\nTraining Model...")
history = model.fit(
    X_train, y_train,
    epochs=35,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ==========================================
# 6. PARAMETER-SPECIFIC EVALUATION
# ==========================================
print("\n=============================================")
print("  MODEL PERFORMANCE BY INDIVIDUAL PARAMETER  ")
print("=============================================\n")

# Make predictions on the test set
y_pred_scaled = model.predict(X_test)

# Reverse the MinMax scaling to get real-world sensor values back
y_test_real = scaler_y.inverse_transform(y_test)
y_pred_real = scaler_y.inverse_transform(y_pred_scaled)

# Define custom tolerances for what counts as an "Accurate" prediction for each specific sensor
# (e.g., being off by 10 TDS is acceptable, but being off by 10 pH is a disaster)
tolerances = {
    'water_level': 1.0,   # Accurate if within 1 liter/cm
    'DHT_temp': 0.5,      # Accurate if within 0.5 degrees
    'TDS': 20.0,          # Accurate if within 20 PPM
    'pH': 0.1,            # Accurate if within 0.1 pH
    'DHT_humidity': 2.0   # Accurate if within 2% humidity
}

# Loop through each parameter to calculate its specific error
for i, feature in enumerate(features):
    # Extract just the data for this specific feature
    actual_values = y_test_real[:, i]
    predicted_values = y_pred_real[:, i]

    # Calculate Standard Metrics
    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    # Calculate Custom Accuracy based on the tolerance dictionary above
    tol = tolerances[feature]
    accurate_predictions = np.sum(np.abs(actual_values - predicted_values) <= tol)
    custom_accuracy = (accurate_predictions / len(actual_values)) * 100

    # Print the isolated results
    print(f"--- Sensor: {feature} ---")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  Accuracy (±{tol}): {custom_accuracy:.2f}%\n")

# ==========================================
# 7. GRAPHING / VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='red')
plt.title('Multi-Output CNN-LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# 1. Save the neural network weights
model.save('hydroponics_ai_model.keras')

# 2. Save the scalers so you can process live data exactly like the training data
joblib.dump(scaler_X, 'scaler_features.pkl')
joblib.dump(scaler_y, 'scaler_targets.pkl')

print("Model and scalers saved successfully!")
```

## 2. Exact interpretation of the current Time-Series setup

### Features

The model uses 5 variables at every observation:

- `water_level`
- `DHT_temp`
- `TDS`
- `pH`
- `DHT_humidity`

Therefore the task is **multivariate time-series forecasting**.

### Lookback/window

```python
time_steps = 15
```

This means **15 consecutive observations**, not inherently 15 minutes.

The actual time represented by 15 observations depends on the dataset sampling interval.

Examples:

- 1 observation/minute → 15 observations = 15 minutes
- 1 observation/10 seconds → 15 observations = 2.5 minutes
- 1 observation/30 seconds → 15 observations = 7.5 minutes

Do NOT state that the model forecasts 15 minutes ahead unless the dataset's actual sampling interval proves this.

### Sliding-window construction

The code does:

```python
X.append(data_X[i:(i + time_steps)])
y.append(data_y[i + time_steps])
```

Therefore:

```text
R1 ... R15  -> predict R16
R2 ... R16  -> predict R17
R3 ... R17  -> predict R18
...
```

Each `R` contains 5 sensor values.

So, conceptually:

```text
Input shape per sample = (15, 5)
Output shape per sample = (5,)
```

This is **one-step-ahead, multi-output, multivariate forecasting**.

### Current task definition

The model is NOT directly trained to classify a nutrient deficiency or decide which chemical/pump to activate.

It predicts the next sensor state:

```text
Past 15 observations
        ↓
   CNN + BiLSTM
        ↓
Next observation
[water_level, DHT_temp, TDS, pH, DHT_humidity]
```

Any stress detection or dosing decision is downstream logic and must be inspected separately.

## 3. Current architecture from code

```text
Input: 15 timesteps × 5 features
        ↓
Conv1D(filters=128, kernel_size=3, ReLU)
        ↓
MaxPooling1D(pool_size=2)
        ↓
Bidirectional(LSTM(50))
        ↓
Dropout(0.4)
        ↓
Dense(5, linear)
        ↓
5 predicted sensor values
```

Training configuration from the source code:

- Optimizer: Adam
- Learning rate: `0.0001`
- Loss: MSE
- Metric during training: MAE
- Epochs: `35`
- Batch size: `32`
- Validation data: `X_test, y_test`

Note: the project report may mention different training settings (for example 100 epochs). The source code supplied here uses **35 epochs**. This discrepancy must be investigated rather than silently reconciled.

## 4. Preprocessing actually implemented

The code first performs:

```python
df.interpolate(method='linear').dropna()
```

Then it fits MinMaxScaler to the **entire interpolated dataset** before the train/test split:

```python
scaled_features = scaler_X.fit_transform(df_interpolated[features])
scaled_target = scaler_y.fit_transform(df_interpolated[target_cols])
```

Then sequences are generated and only afterward an 80/20 chronological split is performed:

```python
split_idx = int(len(X_seq) * 0.8)
X_train = X_seq[:split_idx]
X_test = X_seq[split_idx:]
```

Important audit point:

- The split itself is chronological, which is appropriate for a basic time-series evaluation.
- However, the scalers are fitted on the full dataset before splitting. This allows information from the future/test period to influence the scaling parameters and should be corrected for a rigorous paper evaluation.

Preferred evaluation pipeline to investigate:

```text
Chronological split
       ↓
Fit scaler ONLY on training period
       ↓
Transform train/validation/test using train-fitted scaler
       ↓
Create sequences with careful boundary handling
```

## 5. Important evaluation issue

The code passes the test set as validation data during training:

```python
validation_data=(X_test, y_test)
```

Then the same `X_test/y_test` is used for final performance reporting:

```python
y_pred_scaled = model.predict(X_test)
```

For a conference-quality evaluation, the test set should ideally remain untouched until the final evaluation. A more defensible setup would be chronological train/validation/test partitions, for example:

```text
Train → model fitting
Validation → model selection / hyperparameter tuning
Test → final one-time evaluation
```

The exact percentages should be chosen based on dataset size and experimental structure.

## 6. Custom accuracy metric — important interpretation

The code does NOT calculate conventional classification accuracy.

For each sensor it counts a prediction as accurate if:

```python
abs(actual - predicted) <= tolerance
```

with these tolerances:

| Sensor | Tolerance |
|---|---:|
| water_level | 1.0 |
| DHT_temp | 0.5 |
| TDS | 20.0 |
| pH | 0.1 |
| DHT_humidity | 2.0 |

Then:

```python
custom_accuracy = accurate_predictions / len(actual_values) * 100
```

Therefore any reported "accuracy %" from this calculation should be described as a **tolerance-based prediction accuracy / within-tolerance rate**, NOT generic classification accuracy.

The scientific validity of the chosen tolerances must be justified from sensor specifications, agronomic/biological requirements, or operational requirements rather than chosen arbitrarily.

## 7. Current audit questions to answer before changing the model

The AI agent should investigate these items in order:

1. What is the actual sampling interval in `IoTData_25K_without_interpolation.csv`?
2. Does the CSV contain timestamps? If yes, verify ordering, duplicates, gaps, and sampling regularity.
3. Does one row represent one sensor reading at one time point?
4. Are there multiple experiments, plants, hydroponic runs, stress conditions, or independent sequences mixed into the CSV?
5. Is interpolation performed across experiment/run boundaries or only within continuous runs?
6. Does the dataset contain a true train/test temporal boundary or are multiple independent runs concatenated?
7. Are the 15-step windows crossing meaningful experiment boundaries?
8. What are the exact distributions/ranges of the five features?
9. Are there missing values, duplicated timestamps, outliers, sensor spikes, impossible physical values, or flatlined sensors?
10. What exactly produces the project's reported stress labels/threshold decisions?
11. Are the reported 96% figures generated by this exact code, an earlier code version, or another evaluation script?
12. Where does the claimed "6% mean predictive error" come from? It is not directly produced by the code above unless another calculation exists elsewhere.
13. What is the exact real-world sampling interval from the ESP32/Firebase pipeline?
14. Does live inference use exactly the same preprocessing/scalers as training?
15. How are missing live readings handled?
16. Is the model actually used to trigger pumps automatically, and what safety/rule layer sits between prediction and actuation?

## 8. Potential research improvements — DO NOT IMPLEMENT YET

These are hypotheses/directions for later benchmarking, not conclusions that the current model is wrong:

### Evaluation rigor
- Fit preprocessing only on training data.
- Use separate train/validation/test periods.
- Preserve chronological order.
- If multiple independent runs exist, consider group/run-based evaluation to test generalization to unseen runs.
- Report MAE, RMSE, R² and other appropriate forecasting metrics per sensor.
- For stress detection, report precision, recall, F1, macro F1, confusion matrix, PR-AUC where applicable.

### Forecasting baselines
Benchmark the existing model against simpler and alternative temporal models under the same split and preprocessing protocol:

- persistence / last-value baseline
- linear or classical forecasting baseline where appropriate
- LSTM
- GRU
- TCN
- CNN-LSTM
- BiLSTM
- temporal attention / Transformer only if justified by dataset size and task

Do not claim a new model is better without fair baseline comparison.

### Ablation studies
Potential experiments:

- window sizes: 5 / 10 / 15 / 30 / 60
- CNN vs no CNN
- LSTM vs BiLSTM
- GRU alternative
- attention vs no attention
- all sensors vs selected sensor subsets
- single-task vs multi-output forecasting
- interpolation strategies

### Robustness and deployment
Investigate:

- missing sensor readings
- noisy sensor values
- calibration drift
- domain shift between Kaggle/training data and ESP32 real-world data
- inference latency
- communication/Firebase delay
- sensor failure / flatline behavior

### Uncertainty and safety
Because predictions can influence physical dosing, consider uncertainty-aware predictions and a safety layer. A possible policy to evaluate later:

```text
High-confidence prediction → automatic action if within safety constraints
Medium confidence → alert / human confirmation
Low confidence → do not automatically dose
```

This is a research direction, not an existing project feature unless confirmed by code.

### Explainability
Potentially provide explanations such as:

```text
Predicted pH decrease
because recent pH trend is downward
and TDS/temperature pattern changed
```

Actual implementation should use a technically defensible explainability method rather than manually generated explanations.

## 9. Recommended AI-development workflow for this project

Do NOT immediately replace the CNN-BiLSTM.

Recommended sequence:

```text
1. Understand existing implementation
        ↓
2. Reproduce current reported results
        ↓
3. Audit dataset + preprocessing + split + metrics
        ↓
4. Establish strong baselines
        ↓
5. Identify the real research weakness
        ↓
6. Propose improved architecture/method
        ↓
7. Run controlled experiments + ablations
        ↓
8. Validate on real ESP32/hydroponic data
        ↓
9. Analyze robustness + uncertainty + safety
        ↓
10. Prepare conference-paper methodology/results
```

The immediate deliverable should be an **AI Audit / Reproduction Report**, not a new neural network.

## 10. Key terminology for the AI agent/user

- **Time Series:** observations ordered by time where temporal order matters.
- **Multivariate Time Series:** multiple variables measured over time.
- **Lookback / Window:** number of past observations fed into the model.
- **Forecast Horizon:** how far into the future the model predicts.
- **One-step forecasting:** predict the next observation.
- **Sliding Window:** repeatedly move the historical input window forward one timestep.
- **Multivariate multi-output regression:** predict several continuous variables simultaneously.
- **Data leakage:** information from evaluation/future data influences model fitting or preprocessing.
- **Chronological split:** earlier observations for training, later observations for validation/test.
- **Within-tolerance accuracy:** fraction of predictions whose absolute error is below a predefined tolerance.

## 11. Important correction to the previous project understanding

The current source code makes the following facts clear:

- `time_steps=15` means 15 observations, not automatically 15 minutes.
- The model predicts the **next single timestep**, not 15 future timesteps.
- Each input timestep contains 5 sensor features.
- The model performs multi-output regression for those 5 sensor values.
- The current split is chronological 80/20, not random.
- However, scalers are fitted before the split, and the test set is also used as validation during training; both require attention in a rigorous evaluation.
- The custom "accuracy" is a tolerance-based regression metric, not ordinary classification accuracy.
- The source code uses 35 epochs, so any report claiming 100 epochs must be reconciled with the actual experiment/version.

