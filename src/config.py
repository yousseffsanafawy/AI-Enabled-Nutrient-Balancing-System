"""
config.py — Central configuration for the AI-Enabled Nutrient Balancing System.

All tunable constants live here. Import this module instead of hard-coding values.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR        = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
MODELS_DIR      = os.path.join(ROOT_DIR, "saved_models")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
REPORTS_DIR     = os.path.join(ROOT_DIR, "reports")
NOTEBOOKS_DIR   = os.path.join(ROOT_DIR, "notebooks")

# Primary dataset (without interpolation — use raw for audit)
RAW_CSV         = os.path.join(RAW_DATA_DIR, "IoTData_25K_without_interpolation.csv")
RAW_CSV_INTERP  = os.path.join(RAW_DATA_DIR, "IoTData_25K_with_interpolation.csv")

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# FEATURES & TARGETS
# ─────────────────────────────────────────────
FEATURES = [
    "water_level",
    "DHT_temp",
    "TDS",
    "pH",
    "DHT_humidity",
]
TARGET_COLS = FEATURES  # multi-output: predict all 5 sensors

N_FEATURES = len(FEATURES)

# ─────────────────────────────────────────────
# SEQUENCE / WINDOW SETTINGS
# ─────────────────────────────────────────────
TIME_STEPS      = 15    # lookback window (steps)
FORECAST_HORIZON = 1    # one-step-ahead forecasting

# ─────────────────────────────────────────────
# DATA SPLIT RATIOS
# ─────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
# Note: split is always CHRONOLOGICAL — never random for time-series

# ─────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────
SCALE_RANGE = (0, 1)   # MinMaxScaler range
# IMPORTANT: scalers must be fit ONLY on the training partition

# ─────────────────────────────────────────────
# TRAINING DEFAULTS (baseline model)
# ─────────────────────────────────────────────
LEARNING_RATE = 1e-4
BATCH_SIZE    = 32
EPOCHS        = 35
DROPOUT       = 0.4
LOSS          = "mse"

# ─────────────────────────────────────────────
# MODEL ARCHITECTURE DEFAULTS (baseline CNN-BiLSTM)
# ─────────────────────────────────────────────
CNN_FILTERS     = 128
CNN_KERNEL_SIZE = 3
LSTM_UNITS      = 50
POOL_SIZE       = 2

# ─────────────────────────────────────────────
# TOLERANCE-BASED ACCURACY (operational metric)
# Source: confirmed from AI_PBL (1).ipynb
# These are domain/sensor-specific tolerances, NOT classification thresholds
# ─────────────────────────────────────────────
TOLERANCES = {
    "water_level": 1.0,    # ± 1 litre/cm
    "DHT_temp":    0.5,    # ± 0.5 °C
    "TDS":         20.0,   # ± 20 ppm
    "pH":          0.1,    # ± 0.1 pH units
    "DHT_humidity": 2.0,   # ± 2 % RH
}

# ─────────────────────────────────────────────
# STRESS / SAFETY THRESHOLDS (hydroponic domain)
# Source: general hydroponics agronomic ranges
# Must be verified / adjusted per plant species
# ─────────────────────────────────────────────
SAFE_RANGES = {
    "pH":          (5.5, 6.5),    # optimal hydroponic pH range
    "TDS":         (600, 1400),   # ppm — general leafy greens
    "DHT_temp":    (18.0, 28.0),  # °C ambient
    "water_level": (5.0, 30.0),   # cm (system-dependent)
    "DHT_humidity":(50.0, 80.0),  # % RH
}

# ─────────────────────────────────────────────
# UNCERTAINTY / SAFETY LAYER
# ─────────────────────────────────────────────
MC_DROPOUT_SAMPLES     = 50     # number of forward passes for MC Dropout
HIGH_UNCERTAINTY_SIGMA = 0.15   # σ above this → no auto-dosing
MED_UNCERTAINTY_SIGMA  = 0.07   # σ above this → alert human

# ─────────────────────────────────────────────
# MULTI-TASK LOSS WEIGHT
# ─────────────────────────────────────────────
LAMBDA_STRESS = 0.2   # weight of stress classification loss
# total_loss = (1 - LAMBDA_STRESS) * regression_mse + LAMBDA_STRESS * stress_ce

# ─────────────────────────────────────────────
# ABLATION WINDOW SIZES TO TEST
# ─────────────────────────────────────────────
ABLATION_WINDOWS = [5, 10, 15, 30, 60]
