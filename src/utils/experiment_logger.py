"""
utils/experiment_logger.py — Structured JSON experiment logging.

Every run should call log_experiment() to persist its metadata and results.
Logs are written to experiments/<experiment_id>.json.
"""

import json
import os
import datetime
from typing import Any, Dict, Optional

# Resolve experiments dir relative to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.join(_THIS_DIR, "..", "..", "experiments")


def log_experiment(
    experiment_id: str,
    model_name: str,
    split_strategy: str,
    window_size: int,
    features: list,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, Any],
    random_seed: int = 42,
    dataset: str = "IoTData_25K_without_interpolation.csv",
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Persist experiment metadata and results as a JSON file.

    Args:
        experiment_id : Unique ID, e.g. "exp_001_chronological_cnn_bilstm".
        model_name    : Name of the model architecture.
        split_strategy: e.g. "chronological_70_10_20" | "random_80_20" | "group".
        window_size   : Lookback window (number of timesteps).
        features      : List of feature names used.
        hyperparameters: Dict of all training hyperparams.
        metrics       : Dict of evaluation metrics (fill with None if not yet run).
        random_seed   : Seed used for this run.
        dataset       : Dataset filename.
        notes         : Free-text notes about this experiment.
        extra         : Any additional metadata.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset,
        "split_strategy": split_strategy,
        "window_size": window_size,
        "features": features,
        "model": model_name,
        "hyperparameters": hyperparameters,
        "random_seed": random_seed,
        "metrics": metrics,
        "notes": notes,
    }
    if extra:
        record["extra"] = extra

    file_path = os.path.join(_EXPERIMENTS_DIR, f"{experiment_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(f"[experiment_logger] Saved → {file_path}")
    return file_path


def load_experiment(experiment_id: str) -> Dict[str, Any]:
    """Load a previously logged experiment by ID."""
    file_path = os.path.join(_EXPERIMENTS_DIR, f"{experiment_id}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_experiments() -> list:
    """Return sorted list of all experiment IDs in experiments/."""
    if not os.path.exists(_EXPERIMENTS_DIR):
        return []
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(_EXPERIMENTS_DIR)
        if f.endswith(".json")
    )
