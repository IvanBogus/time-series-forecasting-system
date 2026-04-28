"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, test: pd.Series) -> np.ndarray:
    """Forecast one step ahead using the previous observed value."""
    train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    if len(train_values) == 0:
        raise ValueError("Train series must not be empty.")
    if len(test_values) == 0:
        return np.array([], dtype=float)

    predictions = np.empty(len(test_values), dtype=float)
    predictions[0] = train_values[-1]
    if len(test_values) > 1:
        predictions[1:] = test_values[:-1]
    return predictions


def naive_recursive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Forecast recursively by carrying the last train value forward."""
    train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    if len(train_values) == 0:
        raise ValueError("Train series must not be empty.")
    if horizon < 0:
        raise ValueError("Forecast horizon must not be negative.")

    return np.full(horizon, train_values[-1], dtype=float)
