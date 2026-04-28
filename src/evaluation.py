"""Forecast evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_numeric_array(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Convert input values to a finite numeric numpy array."""
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return array


def mae(y_true: pd.Series | np.ndarray | list[float], y_pred: pd.Series | np.ndarray | list[float]) -> float:
    """Calculate mean absolute error."""
    true = _as_numeric_array(y_true)
    pred = _as_numeric_array(y_pred)
    return float(np.mean(np.abs(true - pred)))


def rmse(y_true: pd.Series | np.ndarray | list[float], y_pred: pd.Series | np.ndarray | list[float]) -> float:
    """Calculate root mean squared error."""
    true = _as_numeric_array(y_true)
    pred = _as_numeric_array(y_pred)
    return float(math.sqrt(np.mean((true - pred) ** 2)))


def mape(y_true: pd.Series | np.ndarray | list[float], y_pred: pd.Series | np.ndarray | list[float]) -> float:
    """Calculate mean absolute percentage error, ignoring zero actual values."""
    true = _as_numeric_array(y_true)
    pred = _as_numeric_array(y_pred)
    non_zero_mask = true != 0
    if not np.any(non_zero_mask):
        return float("nan")
    return float(np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / true[non_zero_mask])) * 100)


def r2_score(y_true: pd.Series | np.ndarray | list[float], y_pred: pd.Series | np.ndarray | list[float]) -> float:
    """Calculate the coefficient of determination."""
    true = _as_numeric_array(y_true)
    pred = _as_numeric_array(y_pred)
    residual_sum = np.sum((true - pred) ** 2)
    total_sum = np.sum((true - np.mean(true)) ** 2)
    if total_sum == 0:
        return float("nan")
    return float(1 - residual_sum / total_sum)


def evaluate_forecast(
    y_true: pd.Series | np.ndarray | list[float],
    y_pred: pd.Series | np.ndarray | list[float],
) -> dict[str, float]:
    """Calculate all supported forecast metrics."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }

