"""Moving average and exponential moving average forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecast

ApproximationMethod = Literal["moving_average", "exponential_moving_average"]


@dataclass(frozen=True)
class ApproximationSelectionResult:
    """Result of MA/EMA parameter selection."""

    method: ApproximationMethod
    best_parameter: int
    predictions: np.ndarray
    metrics_by_parameter: dict[str, dict[str, float]]


def moving_average_forecast(train: pd.Series, horizon: int, window: int) -> np.ndarray:
    """Forecast recursively using a trailing moving average."""
    if window < 1:
        raise ValueError("Moving average window must be at least 1.")

    history = pd.to_numeric(train, errors="coerce").dropna().to_list()
    if not history:
        raise ValueError("Train series must not be empty.")

    predictions: list[float] = []
    for _ in range(horizon):
        prediction = float(np.mean(history[-window:]))
        predictions.append(prediction)
        history.append(prediction)
    return np.array(predictions, dtype=float)


def moving_average_one_step_forecast(
    train: pd.Series,
    test: pd.Series,
    window: int,
) -> np.ndarray:
    """Forecast one step ahead with moving average and update history with actuals."""
    if window < 1:
        raise ValueError("Moving average window must be at least 1.")

    history = pd.to_numeric(train, errors="coerce").dropna().to_list()
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    if not history:
        raise ValueError("Train series must not be empty.")

    predictions: list[float] = []
    for actual_value in test_values:
        predictions.append(float(np.mean(history[-window:])))
        history.append(float(actual_value))
    return np.array(predictions, dtype=float)


def exponential_moving_average_forecast(train: pd.Series, horizon: int, span: int) -> np.ndarray:
    """Forecast recursively using the latest exponential moving average."""
    if span < 1:
        raise ValueError("EMA span must be at least 1.")

    train_values = pd.to_numeric(train, errors="coerce").dropna()
    if train_values.empty:
        raise ValueError("Train series must not be empty.")

    alpha = 2 / (span + 1)
    ema_value = float(train_values.iloc[0])
    for value in train_values.iloc[1:]:
        ema_value = alpha * float(value) + (1 - alpha) * ema_value

    predictions: list[float] = []
    for _ in range(horizon):
        prediction = ema_value
        predictions.append(prediction)
        ema_value = alpha * prediction + (1 - alpha) * ema_value
    return np.array(predictions, dtype=float)


def exponential_moving_average_one_step_forecast(
    train: pd.Series,
    test: pd.Series,
    span: int,
) -> np.ndarray:
    """Forecast one step ahead with EMA and update the EMA with actual values."""
    if span < 1:
        raise ValueError("EMA span must be at least 1.")

    train_values = pd.to_numeric(train, errors="coerce").dropna()
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    if train_values.empty:
        raise ValueError("Train series must not be empty.")

    alpha = 2 / (span + 1)
    ema_value = float(train_values.iloc[0])
    for value in train_values.iloc[1:]:
        ema_value = alpha * float(value) + (1 - alpha) * ema_value

    predictions: list[float] = []
    for actual_value in test_values:
        predictions.append(ema_value)
        ema_value = alpha * float(actual_value) + (1 - alpha) * ema_value
    return np.array(predictions, dtype=float)


def select_moving_average_window(
    train: pd.Series,
    validation: pd.Series,
    windows: Iterable[int],
) -> ApproximationSelectionResult:
    """Select the best moving average window by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_window: dict[str, dict[str, float]] = {}
    predictions_by_window: dict[int, np.ndarray] = {}

    for window in windows:
        predictions = moving_average_forecast(train, len(validation_values), window)
        predictions_by_window[window] = predictions
        metrics_by_window[str(window)] = evaluate_forecast(validation_values, predictions)

    best_window = int(min(metrics_by_window, key=lambda key: metrics_by_window[key]["RMSE"]))
    return ApproximationSelectionResult(
        method="moving_average",
        best_parameter=best_window,
        predictions=predictions_by_window[best_window],
        metrics_by_parameter=metrics_by_window,
    )


def select_moving_average_one_step_window(
    train: pd.Series,
    validation: pd.Series,
    windows: Iterable[int],
) -> ApproximationSelectionResult:
    """Select the best one-step moving average window by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_window: dict[str, dict[str, float]] = {}
    predictions_by_window: dict[int, np.ndarray] = {}

    for window in windows:
        predictions = moving_average_one_step_forecast(train, validation, window)
        predictions_by_window[window] = predictions
        metrics_by_window[str(window)] = evaluate_forecast(validation_values, predictions)

    best_window = int(min(metrics_by_window, key=lambda key: metrics_by_window[key]["RMSE"]))
    return ApproximationSelectionResult(
        method="moving_average",
        best_parameter=best_window,
        predictions=predictions_by_window[best_window],
        metrics_by_parameter=metrics_by_window,
    )


def select_exponential_moving_average_span(
    train: pd.Series,
    validation: pd.Series,
    spans: Iterable[int],
) -> ApproximationSelectionResult:
    """Select the best EMA span by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_span: dict[str, dict[str, float]] = {}
    predictions_by_span: dict[int, np.ndarray] = {}

    for span in spans:
        predictions = exponential_moving_average_forecast(train, len(validation_values), span)
        predictions_by_span[span] = predictions
        metrics_by_span[str(span)] = evaluate_forecast(validation_values, predictions)

    best_span = int(min(metrics_by_span, key=lambda key: metrics_by_span[key]["RMSE"]))
    return ApproximationSelectionResult(
        method="exponential_moving_average",
        best_parameter=best_span,
        predictions=predictions_by_span[best_span],
        metrics_by_parameter=metrics_by_span,
    )


def select_exponential_moving_average_one_step_span(
    train: pd.Series,
    validation: pd.Series,
    spans: Iterable[int],
) -> ApproximationSelectionResult:
    """Select the best one-step EMA span by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_span: dict[str, dict[str, float]] = {}
    predictions_by_span: dict[int, np.ndarray] = {}

    for span in spans:
        predictions = exponential_moving_average_one_step_forecast(train, validation, span)
        predictions_by_span[span] = predictions
        metrics_by_span[str(span)] = evaluate_forecast(validation_values, predictions)

    best_span = int(min(metrics_by_span, key=lambda key: metrics_by_span[key]["RMSE"]))
    return ApproximationSelectionResult(
        method="exponential_moving_average",
        best_parameter=best_span,
        predictions=predictions_by_span[best_span],
        metrics_by_parameter=metrics_by_span,
    )
