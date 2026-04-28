"""Polynomial regression forecasting with numpy.polyfit."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecast


@dataclass(frozen=True)
class PolynomialSelectionResult:
    """Result of polynomial degree selection."""

    best_degree: int
    predictions: np.ndarray
    metrics_by_degree: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LocalPolynomialSelectionResult:
    """Result of local polynomial window and degree selection."""

    best_window: int | str
    best_degree: int
    predictions: np.ndarray
    metrics_by_configuration: dict[str, dict[str, float]]


def _fit_normalized_polynomial(values: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    """Fit a polynomial using normalized x values for numerical stability."""
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")
    if len(values) <= degree:
        raise ValueError("Train series length must be greater than polynomial degree.")

    x_train = np.arange(len(values), dtype=float)
    x_mean = float(x_train.mean())
    x_std = float(x_train.std())
    if x_std == 0:
        x_std = 1.0

    x_train_normalized = (x_train - x_mean) / x_std
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficients = np.polyfit(x_train_normalized, values, degree)
    return coefficients, x_mean, x_std


def _predict_normalized_polynomial(
    coefficients: np.ndarray,
    x_mean: float,
    x_std: float,
    start_index: int,
    horizon: int,
) -> np.ndarray:
    """Predict future values from a polynomial fitted on normalized x values."""
    x_forecast = np.arange(start_index, start_index + horizon, dtype=float)
    x_forecast_normalized = (x_forecast - x_mean) / x_std
    return np.polyval(coefficients, x_forecast_normalized)


def polynomial_forecast(train: pd.Series, horizon: int, degree: int) -> np.ndarray:
    """Fit a global polynomial trend on train data and forecast a fixed horizon."""
    train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    coefficients, x_mean, x_std = _fit_normalized_polynomial(train_values, degree)
    return _predict_normalized_polynomial(
        coefficients,
        x_mean,
        x_std,
        start_index=len(train_values),
        horizon=horizon,
    )


def local_polynomial_forecast(
    train: pd.Series,
    horizon: int,
    degree: int,
    window: int | str,
) -> np.ndarray:
    """Fit a polynomial on the last N train points and forecast a fixed horizon."""
    train_values = pd.to_numeric(train, errors="coerce").to_numpy(dtype=float)
    if window == "all":
        local_values = train_values
    else:
        local_values = train_values[-int(window) :]

    coefficients, x_mean, x_std = _fit_normalized_polynomial(local_values, degree)
    return _predict_normalized_polynomial(
        coefficients,
        x_mean,
        x_std,
        start_index=len(local_values),
        horizon=horizon,
    )


def local_polynomial_one_step_forecast(
    train: pd.Series,
    test: pd.Series,
    degree: int,
    window: int | str,
) -> np.ndarray:
    """Forecast one step ahead with a local polynomial and update with actuals."""
    history = pd.to_numeric(train, errors="coerce").dropna().to_list()
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float)
    if not history:
        raise ValueError("Train series must not be empty.")

    predictions: list[float] = []
    for actual_value in test_values:
        history_series = pd.Series(history, dtype=float)
        prediction = local_polynomial_forecast(history_series, 1, degree, window)[0]
        predictions.append(float(prediction))
        history.append(float(actual_value))
    return np.array(predictions, dtype=float)


def local_polynomial_recursive_forecast(
    train: pd.Series,
    horizon: int,
    degree: int,
    window: int | str,
) -> np.ndarray:
    """Forecast recursively with a local polynomial and update with predictions."""
    history = pd.to_numeric(train, errors="coerce").dropna().to_list()
    if not history:
        raise ValueError("Train series must not be empty.")

    predictions: list[float] = []
    for _ in range(horizon):
        history_series = pd.Series(history, dtype=float)
        prediction = local_polynomial_forecast(history_series, 1, degree, window)[0]
        predictions.append(float(prediction))
        history.append(float(prediction))
    return np.array(predictions, dtype=float)


def select_polynomial_degree(
    train: pd.Series,
    validation: pd.Series,
    degrees: Iterable[int] = range(1, 9),
) -> PolynomialSelectionResult:
    """Select the best polynomial degree by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_degree: dict[str, dict[str, float]] = {}
    predictions_by_degree: dict[int, np.ndarray] = {}

    for degree in degrees:
        predictions = polynomial_forecast(train, len(validation_values), degree)
        predictions_by_degree[degree] = predictions
        metrics_by_degree[str(degree)] = evaluate_forecast(validation_values, predictions)

    best_degree = min(
        metrics_by_degree,
        key=lambda degree_key: metrics_by_degree[degree_key]["RMSE"],
    )
    selected_degree = int(best_degree)
    return PolynomialSelectionResult(
        best_degree=selected_degree,
        predictions=predictions_by_degree[selected_degree],
        metrics_by_degree=metrics_by_degree,
    )


def select_local_polynomial_configuration(
    train: pd.Series,
    validation: pd.Series,
    degrees: Iterable[int] = range(1, 9),
    windows: Iterable[int | str] = (30, 60, 90, 120, 180, "all"),
) -> LocalPolynomialSelectionResult:
    """Select the best local polynomial window and degree by validation RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_configuration: dict[str, dict[str, float]] = {}
    predictions_by_configuration: dict[str, np.ndarray] = {}

    for window in windows:
        for degree in degrees:
            try:
                predictions = local_polynomial_forecast(
                    train,
                    len(validation_values),
                    degree,
                    window,
                )
            except ValueError:
                continue
            key = f"window={window}|degree={degree}"
            predictions_by_configuration[key] = predictions
            metrics_by_configuration[key] = evaluate_forecast(validation_values, predictions)

    if not metrics_by_configuration:
        raise ValueError("No valid local polynomial configurations were produced.")

    best_key = min(
        metrics_by_configuration,
        key=lambda config_key: metrics_by_configuration[config_key]["RMSE"],
    )
    best_parts = dict(part.split("=", 1) for part in best_key.split("|"))
    best_window: int | str
    best_window = (
        best_parts["window"]
        if best_parts["window"] == "all"
        else int(best_parts["window"])
    )
    best_degree = int(best_parts["degree"])

    return LocalPolynomialSelectionResult(
        best_window=best_window,
        best_degree=best_degree,
        predictions=predictions_by_configuration[best_key],
        metrics_by_configuration=metrics_by_configuration,
    )


def select_local_polynomial_one_step_configuration(
    train: pd.Series,
    validation: pd.Series,
    degrees: Iterable[int] = range(1, 9),
    windows: Iterable[int | str] = (30, 60, 90, 120, 180, "all"),
) -> LocalPolynomialSelectionResult:
    """Select the best local polynomial one-step configuration by RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_configuration: dict[str, dict[str, float]] = {}
    predictions_by_configuration: dict[str, np.ndarray] = {}

    for window in windows:
        for degree in degrees:
            try:
                predictions = local_polynomial_one_step_forecast(
                    train,
                    validation,
                    degree,
                    window,
                )
            except ValueError:
                continue
            key = f"window={window}|degree={degree}"
            predictions_by_configuration[key] = predictions
            metrics_by_configuration[key] = evaluate_forecast(validation_values, predictions)

    if not metrics_by_configuration:
        raise ValueError("No valid local polynomial one-step configurations were produced.")

    best_key = min(
        metrics_by_configuration,
        key=lambda config_key: metrics_by_configuration[config_key]["RMSE"],
    )
    best_parts = dict(part.split("=", 1) for part in best_key.split("|"))
    best_window: int | str
    best_window = (
        best_parts["window"]
        if best_parts["window"] == "all"
        else int(best_parts["window"])
    )
    best_degree = int(best_parts["degree"])

    return LocalPolynomialSelectionResult(
        best_window=best_window,
        best_degree=best_degree,
        predictions=predictions_by_configuration[best_key],
        metrics_by_configuration=metrics_by_configuration,
    )


def select_local_polynomial_recursive_configuration(
    train: pd.Series,
    validation: pd.Series,
    degrees: Iterable[int] = range(1, 9),
    windows: Iterable[int | str] = (30, 60, 90, 120, 180, "all"),
) -> LocalPolynomialSelectionResult:
    """Select the best local polynomial recursive configuration by RMSE."""
    validation_values = pd.to_numeric(validation, errors="coerce").to_numpy(dtype=float)
    metrics_by_configuration: dict[str, dict[str, float]] = {}
    predictions_by_configuration: dict[str, np.ndarray] = {}

    for window in windows:
        for degree in degrees:
            try:
                predictions = local_polynomial_recursive_forecast(
                    train,
                    len(validation_values),
                    degree,
                    window,
                )
            except ValueError:
                continue
            key = f"window={window}|degree={degree}"
            predictions_by_configuration[key] = predictions
            metrics_by_configuration[key] = evaluate_forecast(validation_values, predictions)

    if not metrics_by_configuration:
        raise ValueError("No valid local polynomial recursive configurations were produced.")

    best_key = min(
        metrics_by_configuration,
        key=lambda config_key: metrics_by_configuration[config_key]["RMSE"],
    )
    best_parts = dict(part.split("=", 1) for part in best_key.split("|"))
    best_window: int | str
    best_window = (
        best_parts["window"]
        if best_parts["window"] == "all"
        else int(best_parts["window"])
    )
    best_degree = int(best_parts["degree"])

    return LocalPolynomialSelectionResult(
        best_window=best_window,
        best_degree=best_degree,
        predictions=predictions_by_configuration[best_key],
        metrics_by_configuration=metrics_by_configuration,
    )
