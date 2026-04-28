"""Alpha-beta recurrent smoothing and forecasting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecast


DEFAULT_ALPHA_GRID = [0.6, 0.7, 0.8, 0.85, 0.9]
DEFAULT_BETA_GRID = [0.001, 0.003, 0.005, 0.01]


def _as_clean_array(y: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    """Convert a time series to a non-empty finite numeric array."""
    values = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Time series must contain at least one numeric value.")
    return values


def alpha_beta_filter(
    y: pd.Series | np.ndarray | list[float],
    alpha: float = 0.85,
    beta: float = 0.005,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate smoothed level and velocity with an alpha-beta filter."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    values = _as_clean_array(y)
    levels = np.empty(len(values), dtype=float)
    velocities = np.empty(len(values), dtype=float)

    level = float(values[0])
    velocity = float((values[1] - values[0]) / dt) if len(values) > 1 else 0.0

    for index, measurement in enumerate(values):
        if index > 0:
            predicted_level = level + velocity * dt
            residual = float(measurement) - predicted_level
            level = predicted_level + alpha * residual
            velocity = velocity + (beta / dt) * residual

        levels[index] = level
        velocities[index] = velocity

    return levels, velocities


def forecast_alpha_beta(
    y: pd.Series | np.ndarray | list[float],
    steps: int,
    alpha: float = 0.85,
    beta: float = 0.005,
    dt: float = 1.0,
) -> np.ndarray:
    """Forecast recursively from the last estimated level and velocity."""
    if steps < 0:
        raise ValueError("steps must not be negative.")
    if steps == 0:
        return np.array([], dtype=float)

    levels, velocities = alpha_beta_filter(y, alpha=alpha, beta=beta, dt=dt)
    level = float(levels[-1])
    velocity = float(velocities[-1])

    predictions: list[float] = []
    for _ in range(steps):
        level = level + velocity * dt
        predictions.append(level)

    return np.array(predictions, dtype=float)


def forecast_alpha_beta_one_step(
    train: pd.Series | np.ndarray | list[float],
    test: pd.Series | np.ndarray | list[float],
    alpha: float = 0.85,
    beta: float = 0.005,
    dt: float = 1.0,
) -> np.ndarray:
    """Forecast one step ahead and update the filter with each actual value."""
    if dt <= 0:
        raise ValueError("dt must be positive.")

    test_values = pd.to_numeric(pd.Series(test), errors="coerce").to_numpy(dtype=float)
    if len(test_values) == 0:
        return np.array([], dtype=float)

    levels, velocities = alpha_beta_filter(train, alpha=alpha, beta=beta, dt=dt)
    level = float(levels[-1])
    velocity = float(velocities[-1])
    predictions: list[float] = []

    for actual_value in test_values:
        predicted_level = level + velocity * dt
        predictions.append(predicted_level)
        residual = float(actual_value) - predicted_level
        level = predicted_level + alpha * residual
        velocity = velocity + (beta / dt) * residual

    return np.array(predictions, dtype=float)


def optimize_alpha_beta(
    y_train: pd.Series | np.ndarray | list[float],
    y_valid: pd.Series | np.ndarray | list[float],
    alpha_grid: Iterable[float] | None = None,
    beta_grid: Iterable[float] | None = None,
) -> dict[str, object]:
    """Select alpha and beta with minimum validation MAE."""
    train_values = _as_clean_array(y_train)
    valid_values = _as_clean_array(y_valid)
    alphas = list(DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid)
    betas = list(DEFAULT_BETA_GRID if beta_grid is None else beta_grid)
    if not alphas or not betas:
        raise ValueError("alpha_grid and beta_grid must not be empty.")

    best_alpha = float(alphas[0])
    best_beta = float(betas[0])
    best_mae = float("inf")
    best_predictions = np.array([], dtype=float)
    metrics_by_configuration: dict[str, dict[str, float]] = {}

    for alpha in alphas:
        for beta in betas:
            predictions = forecast_alpha_beta(
                train_values,
                steps=len(valid_values),
                alpha=float(alpha),
                beta=float(beta),
            )
            metrics = evaluate_forecast(valid_values, predictions)
            configuration = f"alpha={float(alpha)}|beta={float(beta)}"
            metrics_by_configuration[configuration] = metrics
            if metrics["MAE"] < best_mae:
                best_alpha = float(alpha)
                best_beta = float(beta)
                best_mae = metrics["MAE"]
                best_predictions = predictions

    return {
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "validation_predictions": best_predictions,
        "metrics_by_configuration": metrics_by_configuration,
    }
