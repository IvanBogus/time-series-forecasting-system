"""Synthetic time series generation and verification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SyntheticTrendType = Literal["linear", "quadratic", "exponential"]


@dataclass(frozen=True)
class SyntheticSeriesConfig:
    """Configuration for generating a synthetic time series."""

    trend_type: SyntheticTrendType
    n_points: int = 240
    noise_std: float = 0.35
    anomaly_fraction: float = 0.05
    anomaly_scale: float = 4.0
    random_seed: int = 42


def generate_true_trend(trend_type: SyntheticTrendType, n_points: int) -> np.ndarray:
    """Generate a known deterministic trend."""
    x = np.arange(n_points, dtype=float)
    if trend_type == "linear":
        return 10.0 + 0.05 * x
    if trend_type == "quadratic":
        centered_x = x - n_points / 2
        return 14.0 + 0.03 * x + 0.0009 * centered_x**2
    if trend_type == "exponential":
        return 8.0 + 2.0 * np.exp(0.008 * x)
    raise ValueError(f"Unsupported synthetic trend type: {trend_type}")


def generate_synthetic_series(config: SyntheticSeriesConfig) -> pd.DataFrame:
    """Generate synthetic trend, noisy observations, and injected anomalies."""
    rng = np.random.default_rng(config.random_seed)
    dates = pd.date_range("2024-01-01", periods=config.n_points, freq="D")
    true_trend = generate_true_trend(config.trend_type, config.n_points)
    noisy_values = true_trend + rng.normal(0.0, config.noise_std, config.n_points)

    anomaly_count = max(1, int(config.n_points * config.anomaly_fraction))
    anomaly_indices = rng.choice(config.n_points, size=anomaly_count, replace=False)
    anomaly_directions = rng.choice([-1.0, 1.0], size=anomaly_count)
    observed_values = noisy_values.copy()
    observed_values[anomaly_indices] += (
        anomaly_directions * config.anomaly_scale * config.noise_std
    )

    return pd.DataFrame(
        {
            "date": dates,
            "true_trend": true_trend,
            "noisy_value": noisy_values,
            "value": observed_values,
            "is_injected_anomaly": np.isin(np.arange(config.n_points), anomaly_indices),
        }
    )


def synthetic_train_test_split(data: pd.DataFrame, train_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split synthetic data chronologically into train and test sets."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")
    split_index = int(len(data) * train_size)
    if split_index == 0 or split_index >= len(data):
        raise ValueError("Both train and test splits must contain at least one row.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()

