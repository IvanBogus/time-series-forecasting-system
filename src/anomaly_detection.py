"""Anomaly detection and replacement methods for time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

ReplacementStrategy = Literal["interpolation", "rolling_median"]


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Container for anomaly detection output."""

    cleaned_data: pd.DataFrame
    anomaly_mask: pd.Series
    anomaly_count: int
    method: str


def _numeric_series(data: pd.DataFrame, value_column: str) -> pd.Series:
    """Return a numeric copy of the target value column."""
    series = pd.to_numeric(data[value_column], errors="coerce")
    if series.dropna().empty:
        raise ValueError(f"Column '{value_column}' does not contain numeric values.")
    return series


def _zero_missing_mask(series: pd.Series) -> pd.Series:
    """Mark zero values as missing observations for exchange-rate series."""
    return series == 0


def _interpolate_missing_values(series: pd.Series) -> pd.Series:
    """Replace missing values by interpolation with fill fallback."""
    cleaned = series.replace(0, np.nan)
    cleaned = cleaned.interpolate(method="linear", limit_direction="both")
    cleaned = cleaned.ffill().bfill()
    return cleaned


def _prepare_data_for_detection(
    data: pd.DataFrame,
    value_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare data by treating zeros as missing before anomaly detection."""
    prepared = data.copy()
    series = _numeric_series(prepared, value_column)
    zero_mask = _zero_missing_mask(series)
    prepared[value_column] = _interpolate_missing_values(series)
    return prepared, zero_mask


def _rolling_median(series: pd.Series, window: int) -> pd.Series:
    """Calculate centered rolling median with edge support."""
    return series.rolling(window=window, center=True, min_periods=1).median()


def _replace_anomalies(
    data: pd.DataFrame,
    value_column: str,
    anomaly_mask: pd.Series,
    replacement: ReplacementStrategy,
    window: int = 7,
) -> pd.DataFrame:
    """Replace anomalous values while preserving time series length."""
    cleaned = data.copy()
    series = _numeric_series(cleaned, value_column)

    if replacement == "rolling_median":
        replacement_values = _rolling_median(series, window)
        cleaned.loc[anomaly_mask, value_column] = replacement_values.loc[anomaly_mask]
        return cleaned

    series_with_gaps = series.mask(anomaly_mask)
    cleaned[value_column] = series_with_gaps.interpolate(
        method="linear",
        limit_direction="both",
    )
    return cleaned


def detect_z_score_anomalies(
    data: pd.DataFrame,
    value_column: str,
    threshold: float = 3.0,
    replacement: ReplacementStrategy = "interpolation",
) -> AnomalyDetectionResult:
    """Detect anomalies using the z-score / three-sigma rule."""
    prepared_data, zero_mask = _prepare_data_for_detection(data, value_column)
    series = _numeric_series(prepared_data, value_column)
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or pd.isna(std):
        anomaly_mask = pd.Series(False, index=data.index)
    else:
        anomaly_mask = ((series - mean).abs() / std) > threshold

    anomaly_mask = anomaly_mask | zero_mask
    cleaned = _replace_anomalies(prepared_data, value_column, anomaly_mask, replacement)
    return AnomalyDetectionResult(
        cleaned_data=cleaned,
        anomaly_mask=anomaly_mask,
        anomaly_count=int(anomaly_mask.sum()),
        method="z_score",
    )


def detect_iqr_anomalies(
    data: pd.DataFrame,
    value_column: str,
    multiplier: float = 1.5,
    replacement: ReplacementStrategy = "interpolation",
) -> AnomalyDetectionResult:
    """Detect anomalies using the interquartile range rule."""
    prepared_data, zero_mask = _prepare_data_for_detection(data, value_column)
    series = _numeric_series(prepared_data, value_column)
    quartile_1 = series.quantile(0.25)
    quartile_3 = series.quantile(0.75)
    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - multiplier * iqr
    upper_bound = quartile_3 + multiplier * iqr
    anomaly_mask = (series < lower_bound) | (series > upper_bound)

    anomaly_mask = anomaly_mask | zero_mask
    cleaned = _replace_anomalies(prepared_data, value_column, anomaly_mask, replacement)
    return AnomalyDetectionResult(
        cleaned_data=cleaned,
        anomaly_mask=anomaly_mask,
        anomaly_count=int(anomaly_mask.sum()),
        method="iqr",
    )


def detect_rolling_median_anomalies(
    data: pd.DataFrame,
    value_column: str,
    window: int = 7,
    threshold: float = 3.0,
    replacement: ReplacementStrategy = "rolling_median",
) -> AnomalyDetectionResult:
    """Detect anomalies by comparing values with a centered rolling median."""
    if window < 3:
        raise ValueError("Rolling window must be at least 3.")

    prepared_data, zero_mask = _prepare_data_for_detection(data, value_column)
    series = _numeric_series(prepared_data, value_column)
    rolling_median = _rolling_median(series, window)
    residuals = (series - rolling_median).abs()
    residual_scale = residuals.median()

    if residual_scale == 0 or pd.isna(residual_scale):
        residual_scale = residuals.mean()

    if residual_scale == 0 or pd.isna(residual_scale):
        anomaly_mask = pd.Series(False, index=data.index)
    else:
        anomaly_mask = residuals > threshold * residual_scale

    anomaly_mask = anomaly_mask | zero_mask
    cleaned = _replace_anomalies(
        prepared_data,
        value_column,
        anomaly_mask,
        replacement,
        window=window,
    )
    return AnomalyDetectionResult(
        cleaned_data=cleaned,
        anomaly_mask=anomaly_mask,
        anomaly_count=int(anomaly_mask.sum()),
        method="rolling_median",
    )


def run_anomaly_detection(
    data: pd.DataFrame,
    value_column: str,
    rolling_window: int = 7,
) -> dict[str, AnomalyDetectionResult]:
    """Run all supported anomaly detection methods."""
    return {
        "z_score": detect_z_score_anomalies(data, value_column),
        "iqr": detect_iqr_anomalies(data, value_column),
        "rolling_median": detect_rolling_median_anomalies(
            data,
            value_column,
            window=rolling_window,
        ),
    }
