"""Basic descriptive statistics for numerical time series."""

from typing import Any

import pandas as pd


def calculate_basic_statistics(series: pd.Series) -> dict[str, Any]:
    """Calculate mean, variance, std, min, and max for a numeric series."""
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        raise ValueError("Cannot calculate statistics for an empty numeric series.")

    return {
        "count": int(numeric_series.count()),
        "mean": float(numeric_series.mean()),
        "variance": float(numeric_series.var(ddof=0)),
        "std": float(numeric_series.std(ddof=0)),
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max()),
    }

