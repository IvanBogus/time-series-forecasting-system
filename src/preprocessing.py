"""Preprocessing helpers for time series datasets."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def normalize_column_name(column_name: object) -> str:
    """Convert a column name to a stable snake_case identifier."""
    normalized = str(column_name).strip().lower()
    normalized = normalized.replace("№", "number")
    normalized = re.sub(r"[^\w]+", "_", normalized, flags=re.UNICODE)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unnamed"


def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with normalized column names."""
    result = data.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def find_date_column(data: pd.DataFrame, candidates: Iterable[str] | None = None) -> str:
    """Find the most likely date column in a DataFrame."""
    candidate_names = tuple(candidates or ("date", "дата", "data"))
    for column in data.columns:
        column_lower = str(column).lower()
        if any(candidate in column_lower for candidate in candidate_names):
            return str(column)

    raise ValueError(
        "Could not detect a date column. Pass the correct column name explicitly."
    )


def parse_date_column(data: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    """Parse a date column and remove rows with invalid dates."""
    result = data.copy()
    resolved_date_column = date_column or find_date_column(result)

    result[resolved_date_column] = pd.to_datetime(
        result[resolved_date_column],
        errors="coerce",
        dayfirst=True,
    )
    result = result.dropna(subset=[resolved_date_column])
    return result


def sort_by_date(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Sort a DataFrame by date and reset the row index."""
    return data.sort_values(date_column).reset_index(drop=True)


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using time-series-friendly defaults."""
    result = data.copy()
    numeric_columns = result.select_dtypes(include="number").columns
    non_numeric_columns = result.columns.difference(numeric_columns)

    if len(numeric_columns) > 0:
        result[numeric_columns] = result[numeric_columns].interpolate(
            method="linear",
            limit_direction="both",
        )

    if len(non_numeric_columns) > 0:
        result[non_numeric_columns] = result[non_numeric_columns].ffill().bfill()

    return result


def preprocess_time_series(
    data: pd.DataFrame,
    date_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Normalize, parse dates, sort chronologically, and fill missing values."""
    normalized = normalize_column_names(data)
    parsed = parse_date_column(normalized, date_column)
    resolved_date_column = date_column or find_date_column(parsed)
    sorted_data = sort_by_date(parsed, resolved_date_column)
    cleaned = handle_missing_values(sorted_data)
    return cleaned, resolved_date_column

