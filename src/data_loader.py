"""Data loading utilities for the Oschadbank USD time series."""

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def load_excel(file_path: PathLike, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load an Excel dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path:
        Path to the Excel file.
    sheet_name:
        Sheet name or index passed to ``pandas.read_excel``.

    Returns
    -------
    pd.DataFrame
        Raw dataset as loaded from Excel.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_excel(path, sheet_name=sheet_name)

