"""Time-aware splitting and evaluation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET = "next_hour_requests"


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the latest observations into the test set."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = data.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def regression_metrics(y_true: pd.Series, y_pred: object) -> dict[str, float]:
    """Return metrics with stable names for reports and CI checks."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }
