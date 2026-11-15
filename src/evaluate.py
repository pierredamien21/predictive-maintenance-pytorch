"""Evaluation metrics and plotting helpers."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics for RUL predictions."""
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
    }
