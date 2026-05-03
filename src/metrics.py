"""Evaluation metrics."""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """Mean squared error between two arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))
