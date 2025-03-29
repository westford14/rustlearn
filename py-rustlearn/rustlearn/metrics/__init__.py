"""Expose the metrics API."""

from rustlearn.metrics.mean_absolute_error import mean_absolute_error
from rustlearn.metrics.mean_squared_error import (
    mean_squared_error,
    root_mean_squared_error,
)


__all__ = [
    # Metrics
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
]
