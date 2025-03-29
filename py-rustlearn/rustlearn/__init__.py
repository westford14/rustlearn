"""Expose specific APIs."""

from rustlearn.named_array import NamedArray
from rustlearn.metrics.mean_absolute_error import mean_absolute_error
from rustlearn.metrics.mean_squared_error import (
    mean_squared_error,
    root_mean_squared_error
)
from rustlearn.exceptions import (
    RustLearnError
)

__all__ = [
    # Classes
    "NamedArray",
]
