"""Exposing the linear-model package."""

from rustlearn.linear_model.linear_regression import LinearRegression
from rustlearn.linear_model.types import LinearRegressionReturn

__all__ = [
    # Linear Model Classes
    "LinearRegression",
    # Return Objects
    "LinearRegressionReturn",
]
