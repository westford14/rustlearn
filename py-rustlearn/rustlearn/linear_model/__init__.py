"""Exposing the linear-model package."""

from rustlearn.linear_model.linear_regression import SimpleLinearRegression
from rustlearn.linear_model.types import LinearRegressionReturn

__all__ = [
    "SimpleLinearRegression",
    # Return Objects
    "LinearRegressionReturn",
]
