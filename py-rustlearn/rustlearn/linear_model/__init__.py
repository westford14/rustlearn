"""Exposing the linear-model package."""

from rustlearn.linear_model.linear_regression import SimpleLinearRegression
from rustlearn.linear_model.multiple_regression import MultipleLinearRegression
from rustlearn.linear_model.types import LinearRegressionReturn

__all__ = [
    # Linear Model Classes
    "SimpleLinearRegression",
    "MultipleLinearRegression",
    # Return Objects
    "LinearRegressionReturn",
]
