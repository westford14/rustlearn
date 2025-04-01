"""Test the MultipleLinearRegression."""

import os
from math import isclose
from typing import List

import pytest

from rustlearn import NamedArray
from rustlearn.linear_model import MultipleLinearRegression, LinearRegressionReturn
from rustlearn.exceptions import ValidationError


CURR_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURE_PATH = os.path.join(CURR_PATH, "fixtures")


@pytest.fixture
def x_2d() -> List[NamedArray]:
    return [
        NamedArray("age", [0.038076, -0.001882, 0.085299, -0.089063, 0.005383]),
        NamedArray("bmi", [0.061696, -0.051474, 0.044451, -0.011595, -0.036385]),
    ]


@pytest.fixture
def y_full() -> NamedArray:
    return NamedArray("target", [151.0, 75.0, 141.0, 206.0, 135.0])


@pytest.fixture
def y_missing() -> NamedArray:
    return NamedArray("target", [151.0, 75.0, 141.0, 206.0])


def test_instantiation(x_2d, y_full) -> None:
    MultipleLinearRegression(x=x_2d, y=y_full)


def test_raises_mismatch_error(x_2d, y_missing) -> None:
    with pytest.raises(ValidationError):
        MultipleLinearRegression(x=x_2d, y=y_missing)


def test_fit(x_2d, y_full) -> None:
    lin_reg = MultipleLinearRegression(x=x_2d, y=y_full)
    res = lin_reg.fit()
    expected = LinearRegressionReturn(
        intercept=145.653177, beta_values={"age": -684.31017707, "bmi": 838.08945541}
    )
    assert isclose(res.intercept, expected.intercept, rel_tol=0.001)
    for k, v in res.beta_values.items():
        assert isclose(expected.beta_values[k], v, rel_tol=0.001)
