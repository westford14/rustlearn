"""Test the SimpleLinearRegression."""

import os
import json
from math import isclose

import pytest

from rustlearn import NamedArray
from rustlearn.linear_model import SimpleLinearRegression, LinearRegressionReturn
from rustlearn.exceptions import ValidationError


CURR_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURE_PATH = os.path.join(CURR_PATH, "fixtures")


@pytest.fixture
def x_simple() -> NamedArray:
    return NamedArray("x", [1, 2, 3, 4])


@pytest.fixture
def y_simple() -> NamedArray:
    return NamedArray("y", [10, 20, 30, 40])


def test_instantiation(x_simple, y_simple) -> None:
    SimpleLinearRegression(x=x_simple, y=y_simple)


def test_raises_mismatch_error(x_simple) -> None:
    with pytest.raises(ValidationError):
        SimpleLinearRegression(x=x_simple, y=NamedArray("y", [10, 20, 30]))


def test_fit_simple(x_simple, y_simple) -> None:
    lin_reg = SimpleLinearRegression(x=x_simple, y=y_simple)
    res = lin_reg.fit()
    expected = LinearRegressionReturn(intercept=0, beta_values={"x": 10})
    assert isclose(res.intercept, expected.intercept)
    assert res.beta_values == expected.beta_values


def test_fit_complex() -> None:
    with open(os.path.join(FIXTURE_PATH, "simple.json"), "r") as f:
        data = json.load(f)

    x = NamedArray("x", data["x"])
    y = NamedArray("y", data["y"])

    expected = LinearRegressionReturn(
        intercept=152.13348416289617, beta_values={"x": 949.43526038}
    )
    lin_reg = SimpleLinearRegression(x, y)
    res = lin_reg.fit()

    assert isclose(res.intercept, expected.intercept)
    assert isclose(res.beta_values["x"], expected.beta_values["x"])
