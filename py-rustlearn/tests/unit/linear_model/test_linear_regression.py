"""Test the SimpleLinearRegression."""

import os
import json
from math import isclose
from typing import List

import pytest

from rustlearn import NamedArray
from rustlearn.linear_model import LinearRegression, LinearRegressionReturn
from rustlearn.exceptions import ValidationError


CURR_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURE_PATH = os.path.join(CURR_PATH, "fixtures")


@pytest.fixture
def x_simple() -> NamedArray:
    return [NamedArray("x", [1, 2, 3, 4])]


@pytest.fixture
def y_simple() -> NamedArray:
    return NamedArray("y", [10, 20, 30, 40])


def test_instantiation(x_simple, y_simple) -> None:
    LinearRegression(x=x_simple, y=y_simple)


def test_x_empty() -> None:
    with pytest.raises(ValidationError):
        LinearRegression(x=[NamedArray("x", [])], y=NamedArray("y", [0.0]))


def test_y_empty() -> None:
    with pytest.raises(RuntimeError):
        LinearRegression(x=[NamedArray("x", [0.0])], y=NamedArray("y", []))


def test_raises_mismatch_error(x_simple) -> None:
    with pytest.raises(ValidationError):
        LinearRegression(x=x_simple, y=NamedArray("y", [10, 20, 30]))


def test_fit_simple(x_simple, y_simple) -> None:
    lin_reg = LinearRegression(x=x_simple, y=y_simple)
    res = lin_reg.fit()
    expected = LinearRegressionReturn(intercept=0, beta_values={"x": 10})
    assert isclose(res.intercept, expected.intercept)
    assert res.beta_values == expected.beta_values


def test_fit_complex() -> None:
    with open(os.path.join(FIXTURE_PATH, "simple.json"), "r") as f:
        data = json.load(f)

    x = [NamedArray("x", data["x"])]
    y = NamedArray("y", data["y"])

    expected = LinearRegressionReturn(
        intercept=152.13348416289617, beta_values={"x": 949.43526038}
    )
    lin_reg = LinearRegression(x, y)
    res = lin_reg.fit()

    assert isclose(res.intercept, expected.intercept)
    assert isclose(res.beta_values["x"], expected.beta_values["x"])


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


@pytest.fixture
def two_d_prediction() -> NamedArray:
    return NamedArray(
        "predictions",
        [
            145.653177,
            299.43245535,
            453.2117337,
            606.99101204,
            760.77029039,
        ],
    )


def test_instantiation(x_2d, y_full) -> None:
    LinearRegression(x=x_2d, y=y_full)


def test_raises_mismatch_error(x_2d, y_missing) -> None:
    with pytest.raises(ValidationError):
        LinearRegression(x=x_2d, y=y_missing)


def test_fit(x_2d, y_full) -> None:
    lin_reg = LinearRegression(x=x_2d, y=y_full)
    res = lin_reg.fit()
    expected = LinearRegressionReturn(
        intercept=145.653177, beta_values={"age": -684.31017707, "bmi": 838.08945541}
    )
    assert isclose(res.intercept, expected.intercept, rel_tol=0.001)
    for k, v in res.beta_values.items():
        assert isclose(expected.beta_values[k], v, rel_tol=0.001)


def test_predict(x_2d, y_full, two_d_prediction) -> None:
    lin_reg = LinearRegression(x=x_2d, y=y_full)
    res = lin_reg.fit()
    new_x = [
        NamedArray("age", [0.0, 1.0, 2.0, 3.0, 4.0]),
        NamedArray("bmi", [0.0, 1.0, 2.0, 3.0, 4.0]),
    ]
    pred = lin_reg.predict(new_x, res)

    for i, val in enumerate(pred.data):
        isclose(val, two_d_prediction.data[i], rel_tol=0.001)
