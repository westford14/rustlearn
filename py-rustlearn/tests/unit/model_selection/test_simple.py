"""Test the SimpleTrainTestSplitter."""

import os
import json
from math import isclose
from typing import List

import pytest

from rustlearn import NamedArray
from rustlearn.model_selection import SimpleTrainTestSplit, TrainTestSplitReturn
from rustlearn.exceptions import ValidationError


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
    SimpleTrainTestSplit(x=x_2d, y=y_full, train_proportion=0.5)


def test_raises_mismatch_error(x_2d, y_missing) -> None:
    with pytest.raises(ValidationError):
        SimpleTrainTestSplit(x=x_2d, y=y_missing, train_proportion=0.5)


def test_train_proportion_too_high(x_2d, y_full) -> None:
    with pytest.raises(ValidationError):
        SimpleTrainTestSplit(x=x_2d, y=y_full, train_proportion=1.1)


def test_split(x_2d, y_full) -> None:
    splitter = SimpleTrainTestSplit(x=x_2d, y=y_full, train_proportion=0.5)
    res = splitter.split()
    expected = TrainTestSplitReturn(
        x_train=[
            NamedArray("age", [0.038076, -0.001882]),
            NamedArray("bmi", [0.061696, -0.051474]),
        ],
        x_test=[
            NamedArray("age", [0.085299, -0.089063, 0.005383]),
            NamedArray("bmi", [0.044451, -0.011595, -0.036385]),
        ],
        y_train=NamedArray("target", [151.0, 75.0]),
        y_test=NamedArray("target", [141.0, 206.0, 135.0]),
    )

    assert len(res.x_train) == len(expected.x_train)
    assert len(res.x_train[0].data) == len(expected.x_train[0].data)
    assert len(res.x_test) == len(expected.x_test)
    assert len(res.x_test[0].data) == len(expected.x_test[0].data)
    assert len(res.y_train.data) == len(expected.y_train.data)
    assert len(res.y_test.data) == len(expected.y_test.data)
    for i, x in enumerate(res.x_train):
        for j, t in enumerate(x.data):
            assert isclose(t, expected.x_train[i].data[j], rel_tol=0.001)

    for i, x in enumerate(res.x_test):
        for j, t in enumerate(x.data):
            assert isclose(t, expected.x_test[i].data[j], rel_tol=0.001)

    for j, t in enumerate(res.y_train.data):
        assert isclose(t, expected.y_train.data[j], rel_tol=0.001)
    for j, t in enumerate(res.y_test.data):
        assert isclose(t, expected.y_test.data[j], rel_tol=0.001)
