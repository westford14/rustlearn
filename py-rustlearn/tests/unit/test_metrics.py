"""Test the metrics."""

from math import isclose

import pytest

from rustlearn import NamedArray
from rustlearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from rustlearn.exceptions import ValidationError


@pytest.fixture
def y_true() -> NamedArray:
    """Fixture for the y_true NamedArray."""
    return NamedArray("y_true", [1, 2, 3, 4])


def test_mean_absolute_error_same(y_true) -> None:
    """Simple test of two equivalent NamedArrays."""
    assert mean_absolute_error(y_true, y_true) == 0


def test_mean_absolute_error(y_true) -> None:
    """Simple test of two mean_absolute_errors."""
    y_pred = NamedArray("y_pred", [5, 10, 15, 20])
    assert isclose(mean_absolute_error(y_true, y_pred), 10)


def test_mean_absolute_error_raises(y_true) -> None:
    """Simple test for raising the AttributeError."""
    test = NamedArray("y_pred", [1, 2, 3])
    with pytest.raises(ValidationError):
        mean_absolute_error(y_true, test)


def test_mean_squared_error_same(y_true) -> None:
    """Simple test of two equivalent NamedArrays."""
    assert mean_squared_error(y_true, y_true) == 0


def test_mean_squared_error(y_true) -> None:
    """Simple test of mean_squared_error."""
    y_pred = NamedArray("y_pred", [5, 10, 15, 20])
    assert isclose(mean_squared_error(y_true, y_pred), 120)


def test_mean_squared_error_raises(y_true) -> None:
    """Simple test for raising the AttributeError."""
    test = NamedArray("y_pred", [1, 2, 3])
    with pytest.raises(ValidationError):
        mean_squared_error(y_true, test)


def test_root_mean_squared_error_same(y_true) -> None:
    """Simple test of two equivalent NamedArrays."""
    assert root_mean_squared_error(y_true, y_true) == 0


def test_root_mean_squared_error(y_true) -> None:
    """Simple test of root_mean_squared_error."""
    y_pred = NamedArray("y_pred", [5, 10, 15, 20])
    assert isclose(root_mean_squared_error(y_true, y_pred), 10.954451150103322)


def test_root_mean_squared_error_raises(y_true) -> None:
    """Simple test for raising the AttributeError."""
    test = NamedArray("y_pred", [1, 2, 3])
    with pytest.raises(ValidationError):
        root_mean_squared_error(y_true, test)
