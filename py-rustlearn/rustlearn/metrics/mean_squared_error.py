"""Mean squared error and root mean squared error."""

from rustlearn import NamedArray
from rustylearn import py_mean_squared_error, py_root_mean_squared_error


def mean_squared_error(y_true: NamedArray, y_pred: NamedArray) -> float:
    """
    Mean squared error calculation.

    Simple method for calculating the mean squared error between two
    NamedArrays. These arrays must be the same length.

    :param:
        y_true: (NamedArray) the true values
        y_pred: (NamedArray) the predicted values
    :return:
        (float)
    """
    return py_mean_squared_error(y_true, y_pred)


def root_mean_squared_error(y_true: NamedArray, y_pred: NamedArray) -> float:
    """
    Root mean squared error calculation.

    Simple method for calculating the root mean squared error between two
    NamedArrays. These arrays must be the same length.

    :param:
        y_true: (NamedArray) the true values
        y_pred: (NamedArray) the predicted values
    :return:
        (float)
    """
    return py_root_mean_squared_error(y_true, y_pred)
