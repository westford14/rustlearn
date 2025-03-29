"""Mean absolute error metric."""

from rustlearn import NamedArray
from rustylearn import py_mean_absolute_error


def mean_absolute_error(y_true: NamedArray, y_pred: NamedArray) -> float:
    """
    Mean absolute error calculation.

    Simple method for calculating the mean absolute error between two
    NamedArrays. These arrays must be the same length.

    :param:
        y_true: (NamedArray) the true values
        y_pred: (NamedArray) the predicted values
    :return:
        (float)
    """
    return py_mean_absolute_error(y_true, y_pred)
