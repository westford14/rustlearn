"""R2 calculation from two NamedArrays."""

from rustlearn import NamedArray
from rustylearn import py_r_2


def r_2(y_true: NamedArray, y_pred: NamedArray) -> float:
    """
    R2 calculation.

    Simple method for calculating the r2 value between two
    NamedArrays. These arrays must be the same length.

    :param:
        y_true: (NamedArray) the true values
        y_pred: (NamedArray) the predicted values
    :return:
        (float)
    """
    return py_r_2(y_true, y_pred)
