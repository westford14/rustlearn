"""Linear Regression."""

from typing import Self

from rustlearn.linear_model.types import LinearRegressionReturn
from rustlearn.named_array import NamedArray
from rustylearn import PySimpleLinearRegression


class SimpleLinearRegression:
    """Simple linear regression class."""

    _n: PySimpleLinearRegression = None

    def __init__(self, x: NamedArray, y: NamedArray) -> None:
        """Instantiate the class.

        :params:
            x: NamedArray - the independent variable
            y: NamedArray - the depdendent variable
        :returns:
            None
        """
        self.x = x._n
        self.y = y._n
        self._n = PySimpleLinearRegression(x=self.x, y=self.y)
        self.assert_equal_length(self.x, self.y)

    @classmethod
    def _from_py_simple_linear_regression(
        cls, py_simple_linear_regression: PySimpleLinearRegression
    ) -> Self:
        """Convert a PySimpleLinearRegression to SimpleLinearRegression.

        :params:
            py_simple_linear_regression: (PySimpleLinearRegression)
        :return:
            SimpleLinearRegression
        """
        named_array = cls.__new__(cls)
        named_array._n = py_simple_linear_regression
        return named_array

    def assert_equal_length(self, x: NamedArray, y: NamedArray) -> None:
        """Assert two NamedArrays are equal in length.

        :params:
            x: (NamedArray)
            y: (NamedArray)
        :return:
            None
        """
        self._n.assert_equal_length(x, y)

    def fit(self) -> LinearRegressionReturn:
        """Fit the LinearRegression.

        :return: LinearRegressionReturn
        """
        result = self._n.fit()
        return LinearRegressionReturn(result.intercept(), result.beta_values())
