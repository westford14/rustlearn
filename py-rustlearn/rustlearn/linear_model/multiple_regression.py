"""Multiple Linear Regression fitter."""

from typing import List, Self

from rustlearn import NamedArray
from rustlearn.linear_model.types import LinearRegressionReturn
from rustylearn import PyMultipleLinearRegression


class MultipleLinearRegression:
    """Simple multiple linear regression class."""

    _n: PyMultipleLinearRegression = None

    def __init__(self, x: List[NamedArray], y: NamedArray) -> None:
        """Instantiate the class.

        :params:
            x: List[NamedArray]
            y: NamedArray
        """
        self.x = [nm._n for nm in x]
        self.y = y._n
        self._n = PyMultipleLinearRegression(x=self.x, y=self.y)
        self.assert_equal_length(x, y)

    @classmethod
    def _from_py_multiple_linear_regression(
        cls, py_multiple_linear_regression: PyMultipleLinearRegression
    ) -> Self:
        """Convert a PyMultipleLinearRegression to MultipleLinearRegression.

        :params:
            py_multiple_linear_regression: (PyMultipleLinearRegression)
        :return:
            MultipleLinearRegression
        """
        multiple_lin_reg = cls.__new__(cls)
        multiple_lin_reg._n = py_multiple_linear_regression
        return multiple_lin_reg

    def assert_equal_length(self, x: List[NamedArray], y: NamedArray) -> None:
        """Assert two NamedArrays are equal in length.

        :params:
            x: (List[NamedArray])
            y: (NamedArray)
        :return:
            None
        """
        self._n.assert_equal_length([nm._n for nm in x], y._n)

    def fit(self) -> LinearRegressionReturn:
        """Fit the LinearRegression.

        :return: LinearRegressionReturn
        """
        result = self._n.fit()
        return LinearRegressionReturn(result.intercept(), result.beta_values())
