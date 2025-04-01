"""Linear Regression."""

from typing import Self

from rustlearn.linear_model.types import LinearRegressionReturn
from rustlearn.named_array import NamedArray
from rustylearn import PyLinearRegression


class LinearRegression:
    """Linear regression class."""

    _n: PyLinearRegression = None

    def __init__(self, x: NamedArray, y: NamedArray) -> None:
        """Instantiate the class.

        :params:
            x: NamedArray - the independent variable
            y: NamedArray - the depdendent variable
        :returns:
            None
        """
        self.x = [nm._n for nm in x]
        self.y = y._n
        self._n = PyLinearRegression(x=self.x, y=self.y)
        self.assert_equal_length(x, y)

    @classmethod
    def _from_py_linear_regression(
        cls, py_linear_regression: PyLinearRegression
    ) -> Self:
        """Convert a PyLinearRegression to LinearRegression.

        :params:
            py_linear_regression: (PyLinearRegression)
        :return:
            LinearRegression
        """
        lin_reg = cls.__new__(cls)
        lin_reg._n = py_linear_regression
        return lin_reg

    def assert_equal_length(self, x: NamedArray, y: NamedArray) -> None:
        """Assert two NamedArrays are equal in length.

        :params:
            x: (NamedArray)
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
