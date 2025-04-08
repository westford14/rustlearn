"""Simple return type for LinearRegression."""

from typing import Dict

from rustylearn import PyLinearRegressionReturn


class LinearRegressionReturn:
    """Object that holds the return from a LinearRegression."""

    _n: PyLinearRegressionReturn = None

    def __init__(self, intercept: float, beta_values: Dict[str, float]) -> None:
        """Instantiate the class.

        :params:
            intercept: (float)
            beta_values: (Dict[str, float])
        :return:
            None
        """
        self.intercept = intercept
        self.beta_values = beta_values
        self.coefficients = self.beta_values.copy()
        self.coefficients["intercept"] = self.intercept
        self._n = PyLinearRegressionReturn(self.coefficients)
