"""Simple return type for TrainTestSplit."""

from typing import List

from rustlearn import NamedArray
from rustylearn import PyTrainTestSplitReturn


class TrainTestSplitReturn:
    """Object that holds the return from a TrainTestSplit."""

    _n: PyTrainTestSplitReturn = None

    def __init__(
        self,
        x_train: List[NamedArray],
        x_test: List[NamedArray],
        y_train: NamedArray,
        y_test: NamedArray,
    ) -> None:
        """Instantiate the class.

        :params:
            x_train: List[NamedArray]
            x_test: List[NamedArray]
            y_train: List[NamedArray]
            y_train: List[NamedArray]
        :return:
            None
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self._n = PyTrainTestSplitReturn(
            x_train=[nm._n for nm in x_train],
            x_test=[nm._n for nm in x_test],
            y_train=self.y_train._n,
            y_test=self.y_test._n,
        )
