"""Simple train test splitter."""

from typing import List, Self

from rustlearn import NamedArray
from rustlearn.model_selection.types import TrainTestSplitReturn
from rustylearn import PySimpleTrainTestSplit


class SimpleTrainTestSplit:
    """Class for a naive train test splitter."""

    _n: PySimpleTrainTestSplit = None

    def __init__(
        self, x: List[NamedArray], y: NamedArray, train_proportion: float
    ) -> None:
        """Initialize the class.

        :params:
            x: (List[NamedArray])
            y: (NamedArray)
            train_proportion: float
        :return:
            None
        """
        self.x = [nm._n for nm in x]
        self.y = y._n
        self.train_proportion = train_proportion
        self._n = PySimpleTrainTestSplit(
            x=self.x, y=self.y, train_proportion=self.train_proportion
        )
        self.assert_equal_length(x, y)
        self.assert_logical_train_proportion(train_proportion)

    @classmethod
    def _from_py_simple_train_test_split(
        cls, py_train_test_split: PySimpleTrainTestSplit
    ) -> Self:
        """Convert a PySimpleTrainTestSplit to LinearRegression.

        :params:
            py_train_test_split: (SimpleTrainTestSplit)
        :return:
            LinearRegression
        """
        splitter = cls.__new__(cls)
        splitter._n = py_train_test_split
        return splitter

    def assert_equal_length(self, x: NamedArray, y: NamedArray) -> None:
        """Assert two NamedArrays are equal in length.

        :params:
            x: (NamedArray)
            y: (NamedArray)
        :return:
            None
        """
        self._n.assert_equal_length([nm._n for nm in x], y._n)

    def assert_logical_train_proportion(self, train_proportion: float) -> None:
        """Assert the train proportion makes sense.

        :params:
            train_proportion: float
        :return:
            None
        """
        self._n.assert_logical_train_proportion(train_proportion)

    def split(self) -> TrainTestSplitReturn:
        """Split the SimpleTrainTestSplit.

        :return: TrainTestSplitReturn
        """
        result = self._n.split()
        return TrainTestSplitReturn(
            x_train=[NamedArray._from_py_named_array(x) for x in result.x_train()],
            x_test=[NamedArray._from_py_named_array(x) for x in result.x_test()],
            y_train=NamedArray._from_py_named_array(result.y_train()),
            y_test=NamedArray._from_py_named_array(result.y_test()),
        )
