"""NamedArray definitions."""

from typing import Self

from rustylearn import PyNamedArray


class NamedArray:
    """A NamedArray represents a vector object that has a name associated with."""

    _n: PyNamedArray = None

    def __init__(self, name: str, data: list) -> None:
        """Instantiate the class.

        params:
            name: (str) the name to give to the NamedArray
            data: (list) the values of the data
        returns:
            (None)
        """
        self.name = name
        self.data = data
        self._n = PyNamedArray(self.name, self.data)

    @classmethod
    def _from_py_named_array(cls, py_named_array: PyNamedArray) -> Self:
        """Convert a PyNamedArray to NamedArray.

        :params:
            py_named_array: (PyNamedArray)
        :return:
            NamedArray
        """
        named_array = cls.__new__(cls)
        named_array._n = py_named_array
        named_array.name = named_array._n.name()
        named_array.data = named_array._n.data()
        return named_array

    def is_empty(self) -> bool:
        """Check if the NamedArray is empty.

        :return: bool
        """
        return self._n.is_empty()

    def len(self) -> int:
        """Get the length of the NamedArray.

        :return: int
        """
        return self._n.len()

    def mean(self) -> float:
        """Get the mean of the NamedArray.

        :return: float
        """
        return self._n.mean()

    def dot(self, other: Self) -> float:
        """Calculate the dot product between two NamedArrays.

        :params:
            other: (NamedArray)
        :returns:
            float
        """
        other_n: PyNamedArray = PyNamedArray(name=other.name, data=other.data)
        return self._n.dot(other_n)
