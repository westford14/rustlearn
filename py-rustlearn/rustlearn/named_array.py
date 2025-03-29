from typing import Self

from rustylearn import PyNamedArray


class NamedArray:
    """
    A NamedArray represents a vector object that has a name associated with.

    params:
        name: (str) the name to give to the NamedArray
        data: (list) the values of the data
    returns:
        (None)
    """

    _n: PyNamedArray = None

    def __init__(self, name: str, data: list) -> None:
        self.name = name
        self.data = data
        self._n = PyNamedArray(self.name, self.data)

    @classmethod
    def _from_py_named_array(cls, py_named_array: PyNamedArray) -> Self:
        named_array = cls.__new__(cls)
        named_array._named_array = py_named_array
        return named_array

    def is_empty(self) -> bool:
        return self._n.is_empty()

    def len(self) -> int:
        return self._n.len()

    def mean(self) -> float:
        return self._n.mean()

    def dot(self, other: Self) -> float:
        other_n: PyNamedArray = PyNamedArray(name=other.name, data=other.data)
        return self._n.dot(other_n)
