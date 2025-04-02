use crate::array::PyNamedArray;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PySimpleTrainTestSplit {
    pub x: Vec<PyNamedArray>,
    pub y: PyNamedArray,
}
