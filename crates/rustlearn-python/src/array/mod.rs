use pyo3::prelude::*;
use rustlearn_array::namedarray::NamedArray;

use crate::errors::PyRustLearnError;

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNamedArray {
    pub named_array: NamedArray<f64>,
}

impl From<NamedArray<f64>> for PyNamedArray {
    fn from(named_array: NamedArray<f64>) -> Self {
        PyNamedArray { named_array }
    }
}

#[pymethods]
impl PyNamedArray {
    #[new]
    pub fn __init__(name: String, data: Vec<f64>) -> PyResult<Self> {
        let named_array = NamedArray { name, data };
        Ok(PyNamedArray { named_array })
    }

    pub fn is_empty(&self) -> bool {
        self.named_array.is_empty()
    }

    pub fn len(&self) -> usize {
        self.named_array.len()
    }

    pub fn mean(&self) -> PyResult<f64> {
        let mean = self.named_array.mean();
        match mean {
            Ok(mean) => Ok(mean),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    pub fn dot(&self, other: PyNamedArray) -> PyResult<f64> {
        let dot_product = self.named_array.dot(other.named_array);
        match dot_product {
            Ok(dot_product) => Ok(dot_product),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }
}
