use pyo3::prelude::*;
use rustlearn_array::namedarray::NamedArray;

use crate::errors::PyRustLearnError;

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
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

    pub fn name(&self) -> String {
        self.named_array.name.clone()
    }

    pub fn data(&self) -> Vec<f64> {
        self.named_array.data.clone()
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

#[cfg(test)]
mod array_tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn named_array_fixture() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "y".to_string(),
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        }
    }

    #[rstest]
    fn test_instantiation(named_array_fixture: PyNamedArray) {
        drop(named_array_fixture);
    }

    #[rstest]
    fn test_is_empty(named_array_fixture: PyNamedArray) {
        assert!(!named_array_fixture.is_empty());
    }

    #[rstest]
    fn test_len(named_array_fixture: PyNamedArray) {
        assert_eq!(named_array_fixture.len(), 4)
    }

    #[rstest]
    fn test_mean(named_array_fixture: PyNamedArray) {
        let val = named_array_fixture.mean().unwrap();
        assert_eq!(val, 2.5)
    }

    #[rstest]
    fn test_dot_product(named_array_fixture: PyNamedArray) {
        let ans = named_array_fixture
            .dot(named_array_fixture.clone())
            .unwrap();
        assert_eq!(ans, 30.0)
    }
}
