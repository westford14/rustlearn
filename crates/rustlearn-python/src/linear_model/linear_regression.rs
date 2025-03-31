use pyo3::prelude::*;
use rustlearn_linear_model::ols::linear_regression::SimpleLinearRegression;

use crate::array::PyNamedArray;
use crate::errors::PyRustLearnError;

use super::PyLinearRegressionReturn;

#[pyclass]
#[derive(Clone)]
pub struct PySimpleLinearRegression {
    pub x: PyNamedArray,
    pub y: PyNamedArray,
}

impl From<SimpleLinearRegression<f64>> for PySimpleLinearRegression {
    fn from(simple_linear_regression: SimpleLinearRegression<f64>) -> Self {
        PySimpleLinearRegression {
            x: PyNamedArray {
                named_array: simple_linear_regression.x,
            },
            y: PyNamedArray {
                named_array: simple_linear_regression.y,
            },
        }
    }
}

#[pymethods]
impl PySimpleLinearRegression {
    #[new]
    pub fn __init__(x: PyNamedArray, y: PyNamedArray) -> PyResult<Self> {
        let simple_linear = PySimpleLinearRegression {
            x: x.clone(),
            y: y.clone(),
        };
        let _ = Self::assert_equal_length(x.clone(), y.clone());

        Ok(simple_linear)
    }

    #[staticmethod]
    pub fn assert_equal_length(x: PyNamedArray, y: PyNamedArray) -> PyResult<()> {
        let asserted = SimpleLinearRegression::assert_equal_length(x.named_array, y.named_array);
        match asserted {
            Ok(_asserted) => Ok(()),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    pub fn fit(&self) -> PyResult<PyLinearRegressionReturn> {
        let simple = SimpleLinearRegression {
            x: self.x.named_array.clone(),
            y: self.y.named_array.clone(),
        };
        let res = simple.fit().unwrap();
        Ok(PyLinearRegressionReturn {
            intercept: res.intercept,
            beta_values: res.beta_values.clone(),
        })
    }
}
