use pyo3::prelude::*;
use rustlearn_array::namedarray::NamedArray;
use rustlearn_linear_model::ols::multiple_regression::MultipleLinearRegression;

use crate::array::PyNamedArray;
use crate::errors::PyRustLearnError;

use super::PyLinearRegressionReturn;

#[pyclass]
#[derive(Clone)]
pub struct PyMultipleLinearRegression {
    pub x: Vec<PyNamedArray>,
    pub y: PyNamedArray,
}

impl From<MultipleLinearRegression<f64>> for PyMultipleLinearRegression {
    fn from(multiple_linear_regression: MultipleLinearRegression<f64>) -> Self {
        let mut x: Vec<PyNamedArray> = Vec::new();
        for nm in multiple_linear_regression.x {
            x.push(PyNamedArray { named_array: nm });
        }
        PyMultipleLinearRegression {
            x,
            y: PyNamedArray {
                named_array: multiple_linear_regression.y,
            },
        }
    }
}

#[pymethods]
impl PyMultipleLinearRegression {
    #[new]
    pub fn __init__(x: Vec<PyNamedArray>, y: PyNamedArray) -> PyResult<Self> {
        let multiple_linear = PyMultipleLinearRegression {
            x: x.clone(),
            y: y.clone(),
        };
        let _ = Self::assert_equal_length(x.clone(), y.clone());

        Ok(multiple_linear)
    }

    #[staticmethod]
    pub fn assert_equal_length(x: Vec<PyNamedArray>, y: PyNamedArray) -> PyResult<()> {
        let mut n_x: Vec<NamedArray<f64>> = Vec::new();
        for t in x {
            n_x.push(t.named_array);
        }
        let asserted = MultipleLinearRegression::assert_equal_length(n_x, y.named_array);
        match asserted {
            Ok(_asserted) => return Ok(()),
            Err(e) => return Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    pub fn fit(&self) -> PyResult<PyLinearRegressionReturn> {
        let mut x_clone: Vec<NamedArray<f64>> = Vec::new();
        for v in self.x.clone().iter() {
            x_clone.push(v.named_array.clone())
        }
        let simple = MultipleLinearRegression {
            x: x_clone.clone(),
            y: self.y.named_array.clone(),
        };
        let res = simple.fit().unwrap();
        Ok(PyLinearRegressionReturn {
            intercept: res.intercept,
            beta_values: res.beta_values.clone(),
        })
    }
}
