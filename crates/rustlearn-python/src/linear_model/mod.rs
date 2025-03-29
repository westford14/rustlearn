use pyo3::prelude::*;
use std::collections::HashMap;

use rustlearn_linear_model::types::LinearRegressionReturn;

pub mod linear_regression;

#[pyclass]
pub struct PyLinearRegressionReturn {
    pub intercept: f64,
    pub beta_values: HashMap<String, f64>,
}

impl From<LinearRegressionReturn> for PyLinearRegressionReturn {
    fn from(linear_regression_return: LinearRegressionReturn) -> Self {
        PyLinearRegressionReturn {
            intercept: linear_regression_return.intercept,
            beta_values: linear_regression_return.beta_values,
        }
    }
}

#[pymethods]
impl PyLinearRegressionReturn {
    #[new]
    pub fn __init__(intercept: f64, beta_values: HashMap<String, f64>) -> PyResult<Self> {
        let ret = PyLinearRegressionReturn {
            intercept,
            beta_values,
        };

        Ok(ret)
    }

    pub fn intercept(&self) -> PyResult<f64> {
        Ok(self.intercept)
    }

    pub fn beta_values(&self) -> PyResult<HashMap<String, f64>> {
        Ok(self.beta_values.clone())
    }
}
