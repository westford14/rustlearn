use pyo3::prelude::*;
use std::collections::HashMap;

use rustlearn_linear_model::types::LinearRegressionReturn;

pub mod linear_regression;

#[derive(Debug, PartialEq, FromPyObject)]
#[pyclass]
pub struct PyLinearRegressionReturn {
    pub coefficients: HashMap<String, f64>,
}

impl From<LinearRegressionReturn> for PyLinearRegressionReturn {
    fn from(linear_regression_return: LinearRegressionReturn) -> Self {
        let mut temp: HashMap<String, f64> = HashMap::new();
        temp.insert("intercept".to_string(), linear_regression_return.intercept);
        for (k, v) in linear_regression_return.beta_values.iter() {
            temp.insert(k.to_string(), v.to_owned());
        }
        PyLinearRegressionReturn { coefficients: temp }
    }
}

#[pymethods]
impl PyLinearRegressionReturn {
    #[new]
    pub fn __init__(coefficients: HashMap<String, f64>) -> PyResult<Self> {
        Ok(PyLinearRegressionReturn { coefficients })
    }

    pub fn _intercept(&self) -> PyResult<f64> {
        Ok(self.coefficients.get("intercept").unwrap().to_owned())
    }

    pub fn _beta_values(&self) -> PyResult<HashMap<String, f64>> {
        let mut betas: HashMap<String, f64> = HashMap::new();
        for (k, v) in self.coefficients.iter() {
            if *k != "intercept" {
                betas.insert(k.to_string(), v.to_owned());
            }
        }
        Ok(betas)
    }
}
