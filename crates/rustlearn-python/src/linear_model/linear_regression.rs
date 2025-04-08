use pyo3::prelude::*;
use rustlearn_array::namedarray::NamedArray;
use rustlearn_linear_model::ols::linear_regression::LinearRegression;
use rustlearn_linear_model::types::LinearRegressionReturn;

use crate::array::PyNamedArray;
use crate::errors::PyRustLearnError;

use super::PyLinearRegressionReturn;

#[pyclass]
#[derive(Clone)]
pub struct PyLinearRegression {
    pub x: Vec<PyNamedArray>,
    pub y: PyNamedArray,
}

impl From<LinearRegression<f64>> for PyLinearRegression {
    fn from(multiple_linear_regression: LinearRegression<f64>) -> Self {
        let mut x: Vec<PyNamedArray> = Vec::new();
        for nm in multiple_linear_regression.x {
            x.push(PyNamedArray { named_array: nm });
        }
        PyLinearRegression {
            x,
            y: PyNamedArray {
                named_array: multiple_linear_regression.y,
            },
        }
    }
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    pub fn __init__(x: Vec<PyNamedArray>, y: PyNamedArray) -> PyResult<Self> {
        if x.is_empty() {
            return Err(PyErr::from(PyRustLearnError::Other(
                "x is empty".to_string(),
            )));
        }
        if y.is_empty() {
            return Err(PyErr::from(PyRustLearnError::Other(
                "y is empty".to_string(),
            )));
        }
        let multiple_linear = PyLinearRegression {
            x: x.clone(),
            y: y.clone(),
        };
        Self::assert_equal_length(x.clone(), y.clone())?;

        Ok(multiple_linear)
    }

    #[staticmethod]
    pub fn assert_equal_length(x: Vec<PyNamedArray>, y: PyNamedArray) -> PyResult<()> {
        let mut n_x: Vec<NamedArray<f64>> = Vec::new();
        for t in x {
            n_x.push(t.named_array);
        }
        let asserted = LinearRegression::assert_equal_length(n_x, y.named_array);
        match asserted {
            Ok(_asserted) => Ok(()),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    pub fn fit(&self) -> PyResult<PyLinearRegressionReturn> {
        let mut x_clone: Vec<NamedArray<f64>> = Vec::new();
        for v in self.x.clone().iter() {
            x_clone.push(v.named_array.clone())
        }
        let simple = LinearRegression {
            x: x_clone.clone(),
            y: self.y.named_array.clone(),
        };
        let res = simple.fit().unwrap();
        Ok(PyLinearRegressionReturn::from(res))
    }

    pub fn predict(
        &self,
        new_x: Vec<PyNamedArray>,
        return_object: &PyLinearRegressionReturn,
    ) -> PyResult<PyNamedArray> {
        let mut x_clone: Vec<NamedArray<f64>> = Vec::new();
        for v in new_x.clone().iter() {
            x_clone.push(v.named_array.clone())
        }
        let simple = LinearRegression {
            x: x_clone.clone(),
            y: self.y.named_array.clone(),
        };
        let intercept = return_object._intercept().unwrap();
        let beta_values = return_object._beta_values().unwrap();
        let res_predict = simple.predict(
            x_clone,
            LinearRegressionReturn {
                intercept,
                beta_values,
            },
        );
        Ok(PyNamedArray {
            named_array: res_predict.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::PyLinearRegressionReturn;
    use assert_float_eq::assert_float_relative_eq;
    use rstest::*;
    use std::collections::HashMap;

    #[fixture]
    fn input_named_array() -> Vec<PyNamedArray> {
        vec![PyNamedArray {
            named_array: NamedArray {
                name: "age".to_string(),
                data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
            },
        }]
    }

    #[fixture]
    fn target_named_array() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0, 141.0, 206.0, 135.0],
            },
        }
    }

    #[fixture]
    fn target_named_array_missing() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0, 141.0, 206.0],
            },
        }
    }

    #[fixture]
    fn one_d_prediction() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "predictions".to_string(),
                data: vec![
                    141.12926309,
                    492.79287226,
                    844.45648143,
                    1196.1200906,
                    1547.78369977,
                ],
            },
        }
    }

    #[rstest]
    fn test_instantiation(input_named_array: Vec<PyNamedArray>, target_named_array: PyNamedArray) {
        let _split: PyLinearRegression =
            PyLinearRegression::__init__(input_named_array, target_named_array).unwrap();
    }

    #[rstest]
    fn test_y_empty() {
        let x: Vec<PyNamedArray> = vec![PyNamedArray {
            named_array: NamedArray {
                name: "x".to_string(),
                data: Vec::new(),
            },
        }];
        let y: PyNamedArray = PyNamedArray {
            named_array: NamedArray {
                name: "y".to_string(),
                data: Vec::new(),
            },
        };
        assert!(PyLinearRegression::__init__(x, y).is_err());
    }

    #[rstest]
    fn test_x_empty() {
        let x: Vec<NamedArray<f64>> = vec![NamedArray {
            name: "x".to_string(),
            data: Vec::new(),
        }];
        let y: NamedArray<f64> = NamedArray {
            name: "y".to_string(),
            data: vec![0.0],
        };
        assert!(LinearRegression::new(x, y).is_err());
    }

    #[rstest]
    fn test_x_and_y_not_same_length(
        input_named_array: Vec<PyNamedArray>,
        target_named_array_missing: PyNamedArray,
    ) {
        assert!(
            PyLinearRegression::__init__(input_named_array, target_named_array_missing).is_err()
        );
    }

    #[rstest]
    fn test_single_fit(input_named_array: Vec<PyNamedArray>, target_named_array: PyNamedArray) {
        let lin_reg: PyLinearRegression =
            PyLinearRegression::__init__(input_named_array, target_named_array).unwrap();
        let res = lin_reg.fit().unwrap();

        assert_eq!(
            res,
            PyLinearRegressionReturn {
                coefficients: HashMap::from([
                    ("intercept".into(), 141.12926309276477),
                    ("age".into(), 351.66360917020086)
                ])
            }
        )
    }

    #[rstest]
    fn test_single_predict(
        input_named_array: Vec<PyNamedArray>,
        target_named_array: PyNamedArray,
        one_d_prediction: PyNamedArray,
    ) {
        let lin_reg: PyLinearRegression =
            PyLinearRegression::__init__(input_named_array, target_named_array).unwrap();
        let res = lin_reg.clone().fit().unwrap();
        let new_x: PyNamedArray = PyNamedArray {
            named_array: NamedArray {
                name: "new_x".to_string(),
                data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            },
        };

        let pred = lin_reg.clone().predict(vec![new_x], &res).unwrap();
        for (i, val) in pred.named_array.data.iter().enumerate() {
            assert_float_relative_eq!(val.to_owned(), one_d_prediction.named_array.data[i], 0.001)
        }
    }

    #[fixture]
    fn input_named_array_multi() -> Vec<PyNamedArray> {
        let age = PyNamedArray {
            named_array: NamedArray {
                name: "age".to_string(),
                data: vec![0.038076, -0.001882, 0.085299, -0.089063, 0.005383],
            },
        };
        let bmi = PyNamedArray {
            named_array: NamedArray {
                name: "bmi".to_string(),
                data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
            },
        };
        return vec![age, bmi];
    }

    #[fixture]
    fn target_named_array_multi() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0, 141.0, 206.0, 135.0],
            },
        }
    }

    #[fixture]
    fn target_named_array_missing_multi() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0, 141.0, 206.0],
            },
        }
    }

    #[fixture]
    fn two_d_prediction() -> PyNamedArray {
        PyNamedArray {
            named_array: NamedArray {
                name: "predictions".to_string(),
                data: vec![
                    145.653177,
                    299.43245535,
                    453.2117337,
                    606.99101204,
                    760.77029039,
                ],
            },
        }
    }

    #[rstest]
    fn test_instantiation_multi(
        input_named_array_multi: Vec<PyNamedArray>,
        target_named_array_multi: PyNamedArray,
    ) {
        let _split: PyLinearRegression =
            PyLinearRegression::__init__(input_named_array_multi, target_named_array_multi)
                .unwrap();
    }

    #[rstest]
    fn test_x_and_y_not_same_length_multi(
        input_named_array_multi: Vec<PyNamedArray>,
        target_named_array_missing_multi: PyNamedArray,
    ) {
        assert!(PyLinearRegression::__init__(
            input_named_array_multi,
            target_named_array_missing_multi
        )
        .is_err());
    }

    #[rstest]
    fn test_fit_multiple(
        input_named_array_multi: Vec<PyNamedArray>,
        target_named_array_multi: PyNamedArray,
    ) {
        let expected: PyLinearRegressionReturn = PyLinearRegressionReturn {
            coefficients: HashMap::from([
                ("intercept".into(), 145.653177),
                ("age".into(), -684.31017707),
                ("bmi".into(), 838.08945541),
            ]),
        };
        let lin_reg =
            PyLinearRegression::__init__(input_named_array_multi, target_named_array_multi)
                .unwrap();
        let res = lin_reg.fit().unwrap();

        assert_float_relative_eq!(
            res._intercept().unwrap(),
            expected._intercept().unwrap(),
            0.001
        );
        for (coef, val) in expected._beta_values().unwrap().iter() {
            let v = val.to_owned();
            let res_v = res._beta_values().unwrap().get(coef).unwrap().to_owned();
            assert_float_relative_eq!(res_v, v, 0.001)
        }
    }

    #[rstest]
    fn test_multi_predict(
        input_named_array_multi: Vec<PyNamedArray>,
        target_named_array: PyNamedArray,
        two_d_prediction: PyNamedArray,
    ) {
        let lin_reg: PyLinearRegression =
            PyLinearRegression::__init__(input_named_array_multi, target_named_array).unwrap();
        let res = lin_reg.clone().fit().unwrap();
        let new_x: Vec<PyNamedArray> = vec![
            PyNamedArray {
                named_array: NamedArray {
                    name: "age".to_string(),
                    data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                },
            },
            PyNamedArray {
                named_array: NamedArray {
                    name: "bmi".to_string(),
                    data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                },
            },
        ];

        let pred = lin_reg.clone().predict(new_x, &res).unwrap();
        for (i, val) in pred.named_array.data.iter().enumerate() {
            assert_float_relative_eq!(val.to_owned(), two_d_prediction.named_array.data[i], 0.001)
        }
    }
}
