use std::collections::HashMap;

use crate::types::LinearRegressionReturn;
use ndarray::prelude::*;
use ndarray::{Array, ArrayView};
use ndarray_linalg::solve::Inverse;
use num::{Num, ToPrimitive};
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::RustLearnError::ValidationError;
use rustlearn_errors::{ErrString, RustLearnError};
use std::iter;

#[derive(Debug, PartialEq, Clone)]
pub struct LinearRegression<T> {
    pub x: Vec<NamedArray<T>>,
    pub y: NamedArray<T>,
}

pub type Result<LinearRegression> = std::result::Result<LinearRegression, RustLearnError>;

impl<T> LinearRegression<T>
where
    T: Copy,
    T: Clone,
{
    pub fn new(x: Vec<NamedArray<T>>, y: NamedArray<T>) -> Result<Self>
    where
        T: ToPrimitive,
        T: Num,
        T: Copy,
    {
        if y.data.is_empty() {
            return Err(ValidationError(ErrString::from(
                "target is an empty vector",
            )));
        }
        if x.is_empty() {
            return Err(ValidationError(ErrString::from("no x-values provided")));
        }
        Self::assert_equal_length(x.clone(), y.clone())?;
        Ok(Self { x, y })
    }

    pub fn assert_equal_length(x: Vec<NamedArray<T>>, y: NamedArray<T>) -> Result<()>
    where
        T: ToPrimitive,
        T: Num,
        T: Clone,
        T: Copy,
    {
        for nm in x.iter() {
            if nm.len() != y.len() {
                return Err(ValidationError(ErrString::from("mismatch x and y lengths")));
            }
        }
        Ok(())
    }

    fn single_linear_regression_estimate(self) -> Result<LinearRegressionReturn>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
    {
        let x = self.x[0].clone();
        let col_name = x.name.clone();
        let n: f64 = x.clone().len().to_f64().unwrap();

        let m_x: f64 = x.clone().mean()?;
        let m_y: f64 = self.y.clone().mean()?;

        let ss_xy: f64 = x.clone().dot(self.y.clone()).unwrap() - n * m_y * m_x;
        let ss_xx: f64 = x.clone().dot(x.clone()).unwrap() - n * m_x * m_x;

        let b_1 = ss_xy / ss_xx;
        let b_0 = m_y - b_1 * m_x;

        Ok(LinearRegressionReturn {
            intercept: b_0,
            beta_values: HashMap::from([(col_name, b_1)]),
        })
    }

    fn multiple_linear_regression_estimate(self) -> Result<LinearRegressionReturn>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
    {
        let mut x_copy: Vec<NamedArray<f64>> = Vec::new();
        let mut col_names: Vec<String> = Vec::new();
        for nm in self.x.clone().iter() {
            let updated = nm.data.iter().map(|x| x.to_f64().unwrap()).collect();
            x_copy.push(NamedArray {
                name: nm.name.clone(),
                data: updated,
            });
            col_names.push(nm.name.clone());
        }
        let zeros: Vec<f64> = iter::repeat(1.0).take(x_copy[0].len()).collect();
        let intercepts = NamedArray {
            name: "intercepts".to_string(),
            data: zeros,
        };
        x_copy.insert(0, intercepts);

        let mut x_matrix = Array::<f64, _>::zeros((5, 0).f());

        for col in x_copy.iter() {
            x_matrix.push_column(ArrayView::from(&col.data)).unwrap();
        }

        let x_transpose = x_matrix.t();
        let x_transpose_x = x_transpose.dot(&x_matrix);

        let inverse = x_transpose_x.inv().unwrap();
        let mut y_vec: Vec<f64> = Vec::new();
        for val in self.y.clone().data {
            y_vec.push(val.to_f64().unwrap());
        }
        let y_array = ArrayView::from(&y_vec);
        let final_multi = inverse.dot(&x_transpose);
        let final_y = final_multi.dot(&y_array);

        let mut intercepts: f64 = 0.0;
        let mut coefs: HashMap<String, f64> = HashMap::new();

        for col in final_y.columns() {
            let temp_col: Vec<f64> = col.to_vec();
            for (i, v) in temp_col.iter().enumerate() {
                if i == 0 {
                    intercepts = *v;
                } else {
                    coefs.insert(col_names[i - 1].clone(), *v);
                }
            }
        }

        Ok(LinearRegressionReturn {
            intercept: intercepts,
            beta_values: coefs,
        })
    }

    pub fn fit(self) -> Result<LinearRegressionReturn>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
        T: Copy,
    {
        if self.x.len() == 1 {
            Self::single_linear_regression_estimate(self)
        } else {
            Self::multiple_linear_regression_estimate(self)
        }
    }

    pub fn single_linear_regression_predict(
        self,
        new_x: Vec<NamedArray<f64>>,
        return_object: LinearRegressionReturn,
    ) -> NamedArray<f64>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
        T: Copy,
    {
        let x_clone = new_x.clone();
        let intercept = return_object.intercept;
        let coef = return_object.beta_values;
        let ret: Vec<f64> = x_clone[0]
            .data
            .iter()
            .map(|x| x.to_f64().unwrap())
            .collect();
        let mut transformed: Vec<f64> = Vec::new();

        for (_, v) in coef.iter() {
            transformed = ret.iter().map(|x| x * v + intercept).collect();
        }

        NamedArray {
            name: "predictions".to_string(),
            data: transformed,
        }
    }

    pub fn multiple_linear_regression_predict(
        self,
        new_x: Vec<NamedArray<f64>>,
        return_object: LinearRegressionReturn,
    ) -> NamedArray<f64>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
        T: Copy,
    {
        let x_clone = new_x.clone();
        let intercept = return_object.intercept;
        let coef = return_object.beta_values;
        let mut convenience: HashMap<String, Vec<f64>> = HashMap::new();

        for x in x_clone.iter() {
            convenience.insert(
                x.name.clone(),
                x.data.iter().map(|x| x.to_f64().unwrap()).collect(),
            );
        }

        let mut data: Vec<Vec<f64>> = Vec::new();
        for (k, v) in coef.iter() {
            let transformed: Vec<f64> = convenience.get(k).unwrap().iter().map(|x| x * v).collect();
            data.push(transformed)
        }

        let vec_leng: usize = data[0].len();
        let mut sums = vec![0.0; vec_leng];
        for v in data {
            for (i, x) in v.into_iter().enumerate() {
                sums[i] += x;
            }
        }
        sums = sums.iter().map(|x| x + intercept).collect();

        NamedArray {
            name: "predictions".to_string(),
            data: sums,
        }
    }

    pub fn predict(
        self,
        new_x: Vec<NamedArray<f64>>,
        return_object: LinearRegressionReturn,
    ) -> NamedArray<f64>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
        T: Copy,
    {
        if new_x.len() == 1 {
            Self::single_linear_regression_predict(self, new_x, return_object)
        } else {
            Self::multiple_linear_regression_predict(self, new_x, return_object)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::assert_float_relative_eq;
    use rstest::*;

    #[fixture]
    fn input_named_array() -> Vec<NamedArray<f64>> {
        vec![NamedArray {
            name: "age".to_string(),
            data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
        }]
    }

    #[fixture]
    fn target_named_array() -> NamedArray<f64> {
        NamedArray {
            name: "target".to_string(),
            data: vec![151.0, 75.0, 141.0, 206.0, 135.0],
        }
    }

    #[fixture]
    fn target_named_array_missing() -> NamedArray<f64> {
        NamedArray {
            name: "target".to_string(),
            data: vec![151.0, 75.0, 141.0, 206.0],
        }
    }

    #[fixture]
    fn one_d_prediction() -> NamedArray<f64> {
        NamedArray {
            name: "predictions".to_string(),
            data: vec![
                141.12926309,
                492.79287226,
                844.45648143,
                1196.1200906,
                1547.78369977,
            ],
        }
    }

    #[rstest]
    fn test_instantiation(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
    ) {
        let _split: LinearRegression<f64> =
            LinearRegression::new(input_named_array, target_named_array).unwrap();
    }

    #[rstest]
    fn test_y_empty() {
        let x: Vec<NamedArray<f64>> = vec![NamedArray {
            name: "x".to_string(),
            data: Vec::new(),
        }];
        let y: NamedArray<f64> = NamedArray {
            name: "y".to_string(),
            data: Vec::new(),
        };
        assert!(LinearRegression::new(x, y).is_err());
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
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array_missing: NamedArray<f64>,
    ) {
        assert!(LinearRegression::new(input_named_array, target_named_array_missing).is_err());
    }

    #[rstest]
    fn test_single_fit(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
    ) {
        let lin_reg: LinearRegression<f64> =
            LinearRegression::new(input_named_array, target_named_array).unwrap();
        let res = lin_reg.fit().unwrap();

        assert_eq!(
            res,
            LinearRegressionReturn {
                intercept: 141.12926309276477,
                beta_values: HashMap::from([("age".into(), 351.66360917020086)])
            }
        )
    }

    #[rstest]
    fn test_single_predict(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
        one_d_prediction: NamedArray<f64>,
    ) {
        let lin_reg: LinearRegression<f64> =
            LinearRegression::new(input_named_array, target_named_array).unwrap();
        let res = lin_reg.clone().fit().unwrap();
        let new_x: NamedArray<f64> = NamedArray {
            name: "new_x".to_string(),
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
        };

        let pred = lin_reg.clone().predict(vec![new_x], res);
        for (i, val) in pred.data.iter().enumerate() {
            assert_float_relative_eq!(val.to_owned(), one_d_prediction.data[i], 0.001)
        }
    }

    #[fixture]
    fn input_named_array_multi() -> Vec<NamedArray<f64>> {
        let age = NamedArray {
            name: "age".to_string(),
            data: vec![0.038076, -0.001882, 0.085299, -0.089063, 0.005383],
        };
        let bmi = NamedArray {
            name: "bmi".to_string(),
            data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
        };
        return vec![age, bmi];
    }

    #[fixture]
    fn target_named_array_multi() -> NamedArray<f64> {
        NamedArray {
            name: "target".to_string(),
            data: vec![151.0, 75.0, 141.0, 206.0, 135.0],
        }
    }

    #[fixture]
    fn target_named_array_missing_multi() -> NamedArray<f64> {
        NamedArray {
            name: "target".to_string(),
            data: vec![151.0, 75.0, 141.0, 206.0],
        }
    }

    #[fixture]
    fn two_d_prediction() -> NamedArray<f64> {
        NamedArray {
            name: "predictions".to_string(),
            data: vec![
                145.653177,
                299.43245535,
                453.2117337,
                606.99101204,
                760.77029039,
            ],
        }
    }

    #[rstest]
    fn test_instantiation_multi(
        input_named_array_multi: Vec<NamedArray<f64>>,
        target_named_array_multi: NamedArray<f64>,
    ) {
        let _split: LinearRegression<f64> =
            LinearRegression::new(input_named_array_multi, target_named_array_multi).unwrap();
    }

    #[rstest]
    fn test_x_and_y_not_same_length_multi(
        input_named_array_multi: Vec<NamedArray<f64>>,
        target_named_array_missing_multi: NamedArray<f64>,
    ) {
        assert!(
            LinearRegression::new(input_named_array_multi, target_named_array_missing_multi)
                .is_err()
        );
    }

    #[rstest]
    fn test_fit_multiple(
        input_named_array_multi: Vec<NamedArray<f64>>,
        target_named_array_multi: NamedArray<f64>,
    ) {
        let expected: LinearRegressionReturn = LinearRegressionReturn {
            intercept: 145.653177,
            beta_values: HashMap::from([
                ("age".to_string(), -684.31017707),
                ("bmi".to_string(), 838.08945541),
            ]),
        };
        let lin_reg =
            LinearRegression::new(input_named_array_multi, target_named_array_multi).unwrap();
        let res = lin_reg.fit().unwrap();

        assert_float_relative_eq!(res.intercept, expected.intercept, 0.001);
        for (coef, val) in expected.beta_values.iter() {
            let v = val.to_owned();
            let res_v = res.beta_values.get(coef).unwrap().to_owned();
            assert_float_relative_eq!(res_v, v, 0.001)
        }
    }

    #[rstest]
    fn test_multi_predict(
        input_named_array_multi: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
        two_d_prediction: NamedArray<f64>,
    ) {
        let lin_reg: LinearRegression<f64> =
            LinearRegression::new(input_named_array_multi, target_named_array).unwrap();
        let res = lin_reg.clone().fit().unwrap();
        let new_x: Vec<NamedArray<f64>> = vec![
            NamedArray {
                name: "age".to_string(),
                data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            },
            NamedArray {
                name: "bmi".to_string(),
                data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            },
        ];

        let pred = lin_reg.clone().predict(new_x, res);
        for (i, val) in pred.data.iter().enumerate() {
            assert_float_relative_eq!(val.to_owned(), two_d_prediction.data[i], 0.001)
        }
    }
}
