use std::collections::HashMap;

use crate::core::types::LinearRegressionReturn;
use num::{Num, ToPrimitive};
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::RustLearnError::ValidationError;
use rustlearn_errors::{ErrString, RustLearnError};

#[derive(Debug, PartialEq)]
pub struct SimpleLinearRegression<T> {
    pub x: NamedArray<T>,
    pub y: NamedArray<T>,
}

pub type Result<SimpleLinearRegression> =
    std::result::Result<SimpleLinearRegression, RustLearnError>;

impl<T> SimpleLinearRegression<T>
where
    T: Copy,
    T: Clone,
{
    pub fn new(x: NamedArray<T>, y: NamedArray<T>) -> Result<Self>
    where
        T: ToPrimitive,
        T: Num,
        T: Copy,
    {
        Self::assert_equal_length(x.clone(), y.clone())?;
        Ok(Self { x, y })
    }

    pub fn assert_equal_length(x: NamedArray<T>, y: NamedArray<T>) -> Result<()>
    where
        T: ToPrimitive,
        T: Num,
        T: Copy,
    {
        if x.len() != y.len() {
            return Err(ValidationError(ErrString::from("mismatch x and y lengths")));
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
        let col_name = self.x.name.clone();
        let n: f64 = self.x.clone().len().to_f64().unwrap();

        let m_x: f64 = match self.x.clone().mean() {
            Ok(m_x) => m_x,
            Err(e) => return Err(e),
        };
        let m_y: f64 = match self.y.clone().mean() {
            Ok(m_y) => m_y,
            Err(e) => return Err(e),
        };

        let ss_xy: f64 = self.x.clone().dot(self.y.clone()).unwrap() - n * m_y * m_x;
        let ss_xx: f64 = self.x.clone().dot(self.x.clone()).unwrap() - n * m_x * m_x;

        let b_1 = ss_xy / ss_xx;
        let b_0 = m_y - b_1 * m_x;

        return Ok(LinearRegressionReturn {
            intercept: b_0,
            beta_values: HashMap::from([(col_name, b_1)]),
        });
    }

    pub fn fit(self) -> Result<LinearRegressionReturn>
    where
        T: Num,
        T: ToPrimitive,
        T: Clone,
        T: Into<f64>,
    {
        return Self::single_linear_regression_estimate(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn input_named_array() -> NamedArray<f64> {
        NamedArray {
            name: "age".to_string(),
            data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
        }
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

    #[rstest]
    fn test_instantiation(input_named_array: NamedArray<f64>, target_named_array: NamedArray<f64>) {
        let _split: SimpleLinearRegression<f64> =
            SimpleLinearRegression::new(input_named_array, target_named_array).unwrap();
    }

    #[rstest]
    fn test_x_and_y_not_same_length(
        input_named_array: NamedArray<f64>,
        target_named_array_missing: NamedArray<f64>,
    ) {
        assert!(
            SimpleLinearRegression::new(input_named_array, target_named_array_missing).is_err()
        );
    }

    #[rstest]
    fn test_single_fit(input_named_array: NamedArray<f64>, target_named_array: NamedArray<f64>) {
        let lin_reg: SimpleLinearRegression<f64> =
            SimpleLinearRegression::new(input_named_array, target_named_array).unwrap();
        let res = lin_reg.fit().unwrap();

        assert_eq!(
            res,
            LinearRegressionReturn {
                intercept: 141.12926309276477,
                beta_values: HashMap::from([("age".into(), 351.66360917020086)])
            }
        )
    }
}
