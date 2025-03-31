use crate::core::base::checks;
use num::{Num, ToPrimitive};
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::RustLearnError;

pub fn mean_absolute_error<T>(
    y_true: NamedArray<T>,
    y_pred: NamedArray<T>,
) -> Result<f64, RustLearnError>
where
    T: Num,
    T: ToPrimitive,
    T: Copy,
{
    let checked = checks(y_true.clone(), y_pred.clone());
    match checked {
        Some(checked) => Err(checked),
        None => {
            let mut total: f64 = 0.0;
            let vec_true: Vec<T> = y_true.data;
            let vec_pred: Vec<T> = y_pred.data;
            for (i, val) in vec_true.iter().enumerate() {
                total += (val.to_f64().unwrap() - vec_pred[i].to_f64().unwrap()).abs();
            }
            Ok(total / vec_true.len() as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[fixture]
    fn target_named_array() -> NamedArray<f64> {
        NamedArray {
            name: "target".to_string(),
            data: vec![151.0, 75.0, 141.0, 206.0, 135.0],
        }
    }

    #[fixture]
    fn predictions_named_array() -> NamedArray<f64> {
        NamedArray {
            name: "predictions".to_string(),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        }
    }

    #[fixture]
    fn predictions_named_array_missing() -> NamedArray<f64> {
        NamedArray {
            name: "predictions".to_string(),
            data: vec![1.0, 2.0, 3.0, 4.0],
        }
    }

    #[rstest]
    fn test_equivalent(target_named_array: NamedArray<f64>) {
        assert_eq!(
            mean_absolute_error(target_named_array.clone(), target_named_array.clone()).unwrap(),
            0.0
        );
    }

    #[rstest]
    fn test_mismatch_error(
        target_named_array: NamedArray<f64>,
        predictions_named_array_missing: NamedArray<f64>,
    ) {
        assert!(mean_absolute_error(target_named_array, predictions_named_array_missing).is_err());
    }

    #[rstest]
    fn test_mae(target_named_array: NamedArray<f64>, predictions_named_array: NamedArray<f64>) {
        assert_eq!(
            mean_absolute_error(target_named_array, predictions_named_array).unwrap(),
            138.6
        );
    }
}
