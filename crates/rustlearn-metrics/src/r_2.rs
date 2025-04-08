use crate::core::base::checks;
use num::{pow::Pow, Num, ToPrimitive};
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::RustLearnError;

pub fn r_2<T>(y_true: NamedArray<T>, y_pred: NamedArray<T>) -> Result<f64, RustLearnError>
where
    T: Num,
    T: ToPrimitive,
    T: Copy,
    T: Into<f64>,
{
    let checked = checks(y_true.clone(), y_pred.clone());
    match checked {
        Some(checked) => Err(checked),
        None => {
            let y_mean = y_true.mean().unwrap();
            let mut ss_reg: Vec<f64> = Vec::new();
            for (i, v) in y_true.data.clone().iter().enumerate() {
                ss_reg.push((v.to_f64().unwrap() - y_pred.data[i].to_f64().unwrap()).pow(2))
            }
            let ss_total: f64 = y_true
                .data
                .iter()
                .map(|x| (x.to_f64().unwrap() - y_mean).pow(2))
                .collect::<Vec<f64>>()
                .into_iter()
                .sum();
            Ok(1.0 - (ss_reg.into_iter().sum::<f64>() / ss_total))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_float_eq::assert_float_relative_eq;
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
            data: vec![140.0, 86.0, 120.0, 240.0, 140.0],
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
            r_2(target_named_array.clone(), target_named_array.clone()).unwrap(),
            1.0
        );
    }

    #[rstest]
    fn test_mismatch_error(
        target_named_array: NamedArray<f64>,
        predictions_named_array_missing: NamedArray<f64>,
    ) {
        assert!(r_2(target_named_array, predictions_named_array_missing).is_err());
    }

    #[rstest]
    fn test_r_2(target_named_array: NamedArray<f64>, predictions_named_array: NamedArray<f64>) {
        assert_float_relative_eq!(
            r_2(target_named_array, predictions_named_array).unwrap(),
            0.7861208004406095,
            0.001
        );
    }
}
