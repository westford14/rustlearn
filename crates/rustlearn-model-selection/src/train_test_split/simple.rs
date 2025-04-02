use crate::types::TrainTestSplitReturn;
use num::*;
use rustlearn_array::namedarray::NamedArray;
use rustlearn_errors::RustLearnError::ValidationError;
use rustlearn_errors::{ErrString, RustLearnError};

#[derive(Debug, PartialEq, Clone)]
pub struct SimpleTrainTestSplit<T> {
    pub x: Vec<NamedArray<T>>,
    pub y: NamedArray<T>,
    pub train_proportion: f64,
}

pub type Result<SimpleTrainTestSplit> = std::result::Result<SimpleTrainTestSplit, RustLearnError>;

impl<T> SimpleTrainTestSplit<T>
where
    T: Copy,
    T: Clone,
    T: ToPrimitive,
{
    pub fn new(x: Vec<NamedArray<T>>, y: NamedArray<T>, train_proportion: f64) -> Result<Self>
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
        Self::assert_logical_train_proportion(train_proportion)?;
        Ok(Self {
            x,
            y,
            train_proportion,
        })
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

    pub fn assert_logical_train_proportion(train_proportion: f64) -> Result<()> {
        if train_proportion >= 1.0 || train_proportion <= 0.0 {
            return Err(ValidationError(ErrString::from(
                "train_proportion must be between 0 and 1",
            )));
        }
        Ok(())
    }

    pub fn split(&mut self) -> Result<TrainTestSplitReturn> {
        let x_clone = self.x.clone();
        let y_clone = self.y.clone();
        let index: usize = (y_clone.data.len().to_f64().unwrap() * self.train_proportion) as usize;

        let mut x_train: Vec<NamedArray<f64>> = Vec::new();
        let mut x_test: Vec<NamedArray<f64>> = Vec::new();

        for x in x_clone.iter() {
            let temp_train = x.data[0..index]
                .iter()
                .map(|x| x.to_f64().unwrap())
                .collect();
            let temp_test = x.data[index..x.data.len()]
                .iter()
                .map(|x| x.to_f64().unwrap())
                .collect();
            x_train.push(NamedArray {
                name: x.clone().name,
                data: temp_train,
            });
            x_test.push(NamedArray {
                name: x.clone().name,
                data: temp_test,
            });
        }

        let y_train: NamedArray<f64> = NamedArray {
            name: y_clone.clone().name,
            data: y_clone.data[0..index]
                .iter()
                .map(|x| x.to_f64().unwrap())
                .collect(),
        };
        let y_test: NamedArray<f64> = NamedArray {
            name: y_clone.clone().name,
            data: y_clone.data[index..y_clone.data.len()]
                .iter()
                .map(|x| x.to_f64().unwrap())
                .collect(),
        };

        Ok(TrainTestSplitReturn {
            x_train,
            x_test,
            y_train,
            y_test,
        })
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
    fn input_named_array_2d() -> Vec<NamedArray<f64>> {
        vec![
            NamedArray {
                name: "age".to_string(),
                data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
            },
            NamedArray {
                name: "bmi".to_string(),
                data: vec![0.061696, -0.051474, 0.044451, -0.011595, -0.036385],
            },
        ]
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
    fn test_instantiation(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
    ) {
        let _split: SimpleTrainTestSplit<f64> =
            SimpleTrainTestSplit::new(input_named_array, target_named_array, 0.5).unwrap();
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
        assert!(SimpleTrainTestSplit::new(x, y, 0.5).is_err());
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
        assert!(SimpleTrainTestSplit::new(x, y, 0.5).is_err());
    }

    #[rstest]
    fn test_x_and_y_not_same_length(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array_missing: NamedArray<f64>,
    ) {
        assert!(
            SimpleTrainTestSplit::new(input_named_array, target_named_array_missing, 0.5).is_err()
        );
    }

    #[rstest]
    fn test_train_proportion_wrong(
        input_named_array: Vec<NamedArray<f64>>,
        target_named_array_missing: NamedArray<f64>,
    ) {
        assert!(SimpleTrainTestSplit::new(
            input_named_array.clone(),
            target_named_array_missing.clone(),
            1.1
        )
        .is_err());
        assert!(SimpleTrainTestSplit::new(
            input_named_array.clone(),
            target_named_array_missing.clone(),
            0.0
        )
        .is_err());
    }

    #[rstest]
    fn test_split_1d(input_named_array: Vec<NamedArray<f64>>, target_named_array: NamedArray<f64>) {
        let mut splitter =
            SimpleTrainTestSplit::new(input_named_array.clone(), target_named_array.clone(), 0.5)
                .unwrap();
        let expected = TrainTestSplitReturn {
            x_train: vec![NamedArray {
                name: "age".to_string(),
                data: vec![0.061696, -0.051474],
            }],
            x_test: vec![NamedArray {
                name: "age".to_string(),
                data: vec![0.044451, -0.011595, -0.036385],
            }],
            y_train: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0],
            },
            y_test: NamedArray {
                name: "target".to_string(),
                data: vec![141.0, 206.0, 135.0],
            },
        };
        let res = splitter.split().unwrap();

        assert_eq!(res.x_train.len(), expected.x_train.len());
        assert_eq!(res.x_train[0].len(), expected.x_train[0].len());
        assert_eq!(res.x_test.len(), expected.x_test.len());
        assert_eq!(res.x_test[0].len(), expected.x_test[0].len());
        assert_eq!(res.y_train.len(), expected.y_train.len());
        assert_eq!(res.y_test.len(), expected.y_test.len());

        for (j, x) in res.x_train.iter().enumerate() {
            for (i, t) in x.data.iter().enumerate() {
                assert_float_relative_eq!(t.to_owned(), expected.x_train[j].data[i], 0.001)
            }
        }
        for (j, x) in res.x_test.iter().enumerate() {
            for (i, t) in x.data.iter().enumerate() {
                assert_float_relative_eq!(t.to_owned(), expected.x_test[j].data[i], 0.001)
            }
        }
        for (i, t) in res.y_train.data.iter().enumerate() {
            assert_float_relative_eq!(t.to_owned(), expected.y_train.data[i], 0.001)
        }
        for (i, t) in res.y_test.data.iter().enumerate() {
            assert_float_relative_eq!(t.to_owned(), expected.y_test.data[i], 0.001)
        }
    }

    #[rstest]
    fn test_split_2d(
        input_named_array_2d: Vec<NamedArray<f64>>,
        target_named_array: NamedArray<f64>,
    ) {
        let mut splitter = SimpleTrainTestSplit::new(
            input_named_array_2d.clone(),
            target_named_array.clone(),
            0.5,
        )
        .unwrap();
        let expected = TrainTestSplitReturn {
            x_train: vec![
                NamedArray {
                    name: "age".to_string(),
                    data: vec![0.061696, -0.051474],
                },
                NamedArray {
                    name: "bmi".to_string(),
                    data: vec![0.061696, -0.051474],
                },
            ],
            x_test: vec![
                NamedArray {
                    name: "age".to_string(),
                    data: vec![0.044451, -0.011595, -0.036385],
                },
                NamedArray {
                    name: "bmi".to_string(),
                    data: vec![0.044451, -0.011595, -0.036385],
                },
            ],
            y_train: NamedArray {
                name: "target".to_string(),
                data: vec![151.0, 75.0],
            },
            y_test: NamedArray {
                name: "target".to_string(),
                data: vec![141.0, 206.0, 135.0],
            },
        };
        let res = splitter.split().unwrap();

        assert_eq!(res.x_train.len(), expected.x_train.len());
        assert_eq!(res.x_train[0].len(), expected.x_train[0].len());
        assert_eq!(res.x_test.len(), expected.x_test.len());
        assert_eq!(res.x_test[0].len(), expected.x_test[0].len());
        assert_eq!(res.y_train.len(), expected.y_train.len());
        assert_eq!(res.y_test.len(), expected.y_test.len());

        for (j, x) in res.x_train.iter().enumerate() {
            for (i, t) in x.data.iter().enumerate() {
                assert_float_relative_eq!(t.to_owned(), expected.x_train[j].data[i], 0.001)
            }
        }
        for (j, x) in res.x_test.iter().enumerate() {
            for (i, t) in x.data.iter().enumerate() {
                assert_float_relative_eq!(t.to_owned(), expected.x_test[j].data[i], 0.001)
            }
        }
        for (i, t) in res.y_train.data.iter().enumerate() {
            assert_float_relative_eq!(t.to_owned(), expected.y_train.data[i], 0.001)
        }
        for (i, t) in res.y_test.data.iter().enumerate() {
            assert_float_relative_eq!(t.to_owned(), expected.y_test.data[i], 0.001)
        }
    }
}
