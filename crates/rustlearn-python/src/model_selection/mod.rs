use pyo3::prelude::*;

use crate::array::PyNamedArray;
use rustlearn_model_selection::types::TrainTestSplitReturn;

pub mod simple;

#[pyclass]
pub struct PyTrainTestSplitReturn {
    pub x_train: Vec<PyNamedArray>,
    pub x_test: Vec<PyNamedArray>,
    pub y_train: PyNamedArray,
    pub y_test: PyNamedArray,
}

impl From<TrainTestSplitReturn> for PyTrainTestSplitReturn {
    fn from(train_test_split_return: TrainTestSplitReturn) -> Self {
        let mut new_x_train: Vec<PyNamedArray> = Vec::new();
        let mut new_x_test: Vec<PyNamedArray> = Vec::new();

        for t in train_test_split_return.x_train.clone() {
            new_x_train.push(PyNamedArray { named_array: t });
        }

        for t in train_test_split_return.x_test.clone() {
            new_x_test.push(PyNamedArray { named_array: t });
        }
        PyTrainTestSplitReturn {
            x_train: new_x_train,
            x_test: new_x_test,
            y_train: PyNamedArray {
                named_array: train_test_split_return.y_train,
            },
            y_test: PyNamedArray {
                named_array: train_test_split_return.y_test,
            },
        }
    }
}

#[pymethods]
impl PyTrainTestSplitReturn {
    #[new]
    pub fn __init__(
        x_train: Vec<PyNamedArray>,
        x_test: Vec<PyNamedArray>,
        y_train: PyNamedArray,
        y_test: PyNamedArray,
    ) -> PyResult<Self> {
        let ret = PyTrainTestSplitReturn {
            x_train,
            x_test,
            y_train,
            y_test,
        };

        Ok(ret)
    }

    pub fn x_train(&self) -> PyResult<Vec<PyNamedArray>> {
        Ok(self.x_train.clone())
    }

    pub fn x_test(&self) -> PyResult<Vec<PyNamedArray>> {
        Ok(self.x_test.clone())
    }

    pub fn y_train(&self) -> PyResult<PyNamedArray> {
        Ok(self.y_train.clone())
    }

    pub fn y_test(&self) -> PyResult<PyNamedArray> {
        Ok(self.y_test.clone())
    }
}
