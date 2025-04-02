use super::PyTrainTestSplitReturn;
use crate::array::PyNamedArray;
use crate::errors::PyRustLearnError;
use pyo3::prelude::*;
use rustlearn_array::namedarray::NamedArray;
use rustlearn_model_selection::train_test_split::simple::SimpleTrainTestSplit;

#[pyclass]
#[derive(Clone)]
pub struct PySimpleTrainTestSplit {
    pub x: Vec<PyNamedArray>,
    pub y: PyNamedArray,
    pub train_proportion: f64,
}

impl From<SimpleTrainTestSplit<f64>> for PySimpleTrainTestSplit {
    fn from(simple_train_test_split: SimpleTrainTestSplit<f64>) -> Self {
        let mut x: Vec<PyNamedArray> = Vec::new();
        for nm in simple_train_test_split.x {
            x.push(PyNamedArray { named_array: nm });
        }
        PySimpleTrainTestSplit {
            x,
            y: PyNamedArray {
                named_array: simple_train_test_split.y,
            },
            train_proportion: simple_train_test_split.train_proportion,
        }
    }
}

#[pymethods]
impl PySimpleTrainTestSplit {
    #[new]
    pub fn __init__(
        x: Vec<PyNamedArray>,
        y: PyNamedArray,
        train_proportion: f64,
    ) -> PyResult<Self> {
        let multiple_linear = PySimpleTrainTestSplit {
            x: x.clone(),
            y: y.clone(),
            train_proportion,
        };
        let _ = Self::assert_equal_length(x.clone(), y.clone());
        let _ = Self::assert_logical_train_proportion(train_proportion);

        Ok(multiple_linear)
    }

    #[staticmethod]
    pub fn assert_equal_length(x: Vec<PyNamedArray>, y: PyNamedArray) -> PyResult<()> {
        let mut n_x: Vec<NamedArray<f64>> = Vec::new();
        for t in x {
            n_x.push(t.named_array);
        }
        let asserted = SimpleTrainTestSplit::assert_equal_length(n_x, y.named_array);
        match asserted {
            Ok(_asserted) => Ok(()),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    #[staticmethod]
    pub fn assert_logical_train_proportion(train_proportion: f64) -> PyResult<()> {
        let asserted = SimpleTrainTestSplit::<f64>::assert_logical_train_proportion(
            train_proportion.to_owned(),
        );
        match asserted {
            Ok(_asserted) => Ok(()),
            Err(e) => Err(PyErr::from(PyRustLearnError::RustLearn(e))),
        }
    }

    pub fn split(&self) -> PyResult<PyTrainTestSplitReturn> {
        let mut x_clone: Vec<NamedArray<f64>> = Vec::new();
        for v in self.x.clone().iter() {
            x_clone.push(v.named_array.clone())
        }
        let mut simple = SimpleTrainTestSplit {
            x: x_clone.clone(),
            y: self.y.named_array.clone(),
            train_proportion: self.train_proportion,
        };
        let res = simple.split().unwrap();
        Ok(PyTrainTestSplitReturn::from(res))
    }
}
