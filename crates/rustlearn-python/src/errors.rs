use crate::exceptions::{NotYetImplementedError, ValidationError};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::fmt::{Debug, Display, Formatter};

use rustlearn_errors::RustLearnError;

pub enum PyRustLearnError {
    RustLearn(RustLearnError),
    Python(PyErr),
    Other(String),
}

impl std::fmt::Display for PyRustLearnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RustLearn(err) => Display::fmt(err, f),
            Self::Python(err) => Display::fmt(err, f),
            Self::Other(err) => write!(f, "{err}"),
        }
    }
}

impl From<RustLearnError> for PyRustLearnError {
    fn from(err: RustLearnError) -> Self {
        PyRustLearnError::RustLearn(err)
    }
}

impl From<PyErr> for PyRustLearnError {
    fn from(err: PyErr) -> Self {
        PyRustLearnError::Python(err)
    }
}

impl From<PyRustLearnError> for PyErr {
    fn from(err: PyRustLearnError) -> PyErr {
        use PyRustLearnError::*;
        match err {
            RustLearn(err) => match err {
                RustLearnError::ValidationError(err) => ValidationError::new_err(err.to_string()),
                RustLearnError::NotYetImplementedError(err) => {
                    NotYetImplementedError::new_err(err.to_string())
                }
            },
            Python(err) => err,
            err => PyRuntimeError::new_err(format!("{:?}", &err)),
        }
    }
}

impl Debug for PyRustLearnError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use PyRustLearnError::*;
        match self {
            RustLearn(err) => write!(f, "{err:?}"),
            Python(err) => write!(f, "{err:?}"),
            Other(err) => write!(f, "BindingsError: {err:?}"),
        }
    }
}
