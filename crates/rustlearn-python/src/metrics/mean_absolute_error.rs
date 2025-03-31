use pyo3::intern;
use pyo3::prelude::*;
use rustlearn_metrics::mean_absolute_error::mean_absolute_error;

use crate::array::PyNamedArray;
use crate::errors::PyRustLearnError;

#[pyfunction]
pub fn py_mean_absolute_error(
    y_true: &Bound<'_, PyAny>,
    y_pred: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let t_s = y_true.getattr(intern!(y_true.py(), "_n"))?;
    let y_true_series = t_s.extract::<PyNamedArray>().unwrap().named_array;
    let t_p = y_pred.getattr(intern!(y_pred.py(), "_n"))?;
    let y_pred_series = t_p.extract::<PyNamedArray>().unwrap().named_array;
    let res = match mean_absolute_error(y_true_series, y_pred_series) {
        Ok(res) => res,
        Err(e) => return Err(PyErr::from(PyRustLearnError::from(e))),
    };
    Ok(res)
}
