use pyo3::prelude::*;
use rustlearn_python::array::PyNamedArray;
use rustlearn_python::exceptions::{NotYetImplementedError, RustLearnError, ValidationError};
use rustlearn_python::metrics::{mean_absolute_error, mean_squared_error};

#[pymodule]
fn rustylearn(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Base Classes
    m.add_class::<PyNamedArray>().unwrap();

    // Metrics
    m.add_wrapped(wrap_pyfunction!(
        mean_absolute_error::py_mean_absolute_error
    ))
    .unwrap();
    m.add_wrapped(wrap_pyfunction!(mean_squared_error::py_mean_squared_error))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(
        mean_squared_error::py_root_mean_squared_error
    ))
    .unwrap();

    // Exceptions
    m.add("RustLearnError", py.get_type::<RustLearnError>())
        .unwrap();
    m.add("ValidationError", py.get_type::<ValidationError>())
        .unwrap();
    m.add(
        "NotYetImplementedError",
        py.get_type::<NotYetImplementedError>(),
    )
    .unwrap();
    Ok(())
}
