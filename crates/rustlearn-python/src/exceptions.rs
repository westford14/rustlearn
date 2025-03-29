use pyo3::create_exception;
use pyo3::exceptions::PyException;

create_exception!(rust_kit_learn_core.exceptions, RustLearnError, PyException);
create_exception!(
    rust_kit_learn_core.exceptions,
    ValidationError,
    RustLearnError
);
create_exception!(
    rust_kit_learn_core.exceptions,
    NotYetImplementedError,
    RustLearnError
);
