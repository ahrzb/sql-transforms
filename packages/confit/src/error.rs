//! The one error type every interpreter engine reports through.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyErr;

/// The variant picks the Python exception a failure surfaces as, so it is
/// part of the caller's contract: `Build` refuses at construction and `Eval`
/// traps mid-run (both `ValueError`), `MissingKey` raises `KeyError`.
pub enum InterpError {
    Build(String),
    MissingKey(String),
    Eval(String),
}

impl From<InterpError> for PyErr {
    fn from(e: InterpError) -> PyErr {
        match e {
            InterpError::Build(msg) => PyValueError::new_err(msg),
            InterpError::MissingKey(msg) => PyKeyError::new_err(msg),
            InterpError::Eval(msg) => PyValueError::new_err(msg),
        }
    }
}
