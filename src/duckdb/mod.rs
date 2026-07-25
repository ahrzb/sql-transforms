//! The DuckDB-semantics interpreter: `DuckDBInferFn`.
//!
//! Same Python API as `InferFn` minus the transformer callout. Stub: the SQL
//! is parsed in the DuckDB dialect (so the wiring is real) and nothing else
//! happens — `infer()` raises rather than returning results that would
//! silently carry DataFusion semantics.
//!
//! Plan building, expression evaluation and type inference get their own
//! modules here as they land; none of them exist yet.

use std::collections::HashMap;

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use crate::error::InterpError;

#[pyclass]
pub struct DuckDBInferFn {
    /// No output model is derived yet, so this is whatever the caller declared
    /// (or `None`) rather than a synthesized one.
    #[pyo3(get)]
    output_model: Option<Py<PyAny>>,
}

#[pymethods]
impl DuckDBInferFn {
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, output_model=None))]
    fn new(
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        output_model: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = (row_tables, static_tables);
        Parser::parse_sql(&DuckDbDialect {}, &sql)
            .map_err(|e| InterpError::Build(format!("SQL parse error: {e}")))?;
        Ok(DuckDBInferFn { output_model })
    }

    #[pyo3(signature = (tables=None, **kwargs))]
    fn infer(
        &self,
        tables: Option<HashMap<String, Vec<Py<PyAny>>>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let _ = (tables, kwargs);
        Err(PyNotImplementedError::new_err(
            "the DuckDB interpreter is a stub; use InferFn",
        ))
    }
}
