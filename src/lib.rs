//! Native row-at-a-time SQL interpreters.
//!
//! One extension module, one engine per submodule:
//!   * `datafusion` — `InferFn`, the engine differentially tested against
//!     DataFusion. Full expression surface, transformer callouts, joins.
//!   * `duckdb` — `DuckDBInferFn`, DuckDB semantics. Build-only so far.
//!
//! Each engine owns its own plan/expression/type-inference modules so their
//! semantics can diverge freely. What they share sits at the crate root and is
//! deliberately semantics-free: the value representation, the type vocabulary,
//! the error type, and the Python schema/model marshalling.

use pyo3::prelude::*;

mod datafusion;
mod duckdb;
mod error;
mod lookup;
mod schema;
pub mod specializer;
mod types;
mod value;

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<datafusion::InferFn>()?;
    m.add_class::<duckdb::DuckDBInferFn>()?;
    Ok(())
}
