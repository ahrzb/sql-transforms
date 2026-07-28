//! Confit — a partial evaluator for row-at-a-time SQL serving.
//!
//! Fixed SQL plus static tables frozen at fit time are specialized, once, into
//! a native function whose only remaining input is the request row. The
//! contract has exactly two outcomes: serve bit-for-bit identical to DuckDB, or
//! refuse at build time with a named error. There is no third mode.
//!
//! * `duckdb` — `DuckDBInferFn`, the Python boundary.
//! * `specializer` — frontend, IR, verifier, and the interpreter/cranelift
//!   backends.
//!
//! The remaining modules are deliberately semantics-free and public because
//! `sql-transform` links this crate to share them: the error type, the type
//! vocabulary, and the Python schema/model marshalling.

use pyo3::prelude::*;

mod duckdb;
pub mod error;
pub mod schema;
pub mod specializer;
pub mod types;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<duckdb::DuckDBInferFn>()?;
    // Lets benchmarks refuse an unoptimized build (a `maturin develop` debug
    // .pyd shadowing the release wheel once inflated engine rows ~5x).
    m.add(
        "BUILD_PROFILE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
