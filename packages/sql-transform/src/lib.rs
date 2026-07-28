//! Native row-at-a-time SQL interpretation for `sql_transform`.
//!
//! `datafusion` — `InferFn`, the engine differentially tested against
//! DataFusion. Full expression surface, transformer callouts, joins.
//!
//! The DuckDB engine no longer lives here: it is the `confit` package. This
//! crate links `confit` for the shared, semantics-free vocabulary only — the
//! error type, the type vocabulary, and the Python schema/model marshalling —
//! and keeps the value representation and lookup helpers that only this engine
//! uses.

use pyo3::prelude::*;

// confit's lib target is named `_engine` (maturin builds it as
// `confit/_engine.pyd`), so the crate is renamed here to read as itself.
extern crate _engine as confit;

// Re-exported under the names the engine modules already use, so `crate::error`
// and friends keep resolving after the split.
pub use confit::{error, schema, types};

mod datafusion;
mod lookup;
mod value;

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<datafusion::InferFn>()?;
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
