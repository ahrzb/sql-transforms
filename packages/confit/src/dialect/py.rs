//! Python boundary for the dialect plan — enough surface for the corpus
//! gate (tests/test_dialect_corpus_gate.py) and fit-side experimentation.
//!
//! Catalog wire form: `[(table, [(col, duckdb_type, nullable), ...]), ...]`
//! — what a `DESCRIBE` walk produces, no JSON in between. Errors cross as
//! `ValueError` carrying the [`DialectError`] display text, so Python-side
//! classification can key on the `unsupported:` / `bind error:` prefixes,
//! the same words the specializer boundary uses.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::plan::{Catalog, ColDef, Table};
use super::{duckdb, text, DialectError};

type PyCatalog = Vec<(String, Vec<(String, String, bool)>)>;

fn build_catalog(tables: PyCatalog) -> Result<Catalog, DialectError> {
    let mut out = Vec::new();
    for (name, cols) in tables {
        let mut defs = Vec::new();
        for (col, ty, nullable) in cols {
            defs.push(ColDef {
                name: col,
                ty: super::ty::DTy::from_duckdb(&ty)?,
                nullable,
            });
        }
        out.push(Table { name, cols: defs });
    }
    Ok(Catalog { tables: out })
}

fn err(e: DialectError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// DuckDB SQL → canonical plan text (bound, typed, verified).
#[pyfunction]
pub fn dialect_parse(sql: &str, tables: PyCatalog) -> PyResult<String> {
    let cat = build_catalog(tables).map_err(err)?;
    let rel = duckdb::parse_sql(sql, &cat).map_err(err)?;
    Ok(text::print(&rel))
}

/// Canonical plan text → SQL in the target dialect. v0 targets: "duckdb".
#[pyfunction]
pub fn dialect_print(plan_text: &str, target: &str, tables: PyCatalog) -> PyResult<String> {
    let cat = build_catalog(tables).map_err(err)?;
    let rel = text::parse(plan_text).map_err(err)?;
    match target {
        "duckdb" => duckdb::print_sql(&rel, &cat).map_err(err),
        other => Err(PyValueError::new_err(format!(
            "unsupported: print target {other} (phase 3+: spark, bigquery)"
        ))),
    }
}
