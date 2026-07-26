//! End-to-end stretch-1 tests: SQL text -> prepare -> interpreter oracle.
//! Expected values follow the DuckDB pins measured 2026-07-26 (`/` is float
//! division, `%` stays integral, overflow traps).

use super::exec::interp::compile;
use super::exec::testutil::{batch, c_f64, c_i64, rows, run_snapshot};
use super::ir::{parse::parse, print::print, Col, ColTy, Ty};
use super::{prepare, PrepareError};

fn cols(spec: &[(&str, Ty, bool)]) -> Vec<Col> {
    spec.iter()
        .map(|(n, t, null)| Col { name: n.to_string(), ty: ColTy { ty: *t, nullable: *null } })
        .collect()
}

fn run_sql(
    sql: &str,
    in_cols: &[Col],
    input: super::exec::Batch,
) -> Result<Vec<Vec<String>>, String> {
    let p = prepare(sql, in_cols).map_err(|e| e.to_string())?;
    let f = compile(&p, vec![]).map_err(|e| e.to_string())?;
    run_snapshot(&f, &input).map_err(|e| e.to_string())
}

#[test]
fn arithmetic_projection_end_to_end() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a + 1 AS x, a * 2 AS y, a % 2 AS m FROM __THIS__",
        &schema,
        batch(2, vec![c_i64(&[Some(4), Some(7)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["5", "8", "0"], &["8", "14", "1"]]));
}

#[test]
fn division_is_float_division() {
    // DuckDB pin: 5/2 = 2.5 DOUBLE.
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a / 2 AS h FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(5)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["2.5"]]));
}

#[test]
fn null_propagation_through_arithmetic() {
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, true)]);
    let got = run_sql(
        "SELECT a + b AS s FROM __THIS__",
        &schema,
        batch(2, vec![c_i64(&[Some(1), Some(2)]), c_i64(&[Some(10), None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["11"], &["NULL"]]));
}

#[test]
fn where_filters_and_null_predicate_drops() {
    // SQL WHERE keeps only TRUE: false drops, NULL drops.
    let schema = cols(&[("score", Ty::F64, true)]);
    let got = run_sql(
        "SELECT score FROM __THIS__ WHERE score > 0.5",
        &schema,
        batch(3, vec![c_f64(&[Some(0.9), Some(0.1), None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["0.9"]]));
}

#[test]
fn case_insensitive_bind_preserves_query_spelling() {
    let schema = cols(&[("age", Ty::I64, false)]);
    let p = prepare("SELECT AGE FROM __this__", &schema).unwrap();
    assert_eq!(p.out_cols[0].name, "AGE");
    let f = compile(&p, vec![]).unwrap();
    let got = run_snapshot(&f, &batch(1, vec![c_i64(&[Some(3)])])).unwrap();
    assert_eq!(got, rows(&[&["3"]]));
}

#[test]
fn unary_minus_and_literals() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT -a AS n, 1.5 AS f, 'x' AS s, true AS t FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(3)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["-3", "1.5", "x", "true"]]));
}

#[test]
fn integer_overflow_traps_like_duckdb() {
    // DuckDB pin: BIGINT overflow is an error, not a wrap.
    let schema = cols(&[("a", Ty::I64, false)]);
    let err = run_sql(
        "SELECT a + 1 AS x FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(i64::MAX)])]),
    )
    .unwrap_err();
    assert!(err.contains("overflow"), "expected an overflow trap, got: {err}");
}

#[test]
fn prepared_programs_are_canonical_ir() {
    // prepare() output is verified AND round-trips through the text format —
    // the Builder assigns definition-ordered ids, so this holds exactly.
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::F64, true)]);
    let p = prepare(
        "SELECT a * 2 AS x, b / a AS r FROM __THIS__ WHERE b > 0.0",
        &schema,
    )
    .unwrap();
    let text = print(&p);
    assert_eq!(parse(&text).unwrap(), p, "prepared program is not canonical:\n{text}");
}

#[test]
fn column_cache_loads_once_per_block() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let p = prepare("SELECT a + a AS d FROM __THIS__", &schema).unwrap();
    let text = print(&p);
    assert_eq!(
        text.matches("load in.a").count(),
        1,
        "column loaded more than once in one block:\n{text}"
    );
}

#[test]
fn unsupported_constructs_are_named_cleanly() {
    let schema = cols(&[("a", Ty::I64, false)]);
    for (sql, needle) in [
        ("SELECT a FROM __THIS__ JOIN t ON true", "JOIN"),
        // Bare aggregates parse as plain function calls; they reject via the
        // function arm until the catalogue distinguishes aggregation.
        ("SELECT sum(a) FROM __THIS__", "function sum"),
        ("SELECT a FROM __THIS__ GROUP BY a", "aggregation"),
        ("SELECT upper('x') FROM __THIS__", "function"),
        ("SELECT * FROM __THIS__", "star expansion"),
        ("SELECT a FROM __THIS__ WHERE a > 0 AND a < 9", "AND/OR"),
        ("SELECT CASE WHEN a > 0 THEN 1 END FROM __THIS__", "CASE"),
        ("SELECT a FROM __THIS__ ORDER BY a", "ORDER BY"),
        ("SELECT a, a FROM __THIS__", "duplicate output column"),
        ("SELECT NULL FROM __THIS__", "NULL literal"),
        ("SELECT a FROM other_table", "only the dynamic table"),
    ] {
        match prepare(sql, &schema) {
            Err(PrepareError::Unsupported(msg)) => {
                assert!(msg.contains(needle), "'{sql}': wanted '{needle}' in '{msg}'")
            }
            Err(other) => panic!("'{sql}': wrong error kind: {other}"),
            Ok(_) => panic!("'{sql}': unexpectedly prepared"),
        }
    }
}

#[test]
fn bind_errors_are_not_unsupported() {
    let schema = cols(&[("a", Ty::I64, false), ("s", Ty::Str, false)]);
    for (sql, needle) in [
        ("SELECT nope FROM __THIS__", "does not exist"),
        ("SELECT a + s FROM __THIS__", "numeric"),
        ("SELECT a FROM __THIS__ WHERE a + 1", "BOOLEAN"),
        ("SELECT a < s FROM __THIS__", "cannot compare"),
    ] {
        match prepare(sql, &schema) {
            Err(PrepareError::Bind(msg)) => {
                assert!(msg.contains(needle), "'{sql}': wanted '{needle}' in '{msg}'")
            }
            Err(other) => panic!("'{sql}': wrong error kind: {other}"),
            Ok(_) => panic!("'{sql}': unexpectedly prepared"),
        }
    }
}
