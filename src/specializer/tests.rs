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

// -------------------------------------------------------- 3VL (stretch 2) --

fn c_i1v(vals: &[Option<bool>]) -> super::exec::ColData {
    super::exec::testutil::c_i1(vals)
}

/// Kleene truth tables through nullable comparisons: p = (a > 0), q = (b > 0)
/// with NULLs flowing in from the columns.
#[test]
fn kleene_and_or_truth_tables() {
    let schema = cols(&[("a", Ty::I64, true), ("b", Ty::I64, true)]);
    // rows: (T,T) (T,F) (T,N) (F,N) (N,N) (F,F)
    let a = c_i64(&[Some(1), Some(1), Some(1), Some(-1), None, Some(-1)]);
    let b = c_i64(&[Some(1), Some(-1), None, None, None, Some(-1)]);
    let got = run_sql(
        "SELECT (a > 0) AND (b > 0) AS x, (a > 0) OR (b > 0) AS y FROM __THIS__",
        &schema,
        batch(6, vec![a, b]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["true", "true"],
            &["false", "true"],
            &["NULL", "true"],    // T AND N = N ; T OR N = T
            &["false", "NULL"],   // F AND N = F ; F OR N = N
            &["NULL", "NULL"],
            &["false", "false"],
        ])
    );
}

#[test]
fn not_and_is_null() {
    let schema = cols(&[("b", Ty::I64, true)]);
    let got = run_sql(
        "SELECT NOT (b > 0) AS n, b IS NULL AS isn, b IS NOT NULL AS notn FROM __THIS__",
        &schema,
        batch(3, vec![c_i64(&[Some(1), Some(-1), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["false", "false", "true"],
            &["true", "false", "true"],
            &["NULL", "true", "false"],
        ])
    );
}

/// The eager-evaluation hazard: an untaken CASE branch containing a trapping
/// op (`%` by zero) must NOT trap — branches run only on their taken path.
#[test]
fn case_guards_trapping_branches() {
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, false)]);
    let got = run_sql(
        "SELECT CASE WHEN b <> 0 THEN a % b ELSE -1 END AS r FROM __THIS__",
        &schema,
        batch(2, vec![c_i64(&[Some(7), Some(9)]), c_i64(&[Some(4), Some(0)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["3"], &["-1"]]));
}

#[test]
fn case_forms_null_conditions_and_type_unification() {
    let schema = cols(&[("a", Ty::I64, true)]);
    // Searched, no ELSE -> NULL; int/float branch unification -> DOUBLE
    // (measured: CASE WHEN 1=0 THEN 1/0 ELSE -1 END -> -1.0).
    let got = run_sql(
        "SELECT CASE WHEN a > 1 THEN 1 WHEN a = 1 THEN 2.5 END AS u, \
         CASE a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'many' END AS s \
         FROM __THIS__",
        &schema,
        batch(4, vec![c_i64(&[Some(1), Some(2), Some(9), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["2.5", "one"],
            &["1.0", "two"],
            &["1.0", "many"],
            // NULL operand: simple-form conditions are `a = v` -> NULL,
            // never TRUE -> ELSE; searched arms also never TRUE -> NULL.
            &["NULL", "many"],
        ])
    );
}

#[test]
fn cast_matrix() {
    let schema = cols(&[("s", Ty::Str, false), ("f", Ty::F64, false), ("i", Ty::I64, false)]);
    let input = || {
        batch(
            1,
            vec![
                super::exec::testutil::c_str(&[Some(" 5")]),
                c_f64(&[Some(-2.5)]),
                c_i64(&[Some(2)]),
            ],
        )
    };
    let got = run_sql(
        "SELECT s::BIGINT AS a, f::BIGINT AS b, i::BOOLEAN AS c, \
         (i - 2)::BOOLEAN AS d, true::VARCHAR AS e, f::VARCHAR AS g, \
         TRY_CAST('x' AS BIGINT) AS h, TRY_CAST(1e19 AS BIGINT) AS j \
         FROM __THIS__",
        &schema,
        input(),
    )
    .unwrap();
    // ' 5' trims (DuckDB CAST); -2.5 rounds half-away to -3; 2 -> true,
    // 0 -> false; TRY_CAST failures -> NULL.
    assert_eq!(
        got,
        rows(&[&["5", "-3", "true", "false", "true", "-2.5", "NULL", "NULL"]])
    );
}

#[test]
fn cast_failure_traps_and_null_never_traps() {
    let schema = cols(&[("s", Ty::Str, true)]);
    // NULL input: CAST(NULL) is NULL, no trap.
    let got = run_sql(
        "SELECT s::BIGINT AS n FROM __THIS__",
        &schema,
        batch(1, vec![super::exec::testutil::c_str(&[None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL"]]));
    // Real string that does not parse: trap.
    let err = run_sql(
        "SELECT s::BIGINT AS n FROM __THIS__",
        &schema,
        batch(1, vec![super::exec::testutil::c_str(&[Some("abc")])]),
    )
    .unwrap_err();
    assert!(err.contains("Conversion Error"), "wrong trap: {err}");
}

#[test]
fn null_literal_in_context() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a + NULL AS x, a = NULL AS y, NULL IS NULL AS z, \
         CAST(NULL AS BIGINT) AS w, CASE WHEN a > 0 THEN NULL ELSE 5 END AS v \
         FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(3)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL", "NULL", "true", "NULL", "NULL"]]));
}

#[test]
fn where_with_kleene_and_case() {
    let schema = cols(&[("a", Ty::I64, true), ("b", Ty::I64, false)]);
    // NULL AND true -> NULL -> dropped; CASE inside WHERE.
    let got = run_sql(
        "SELECT b FROM __THIS__ WHERE (a > 0) AND (CASE WHEN b > 10 THEN true ELSE b > 2 END)",
        &schema,
        batch(
            4,
            vec![
                c_i64(&[Some(1), Some(1), None, Some(1)]),
                c_i64(&[Some(20), Some(1), Some(20), Some(3)]),
            ],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["20"], &["3"]]));
}

#[test]
fn case_heavy_program_is_canonical() {
    // CASE lowering mints join-param ids before later blocks' instructions;
    // prepare() canonicalizes, so exact round-trip equality holds anyway.
    let schema = cols(&[("a", Ty::I64, true)]);
    let p = prepare(
        "SELECT CASE WHEN a > 0 THEN TRY_CAST(a::VARCHAR AS BIGINT) ELSE a % 2 END AS r \
         FROM __THIS__ WHERE a IS NOT NULL",
        &schema,
    )
    .unwrap();
    let text = super::ir::print::print(&p);
    assert_eq!(parse(&text).unwrap(), p, "prepared program is not canonical:\n{text}");
}

#[test]
fn bool_column_directly_in_where() {
    let schema = cols(&[("ok", Ty::I1, true), ("v", Ty::I64, false)]);
    let got = run_sql(
        "SELECT v FROM __THIS__ WHERE ok",
        &schema,
        batch(3, vec![c_i1v(&[Some(true), Some(false), None]), c_i64(&[Some(1), Some(2), Some(3)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1"]]));
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
