//! End-to-end stretch-1 tests: SQL text -> prepare -> interpreter oracle.
//! Expected values follow the DuckDB pins measured 2026-07-26 (`/` is float
//! division, `%` stays integral, overflow traps).

use super::exec::interp::compile;
use super::exec::testutil::{batch, c_f64, c_i64, c_str, rows, run_snapshot};
use super::exec::{KeyBits, ScalarVal, StaticData};
use super::ir::{parse::parse, print::print, Col, ColTy, Ty};
use super::plan::StaticTable;
use super::{prepare, PrepareError};

fn cols(spec: &[(&str, Ty, bool)]) -> Vec<Col> {
    spec.iter()
        .map(|(n, t, null)| Col {
            name: n.to_string(),
            ty: ColTy {
                ty: *t,
                nullable: *null,
            },
        })
        .collect()
}

/// prepare() for the common no-statics case, unwrapped to the program.
fn prep(sql: &str, in_cols: &[Col]) -> Result<super::ir::Program, PrepareError> {
    prepare(sql, "__THIS__", in_cols, &[]).map(|p| p.program)
}

fn stat(name: &str, spec: &[(&str, Ty, bool)]) -> StaticTable {
    StaticTable {
        name: name.to_string(),
        cols: cols(spec),
    }
}

/// prepare + compile + run with static-table map data.
fn run_join(
    sql: &str,
    in_cols: &[Col],
    statics: &[StaticTable],
    data: Vec<StaticData>,
    input: super::exec::Batch,
) -> Result<Vec<Vec<String>>, String> {
    let p = prepare(sql, "__THIS__", in_cols, statics).map_err(|e| e.to_string())?;
    let f = compile(&p.program, data).map_err(|e| e.to_string())?;
    run_snapshot(&f, &input).map_err(|e| e.to_string())
}

fn run_sql(
    sql: &str,
    in_cols: &[Col],
    input: super::exec::Batch,
) -> Result<Vec<Vec<String>>, String> {
    let p = prep(sql, in_cols).map_err(|e| e.to_string())?;
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
        batch(
            2,
            vec![c_i64(&[Some(1), Some(2)]), c_i64(&[Some(10), None])],
        ),
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
    let p = prep("SELECT AGE FROM __this__", &schema).unwrap();
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
    assert!(
        err.contains("Overflow"),
        "expected an overflow trap, got: {err}"
    );
}

#[test]
fn prepared_programs_are_canonical_ir() {
    // prepare() output is verified AND round-trips through the text format —
    // the Builder assigns definition-ordered ids, so this holds exactly.
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::F64, true)]);
    let p = prep(
        "SELECT a * 2 AS x, b / a AS r FROM __THIS__ WHERE b > 0.0",
        &schema,
    )
    .unwrap();
    let text = print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "prepared program is not canonical:\n{text}"
    );
}

#[test]
fn column_cache_loads_once_per_block() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let p = prep("SELECT a + a AS d FROM __THIS__", &schema).unwrap();
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
            &["NULL", "true"],  // T AND N = N ; T OR N = T
            &["false", "NULL"], // F AND N = F ; F OR N = N
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
        batch(
            2,
            vec![c_i64(&[Some(7), Some(9)]), c_i64(&[Some(4), Some(0)])],
        ),
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
    let schema = cols(&[
        ("s", Ty::Str, false),
        ("f", Ty::F64, false),
        ("i", Ty::I64, false),
    ]);
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
    let p = prep(
        "SELECT CASE WHEN a > 0 THEN TRY_CAST(a::VARCHAR AS BIGINT) ELSE a % 2 END AS r \
         FROM __THIS__ WHERE a IS NOT NULL",
        &schema,
    )
    .unwrap();
    let text = super::ir::print::print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "prepared program is not canonical:\n{text}"
    );
}

#[test]
fn bool_column_directly_in_where() {
    let schema = cols(&[("ok", Ty::I1, true), ("v", Ty::I64, false)]);
    let got = run_sql(
        "SELECT v FROM __THIS__ WHERE ok",
        &schema,
        batch(
            3,
            vec![
                c_i1v(&[Some(true), Some(false), None]),
                c_i64(&[Some(1), Some(2), Some(3)]),
            ],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1"]]));
}

#[test]
fn star_expands_in_declared_order_with_exclude() {
    let schema = cols(&[
        ("a", Ty::I64, false),
        ("b", Ty::F64, true),
        ("s", Ty::Str, false),
    ]);
    let names = |p: &super::ir::Program| {
        p.out_cols
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>()
    };

    let p = prep("SELECT * FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["a", "b", "s"]);
    // Nullability and types ride along from the input schema.
    assert!(p.out_cols[1].ty.nullable && p.out_cols[1].ty.ty == Ty::F64);

    // EXCLUDE is case-insensitive (measured: DuckDB drops k for EXCLUDE (K)),
    // both paren and bare forms; qualified star + mixed items compose.
    let p = prep("SELECT * EXCLUDE (B) FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["a", "s"]);
    let p = prep("SELECT * EXCLUDE b FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["a", "s"]);
    let p = prep("SELECT __THIS__.*, a + 1 AS a2 FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["a", "b", "s", "a2"]);

    // Star + explicit same column: DuckDB-legal duplicate names stay our
    // clean unsupported (the output model needs unique fields).
    match prep("SELECT *, a FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(m)) => assert!(m.contains("duplicate output column")),
        other => panic!("wanted duplicate-name unsupported, got {other:?}"),
    }
    // EXCLUDE of an unknown column is a bind error, mirroring DuckDB's
    // binder message.
    match prep("SELECT * EXCLUDE (nope) FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("EXCLUDE list not found")),
        other => panic!("wanted EXCLUDE bind error, got {other:?}"),
    }
    // Excluding everything leaves an empty SELECT — bind error like DuckDB.
    match prep("SELECT * EXCLUDE (a, b, s) FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("empty")),
        other => panic!("wanted empty-list bind error, got {other:?}"),
    }
}

#[test]
fn star_over_joined_table_rejects_by_name() {
    // Wave-4: joined-table stars expand (key columns reconstruct from the
    // dynamic side); only DUPLICATE output names still reject — DuckDB
    // emits them verbatim, the typed output model cannot hold them.
    let schema = cols(&[("a", Ty::I64, false)]);
    let st = stat("dim", &[("id", Ty::I64, false), ("v", Ty::F64, false)]);
    for (sql, want) in [
        (
            "SELECT * FROM __THIS__ JOIN dim ON a = dim.id",
            vec!["a", "id", "v"],
        ),
        (
            "SELECT dim.* FROM __THIS__ JOIN dim ON a = dim.id",
            vec!["id", "v"],
        ),
    ] {
        let p = prepare(sql, "__THIS__", &schema, std::slice::from_ref(&st))
            .unwrap_or_else(|e| panic!("'{sql}': {e}"))
            .program;
        let names: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, want, "'{sql}'");
    }
    // A star that would produce duplicate names still rejects cleanly.
    // (The bare `id = dim.id` spelling is an ambiguity error in DuckDB
    // too — qualify the dynamic side.)
    let clash = cols(&[("id", Ty::I64, false)]);
    match prepare(
        "SELECT * FROM __THIS__ JOIN dim ON __THIS__.id = dim.id",
        "__THIS__",
        &clash,
        std::slice::from_ref(&st),
    ) {
        Err(PrepareError::Unsupported(m)) => {
            assert!(m.contains("duplicate output column"), "got '{m}'")
        }
        other => panic!("wanted duplicate-name unsup, got {:?}", other.err()),
    }
    // Qualified star over the ROW table under a join is fine.
    let p = prepare(
        "SELECT __THIS__.*, dim.v AS v FROM __THIS__ JOIN dim ON a = dim.id",
        "__THIS__",
        &schema,
        std::slice::from_ref(&st),
    )
    .unwrap();
    assert_eq!(p.program.out_cols.len(), 2);
}

#[test]
fn unsupported_constructs_are_named_cleanly() {
    let schema = cols(&[("a", Ty::I64, false)]);
    for (sql, needle) in [
        (
            "SELECT a FROM __THIS__ FULL OUTER JOIN t ON a = t.a",
            "join type",
        ),
        // Bare aggregates parse as plain function calls; they reject via the
        // function arm until the catalogue distinguishes aggregation.
        ("SELECT sum(a) FROM __THIS__", "aggregate function sum"),
        ("SELECT a FROM __THIS__ GROUP BY a", "aggregation"),
        // Wave-3 named rejects: regex family, grapheme-based reverse.
        ("SELECT regexp_matches('x', 'y') FROM __THIS__", "RE2"),
        ("SELECT reverse('abc') FROM __THIS__", "grapheme"),
        ("SELECT jaro_similarity('x', 'y') FROM __THIS__", "function"),
        // Star now expands; the still-unsupported star forms reject by name.
        ("SELECT * REPLACE (a + 1 AS a) FROM __THIS__", "REPLACE"),
        ("SELECT COLUMNS('a') FROM __THIS__", "function COLUMNS"),
        ("SELECT a FROM __THIS__ ORDER BY a", "ORDER BY"),
        ("SELECT a, a FROM __THIS__", "duplicate output column"),
        ("SELECT NULL FROM __THIS__", "NULL literal"),
        ("SELECT a FROM other_table", "must be the dynamic table"),
    ] {
        match prep(sql, &schema) {
            Err(PrepareError::Unsupported(msg)) => {
                assert!(
                    msg.contains(needle),
                    "'{sql}': wanted '{needle}' in '{msg}'"
                )
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
        ("SELECT a FROM __THIS__ WHERE s", "BOOLEAN"),
        ("SELECT a < s FROM __THIS__", "cannot compare"),
        (
            "SELECT a FROM __THIS__ JOIN t ON a = a",
            "was not provided as a static table",
        ),
    ] {
        match prep(sql, &schema) {
            Err(PrepareError::Bind(msg)) => {
                assert!(
                    msg.contains(needle),
                    "'{sql}': wanted '{needle}' in '{msg}'"
                )
            }
            Err(other) => panic!("'{sql}': wrong error kind: {other}"),
            Ok(_) => panic!("'{sql}': unexpectedly prepared"),
        }
    }
}

// ---------------------------------------------------------------- stretch 3:
// statics — equi-joins lower to probes, constants fold at prepare.

#[test]
fn inner_join_probe_hits_and_misses() {
    // k=2 has no build entry: INNER drops the row.
    let schema = cols(&[("k", Ty::I64, false)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("name", Ty::Str, false)]);
    let data = StaticData::Map(vec![
        (vec![KeyBits::I64(1)], vec![ScalarVal::Str("one".into())]),
        (vec![KeyBits::I64(3)], vec![ScalarVal::Str("three".into())]),
    ]);
    let got = run_join(
        "SELECT k, name FROM __THIS__ JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(3, vec![c_i64(&[Some(1), Some(2), Some(3)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1", "one"], &["3", "three"]]));
}

#[test]
fn left_join_miss_and_null_key_give_null_values() {
    // LEFT keeps every dynamic row; a miss (k=2) and a NULL key (a NULL
    // never equi-matches — DuckDB pin) both yield NULL value columns.
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("name", Ty::Str, false)]);
    let data = StaticData::Map(vec![(
        vec![KeyBits::I64(1)],
        vec![ScalarVal::Str("one".into())],
    )]);
    let got = run_join(
        "SELECT k, name FROM __THIS__ LEFT JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(3, vec![c_i64(&[Some(1), Some(2), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[&["1", "one"], &["2", "NULL"], &["NULL", "NULL"]])
    );
}

#[test]
fn inner_join_null_key_drops_the_row() {
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![(vec![KeyBits::I64(1)], vec![ScalarVal::I64(10)])]);
    let got = run_join(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(2, vec![c_i64(&[Some(1), None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["10"]]));
}

#[test]
fn join_key_promotion_int_dyn_against_float_col() {
    // dyn I64 key vs F64 static col: the dynamic side is promoted, the map
    // keys stay F64 (canonical bits).
    let schema = cols(&[("k", Ty::I64, false)]);
    let dim = stat("dim", &[("id", Ty::F64, false), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![(
        vec![KeyBits::F64(1f64.to_bits())],
        vec![ScalarVal::I64(10)],
    )]);
    let got = run_join(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(2, vec![c_i64(&[Some(1), Some(2)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["10"]]));
}

#[test]
fn join_key_promotion_float_dyn_against_int_col() {
    // dyn F64 key vs I64 static col: the map's key type is the expression's
    // F64 — the StaticSpec tells the materializer to convert the build side.
    let schema = cols(&[("k", Ty::F64, false)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("v", Ty::I64, false)]);
    let p = prepare(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        "__THIS__",
        &schema,
        &[dim],
    )
    .unwrap();
    let spec = &p.statics[0];
    assert_eq!(
        (spec.table.as_str(), &spec.key_cols[..], &spec.val_cols[..]),
        ("dim", &["id".to_string()][..], &["v".to_string()][..])
    );
    let data = StaticData::Map(vec![(
        vec![KeyBits::F64(2f64.to_bits())],
        vec![ScalarVal::I64(20)],
    )]);
    let f = compile(&p.program, vec![data]).unwrap();
    let got = run_snapshot(&f, &batch(2, vec![c_f64(&[Some(2.0), Some(2.5)])])).unwrap();
    assert_eq!(got, rows(&[&["20"]]));
}

#[test]
fn f64_probe_keys_canonicalize_negzero_and_nan() {
    // DuckDB pin: for DOUBLE `=`, 0.0 = -0.0 and NaN = NaN. Build keys are
    // canonicalized at compile (here -0.0 is stored), probes canonicalize
    // the searched value, so every row below hits.
    let schema = cols(&[("k", Ty::F64, false)]);
    let dim = stat("dim", &[("id", Ty::F64, false), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![
        (
            vec![KeyBits::F64((-0.0f64).to_bits())],
            vec![ScalarVal::I64(1)],
        ),
        (
            vec![KeyBits::F64((f64::NAN).to_bits() ^ 1)],
            vec![ScalarVal::I64(2)],
        ),
    ]);
    let got = run_join(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(3, vec![c_f64(&[Some(0.0), Some(-0.0), Some(f64::NAN)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1"], &["1"], &["2"]]));
}

#[test]
fn duplicate_build_keys_are_rejected_at_compile() {
    let schema = cols(&[("k", Ty::I64, false)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![
        (vec![KeyBits::I64(1)], vec![ScalarVal::I64(10)]),
        (vec![KeyBits::I64(1)], vec![ScalarVal::I64(11)]),
    ]);
    let err = run_join(
        "SELECT v FROM __THIS__ JOIN dim ON k = dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(1, vec![c_i64(&[Some(1)])]),
    )
    .unwrap_err();
    assert!(err.contains("duplicate map key"), "got: {err}");
}

#[test]
fn join_programs_are_canonical_ir() {
    let schema = cols(&[("k", Ty::I64, true), ("x", Ty::I64, false)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("name", Ty::Str, false)]);
    let p = prepare(
        "SELECT x, name FROM __THIS__ LEFT JOIN dim ON k = dim.id WHERE x > 0",
        "__THIS__",
        &schema,
        &[dim],
    )
    .unwrap()
    .program;
    let text = print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "join program is not canonical:\n{text}"
    );
}

#[test]
fn join_shape_errors() {
    // Wave-4: both former rejections now prepare — the key column
    // reconstructs from the dynamic side, and all-key (semi) joins probe
    // with zero value lanes.
    let schema = cols(&[("k", Ty::I64, false)]);
    let dim = stat("dim", &[("id", Ty::I64, false), ("name", Ty::Str, false)]);
    let p = prepare(
        "SELECT dim.id FROM __THIS__ JOIN dim ON k = dim.id",
        "__THIS__",
        &schema,
        &[dim],
    )
    .expect("key reconstruction prepares")
    .program;
    assert_eq!(p.out_cols[0].name, "id");
    let keys_only = stat("dim", &[("id", Ty::I64, false)]);
    prepare(
        "SELECT k FROM __THIS__ JOIN dim ON k = dim.id",
        "__THIS__",
        &schema,
        &[keys_only],
    )
    .expect("all-key join prepares");
}

#[test]
fn constant_arithmetic_folds_at_prepare() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let p = prep("SELECT 1 + 2 * 3 AS x FROM __THIS__", &schema).unwrap();
    let text = print(&p);
    assert!(
        text.contains("const.i64 7"),
        "expected folded constant 7:\n{text}"
    );
    assert!(
        !text.contains("iadd") && !text.contains("imul"),
        "expected no arithmetic:\n{text}"
    );
}

#[test]
fn dominating_constant_keeps_the_dynamic_side() {
    // fold() must not rewrite `false AND <dyn>` to false: the dynamic side
    // may trap (a % 0) and folding it away would change behavior.
    let schema = cols(&[("a", Ty::I64, false)]);
    let p = prep("SELECT false AND a % 0 = 0 AS x FROM __THIS__", &schema).unwrap();
    let text = print(&p);
    assert!(
        text.contains("irem"),
        "the trapping rem was folded away:\n{text}"
    );
}

// ---------------------------------------------------------------- stretch 4:
// the builtin catalogue, per the measured pins in
// docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md.

#[test]
fn string_builtins_end_to_end() {
    let schema = cols(&[("s", Ty::Str, true)]);
    let got = run_sql(
        "SELECT upper(s) AS u, lower('AbC') AS l, trim('  x  ') AS t, \
         ltrim('xxa', 'x') AS lt, rtrim('a  ') AS rt, \
         TRIM(LEADING 'x' FROM 'xax') AS tl, substr('hello', 2, 3) AS sub \
         FROM __THIS__",
        &schema,
        batch(2, vec![c_str(&[Some("ab"), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["AB", "abc", "x", "a", "a", "ax", "ell"],
            &["NULL", "abc", "x", "a", "a", "ax", "ell"],
        ])
    );
}

#[test]
fn one_arg_trim_removes_only_spaces() {
    // DuckDB pin: 1-arg trim removes ONLY 0x20 — tabs/newlines stay.
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT trim(' \t a \n ') AS t FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["\t a \n"]]));
}

#[test]
fn substr_window_arithmetic_via_sql() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT substr('hello', 0, 3) AS a, substr('hello', -2) AS b, \
         substr('hello', -10, 8) AS c, substr('hello', 1, -1) AS d, \
         substring('hello' FROM 2 FOR 2) AS e FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["he", "lo", "hello", "", "el"]]));
}

#[test]
fn concat_operator_always_concats_and_propagates_null() {
    // DuckDB pins: 1 || 'x' = '1x' (implicit VARCHAR cast), || propagates
    // NULL; CONCAT skips NULLs and never returns NULL.
    let schema = cols(&[("n", Ty::I64, true)]);
    let got = run_sql(
        "SELECT 1 || 'x' AS a, 'a' || NULL AS b, n || '!' AS c, \
         CONCAT('a', NULL, 1) AS d, CONCAT(n, '-') AS e FROM __THIS__",
        &schema,
        batch(2, vec![c_i64(&[Some(7), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["1x", "NULL", "7!", "a1", "7-"],
            &["1x", "NULL", "NULL", "a1", "-"],
        ])
    );
}

#[test]
fn abs_and_round_semantics() {
    // Pins: abs(i64) traps only on MIN; abs(-0.0) = +0.0; round is half
    // away from zero; integer round is identity (type preserved).
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT abs(-5) AS ai, abs(a) AS av, round(2.5) AS r1, \
         round(-2.5) AS r2, round(a) AS ri FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(-3)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["5", "3", "3.0", "-3.0", "-3"]]));
}

#[test]
fn abs_min_traps_like_duckdb() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let err = run_sql(
        "SELECT abs(a) AS x FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(i64::MIN)])]),
    )
    .unwrap_err();
    assert!(err.contains("Overflow"), "got: {err}");
}

#[test]
fn int_rem_by_zero_is_null_not_error() {
    // DuckDB pin (2026-07-26): 5 % 0 is NULL. MIN % -1 still traps.
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a % b AS r FROM __THIS__",
        &schema,
        batch(
            3,
            vec![
                c_i64(&[Some(5), Some(5), Some(-7)]),
                c_i64(&[Some(0), Some(3), Some(2)]),
            ],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL"], &["2"], &["-1"]]));
}

#[test]
fn float_rem_is_ieee() {
    let schema = cols(&[("x", Ty::F64, false), ("y", Ty::F64, false)]);
    let got = run_sql(
        "SELECT x % y AS r FROM __THIS__",
        &schema,
        batch(
            2,
            vec![
                c_f64(&[Some(-5.5), Some(5.0)]),
                c_f64(&[Some(2.5), Some(0.0)]),
            ],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["-0.5"], &["NaN"]]));
}

#[test]
fn nan_equals_nan_in_where() {
    // DuckDB DOUBLE order: nan = nan is TRUE, nan > 1 is TRUE.
    let schema = cols(&[("x", Ty::F64, false)]);
    let got = run_sql(
        "SELECT x FROM __THIS__ WHERE x = x AND x > 1.0",
        &schema,
        batch(2, vec![c_f64(&[Some(f64::NAN), Some(0.5)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NaN"]]));
}

#[test]
fn coalesce_binds_lazily_and_unifies() {
    let schema = cols(&[("n", Ty::I64, true)]);
    // The CAST in the untaken arm must not trap when n is non-NULL.
    let got = run_sql(
        "SELECT coalesce(n, CAST('nope' AS BIGINT)) AS a, \
         coalesce(NULL, 1, 2) AS b, coalesce(n, 2.5) AS c FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(4)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["4", "1", "4.0"]]));
}

#[test]
fn coalesce_taken_null_arm_falls_through_and_bad_cast_traps() {
    let schema = cols(&[("n", Ty::I64, true)]);
    let err = run_sql(
        "SELECT coalesce(n, CAST('nope' AS BIGINT)) AS a FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[None])]),
    )
    .unwrap_err();
    assert!(
        err.contains("cast") || err.contains("nope") || err.contains("convert"),
        "expected a cast trap, got: {err}"
    );
}

#[test]
fn nullif_compares_promoted_keeps_first_type() {
    let schema = cols(&[("x", Ty::F64, false)]);
    // nullif(1, 1.0) -> NULL (promoted compare); nullif(nan, nan) -> NULL
    // (DuckDB float order); nullif(3, 3.5) -> 3 INTEGER.
    let got = run_sql(
        "SELECT nullif(1, 1.0) AS a, nullif(x, x) AS b, nullif(3, 3.5) AS c \
         FROM __THIS__",
        &schema,
        batch(1, vec![c_f64(&[Some(f64::NAN)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL", "NULL", "3"]]));
}

#[test]
fn builtin_programs_are_canonical_ir() {
    let schema = cols(&[("s", Ty::Str, true), ("x", Ty::F64, false)]);
    let p = prep(
        "SELECT upper(trim(s)) AS u, substr(s, 2) AS m, s || 'x' AS c, \
         abs(x) AS a, round(x) AS r FROM __THIS__ WHERE x % 2.0 > 0.0",
        &schema,
    )
    .unwrap();
    let text = print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "builtin program is not canonical:\n{text}"
    );
}

// ------------------------------------------------- adversarial-fleet fixes:
// divergences found by the 6-agent differential probe (2026-07-26).

#[test]
fn null_divisor_rem_is_null_not_trap() {
    // The `b = 0` guard alone is NULL for a NULL divisor, which fell through
    // to irem on the garbage zero payload. IS NULL now shields it.
    let schema = cols(&[("a", Ty::I64, true), ("b", Ty::I64, true)]);
    let got = run_sql(
        "SELECT a % b AS r FROM __THIS__",
        &schema,
        batch(
            3,
            vec![
                c_i64(&[Some(7), None, Some(7)]),
                c_i64(&[None, Some(0), Some(3)]),
            ],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL"], &["NULL"], &["1"]]));
}

#[test]
fn traps_never_fire_under_a_null_flag() {
    // Computed garbage payloads are unbounded: (NULL + MAX) + MAX would
    // overflow its payload lane. Masking forces the default before any
    // trapping instruction. A real value still traps like DuckDB.
    let schema = cols(&[("a", Ty::I64, true)]);
    let sql = "SELECT a + 9223372036854775807 + 9223372036854775807 AS r FROM __THIS__";
    let got = run_sql(sql, &schema, batch(1, vec![c_i64(&[None])])).unwrap();
    assert_eq!(got, rows(&[&["NULL"]]));
    let err = run_sql(sql, &schema, batch(1, vec![c_i64(&[Some(1)])])).unwrap_err();
    assert!(err.contains("Overflow"), "got: {err}");
}

#[test]
fn numeric_conditions_coerce_to_bool() {
    // DuckDB pins: WHERE/AND/NOT/CASE take numerics — nonzero is true.
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a FROM __THIS__ WHERE a % 2 AND 1",
        &schema,
        batch(4, vec![c_i64(&[Some(1), Some(2), Some(3), Some(4)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1"], &["3"]]));
    let got = run_sql(
        "SELECT CASE WHEN 2 THEN 'a' ELSE 'b' END AS c, NOT 5 AS n FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["a", "false"]]));
}

#[test]
fn trim_default_set_is_unicode_zs() {
    // Adversarial census: the 1-arg trim set is exactly the Zs category —
    // NBSP and ideographic space go, tab and newline stay.
    let schema = cols(&[("s", Ty::Str, false)]);
    let got = run_sql(
        "SELECT trim(s) AS t FROM __THIS__",
        &schema,
        batch(2, vec![c_str(&[Some("\u{A0}a\u{3000}"), Some("\ta\n")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["a"], &["\ta\n"]]));
}

#[test]
fn substr_negative_length_slices_backwards() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT substr('hello', 3, -2) AS a, substr('hello', 6, -5) AS b, \
         substr('hello', 1, -1) AS c FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["he", "hello", ""]]));
}

#[test]
fn substr_range_guard_traps_like_duckdb() {
    let schema = cols(&[("a", Ty::I64, false)]);
    let err = run_sql(
        "SELECT substr('hello', 4294967296) AS x FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap_err();
    assert!(err.contains("offset outside"), "got: {err}");
}

#[test]
fn float_to_varchar_matches_duckdb_rendering() {
    // Pins: explicit exponent sign, two-digit minimum, lowercase nan.
    let schema = cols(&[("x", Ty::F64, false)]);
    let got = run_sql(
        "SELECT x || '' AS s FROM __THIS__",
        &schema,
        batch(
            4,
            vec![c_f64(&[Some(1e300), Some(1e-5), Some(f64::NAN), Some(2.5)])],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1e+300"], &["1e-05"], &["nan"], &["2.5"]]));
}

#[test]
fn rowid_and_lateral_alias_reject_cleanly() {
    let schema = cols(&[("a", Ty::I64, false)]);
    for (sql, needle) in [
        ("SELECT rowid FROM __THIS__", "rowid"),
        ("SELECT a % 2 AS k FROM __THIS__ WHERE k = 1", "lateral"),
    ] {
        match prep(sql, &schema) {
            Err(PrepareError::Unsupported(msg)) => {
                assert!(
                    msg.contains(needle),
                    "'{sql}': wanted '{needle}' in '{msg}'"
                )
            }
            other => panic!("'{sql}': wrong outcome: {:?}", other.err()),
        }
    }
}
