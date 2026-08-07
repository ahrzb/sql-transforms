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

    // Star + explicit same column: duplicates get DuckDB's boundary rename
    // (wave-5 pins: own-name _N, smallest free, case-insensitive check).
    let p = prep("SELECT *, a FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["a", "b", "s", "a_1"]);
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
    // A star producing duplicate names now renames per the wave-5 dup-name
    // contract (DuckDB's own boundary rename). (The bare `id = dim.id`
    // spelling is an ambiguity error in DuckDB too — qualify.)
    let clash = cols(&[("id", Ty::I64, false)]);
    let p = prepare(
        "SELECT * FROM __THIS__ JOIN dim ON __THIS__.id = dim.id",
        "__THIS__",
        &clash,
        std::slice::from_ref(&st),
    )
    .unwrap()
    .program;
    let names: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["id", "id_1", "v"]);
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
        // Scalar regexp serves since wave B; list-valued forms stay named.
        (
            "SELECT regexp_extract_all('x', 'y') FROM __THIS__",
            "list-valued",
        ),
        ("SELECT jaro_similarity('x', 'y') FROM __THIS__", "function"),
        // Bare COLUMNS expands since wave B; expression forms stay named.
        ("SELECT COLUMNS('a') + 1 FROM __THIS__", "COLUMNS"),
        ("SELECT a FROM __THIS__ ORDER BY a", "ORDER BY"),
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

// ------------------------------------------------------- params-join wiring
// (DRAFT-22 step 3): IS NOT DISTINCT FROM join keys — NULL is an ordinary
// key value, encoded as a (validity i1, masked payload) key PAIR on both
// sides — and the keyless always-true LEFT JOIN against a one-row static.

/// INDF i64 key entry: (valid, payload) — NULL is (false, type default).
fn indf_key(v: Option<i64>) -> Vec<KeyBits> {
    vec![KeyBits::I1(v.is_some()), KeyBits::I64(v.unwrap_or(0))]
}

#[test]
fn indf_left_join_null_key_joins_null_bucket() {
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, true), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![
        (indf_key(Some(1)), vec![ScalarVal::I64(10)]),
        (indf_key(None), vec![ScalarVal::I64(99)]),
    ]);
    let got = run_join(
        "SELECT k, v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(3, vec![c_i64(&[Some(1), None, Some(2)])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[&["1", "10"], &["NULL", "99"], &["2", "NULL"]])
    );
}

#[test]
fn indf_inner_join_null_key_hits() {
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, true), ("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![
        (indf_key(Some(1)), vec![ScalarVal::I64(10)]),
        (indf_key(None), vec![ScalarVal::I64(99)]),
    ]);
    let got = run_join(
        "SELECT v FROM __THIS__ JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        &schema,
        &[dim],
        vec![data],
        batch(3, vec![c_i64(&[Some(1), None, Some(2)])]),
    )
    .unwrap();
    // k=2 misses (INNER drops); NULL hits the NULL bucket.
    assert_eq!(got, rows(&[&["10"], &["99"]]));
}

#[test]
fn indf_multi_key_conjunction_mixed_with_eq() {
    // `=` key (NULL never matches) AND INDF key (NULL matches NULL).
    let schema = cols(&[("a", Ty::I64, true), ("b", Ty::Str, true)]);
    let dim = stat(
        "dim",
        &[
            ("x", Ty::I64, false),
            ("y", Ty::Str, true),
            ("v", Ty::I64, false),
        ],
    );
    let entry = |x: i64, y: Option<&str>, v: i64| {
        (
            vec![
                KeyBits::I64(x),
                KeyBits::I1(y.is_some()),
                KeyBits::Str(y.unwrap_or("").to_string()),
            ],
            vec![ScalarVal::I64(v)],
        )
    };
    let data = StaticData::Map(vec![
        entry(1, Some("u"), 10),
        entry(1, None, 11),
        entry(2, Some("w"), 20),
    ]);
    let got = run_join(
        "SELECT v FROM __THIS__ LEFT JOIN dim \
         ON a = dim.x AND (b IS NOT DISTINCT FROM dim.y)",
        &schema,
        &[dim],
        vec![data],
        batch(
            4,
            vec![
                c_i64(&[Some(1), Some(1), None, Some(2)]),
                c_str(&[Some("u"), None, None, None]),
            ],
        ),
    )
    .unwrap();
    // (1,'u') hits 10; (1,NULL) hits the NULL bucket 11; (NULL,NULL)
    // misses — the `=` key never matches on NULL; (2,NULL) misses.
    assert_eq!(got, rows(&[&["10"], &["11"], &["NULL"], &["NULL"]]));
}

#[test]
fn keyless_left_join_on_one_eq_one() {
    let schema = cols(&[("k", Ty::I64, false)]);
    let params = stat("p", &[("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![(vec![], vec![ScalarVal::I64(7)])]);
    let got = run_join(
        "SELECT k, v FROM __THIS__ LEFT JOIN p ON ((1 = 1))",
        &schema,
        &[params],
        vec![data],
        batch(2, vec![c_i64(&[Some(1), Some(2)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1", "7"], &["2", "7"]]));
}

#[test]
fn keyless_join_two_row_build_refuses() {
    let schema = cols(&[("k", Ty::I64, false)]);
    let params = stat("p", &[("v", Ty::I64, false)]);
    let data = StaticData::Map(vec![
        (vec![], vec![ScalarVal::I64(7)]),
        (vec![], vec![ScalarVal::I64(8)]),
    ]);
    let err = run_join(
        "SELECT v FROM __THIS__ LEFT JOIN p ON ((1 = 1))",
        &schema,
        &[params],
        vec![data],
        batch(1, vec![c_i64(&[Some(1)])]),
    )
    .unwrap_err();
    assert!(err.contains("duplicate map key"), "got: {err}");
}

#[test]
fn indf_and_keyless_joins_are_map_shape_provable() {
    // The serving_sql shape (DRAFT-22): both param joins are LEFT, so the
    // exactly-one-row proof holds and shape='map' builds.
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, true), ("v", Ty::I64, true)]);
    let params = stat("p", &[("w", Ty::I64, false)]);
    let p = prepare(
        "SELECT (v + w) AS z FROM __THIS__ \
         LEFT JOIN dim ON (k IS NOT DISTINCT FROM dim.id) \
         LEFT JOIN p ON ((1 = 1))",
        "__THIS__",
        &schema,
        &[dim, params],
    )
    .unwrap();
    assert_eq!(p.one_row_blocker, None);
    // The StaticSpec names which key columns join under INDF, so the
    // materializer knows to KEEP NULL-key rows as (false, default) pairs.
    assert_eq!(p.statics[0].key_indf, vec![true]);
    assert!(p.statics[1].key_indf.is_empty());
    // Chained end to end. `v` is nullable, so values are (validity,
    // payload) pairs (TASK-55 flattening).
    let data0 = StaticData::Map(vec![
        (
            indf_key(Some(1)),
            vec![ScalarVal::I1(true), ScalarVal::I64(10)],
        ),
        (
            indf_key(None),
            vec![ScalarVal::I1(true), ScalarVal::I64(99)],
        ),
    ]);
    let data1 = StaticData::Map(vec![(vec![], vec![ScalarVal::I64(5)])]);
    let f = compile(&p.program, vec![data0, data1]).unwrap();
    let got = run_snapshot(&f, &batch(3, vec![c_i64(&[Some(1), None, Some(2)])])).unwrap();
    assert_eq!(got, rows(&[&["15"], &["104"], &["NULL"]]));
}

#[test]
fn indf_under_shape_many_refuses_by_name() {
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, true), ("v", Ty::I64, false)]);
    match super::prepare_opaque(
        "SELECT v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
        "__THIS__",
        &schema,
        &[],
        &[],
        &[dim],
        true,

        &[],
        &[],
    ) {
        Err(PrepareError::Unsupported(m)) => {
            assert!(m.contains("IS NOT DISTINCT FROM"), "got: {m}")
        }
        Err(other) => panic!("wrong error kind: {other}"),
        Ok(_) => panic!("INDF under shape='many' unexpectedly prepared"),
    }
}

#[test]
fn indf_join_programs_are_canonical_ir() {
    let schema = cols(&[("k", Ty::I64, true)]);
    let dim = stat("dim", &[("id", Ty::I64, true), ("v", Ty::I64, false)]);
    let p = prepare(
        "SELECT v FROM __THIS__ LEFT JOIN dim ON k IS NOT DISTINCT FROM dim.id",
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
        "INDF join program is not canonical:\n{text}"
    );
}

// ------------------------------------------------------------ UDF externs
// (DRAFT-22 step 2): declared UDFs bind as extern calls; unknown functions
// keep the named refusal; width-k calls are bare-item-only and expand to a
// whole-validity lane plus k component lanes for the output boundary.

fn udf(name: &str, params: &[Ty], rets: &[Ty]) -> super::ir::ExternSpec {
    super::ir::ExternSpec {
        name: name.to_string(),
        params: params.to_vec(),
        rets: rets.to_vec(),
        ret_names: Vec::new(),
    }
}

fn imp(
    name: &str,
    f: impl Fn(&[Option<ScalarVal>]) -> Result<Option<Vec<Option<ScalarVal>>>, String> + 'static,
) -> super::exec::ExternImpl {
    super::exec::ExternImpl {
        name: name.into(),
        fun: Box::new(f),
    }
}

fn prep_udfs(
    sql: &str,
    in_cols: &[Col],
    statics: &[StaticTable],
    udfs: &[super::ir::ExternSpec],
) -> Result<super::Prepared, PrepareError> {
    super::prepare_opaque(sql, "__THIS__", in_cols, &[], &[], statics, false, udfs, &[])
}

// ------------------------------------------------------- tree_predict --

fn model(name: &str, features: &[&str]) -> super::plan::ModelTable {
    super::plan::ModelTable {
        name: name.to_string(),
        features: features.iter().map(|s| s.to_string()).collect(),
    }
}

fn prep_models(
    sql: &str,
    in_cols: &[Col],
    statics: &[StaticTable],
    models: &[super::plan::ModelTable],
) -> Result<super::Prepared, PrepareError> {
    super::prepare_opaque(sql, "__THIS__", in_cols, &[], &[], statics, false, &[], models)
}

/// The serving shape: the group's model id comes from a params join, the
/// features from a struct whose field NAMES are checked and reordered into
/// the model's declared order — the one mistake a fitted model cannot catch
/// for itself.
#[test]
fn tree_predict_resolves_struct_features_by_name() {
    let schema = cols(&[
        ("g", Ty::Str, true),
        ("price", Ty::F64, false),
        ("sqft", Ty::I64, false),
    ]);
    let params = stat("p0", &[("g", Ty::Str, true), ("est", Ty::I64, true)]);
    // Fields deliberately given in the WRONG order, and sqft as an i64.
    let p = prep_models(
        "SELECT tree_predict('trees', p.est, struct_pack(sqft := t.sqft, price := t.price)) AS z \
         FROM __THIS__ AS t LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        &schema,
        &[params],
        &[model("trees", &["price", "sqft"])],
    )
    .unwrap();
    assert_eq!(p.models, vec!["trees".to_string()]);
    // The model static is appended AFTER the join static, so the probe's @0
    // is untouched.
    let text = print(&p.program);
    assert!(
        text.contains("static @0: map(") && text.contains("static @1: model<tree_ensemble(2)>"),
        "model static must follow the join static:\n{text}"
    );
    // Exactly one predict, against @1.
    assert_eq!(text.matches("predict @1,").count(), 1, "{text}");
    assert_eq!(feature_position_of_itof(&text), 1, "sqft is declared second:\n{text}");

    // Same SQL, same struct, only the DECLARED order flipped: the operand
    // order must follow the declaration, not the call site. That is the
    // whole point of naming the features.
    let schema = cols(&[
        ("g", Ty::Str, true),
        ("price", Ty::F64, false),
        ("sqft", Ty::I64, false),
    ]);
    let params = stat("p0", &[("g", Ty::Str, true), ("est", Ty::I64, true)]);
    let p = prep_models(
        "SELECT tree_predict('trees', p.est, struct_pack(sqft := t.sqft, price := t.price)) AS z \
         FROM __THIS__ AS t LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        &schema,
        &[params],
        &[model("trees", &["sqft", "price"])],
    )
    .unwrap();
    let text = print(&p.program);
    assert_eq!(feature_position_of_itof(&text), 0, "sqft is declared first:\n{text}");
}

/// Which feature slot of the single `predict` carries the `itof` result.
/// sqft is the only i64 feature, so the promotion identifies it positionally
/// without parsing the whole program.
fn feature_position_of_itof(text: &str) -> usize {
    let itof = text
        .lines()
        .find_map(|l| {
            let (dst, rest) = l.trim().split_once(" = ")?;
            rest.starts_with("itof ").then(|| dst.trim().to_string())
        })
        .expect("an itof promoting the i64 feature");
    let line = text
        .lines()
        .find(|l| l.contains("predict @1,"))
        .expect("a predict line");
    let ops: Vec<&str> = line
        .split_once("predict @1, ")
        .expect("predict operands")
        .1
        .split(", ")
        .map(|s| s.trim())
        .collect();
    // ops[0] is the model id; features follow.
    ops[1..]
        .iter()
        .position(|o| *o == itof)
        .unwrap_or_else(|| panic!("itof result {itof} is not a feature operand of: {line}"))
}

/// A NULL feature is not a NULL result — the model answers for missing.
/// A NULL model id is, because an unseen group has no model at all.
#[test]
fn only_the_model_id_makes_a_prediction_null() {
    let schema = cols(&[("price", Ty::F64, true), ("id", Ty::I64, false)]);
    let p = prep_models(
        "SELECT tree_predict('trees', id, struct_pack(price := price)) AS z FROM __THIS__",
        &schema,
        &[],
        &[model("trees", &["price"])],
    )
    .unwrap();
    // id is NOT NULL, price IS — the output column must still be non-null.
    assert!(
        !p.program.out_cols[0].ty.nullable,
        "a nullable feature must not make the prediction nullable:\n{}",
        print(&p.program)
    );
    // The NULL feature becomes NaN through an explicit select.
    let text = print(&p.program);
    assert!(text.contains("const.f64 nan") && text.contains("select "), "{text}");

    let schema = cols(&[("price", Ty::F64, false), ("id", Ty::I64, true)]);
    let p = prep_models(
        "SELECT tree_predict('trees', id, struct_pack(price := price)) AS z FROM __THIS__",
        &schema,
        &[],
        &[model("trees", &["price"])],
    )
    .unwrap();
    assert!(
        p.program.out_cols[0].ty.nullable,
        "a nullable model id must make the prediction nullable"
    );
    // No feature is nullable here, so no NULL-to-NaN select is emitted.
    assert!(!print(&p.program).contains("const.f64 nan"));
}

#[test]
fn tree_predict_refuses_by_name() {
    let schema = cols(&[("price", Ty::F64, false), ("s", Ty::Str, false), ("id", Ty::I64, false)]);
    let models = [model("trees", &["price", "sqft"])];
    let one = [model("trees", &["price"])];
    let cases: &[(&str, &str, &[super::plan::ModelTable])] = &[
        (
            "was not provided to prepare",
            "SELECT tree_predict('nope', id, struct_pack(price := price)) FROM __THIS__",
            &one,
        ),
        (
            "no feature 'weight'",
            "SELECT tree_predict('trees', id, struct_pack(weight := price)) FROM __THIS__",
            &one,
        ),
        (
            "missing feature(s): sqft",
            "SELECT tree_predict('trees', id, struct_pack(price := price)) FROM __THIS__",
            &models,
        ),
        (
            "given twice",
            "SELECT tree_predict('trees', id, struct_pack(price := price, PRICE := price)) \
             FROM __THIS__",
            &one,
        ),
        (
            "want a number",
            "SELECT tree_predict('trees', id, struct_pack(price := s)) FROM __THIS__",
            &one,
        ),
        (
            "must be a struct_pack",
            "SELECT tree_predict('trees', id, price) FROM __THIS__",
            &one,
        ),
        (
            "exactly 3 arguments",
            "SELECT tree_predict('trees', id) FROM __THIS__",
            &one,
        ),
        (
            "non-constant tree_predict model set",
            "SELECT tree_predict(s, id, struct_pack(price := price)) FROM __THIS__",
            &one,
        ),
    ];
    for (needle, sql, ms) in cases {
        let e = prep_models(sql, &schema, &[], ms)
            .err()
            .unwrap_or_else(|| panic!("accepted a bad call site ({needle}): {sql}"));
        let msg = e.to_string();
        assert!(msg.contains(needle), "wanted '{needle}', got '{msg}' for: {sql}");
    }
}

#[test]
fn udf_call_serves_the_marginalizer_shape() {
    // The DRAFT-22 serving_sql shape: params join by INDF, the transformer
    // call takes the joined instance id plus a row feature (i64 age
    // promoted to the declared f64 like DuckDB's implicit cast).
    let schema = cols(&[("g", Ty::Str, true), ("age", Ty::I64, true)]);
    let params = stat("p0", &[("g", Ty::Str, true), ("est", Ty::I64, true)]);
    let p = prep_udfs(
        "SELECT (tf0(p.est, t.age) + 1.0) AS z FROM __THIS__ AS t \
         LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        &schema,
        &[params],
        &[udf("tf0", &[Ty::I64, Ty::F64], &[Ty::F64])],
    )
    .unwrap();
    assert_eq!(p.one_row_blocker, None);
    let data = StaticData::Map(vec![
        (
            vec![KeyBits::I1(true), KeyBits::Str("de".into())],
            vec![ScalarVal::I1(true), ScalarVal::I64(0)],
        ),
        (
            vec![KeyBits::I1(false), KeyBits::Str(String::new())],
            vec![ScalarVal::I1(true), ScalarVal::I64(1)],
        ),
    ]);
    // tf0(id, x) = x * (id + 1); NULL id or x -> NULL (the callable's
    // convention, not the engine's).
    let tf0 = imp("tf0", |args| {
        let (Some(ScalarVal::I64(id)), Some(ScalarVal::F64(x))) = (&args[0], &args[1]) else {
            return Ok(None);
        };
        Ok(Some(vec![Some(ScalarVal::F64(x * (*id as f64 + 1.0)))]))
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![data], vec![tf0]).unwrap();
    let input = batch(
        4,
        vec![
            c_str(&[Some("de"), None, Some("fr"), Some("de")]),
            c_i64(&[Some(10), Some(20), Some(30), None]),
        ],
    );
    // de: id 0 -> 10*1+1 = 11; NULL g: id 1 -> 20*2+1 = 41; fr: miss ->
    // NULL id -> NULL; de with NULL age -> NULL.
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["11.0"], &["41.0"], &["NULL"], &["NULL"]])
    );
}

#[test]
fn unknown_function_refusal_survives_udf_declarations() {
    let schema = cols(&[("x", Ty::F64, false)]);
    let err = prep_udfs(
        "SELECT nope(x) AS z FROM __THIS__",
        &schema,
        &[],
        &[udf("tf0", &[Ty::F64], &[Ty::F64])],
    )
    .unwrap_err();
    match err {
        PrepareError::Unsupported(m) => {
            assert!(m.contains("nope"), "got: {m}")
        }
        other => panic!("wrong error kind: {other}"),
    }
}

#[test]
fn udf_arity_and_type_mismatches_refuse_by_name() {
    let schema = cols(&[("x", Ty::F64, false), ("s", Ty::Str, false)]);
    let err = prep_udfs(
        "SELECT tf0(x, x) AS z FROM __THIS__",
        &schema,
        &[],
        &[udf("tf0", &[Ty::F64], &[Ty::F64])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Bind(m) if m.contains("tf0") && m.contains("argument")),
        "got: {err}"
    );
    let err = prep_udfs(
        "SELECT tf0(s) AS z FROM __THIS__",
        &schema,
        &[],
        &[udf("tf0", &[Ty::F64], &[Ty::F64])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Bind(m) if m.contains("tf0") && m.contains("str")),
        "got: {err}"
    );
}

#[test]
fn wide_udf_bare_item_expands_to_output_lanes() {
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT tf2(x) AS z, x AS orig FROM __THIS__",
        &schema,
        &[],
        &[udf("tf2", &[Ty::F64], &[Ty::F64, Ty::F64])],
    )
    .unwrap();
    // One wide output: field z assembles from out cols [0..3) —
    // whole-validity lane + 2 nullable component lanes.
    assert_eq!(p.wide_outputs.len(), 1);
    let w = &p.wide_outputs[0];
    assert_eq!((w.name.as_str(), w.first, w.width), ("z", 0, 2));
    assert_eq!(p.program.out_cols.len(), 4);
    assert_eq!(p.program.out_cols[0].ty.ty, Ty::I1);
    assert!(p.program.out_cols[1].ty.nullable);
    assert_eq!(p.program.out_cols[3].name, "orig");
    // Executes: one callable invocation per row feeds every lane.
    use std::cell::Cell;
    use std::rc::Rc;
    let calls = Rc::new(Cell::new(0u32));
    let c2 = calls.clone();
    let tf2 = imp("tf2", move |args| {
        c2.set(c2.get() + 1);
        match &args[0] {
            None => Ok(None),
            Some(ScalarVal::F64(x)) => Ok(Some(vec![
                Some(ScalarVal::F64(x + 1.0)),
                Some(ScalarVal::F64(x * 2.0)),
            ])),
            _ => Err("bad arg".into()),
        }
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![], vec![tf2]).unwrap();
    let input = batch(2, vec![c_f64(&[Some(3.0), None])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[
            &["true", "4.0", "6.0", "3.0"],
            &["false", "NULL", "NULL", "NULL"]
        ])
    );
    assert_eq!(calls.get(), 2, "one callable invocation per row");
}

#[test]
fn wide_udf_mid_expression_refuses_by_name() {
    let schema = cols(&[("x", Ty::F64, false)]);
    let err = prep_udfs(
        "SELECT tf2(x) + 1 AS z FROM __THIS__",
        &schema,
        &[],
        &[udf("tf2", &[Ty::F64], &[Ty::F64, Ty::F64])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Unsupported(m) if m.contains("tf2")),
        "got: {err}"
    );
}

fn udf_named(name: &str, params: &[Ty], rets: &[(&str, Ty)]) -> super::ir::ExternSpec {
    super::ir::ExternSpec {
        name: name.to_string(),
        params: params.to_vec(),
        rets: rets.iter().map(|(_, t)| *t).collect(),
        ret_names: rets.iter().map(|(n, _)| n.to_string()).collect(),
    }
}

#[test]
fn field_access_reads_lanes_off_one_ecall() {
    // TASK-63: k field reads of one width-k call share ONE ecall — the dot
    // spelling and the struct_extract spelling, mid-expression included (a
    // field read is width-1 and composes).
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT (emb(x)).a AS u, (struct_extract(emb(x), 'b') + 1.0) AS v FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap();
    let text = print(&p.program);
    assert_eq!(
        text.matches("ecall").count(),
        1,
        "two field reads must share one ecall:\n{text}"
    );
    use std::cell::Cell;
    use std::rc::Rc;
    let calls = Rc::new(Cell::new(0u32));
    let c2 = calls.clone();
    let emb = imp("emb", move |args| {
        c2.set(c2.get() + 1);
        match &args[0] {
            None => Ok(None),
            Some(ScalarVal::F64(x)) => Ok(Some(vec![
                Some(ScalarVal::F64(x + 1.0)),
                Some(ScalarVal::F64(x * 2.0)),
            ])),
            _ => Err("bad arg".into()),
        }
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![], vec![emb]).unwrap();
    let input = batch(2, vec![c_f64(&[Some(3.0), None])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["4.0", "7.0"], &["NULL", "NULL"]])
    );
    assert_eq!(calls.get(), 2, "one evaluation per row, not one per field");
}

#[test]
fn field_access_unknown_field_refuses_listing_declared() {
    let schema = cols(&[("x", Ty::F64, true)]);
    let err = prep_udfs(
        "SELECT (emb(x)).c AS u FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Bind(m)
            if m.contains("'c'") && m.contains("emb") && m.contains("'a'") && m.contains("'b'")),
        "got: {err}"
    );
}

#[test]
fn field_access_on_a_width1_named_extern_binds_lane_zero() {
    // Struct-valued at every width (the subtraction loop): a named extern
    // registers as a single-field STRUCT in DuckDB, so `.a` reads lane 0.
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT (sc(x)).a AS u FROM __THIS__",
        &schema,
        &[],
        &[udf_named("sc", &[Ty::F64], &[("a", Ty::F64)])],
    )
    .unwrap();
    let sc = imp("sc", |args| match &args[0] {
        None => Ok(None),
        Some(ScalarVal::F64(x)) => Ok(Some(vec![Some(ScalarVal::F64(x * 2.0))])),
        _ => Err("bad arg".into()),
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![], vec![sc]).unwrap();
    let input = batch(2, vec![c_f64(&[Some(3.0), None])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["6.0"], &["NULL"]])
    );
}

#[test]
fn named_extern_mid_expression_refuses() {
    // A named extern is struct-valued — width-1 included: a MID-EXPRESSION
    // position refuses (DuckDB's struct registration would binder-error on
    // the arithmetic). Bare items take the struct boundary (slice 5).
    let schema = cols(&[("x", Ty::F64, true)]);
    let err = prep_udfs(
        "SELECT (sc(x) + 1.0) AS u FROM __THIS__",
        &schema,
        &[],
        &[udf_named("sc", &[Ty::F64], &[("a", Ty::F64)])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Unsupported(m) if m.contains("sc") && m.contains("struct-valued")),
        "got: {err}"
    );
}

#[test]
fn named_extern_bare_item_expands_to_struct_lanes() {
    // Slice 5 (DRAFT-25): a NAMED extern as a bare item serves its whole
    // output struct — whole-validity lane + one lane per declared field,
    // the names riding the WideOut for the boundary to key the struct.
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT emb(x) AS z, x AS orig FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap();
    assert_eq!(p.wide_outputs.len(), 1);
    let w = &p.wide_outputs[0];
    assert_eq!((w.name.as_str(), w.first, w.width), ("z", 0, 2));
    assert_eq!(w.names, ["a", "b"]);
    assert_eq!(p.program.out_cols.len(), 4);
    assert_eq!(p.program.out_cols[0].ty.ty, Ty::I1);
    // One evaluation feeds every lane (same site sharing as the unnamed
    // list boundary).
    use std::cell::Cell;
    use std::rc::Rc;
    let calls = Rc::new(Cell::new(0u32));
    let c2 = calls.clone();
    let emb = imp("emb", move |args| {
        c2.set(c2.get() + 1);
        match &args[0] {
            None => Ok(None),
            Some(ScalarVal::F64(x)) => Ok(Some(vec![
                Some(ScalarVal::F64(x + 1.0)),
                Some(ScalarVal::F64(x * 2.0)),
            ])),
            _ => Err("bad arg".into()),
        }
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![], vec![emb]).unwrap();
    let input = batch(2, vec![c_f64(&[Some(3.0), None])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[
            &["true", "4.0", "6.0", "3.0"],
            &["false", "NULL", "NULL", "NULL"]
        ])
    );
    assert_eq!(calls.get(), 2, "one evaluation per row, not one per lane");
    // Width-1 named externs take the same boundary — struct at every width.
    let p1 = prep_udfs(
        "SELECT sc(x) AS u FROM __THIS__",
        &schema,
        &[],
        &[udf_named("sc", &[Ty::F64], &[("a", Ty::F64)])],
    )
    .unwrap();
    assert_eq!(p1.wide_outputs.len(), 1);
    let w1 = &p1.wide_outputs[0];
    assert_eq!((w1.name.as_str(), w1.first, w1.width), ("u", 0, 1));
    assert_eq!(w1.names, ["a"]);
}

#[test]
fn struct_pack_item_is_a_struct_output() {
    // θ export (slice 6): a struct_pack item takes the wide-lane boundary
    // — one validity lane plus a component lane per named field. Guarded
    // by CASE WHEN g IS NULL THEN NULL, the guard IS the validity lane, so
    // an unseen group's handle is wholly NULL (measured against DuckDB).
    let schema = cols(&[("x", Ty::I64, true)]);
    let plain = prepare(
        "SELECT struct_pack(\"type\" := 'tf0', id := x) AS th FROM __THIS__",
        "__THIS__",
        &schema,
        &[],
    )
    .unwrap();
    assert_eq!(plain.wide_outputs.len(), 1);
    assert_eq!(plain.wide_outputs[0].names, ["type", "id"]);
    assert_eq!(plain.wide_outputs[0].width, 2);

    let guarded = prepare(
        "SELECT CASE WHEN (x IS NULL) THEN (NULL) ELSE \
         struct_pack(\"type\" := 'tf0', id := x) END AS th FROM __THIS__",
        "__THIS__",
        &schema,
        &[],
    )
    .unwrap();
    assert_eq!(guarded.wide_outputs.len(), 1);
    assert_eq!(guarded.wide_outputs[0].names, ["type", "id"]);
    // The validity lane IS the guard: `x IS NOT NULL` folds to the load's
    // own validity flag, so the handle is NULL exactly when x is.
    let text = print(&guarded.program);
    assert!(
        text.contains("%v0, %v1 = load.opt in.x") && text.contains(r#"valid", %v0"#),
        "guard must drive the validity lane:\n{text}"
    );
    // The unguarded spelling cannot have the same validity lane.
    assert_ne!(print(&plain.program), text);

    // Duplicate field names are a BINDER ERROR in the oracle (measured:
    // 'Duplicate struct entry name "a"', case-insensitively) — refuse
    // rather than serve a struct the batch path cannot produce.
    for sql in [
        "SELECT struct_pack(a := x, a := x) AS th FROM __THIS__",
        "SELECT struct_pack(a := x, A := x) AS th FROM __THIS__",
    ] {
        let err = prepare(sql, "__THIS__", &schema, &[]).unwrap_err().to_string();
        assert!(err.to_lowercase().contains("duplicate"), "{sql}: {err}");
    }
    // A _-leading field cannot cross the row-path model boundary (pydantic
    // makes it a private attribute), so it refuses by name instead of
    // vanishing from the served struct.
    let err = prepare(
        "SELECT struct_pack(_a := x, b := x) AS th FROM __THIS__",
        "__THIS__",
        &schema,
        &[],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("_a"), "{err}");
}

#[test]
fn modifiers_on_an_unnest_item_refuse() {
    // Measured: DuckDB refuses every modifier on UNNEST itself (DISTINCT /
    // FILTER / in-call ORDER BY are "not applicable to UNNEST", OVER is a
    // catalog error, IGNORE NULLS a parser error) — the expansion must
    // re-screen them instead of dropping them (review round).
    let schema = cols(&[("x", Ty::F64, true)]);
    let udf = udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)]);
    for sql in [
        "SELECT unnest(DISTINCT emb(x)) FROM __THIS__",
        "SELECT unnest(emb(x) ORDER BY x) FROM __THIS__",
        "SELECT unnest(emb(x)) FILTER (WHERE x > 0.0) FROM __THIS__",
        "SELECT unnest(emb(x)) OVER () FROM __THIS__",
        "SELECT unnest(emb(x) IGNORE NULLS) FROM __THIS__",
    ] {
        let err = prep_udfs(sql, &schema, &[], &[udf.clone()]).unwrap_err();
        let msg = err.to_string().to_lowercase();
        // IGNORE NULLS parses onto the INNER call, so it refuses there —
        // either way the modifier is named, never dropped.
        assert!(
            msg.contains("unnest") || msg.contains("modifier"),
            "{sql}: expected a named modifier refusal, got: {msg}"
        );
    }
}

#[test]
fn unnest_named_extern_expands_to_per_field_columns() {
    // Measured oracle behavior: one column per declared field, named by
    // the field, in place, alias ignored — all off ONE ecall site.
    let schema = cols(&[("x", Ty::F64, true)]);
    let udf = udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)]);
    for sql in [
        "SELECT unnest(emb(x)), x AS keep FROM __THIS__",
        "SELECT unnest(emb(x)) AS ignored, x AS keep FROM __THIS__",
    ] {
        let p = prep_udfs(sql, &schema, &[], &[udf.clone()]).unwrap();
        let names: Vec<&str> = p.program.out_cols.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "keep"], "{sql}");
        // Expanded columns are PLAIN scalars — no wide/struct assembly.
        assert!(p.wide_outputs.is_empty(), "{sql}");
        assert_eq!(print(&p.program).matches("ecall").count(), 1, "{sql}");
    }
}

#[test]
fn whole_item_and_field_read_share_one_ecall() {
    // Review round (P16 single-eval): a whole item and a field read of the
    // SAME call share one ecall site, in either item order — the wide
    // expansion consults the extern_sites cache like extern_field_lane.
    let schema = cols(&[("x", Ty::F64, true)]);
    for sql in [
        "SELECT emb(x) AS s, (emb(x)).a AS z FROM __THIS__",
        "SELECT (emb(x)).a AS z, emb(x) AS s FROM __THIS__",
    ] {
        let p = prep_udfs(
            sql,
            &schema,
            &[],
            &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
        )
        .unwrap();
        let text = print(&p.program);
        assert_eq!(
            text.matches("ecall").count(),
            1,
            "{sql}: whole item and field read must share one ecall:\n{text}"
        );
    }
}

#[test]
fn field_access_without_declared_names_refuses() {
    let schema = cols(&[("x", Ty::F64, true)]);
    let err = prep_udfs(
        "SELECT (tf2(x)).a AS u FROM __THIS__",
        &schema,
        &[],
        &[udf("tf2", &[Ty::F64], &[Ty::F64, Ty::F64])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Unsupported(m) if m.contains("tf2") && m.contains("field names")),
        "got: {err}"
    );
}

#[test]
fn named_extern_programs_are_canonical_ir() {
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT (emb(x)).a AS u, (emb(x)).b AS v FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap()
    .program;
    let text = print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "named-extern program is not canonical:\n{text}"
    );
}

#[test]
fn lateral_reference_to_a_duplicated_alias_refuses() {
    // DuckDB resolves a lateral ref to the LAST definition of a name, so a
    // reference between two definitions is its forward-reference binder
    // error (measured 1.5.5). Confit's binding is per-occurrence, so a
    // shared-site ecall bound through a mutating alias would silently take
    // the FIRST binding — refusing the duplicated-alias reference keeps
    // binding time-invariant and site sharing sound for every accepted
    // query.
    let schema = cols(&[("x", Ty::F64, true)]);
    let err = prep_udfs(
        "SELECT 1.0 AS z, (emb(z)).a AS p, 2.0 AS z, (emb(z)).b AS q FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrepareError::Bind(m) if m.contains("\"z\"")),
        "got: {err}"
    );
}

#[test]
fn distinct_args_to_the_same_extern_get_distinct_ecalls() {
    // The site cache keys on the FULL syntactic call — two reads with
    // different arguments must stay two evaluations (a name-keyed cache
    // would serve ya from emb(x); mutation-tested in review).
    let schema = cols(&[("x", Ty::F64, true), ("y", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT (emb(x)).a AS xa, (emb(y)).a AS ya FROM __THIS__",
        &schema,
        &[],
        &[udf_named("emb", &[Ty::F64], &[("a", Ty::F64), ("b", Ty::F64)])],
    )
    .unwrap();
    let text = print(&p.program);
    assert_eq!(text.matches("ecall").count(), 2, "IR:\n{text}");
    let emb = imp("emb", |args| match &args[0] {
        None => Ok(None),
        Some(ScalarVal::F64(x)) => Ok(Some(vec![
            Some(ScalarVal::F64(x + 1.0)),
            Some(ScalarVal::F64(x * 2.0)),
        ])),
        _ => Err("bad arg".into()),
    });
    let f = super::exec::interp::compile_ext(&p.program, vec![], vec![emb]).unwrap();
    let input = batch(1, vec![c_f64(&[Some(3.0)]), c_f64(&[Some(10.0)])]);
    assert_eq!(
        run_snapshot(&f, &input).unwrap(),
        rows(&[&["4.0", "11.0"]])
    );
}

#[test]
fn udf_programs_are_canonical_ir() {
    let schema = cols(&[("x", Ty::F64, true)]);
    let p = prep_udfs(
        "SELECT tf0(x) AS z FROM __THIS__ WHERE x IS NOT NULL",
        &schema,
        &[],
        &[udf("tf0", &[Ty::F64], &[Ty::F64])],
    )
    .unwrap()
    .program;
    let text = print(&p);
    assert_eq!(
        parse(&text).unwrap(),
        p,
        "udf program is not canonical:\n{text}"
    );
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
fn rowid_rejects_and_lateral_aliases_serve() {
    let schema = cols(&[("a", Ty::I64, false)]);
    match prep("SELECT rowid FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(msg)) => assert!(msg.contains("rowid"), "{msg}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    // Wave-5 pins: lateral aliases — later items and WHERE see the alias;
    // chains fold left-to-right.
    let got = run_sql(
        "SELECT a % 2 AS k, k * 2 AS d FROM __THIS__ WHERE k = 1",
        &schema,
        batch(3, vec![c_i64(&[Some(3), Some(4), Some(7)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["1", "2"], &["1", "2"]]));
    let got = run_sql(
        "SELECT a + 1 AS b, b + 1 AS c, c + 1 AS d FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(1)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["2", "3", "4"]]));
    // The REAL column beats the alias (measured), and a forward reference
    // is the pinned bind error.
    let schema2 = cols(&[("a", Ty::I64, false), ("k", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a + 1 AS k, k * 2 AS d FROM __THIS__",
        &schema2,
        batch(1, vec![c_i64(&[Some(10)]), c_i64(&[Some(100)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["11", "200"]]));
    match prep("SELECT b + 1 AS c, a + 1 AS b FROM __THIS__", &schema) {
        Err(PrepareError::Bind(msg)) => {
            assert!(msg.contains("before it is defined"), "{msg}")
        }
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn colon_prefix_alias_desugars_to_as() {
    // DuckDB `SELECT k: expr` — sqlparser would silently misparse it as a
    // Snowflake JSON path, so rewrite.rs desugars it on the token stream
    // (pins-wave5/sqlparser-spike.json).
    let schema = cols(&[("a", Ty::I64, false)]);
    let p = prep("SELECT k: a + 1, j: a * 2 FROM __THIS__", &schema).unwrap();
    let names: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["k", "j"]);
    let f = compile(&p, vec![]).unwrap();
    let got = run_snapshot(&f, &batch(1, vec![c_i64(&[Some(4)])])).unwrap();
    assert_eq!(got, rows(&[&["5", "8"]]));
}

#[test]
fn colon_alias_quoted_mixed_and_filtered() {
    let schema = cols(&[("a", Ty::I64, false)]);
    // Quoted alias keeps its spelling; mixes with AS items; WHERE ends the
    // projection list correctly (alias flushed before FROM).
    let p = prep(
        "SELECT \"K\": a - 1, a AS plain FROM __THIS__ WHERE a > 0",
        &schema,
    )
    .unwrap();
    let names: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["K", "plain"]);
}

#[test]
fn double_colon_cast_is_not_a_colon_alias() {
    let schema = cols(&[("a", Ty::I64, false)]);
    // `a::BIGINT` tokenizes as DoubleColon — must not trigger the rewrite.
    let p = prep("SELECT a::BIGINT AS c FROM __THIS__", &schema).unwrap();
    assert_eq!(p.out_cols[0].name, "c");
}

#[test]
fn bracket_subscripts_extract_codepoints() {
    // pins-wave5/subscripts-extended.json: 1-based codepoints, negative
    // from-end, 0/out-of-range -> '' (never NULL).
    let schema = cols(&[("s", Ty::Str, false), ("off", Ty::I64, false)]);
    let got = run_sql(
        "SELECT s[2] AS a, s[-1] AS b, s[0] AS z, s[100] AS oor, s[off] AS dy \
         FROM __THIS__",
        &schema,
        batch(
            1,
            vec![c_str(&[Some("h\u{e9}llo")]), c_i64(&[Some(3)])],
        ),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["\u{e9}", "o", "", "", "l"]]));
}

#[test]
fn bracket_slices_open_bounds_and_chain() {
    // pins-wave5/slices.json: both-inclusive, [:b]==[1:b], [a:]==[a:-1],
    // reversed -> ''; a chained extract applies to the slice result.
    let schema = cols(&[("s", Ty::Str, false)]);
    let got = run_sql(
        "SELECT s[2:4] AS m, s[:2] AS a, s[2:] AS b, s[:] AS w, s[4:2] AS inv, \
         s[2:4][1] AS chain FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("hello")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["ell", "he", "ello", "hello", "", "e"]]));
}

#[test]
fn subscript_offset_window_traps_past_2_pow_32() {
    // In-window extremes return ''; one past traps (same window as substr).
    let schema = cols(&[("s", Ty::Str, false)]);
    let got = run_sql(
        "SELECT s[4294967295] AS hi, s[-4294967296] AS lo FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("hello")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["", ""]]));
    let err = run_sql(
        "SELECT s[4294967296] AS boom FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("hello")])]),
    )
    .unwrap_err();
    assert!(err.contains("supported range"), "got: {err}");
}

#[test]
fn slice_with_step_rejects_cleanly() {
    let schema = cols(&[("s", Ty::Str, false)]);
    match prep("SELECT s[1:3:1] AS x FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(msg)) => assert!(msg.contains("step"), "{msg}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn bitwise_flat_precedence_and_values() {
    // pins-wave5/bitwise-int-ops.json: << >> & | are ONE flat left-assoc
    // tier (4|1&1 = (4|1)&1 = 1), arithmetic binds tighter (1<<1+1 = 4).
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT 1 << 3 AS s, 8 >> 1 AS r, 5 & 3 AS n, 5 | 3 AS o, xor(5, 3) AS x, \
         4 | 1 & 1 AS p1, 1 & 3 << 1 AS p2, 8 >> 2 | 1 AS p3, 1 << 1 + 1 AS p4, \
         (-8) >> 1 AS ar, (-1) >> 64 AS oor, 0 << 100 AS zs FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(0)])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[&["8", "4", "1", "7", "6", "1", "2", "3", "4", "-4", "0", "0"]])
    );
}

#[test]
fn left_shift_trap_ladder() {
    // Ladder order per pins: negative value (even << 0), negative count,
    // zero shortcut, count range, overflow — DuckDB texts verbatim.
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, false)]);
    for (x, y, needle) in [
        (-5, 0, "Cannot left-shift negative number -5"),
        (1, -1, "Cannot left-shift by negative number -1"),
        (1, 64, "Left-shift value 64 is out of range"),
        (1, 63, "Overflow in left shift (1 << 63)"),
        (4611686018427387904, 1, "Overflow in left shift"),
    ] {
        let err = run_sql(
            "SELECT a << b AS v FROM __THIS__",
            &schema,
            batch(1, vec![c_i64(&[Some(x)]), c_i64(&[Some(y)])]),
        )
        .unwrap_err();
        assert!(err.contains(needle), "{x} << {y}: got {err}");
    }
    // In-range boundary folds/computes fine; NULL masks the would-trap row.
    let schema2 = cols(&[("b", Ty::I64, true)]);
    let got = run_sql(
        "SELECT 1 << 62 AS big, 1 << b AS masked FROM __THIS__",
        &schema2,
        batch(1, vec![c_i64(&[None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["4611686018427387904", "NULL"]]));
}

#[test]
fn power_caret_stays_unsupported() {
    // DuckDB ^ is pow with a precedence sqlparser can't mirror — clean
    // unsupported, pow()/power() carry the semantics.
    let schema = cols(&[("a", Ty::I64, false)]);
    match prep("SELECT a ^ 2 AS x FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(msg)) => assert!(msg.contains('^'), "{msg}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn caret_at_is_starts_with() {
    let schema = cols(&[("s", Ty::Str, true)]);
    let got = run_sql(
        "SELECT s ^@ 'he' AS p, s ^@ '' AS e FROM __THIS__",
        &schema,
        batch(3, vec![c_str(&[Some("hello"), Some("Hello"), None])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[&["true", "true"], &["false", "true"], &["NULL", "NULL"]])
    );
}

#[test]
fn glob_quirks_per_pins() {
    // pins-wave5/text-operators.json: byte-based ?, '^' literal in classes
    // (only '!' negates), dead patterns match nothing, \ escapes outside
    // classes, '%' is a plain literal, case-sensitive.
    let schema = cols(&[("s", Ty::Str, false)]);
    let cases: &[(&str, &str, bool)] = &[
        ("hello", "h*", true),
        ("Hello", "h*", false),
        ("", "*", true),
        ("", "", true),
        ("a", "", false),
        ("h\u{e9}llo", "h?llo", false), // é is 2 bytes; ? eats ONE
        ("h\u{e9}llo", "h??llo", true),
        ("hello", "[^h]ello", true), // '^' is a literal member
        ("hello", "[!h]ello", false),
        ("jello", "[!h]ello", true),
        ("a", "[a-]", false), // ']' eaten as endpoint -> dead
        ("-", "[-a]", true),  // '-' literal when first
        ("a", "[-a]", true),
        ("b", "[-a]", false),
        ("]", "[A-]]", true), // ']' as a range endpoint
        ("B]", "[A-]]", false),
        ("a*b", "a\\*b", true),
        ("ab", "a\\b", true),
        ("a\\", "a\\", false), // dangling escape -> dead
        ("h", "[", false),     // lone '[' -> dead
        ("h%llo", "h%llo", true),
        ("hello", "h[a-f]llo", true),
    ];
    for (s, p, want) in cases {
        // SQL single-quoted strings pass backslashes through verbatim.
        let sql = format!("SELECT s GLOB '{p}' AS m FROM __THIS__");
        let got = run_sql(&sql, &schema, batch(1, vec![c_str(&[Some(s)])])).unwrap();
        assert_eq!(
            got,
            rows(&[&[if *want { "true" } else { "false" }]]),
            "{s:?} GLOB {p:?}"
        );
    }
}

#[test]
fn not_glob_is_a_parse_error_and_marker_is_reserved() {
    let schema = cols(&[("s", Ty::Str, false)]);
    match prep("SELECT s NOT GLOB 'h*' AS m FROM __THIS__", &schema) {
        Err(PrepareError::Parse(_)) => {}
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    match prep("SELECT __glob_pat('x') FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(msg)) => assert!(msg.contains("__glob_pat"), "{msg}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn star_name_filters_replace_rename_qualified_exclude() {
    let schema = cols(&[
        ("abc", Ty::I64, false),
        ("abd", Ty::I64, false),
        ("xyz", Ty::Str, false),
    ]);
    let names = |p: &super::ir::Program| {
        p.out_cols
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>()
    };
    // Name filters against declared-case names (pins-wave5/star-forms.json):
    // LIKE case-sensitive, ILIKE folds, GLOB byte matcher, NOT LIKE negates.
    let p = prep("SELECT * LIKE 'ab%' FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["abc", "abd"]);
    let p = prep("SELECT * ILIKE 'AB%' FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["abc", "abd"]);
    let p = prep("SELECT * NOT LIKE 'ab%' FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["xyz"]);
    let p = prep("SELECT * GLOB 'ab[cd]' FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["abc", "abd"]);
    // Filter FOLLOWS exclude (grammar order).
    let p = prep("SELECT * EXCLUDE (abd) LIKE 'ab%' FROM __THIS__", &schema).unwrap();
    assert_eq!(names(&p), ["abc"]);
    // Zero matches is an error, never an empty star.
    match prep("SELECT * LIKE 'zz%' FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("empty set of columns"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    // REPLACE keeps position and name, may change type, sees originals.
    let p = prep(
        "SELECT * REPLACE (abc + abd AS abc, 'x' AS xyz) FROM __THIS__",
        &schema,
    )
    .unwrap();
    assert_eq!(names(&p), ["abc", "abd", "xyz"]);
    match prep("SELECT * REPLACE (1 AS nope) FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("REPLACE list not found"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    // RENAME keeps position; nonexistent target silently ignored.
    let p = prep(
        "SELECT * RENAME (abc AS q, nope AS r) FROM __THIS__",
        &schema,
    )
    .unwrap();
    assert_eq!(names(&p), ["q", "abd", "xyz"]);
    // Cross-list conflicts are the pinned parse-class errors.
    match prep(
        "SELECT * EXCLUDE (abc) REPLACE (1 AS abc) FROM __THIS__",
        &schema,
    ) {
        Err(PrepareError::Parse(m)) => assert!(m.contains("EXCLUDE and REPLACE"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn dup_name_rename_matches_duckdb_boundary_algorithm() {
    // pins-wave5/dup-names-client-contract.json: case-insensitive collision
    // check that also covers candidates; a renamed dup can steal a later
    // literal's name.
    let schema = cols(&[("id", Ty::I64, false)]);
    let names = |sql: &str| {
        prep(sql, &schema)
            .unwrap()
            .out_cols
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names("SELECT id, id AS id, id AS id FROM __THIS__"),
        ["id", "id_1", "id_2"]
    );
    assert_eq!(
        names("SELECT id, id AS \"ID\" FROM __THIS__"),
        ["id", "ID_1"]
    );
    assert_eq!(
        names("SELECT id, id AS id, id AS id_1 FROM __THIS__"),
        ["id", "id_1", "id_1_1"]
    );
}

#[test]
fn qualified_exclude_over_join() {
    let schema = cols(&[("id", Ty::I64, false)]);
    let st = stat("dim", &[("id", Ty::I64, false), ("v", Ty::F64, false)]);
    // Qualified EXCLUDE strips ONE table's copy; unqualified strips both.
    let p = prepare(
        "SELECT * EXCLUDE (dim.id) FROM __THIS__ JOIN dim ON __THIS__.id = dim.id",
        "__THIS__",
        &schema,
        std::slice::from_ref(&st),
    )
    .unwrap()
    .program;
    let got: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(got, ["id", "v"]);
    let p = prepare(
        "SELECT * EXCLUDE (id) FROM __THIS__ JOIN dim ON __THIS__.id = dim.id",
        "__THIS__",
        &schema,
        std::slice::from_ref(&st),
    )
    .unwrap()
    .program;
    let got: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(got, ["v"]);
}

#[test]
fn binder_tail_null_ops_natural_join_main_qualifier() {
    // NULL <op> NULL types by operator (wave-5 pins): + -> BIGINT NULL,
    // / -> DOUBLE NULL, = -> BOOLEAN NULL.
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT NULL + NULL AS s, NULL / NULL AS d, NULL = NULL AS e FROM __THIS__",
        &schema,
        batch(1, vec![c_i64(&[Some(1)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL", "NULL", "NULL"]]));
    // Schema qualifiers are registry-noise (TASK-55): any single qualifier
    // resolves when the table part matches; 3-part column refs bind too.
    let p = prep("SELECT a FROM main.__THIS__", &schema).unwrap();
    assert_eq!(p.out_cols[0].name, "a");
    let p = prep("SELECT test.__THIS__.a FROM test.__THIS__", &schema).unwrap();
    assert_eq!(p.out_cols[0].name, "a");
    match prep("SELECT a FROM test.other", &schema) {
        Err(PrepareError::Unsupported(m)) => assert!(m.contains("driving relation"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    // NATURAL JOIN = USING(common cols): merged key, left spelling; no
    // common columns is a hard error.
    let st = stat("dim", &[("a", Ty::I64, false), ("v", Ty::F64, false)]);
    let p = prepare(
        "SELECT * FROM __THIS__ NATURAL JOIN dim",
        "__THIS__",
        &schema,
        std::slice::from_ref(&st),
    )
    .unwrap()
    .program;
    let names: Vec<&str> = p.out_cols.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["a", "v"]);
    let st2 = stat("dim", &[("z", Ty::I64, false), ("v", Ty::F64, false)]);
    match prepare(
        "SELECT * FROM __THIS__ NATURAL JOIN dim",
        "__THIS__",
        &schema,
        std::slice::from_ref(&st2),
    ) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("No columns found"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn between_in_mixed_literals_cast_to_the_numeric_side() {
    // Wave-5 pins: 1 IN ('1', 2) TRUE; 2 = '1.5' rounds half-away (via IN);
    // TRUE IN (1, 0) casts bool -> int; non-numeric strings are conversion
    // bind errors.
    let schema = cols(&[("a", Ty::I64, false)]);
    let got = run_sql(
        "SELECT a IN ('1', 2) AS x, a IN ('1.5') AS h, true IN (1, 0) AS b, \
         a BETWEEN '0' AND '5' AS r FROM __THIS__",
        &schema,
        batch(2, vec![c_i64(&[Some(1), Some(2)])]),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &["true", "false", "true", "true"],
            &["true", "true", "true", "true"]
        ])
    );
    // Non-numeric strings convert at EXECUTION time in DuckDB (an empty
    // input succeeds), so this stays clean-unsupported, not a bind error.
    match prep("SELECT a IN ('abc') AS x FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(m)) => assert!(m.contains("non-numeric"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn column_renaming_table_alias() {
    // t AS u(x, y): prefix rename is legal, old names and the original
    // table name die as qualifiers; too many names errors (wave-5 pins).
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, false)]);
    let got = run_sql(
        "SELECT x + u.y AS s FROM __THIS__ AS u(x, y)",
        &schema,
        batch(1, vec![c_i64(&[Some(1)]), c_i64(&[Some(2)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["3"]]));
    // Partial list: only the prefix renames.
    let got = run_sql(
        "SELECT x + b AS s FROM __THIS__ AS u(x)",
        &schema,
        batch(1, vec![c_i64(&[Some(1)]), c_i64(&[Some(2)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["3"]]));
    match prep("SELECT a FROM __THIS__ AS u(x, y)", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("does not exist"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    match prep("SELECT x FROM __THIS__ AS u(x, y, z)", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("columns specified"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn regexp_family_end_to_end() {
    // Wave-B pins: regexp_matches is a SEARCH, ~ / SIMILAR TO are FULL
    // match on the raw pattern (no % translation), extract returns '' on
    // no match, replace is first-match-only without 'g'.
    let schema = cols(&[("s", Ty::Str, true)]);
    let input = || {
        batch(
            3,
            vec![c_str(&[Some("hello world"), Some("abc123def"), None])],
        )
    };
    let got = run_sql(
        "SELECT regexp_matches(s, 'ell') AS m, s ~ 'ell' AS f, \
         s ~ 'h.*d' AS w, s SIMILAR TO 'h%o' AS pct, \
         regexp_extract(s, '[0-9]+') AS num, \
         regexp_extract(s, '(\\w+) (\\w+)', 2) AS g2, \
         regexp_replace(s, 'l', 'L') AS r1, \
         regexp_replace(s, 'l', 'L', 'g') AS rg FROM __THIS__",
        &schema,
        input(),
    )
    .unwrap();
    assert_eq!(
        got,
        rows(&[
            &[
                "true", "false", "true", "false", "", "world", "heLlo world",
                "heLLo worLd"
            ],
            &[
                "false", "false", "false", "false", "123", "", "abc123def",
                "abc123def"
            ],
            &["NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL", "NULL"],
        ])
    );
    // ASCII \d semantics survive the translation (the differential pin):
    // Arabic-Indic digits do NOT match \d but DO match \p{Nd}.
    let got = run_sql(
        "SELECT regexp_matches(s, '\\d') AS a, regexp_matches(s, '\\p{Nd}') AS u \
         FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("\u{0663}\u{0664}")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["false", "true"]]));
    // Backrefs are backslash-style; $1 is a literal. Options 'i' folds.
    let got = run_sql(
        "SELECT regexp_replace(s, '(h)ello', '[\\1]') AS b, \
         regexp_replace(s, '(h)ello', '[$1]') AS lit, \
         regexp_matches(s, 'HELLO', 'i') AS ci FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("hello world")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["[h] world", "[$1] world", "true"]]));
    // Invalid-rewrite quirks: out-of-range backref = silent no-op.
    let got = run_sql(
        "SELECT regexp_replace(s, '(h)', '\\2') AS noop FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[Some("hello")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["hello"]]));
    // Bad constant pattern errors at PREPARE (pinned eagerness).
    match prep("SELECT regexp_matches(s, '(') AS x FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("Invalid Input Error"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
    // Divergence guard: \B rejects cleanly.
    match prep("SELECT regexp_matches(s, 'a\\B') AS x FROM __THIS__", &schema) {
        Err(PrepareError::Unsupported(m)) => assert!(m.contains("\\B"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn star_similar_to_and_columns() {
    // Wave-B pins: positive * SIMILAR TO = unanchored search over names,
    // NOT form = NOT full-match — NOT complements ('a.*' on a col set
    // where a name merely CONTAINS 'a' lands in both results).
    let schema = cols(&[
        ("abc", Ty::I64, false),
        ("abd", Ty::I64, false),
        ("xyz", Ty::Str, false),
    ]);
    let names = |sql: &str| {
        prep(sql, &schema)
            .unwrap()
            .out_cols
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names("SELECT * SIMILAR TO 'c' FROM __THIS__"), ["abc"]);
    assert_eq!(
        names("SELECT * NOT SIMILAR TO 'c' FROM __THIS__"),
        ["abc", "abd", "xyz"] // NOT full-match: 'c' full-matches no name
    );
    assert_eq!(
        names("SELECT * NOT SIMILAR TO 'ab.' FROM __THIS__"),
        ["xyz"]
    );
    assert_eq!(names("SELECT COLUMNS('ab.') FROM __THIS__"), ["abc", "abd"]);
    assert_eq!(
        names("SELECT COLUMNS(*) FROM __THIS__"),
        ["abc", "abd", "xyz"]
    );
    // Alias stamps every expansion; duplicates feed the dedup rename.
    assert_eq!(
        names("SELECT COLUMNS('ab.') AS x FROM __THIS__"),
        ["x", "x_1"]
    );
    match prep("SELECT COLUMNS('zz.*') FROM __THIS__", &schema) {
        Err(PrepareError::Bind(m)) => assert!(m.contains("No matching columns"), "{m}"),
        other => panic!("wrong outcome: {:?}", other.err()),
    }
}

#[test]
fn opaque_row_columns_reject_only_on_reference() {
    // Model order: a (i64), d (opaque — e.g. a timestamp), s (str). The
    // engine sees d as a positioned name with no lane: referencing it (star
    // expansion included) is the named error; removing it serves.
    let schema = cols(&[("a", Ty::I64, false), ("s", Ty::Str, true)]);
    let opaque = vec![(1usize, "d".to_string())];
    let prep_o = |sql: &str| {
        super::prepare_opaque(sql, "__THIS__", &schema, &opaque, &[], &[], false, &[], &[])
            .map(|p| p.program)
    };

    // Untouched -> serves end to end.
    let p = prep_o("SELECT a + 1 AS x FROM __THIS__").unwrap();
    let f = compile(&p, vec![]).unwrap();
    let got = run_snapshot(
        &f,
        &batch(1, vec![c_i64(&[Some(4)]), c_str(&[Some("k")])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["5"]]));

    // Any reference path rejects with the same named error as before.
    for sql in [
        "SELECT d FROM __THIS__",
        "SELECT __THIS__.d FROM __THIS__",
        "SELECT a FROM __THIS__ WHERE d IS NULL",
        "SELECT * FROM __THIS__",
        "SELECT COLUMNS(*) FROM __THIS__",
        "SELECT COLUMNS('d') FROM __THIS__",
    ] {
        let e = prep_o(sql).unwrap_err().to_string();
        assert!(e.contains("non-scalar type"), "{sql}: {e}");
    }

    // Removed before the output -> serves, in model order.
    let names = |sql: &str| {
        prep_o(sql)
            .unwrap()
            .out_cols
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names("SELECT * EXCLUDE (d) FROM __THIS__"), ["a", "s"]);
    assert_eq!(names("SELECT * LIKE 'a%' FROM __THIS__"), ["a"]);
    assert_eq!(names("SELECT COLUMNS('a|s') FROM __THIS__"), ["a", "s"]);
    // REPLACE keeps d's position but gives it a real lane.
    assert_eq!(names("SELECT * REPLACE (7 AS d) FROM __THIS__"), ["a", "d", "s"]);

    // Column-list alias is positional over the FULL model: stopping before
    // d serves, reaching d rejects, and the count error includes d.
    prep_o("SELECT x FROM __THIS__ AS u(x)").unwrap();
    let e = prep_o("SELECT x FROM __THIS__ AS u(x, y)")
        .unwrap_err()
        .to_string();
    assert!(e.contains("non-scalar type"), "{e}");
    let e = prep_o("SELECT x FROM __THIS__ AS u(w, x, y, z)")
        .unwrap_err()
        .to_string();
    assert!(e.contains("3 columns available but 4"), "{e}");
}

#[test]
fn from_colon_prefix_alias_matches_as_form() {
    // pins-waveA/from-colon-alias.json: `FROM x : T` == `FROM T AS x` in
    // every probed behavior; whitespace around the colon irrelevant.
    let schema = cols(&[("i", Ty::I64, false)]);
    for sql in [
        "SELECT * FROM b : __THIS__",
        "SELECT * FROM \"b\" : __THIS__",
        "SELECT i FROM b:__THIS__",
        "SELECT b.i FROM b : __THIS__",
        "SELECT i FROM b : __THIS__ WHERE b.i = 42",
        "SELECT i FROM b : main.__THIS__",
    ] {
        let got = run_sql(sql, &schema, batch(1, vec![c_i64(&[Some(42)])])).unwrap();
        assert_eq!(got, rows(&[&["42"]]), "{sql}");
    }
    // The original name is hidden, exactly like AS.
    let e = prep("SELECT __THIS__.i FROM b : __THIS__", &schema)
        .unwrap_err()
        .to_string();
    assert!(e.contains("unknown table"), "{e}");
    // Static tables take colon aliases too (alias scopes the join).
    let dim = stat("dim", &[("i", Ty::I64, false), ("v", Ty::I64, false)]);
    let p = prepare(
        "SELECT d.v FROM __THIS__ JOIN d : dim ON __THIS__.i = d.i",
        "__THIS__",
        &schema,
        std::slice::from_ref(&dim),
    )
    .unwrap();
    assert_eq!(p.program.out_cols[0].name, "v");
    // Right sides we don't rewrite stay clean parse errors: chained and
    // postfix-mixed forms (DuckDB parse errors too), table functions.
    for sql in [
        "SELECT * FROM b : c : __THIS__",
        "SELECT * FROM b : __THIS__ AS d",
        "SELECT * FROM r : range(3)",
    ] {
        let e = prep(sql, &schema).unwrap_err().to_string();
        assert!(e.contains("parse error") || e.contains("unsupported"), "{sql}: {e}");
    }
}

#[test]
fn parenless_star_replace_consumes_one_item() {
    // pins-waveA/columns-replace.json: paren-less REPLACE takes exactly
    // ONE `expr AS name`; a comma starts a NEW select item (dup names and
    // all). Multiplication by a column named replace is untouched.
    let schema = cols(&[("i", Ty::I64, false), ("j", Ty::I64, false)]);
    let input = || batch(1, vec![c_i64(&[Some(1)]), c_i64(&[Some(2)])]);
    let got = run_sql("SELECT * REPLACE i+100 AS i FROM __THIS__", &schema, input()).unwrap();
    assert_eq!(got, rows(&[&["101", "2"]]));
    let got = run_sql(
        "SELECT integers.* REPLACE i+100 AS i FROM __THIS__ AS integers",
        &schema,
        input(),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["101", "2"]]));
    // Comma ends the item: third column is a separate j+1, and the dup
    // name goes through the boundary rename (j, j -> j, j_1).
    let p = prep(
        "SELECT * REPLACE i+100 AS i, j+1 AS j FROM __THIS__",
        &schema,
    )
    .unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["i", "j", "j_1"]
    );
    let f = compile(&p, vec![]).unwrap();
    let got = run_snapshot(&f, &input()).unwrap();
    assert_eq!(got, rows(&[&["101", "2", "3"]]));
    // `3 * replace` where replace is a column: not a star modifier.
    let rschema = cols(&[("replace", Ty::I64, false)]);
    let got = run_sql(
        "SELECT 3 * replace AS x FROM __THIS__",
        &rschema,
        batch(1, vec![c_i64(&[Some(5)])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["15"]]));
}

#[test]
fn cast_null_regex_arguments_match_bare_null() {
    // pins-waveA/regex-null-pattern.json: CAST(NULL AS VARCHAR) behaves
    // exactly like a bare NULL literal in every regex argument slot.
    let schema = cols(&[("s", Ty::Str, true)]);
    let input = || batch(2, vec![c_str(&[Some("a"), None])]);
    for sql in [
        "SELECT regexp_matches(s, CAST(NULL AS VARCHAR)) AS r FROM __THIS__",
        "SELECT s SIMILAR TO CAST(NULL AS VARCHAR) AS r FROM __THIS__",
        "SELECT NOT (s ~ CAST(NULL AS VARCHAR)) AS r FROM __THIS__",
    ] {
        let got = run_sql(sql, &schema, input()).unwrap();
        assert_eq!(got, rows(&[&["NULL"], &["NULL"]]), "{sql}");
    }
    for sql in [
        "SELECT regexp_replace(s, CAST(NULL AS VARCHAR), 'x') AS r FROM __THIS__",
        "SELECT regexp_replace(s, 'a', CAST(NULL AS VARCHAR)) AS r FROM __THIS__",
        "SELECT regexp_replace(s, 'a', 'x', CAST(NULL AS VARCHAR)) AS r FROM __THIS__",
        "SELECT regexp_extract(s, CAST(NULL AS VARCHAR)) AS r FROM __THIS__",
    ] {
        let got = run_sql(sql, &schema, input()).unwrap();
        assert_eq!(got, rows(&[&["NULL"], &["NULL"]]), "{sql}");
    }
    // NULL OPTIONS stay the pinned error for the non-replace functions.
    let e = prep(
        "SELECT regexp_matches(s, 'a', CAST(NULL AS VARCHAR)) FROM __THIS__",
        &schema,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("must not be NULL"), "{e}");
    // A NULL pattern of the WRONG type is still a bind error.
    let e = prep(
        "SELECT regexp_matches(s, CAST(NULL AS INTEGER)) FROM __THIS__",
        &schema,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("pattern"), "{e}");
}

#[test]
fn reverse_ascii_byte_path_and_grapheme_path() {
    // pins-waveA/reverse-graphemes.json. All-ASCII inputs BYTE-reverse
    // (CRLF splits — DuckDB's fast path, measured); any non-ASCII char
    // switches to UAX-29 extended grapheme clusters, byte-preserving.
    let schema = cols(&[("s", Ty::Str, true)]);
    let cases: &[(&str, &str)] = &[
        ("", ""),
        ("abc", "cba"),
        ("a\r\nb", "b\n\ra"),          // ASCII: CRLF SPLITS (fast path)
        ("\u{f6}\r\nb", "b\r\n\u{f6}"), // non-ASCII: CRLF holds
        ("Mot\u{f6}rHead", "daeHr\u{f6}toM"),
        ("e\u{301}x", "xe\u{301}"), // combining mark stays attached
        // 3 regional indicators: U+S pair from the left, F trails alone.
        ("\u{1f1fa}\u{1f1f8}\u{1f1eb}", "\u{1f1eb}\u{1f1fa}\u{1f1f8}"),
        // ZWJ family emoji is one cluster.
        (
            "x\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}y",
            "y\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}x",
        ),
        // Hangul jamo LVT stay one cluster, NOT composed.
        (
            "\u{1112}\u{1161}\u{11ab}\u{d55c}",
            "\u{d55c}\u{1112}\u{1161}\u{11ab}",
        ),
        // ZWJ travels with the PRECEDING char.
        ("a\u{200d}b\u{f6}", "\u{f6}ba\u{200d}"),
    ];
    for (input, want) in cases {
        let got = run_sql(
            "SELECT reverse(s) AS r FROM __THIS__",
            &schema,
            batch(1, vec![c_str(&[Some(input)])]),
        )
        .unwrap();
        assert_eq!(got, rows(&[&[want]]), "input {:?}", input);
    }
    // NULL passes through; non-VARCHAR args are binder errors (NO
    // implicit cast — measured: sole overload reverse(VARCHAR)).
    let got = run_sql(
        "SELECT reverse(s) AS r, reverse(NULL) AS n FROM __THIS__",
        &schema,
        batch(1, vec![c_str(&[None])]),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["NULL", "NULL"]]));
    for sql in [
        "SELECT reverse(123) FROM __THIS__",
        "SELECT reverse(1.5) FROM __THIS__",
        "SELECT reverse(true) FROM __THIS__",
    ] {
        let e = prep(sql, &schema).unwrap_err().to_string();
        assert!(e.contains("no function matches reverse"), "{sql}: {e}");
    }
}

#[test]
fn columns_star_with_modifiers() {
    // pins-waveA/columns-replace.json: COLUMNS(* <modifiers>) as a bare
    // select item == * <modifiers> (names, order, values). The regex+
    // modifier combo is a DuckDB parser error and stays rejected.
    let schema = cols(&[("a", Ty::I64, false), ("b", Ty::I64, false)]);
    let input = || batch(1, vec![c_i64(&[Some(1)]), c_i64(&[Some(2)])]);
    let got = run_sql(
        "SELECT COLUMNS(* REPLACE (a + 10 AS a, b + 20 AS b)) FROM __THIS__",
        &schema,
        input(),
    )
    .unwrap();
    assert_eq!(got, rows(&[&["11", "22"]]));
    let p = prep(
        "SELECT COLUMNS(* REPLACE (a + 10 AS a)) FROM __THIS__",
        &schema,
    )
    .unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    for (sql, want) in [
        ("SELECT COLUMNS(* EXCLUDE (b)) FROM __THIS__", vec!["a"]),
        ("SELECT COLUMNS(* EXCLUDE b) FROM __THIS__", vec!["a"]),
    ] {
        let p = prep(sql, &schema).unwrap();
        assert_eq!(
            p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            want,
            "{sql}"
        );
    }
    // Alias stamps every expansion (same rule as COLUMNS('re') AS x).
    let p = prep(
        "SELECT COLUMNS(* REPLACE (a + 10 AS a)) AS x FROM __THIS__",
        &schema,
    )
    .unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["x", "x_1"]
    );
    // COLUMNS('re' REPLACE ...) is a DuckDB parser error — stays rejected.
    let e = prep("SELECT COLUMNS('a' REPLACE (a+1 AS a)) FROM __THIS__", &schema)
        .unwrap_err()
        .to_string();
    assert!(e.contains("parse error") || e.contains("unsupported"), "{e}");
}

fn struct_field(name: &str, node: super::plan::StructNode) -> super::plan::StructField {
    super::plan::StructField {
        name: name.to_string(),
        node,
    }
}

#[test]
fn structs_flatten_to_lanes() {
    use super::plan::{StructCol, StructNode};
    // Model: x (i64), a STRUCT(i INT, j INT) nullable — lanes: x, a.i, a.j.
    let schema = cols(&[
        ("x", Ty::I64, false),
        ("a.i", Ty::I64, true),
        ("a.j", Ty::I64, true),
    ]);
    let structs = vec![StructCol {
        pos: 1,
        name: "a".to_string(),
        fields: vec![
            struct_field("i", StructNode::Leaf(1)),
            struct_field("j", StructNode::Leaf(2)),
        ],
    }];
    let prep_s = |sql: &str| {
        super::prepare_opaque(sql, "__THIS__", &schema, &[], &structs, &[], false, &[], &[])
            .map(|p| p.program)
    };
    let run = |sql: &str| -> Result<Vec<Vec<String>>, String> {
        let p = prep_s(sql).map_err(|e| e.to_string())?;
        let f = compile(&p, vec![]).map_err(|e| e.to_string())?;
        // Rows: (x=9, a={i:1, j:2}), (x=8, a=NULL) -> leaf lanes NULL.
        run_snapshot(
            &f,
            &batch(
                2,
                vec![
                    c_i64(&[Some(9), Some(8)]),
                    c_i64(&[Some(1), None]),
                    c_i64(&[Some(2), None]),
                ],
            ),
        )
        .map_err(|e| e.to_string())
    };

    // a.* expands in place to bare field names; NULL struct -> NULL fields.
    let p = prep_s("SELECT a.* FROM __THIS__").unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["i", "j"]
    );
    assert_eq!(
        run("SELECT a.* FROM __THIS__").unwrap(),
        rows(&[&["1", "2"], &["NULL", "NULL"]])
    );
    // EXCLUDE case-insensitive even quoted; REPLACE alias case wins and
    // its expr sees other columns.
    assert_eq!(
        run("SELECT a.* EXCLUDE(J) FROM __THIS__").unwrap(),
        rows(&[&["1"], &["NULL"]])
    );
    assert_eq!(
        run("SELECT a.* EXCLUDE(\"J\") FROM __THIS__").unwrap(),
        rows(&[&["1"], &["NULL"]])
    );
    let p = prep_s("SELECT a.* REPLACE(a.i + 3 AS I) FROM __THIS__").unwrap();
    assert_eq!(p.out_cols[0].name, "I");
    assert_eq!(
        run("SELECT a.* REPLACE(x + a.i AS i) FROM __THIS__").unwrap(),
        rows(&[&["10", "2"], &["NULL", "NULL"]])
    );
    // Field access paths: this.col.field, schema.this.col.field, bare.
    for sql in [
        "SELECT a.i FROM __THIS__",
        "SELECT __THIS__.a.i FROM __THIS__",
        "SELECT s.__THIS__.a.i FROM __THIS__",
    ] {
        assert_eq!(run(sql).unwrap(), rows(&[&["1"], &["NULL"]]), "{sql}");
    }
    // Output name = last part as written.
    let p = prep_s("SELECT a.I FROM __THIS__").unwrap();
    assert_eq!(p.out_cols[0].name, "I");

    // Whole-struct values, star expansion keeping the struct, and bad
    // fields are named errors.
    for (sql, needle) in [
        ("SELECT a FROM __THIS__", "whole value"),
        ("SELECT __THIS__.a FROM __THIS__", "whole value"),
        ("SELECT * FROM __THIS__", "non-scalar"),
        ("SELECT a.nope FROM __THIS__", "Could not find key"),
        ("SELECT a.i.j FROM __THIS__", "not a struct"),
        ("SELECT x.i FROM __THIS__", "not a struct"),
    ] {
        let e = prep_s(sql).unwrap_err().to_string();
        assert!(e.contains(needle), "{sql}: {e}");
    }
    // Star still serves once the struct is excluded or replaced.
    let p = prep_s("SELECT * EXCLUDE (a) FROM __THIS__").unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["x"]
    );
    let p = prep_s("SELECT * REPLACE (7 AS a) FROM __THIS__").unwrap();
    assert_eq!(
        p.out_cols.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        ["x", "a"]
    );
    // EXCLUDE-all on the struct star is legal beside another item; alone
    // it is the pinned empty-list error.
    assert_eq!(
        run("SELECT x, a.* EXCLUDE(i, j) FROM __THIS__").unwrap(),
        rows(&[&["9"], &["8"]])
    );
    let e = prep_s("SELECT a.* EXCLUDE(i, j) FROM __THIS__")
        .unwrap_err()
        .to_string();
    assert!(e.contains("SELECT list is empty"), "{e}");
}

#[test]
fn nested_struct_resolution_matches_pins() {
    use super::plan::{StructCol, StructNode};
    // Corpus case 2 shape: column t = 5-deep nested struct of t's, table
    // registered as 't' (FROM t.t suffix-matches). Lane: t.t.t.t.t.t.
    let schema = cols(&[("t.t.t.t.t.t", Ty::I64, true)]);
    let nest = |inner: StructNode| vec![struct_field("t", inner)];
    let structs = vec![StructCol {
        pos: 0,
        name: "t".to_string(),
        fields: nest(StructNode::Nested(nest(StructNode::Nested(nest(
            StructNode::Nested(nest(StructNode::Nested(nest(StructNode::Leaf(0))))),
        ))))),
    }];
    let prep_s = |sql: &str| {
        super::prepare_opaque(sql, "t", &schema, &[], &structs, &[], false, &[], &[])
            .map(|p| p.program)
    };
    // 8 parts = schema.table.column + 5 field extracts (corpus verbatim).
    let p = prep_s("SELECT t.t.t.t.t.t.t.t FROM t.t").unwrap();
    assert_eq!(p.out_cols[0].name, "t");
    let f = compile(&p, vec![]).unwrap();
    let got = run_snapshot(&f, &batch(1, vec![c_i64(&[Some(42)])])).unwrap();
    assert_eq!(got, rows(&[&["42"]]));
    // Shorter prefixes hit the whole-column / partial-struct rejections;
    // one part beyond is the pinned hard error.
    for (sql, needle) in [
        ("SELECT t.t.t FROM t.t", "whole value"),
        ("SELECT t.t.t.t FROM t.t", "whole value"), // 1 field: still a struct
        ("SELECT t.t.t.t.t.t.t.t.t FROM t.t", "not a struct"),
    ] {
        let e = prep_s(sql).unwrap_err().to_string();
        assert!(e.contains(needle), "{sql}: {e}");
    }
    // Backtracking: under an alias the schema.table path is hidden and t
    // re-reads as the column (pins: aliasing changes the value) — here
    // t.t = column.field chain.
    let p = prep_s("SELECT z.t.t.t.t.t.t FROM t.t AS z").unwrap();
    assert_eq!(p.out_cols[0].name, "t");
}

#[test]
fn many_shape_dup_key_joins_fan_out() {
    // Stage-B loop lowering (pins-stageB): per-pair emission in probe
    // order outer / build INSERTION order inner; LEFT null-extends
    // zero-match rows (NULL keys and residual-filters-all included);
    // WHERE composes per emitted candidate, null-extension included.
    let schema = cols(&[("pid", Ty::I64, true)]);
    let dim = stat("d", &[("id", Ty::I64, false), ("v", Ty::I64, false)]);
    let prep_many = |sql: &str| {
        super::prepare_opaque(
            sql,
            "__THIS__",
            &schema,
            &[],
            &[],
            std::slice::from_ref(&dim),
            true,
            &[],
            &[],
        )
    };
    let data = || {
        StaticData::Map(vec![
            (vec![KeyBits::I64(1)], vec![ScalarVal::I64(10)]),
            (vec![KeyBits::I64(2)], vec![ScalarVal::I64(20)]),
            (vec![KeyBits::I64(1)], vec![ScalarVal::I64(11)]),
        ])
    };
    let input = || batch(4, vec![c_i64(&[Some(1), Some(2), Some(3), None])]);
    let run_many = |sql: &str| -> Vec<Vec<String>> {
        let p = prep_many(sql).unwrap();
        let f = compile(&p.program, vec![data()]).unwrap();
        run_snapshot(&f, &input()).unwrap()
    };

    // LEFT: dup-key fan-out + null-extension for the miss and NULL key.
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id"),
        rows(&[
            &["1", "10"],
            &["1", "11"],
            &["2", "20"],
            &["3", "NULL"],
            &["NULL", "NULL"],
        ])
    );
    // INNER: zero-match rows drop.
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ JOIN d ON pid = d.id"),
        rows(&[&["1", "10"], &["1", "11"], &["2", "20"]])
    );
    // Residual filters PER MATCH; a row whose matches are all filtered
    // still null-extends (measured).
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id AND d.v > 10"),
        rows(&[&["1", "11"], &["2", "20"], &["3", "NULL"], &["NULL", "NULL"]])
    );
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id AND d.v > 100"),
        rows(&[
            &["1", "NULL"],
            &["2", "NULL"],
            &["3", "NULL"],
            &["NULL", "NULL"],
        ])
    );
    // WHERE applies to every emitted candidate incl. the null-extension.
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id WHERE v IS NULL"),
        rows(&[&["3", "NULL"], &["NULL", "NULL"]])
    );
    assert_eq!(
        run_many("SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id WHERE v > 10"),
        rows(&[&["1", "11"], &["2", "20"]])
    );
    // Under the DEFAULT shape the same dup-key data still errors at
    // materialization (the 1:1 map contract is untouched).
    let p = prepare(
        "SELECT pid, v FROM __THIS__ LEFT JOIN d ON pid = d.id",
        "__THIS__",
        &schema,
        std::slice::from_ref(&dim),
    )
    .unwrap();
    let e = match compile(&p.program, vec![data()]) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("dup keys under the default shape must error"),
    };
    assert!(e.contains("duplicate map key"), "{e}");
    // Multi-join under 'many' is the named stage-B restriction.
    let dim_b = stat("d", &[("id", Ty::I64, false), ("v", Ty::I64, false)]);
    let dim2 = stat("d2", &[("id", Ty::I64, false), ("w", Ty::I64, false)]);
    let e = match super::prepare_opaque(
        "SELECT pid FROM __THIS__ LEFT JOIN d ON pid = d.id LEFT JOIN d2 ON pid = d2.id",
        "__THIS__",
        &schema,
        &[],
        &[],
        &[dim_b, dim2],
        true,

        &[],
        &[],
    ) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("multi-join under many must be the named restriction"),
    };
    assert!(e.contains("one join per query"), "{e}");
}

#[test]
fn many_shape_keyless_and_inequality_joins() {
    // pins-stageB/cross-inequality.json: keyless joins are cross-product-
    // then-filter; LEFT null-extends zero-match rows; ON NULL = 2 matches
    // nothing. All ride the empty-key multimap (whole-table range).
    let schema = cols(&[("pid", Ty::I64, false)]);
    let dim = stat("d", &[("id", Ty::I64, false)]);
    let prep_many = |sql: &str| {
        super::prepare_opaque(
            sql,
            "__THIS__",
            &schema,
            &[],
            &[],
            std::slice::from_ref(&dim),
            true,
            &[],
            &[],
        )
    };
    let data = || {
        StaticData::Map(vec![
            (vec![], vec![ScalarVal::I64(2)]),
            (vec![], vec![ScalarVal::I64(3)]),
            (vec![], vec![ScalarVal::I64(4)]),
        ])
    };
    let input = || batch(3, vec![c_i64(&[Some(1), Some(2), Some(3)])]);
    let run_many = |sql: &str| -> Result<Vec<Vec<String>>, String> {
        let p = prep_many(sql).map_err(|e| e.to_string())?;
        let f = compile(&p.program, vec![data()]).map_err(|e| e.to_string())?;
        run_snapshot(&f, &input()).map_err(|e| e.to_string())
    };

    // Plain comma cross join: 3x3.
    assert_eq!(
        run_many("SELECT pid, id FROM __THIS__, d").unwrap(),
        rows(&[
            &["1", "2"],
            &["1", "3"],
            &["1", "4"],
            &["2", "2"],
            &["2", "3"],
            &["2", "4"],
            &["3", "2"],
            &["3", "3"],
            &["3", "4"],
        ])
    );
    // Inequality INNER: cross + filter.
    assert_eq!(
        run_many("SELECT pid, id FROM __THIS__ JOIN d ON pid > d.id").unwrap(),
        rows(&[&["3", "2"]])
    );
    // Inequality LEFT: null-extension for rows with no match.
    assert_eq!(
        run_many("SELECT pid, id FROM __THIS__ LEFT JOIN d ON pid > d.id").unwrap(),
        rows(&[&["1", "NULL"], &["2", "NULL"], &["3", "2"]])
    );
    // Constant-NULL ON: matches nothing; LEFT null-extends everything.
    assert_eq!(
        run_many("SELECT pid, id FROM __THIS__ LEFT JOIN d ON NULL = 2").unwrap(),
        rows(&[&["1", "NULL"], &["2", "NULL"], &["3", "NULL"]])
    );
}

#[test]
fn many_shape_self_joins() {
    // Stage-B self-joins: the batch is BOTH sides — a keyless batchmap
    // built per call, the whole ON as residual (cross-then-filter is
    // bit-identical under multiplicity; pins-stageB).
    let schema = cols(&[("i", Ty::I64, false), ("j", Ty::I64, false)]);
    let prep_many = |sql: &str| {
        super::prepare_opaque(sql, "__THIS__", &schema, &[], &[], &[], true, &[], &[])
    };
    let input = || {
        batch(
            2,
            vec![c_i64(&[Some(1), Some(2)]), c_i64(&[Some(10), Some(20)])],
        )
    };
    let run_many = |sql: &str| -> Result<Vec<Vec<String>>, String> {
        let p = prep_many(sql).map_err(|e| e.to_string())?;
        let f = compile(&p.program, vec![StaticData::Map(Vec::new())])
            .map_err(|e| e.to_string())?;
        run_snapshot(&f, &input()).map_err(|e| e.to_string())
    };

    // Comma cross self-join: 2x2 in probe-outer/batch-insertion order.
    assert_eq!(
        run_many("SELECT i1.i, i2.j FROM __THIS__ i1, __THIS__ i2").unwrap(),
        rows(&[&["1", "10"], &["1", "20"], &["2", "10"], &["2", "20"]])
    );
    // Equi conjuncts stay WHERE (cross-then-filter).
    assert_eq!(
        run_many("SELECT i1.i, i2.j FROM __THIS__ i1, __THIS__ i2 WHERE i1.i = i2.i")
            .unwrap(),
        rows(&[&["1", "10"], &["2", "20"]])
    );
    // ON self-join, inequality + LEFT null-extension.
    assert_eq!(
        run_many("SELECT i1.i, i2.i FROM __THIS__ i1 JOIN __THIS__ i2 ON i1.i > i2.i")
            .unwrap(),
        rows(&[&["2", "1"]])
    );
    assert_eq!(
        run_many(
            "SELECT i1.i, i2.i FROM __THIS__ i1 LEFT JOIN __THIS__ i2 ON i1.i > i2.i"
        )
        .unwrap(),
        rows(&[&["1", "NULL"], &["2", "1"]])
    );
    // Star EXCLUDE over the self-join (the corpus shapes): unqualified
    // strips both copies; qualified strips one side's.
    let p = prep_many("SELECT * EXCLUDE (i) FROM __THIS__ i1, __THIS__ i2").unwrap();
    assert_eq!(
        p.program
            .out_cols
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["j", "j_1"] // dup names go through the boundary rename
    );
    let p = prep_many("SELECT i1.* EXCLUDE (i), i2.* EXCLUDE (j) FROM __THIS__ i1, __THIS__ i2")
        .unwrap();
    assert_eq!(
        p.program
            .out_cols
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["j", "i"]
    );
    // Default shapes: still the named rejection.
    let e = prep(
        "SELECT i1.i FROM __THIS__ i1, __THIS__ i2",
        &schema,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("dynamic table"), "{e}");
    // USING self-join: named stage-B follow-up.
    let e = match prep_many("SELECT * FROM __THIS__ i1 JOIN __THIS__ i2 USING (i)") {
        Err(e) => e.to_string(),
        Ok(_) => panic!("USING self-join must stay a named rejection"),
    };
    assert!(e.contains("USING/NATURAL"), "{e}");
}
