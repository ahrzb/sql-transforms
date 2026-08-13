//! Plan-core tests: the three mandatory properties, executed.

use super::plan::{BinOp, Catalog, ColDef, Expr, Rel, Table, UnOp};
use super::text;
use super::ty::DTy;
use super::verify::verify;
use super::DialectError;

fn cat() -> Catalog {
    Catalog {
        tables: vec![Table {
            name: "t".into(),
            cols: vec![
                ColDef {
                    name: "a".into(),
                    ty: DTy::I32,
                    nullable: true,
                },
                ColDef {
                    name: "b".into(),
                    ty: DTy::Str,
                    nullable: true,
                },
                ColDef {
                    name: "c".into(),
                    ty: DTy::F64,
                    nullable: false,
                },
            ],
        }],
    }
}

fn col(ordinal: usize, name: &str, ty: DTy) -> Expr {
    Expr::Col {
        ordinal,
        name: name.into(),
        ty,
    }
}

fn lit(ty: DTy, lexeme: &str) -> Expr {
    Expr::Lit {
        lexeme: lexeme.into(),
        ty,
    }
}

/// A plan exercising every v0 node once.
fn kitchen_sink() -> Rel {
    Rel::Project {
        input: Box::new(Rel::Filter {
            input: Box::new(Rel::Scan { table: "t".into() }),
            pred: Expr::Bin {
                op: BinOp::And,
                l: Box::new(Expr::IsNull {
                    negated: true,
                    e: Box::new(col(0, "a", DTy::I32)),
                }),
                r: Box::new(Expr::Bin {
                    op: BinOp::Gt,
                    l: Box::new(col(2, "c", DTy::F64)),
                    r: Box::new(lit(DTy::F64, "1e16")),
                }),
            },
        }),
        items: vec![
            ("a".into(), col(0, "a", DTy::I32)),
            (
                "half".into(),
                Expr::Bin {
                    op: BinOp::FDiv,
                    l: Box::new(col(0, "a", DTy::I32)),
                    r: Box::new(lit(DTy::I32, "2")),
                },
            ),
            (
                "label".into(),
                Expr::Case {
                    whens: vec![(
                        Expr::IsDistinct {
                            negated: true,
                            l: Box::new(col(0, "a", DTy::I32)),
                            r: Box::new(lit(DTy::I32, "1")),
                        },
                        lit(DTy::Str, "one"),
                    )],
                    else_: Some(Box::new(Expr::Bin {
                        op: BinOp::Concat,
                        l: Box::new(col(1, "b", DTy::Str)),
                        r: Box::new(lit(DTy::Str, "!\"\\")),
                    })),
                },
            ),
            (
                "wide".into(),
                Expr::Cast {
                    strict: false,
                    e: Box::new(Expr::Un {
                        op: UnOp::Neg,
                        e: Box::new(col(0, "a", DTy::I32)),
                    }),
                    target: DTy::I64,
                },
            ),
        ],
    }
}

#[test]
fn round_trip_kitchen_sink() {
    let p = kitchen_sink();
    verify(&p, &cat()).unwrap();
    let printed = text::print(&p);
    let reparsed = text::parse(&printed).unwrap();
    assert_eq!(p, reparsed, "parse(print(p)) == p\n{printed}");
    // Canonical: printing the reparse is byte-identical.
    assert_eq!(printed, text::print(&reparsed));
}

#[test]
fn parse_is_whitespace_insensitive() {
    let p = kitchen_sink();
    let squashed: String = text::print(&p)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    assert_eq!(p, text::parse(&squashed).unwrap());
}

#[test]
fn schema_derives() {
    let s = kitchen_sink().schema(&cat()).unwrap();
    assert_eq!(
        s,
        vec![
            ("a".into(), DTy::I32),
            ("half".into(), DTy::F64), // pinned: / is DOUBLE on integers
            ("label".into(), DTy::Str),
            ("wide".into(), DTy::I64),
        ]
    );
}

#[test]
fn verifier_rejects_bad_ordinal_and_drifted_carries() {
    let bad_ordinal = Rel::Project {
        input: Box::new(Rel::Scan { table: "t".into() }),
        items: vec![("x".into(), col(9, "a", DTy::I32))],
    };
    assert!(matches!(
        verify(&bad_ordinal, &cat()),
        Err(DialectError::Internal(_))
    ));

    let drifted_name = Rel::Project {
        input: Box::new(Rel::Scan { table: "t".into() }),
        items: vec![("x".into(), col(0, "zzz", DTy::I32))],
    };
    assert!(matches!(
        verify(&drifted_name, &cat()),
        Err(DialectError::Internal(_))
    ));

    let drifted_ty = Rel::Project {
        input: Box::new(Rel::Scan { table: "t".into() }),
        items: vec![("x".into(), col(0, "a", DTy::I64))],
    };
    assert!(matches!(
        verify(&drifted_ty, &cat()),
        Err(DialectError::Internal(_))
    ));

    // Case-insensitive spelling match is DuckDB semantics, not drift.
    let respelled = Rel::Project {
        input: Box::new(Rel::Scan { table: "t".into() }),
        items: vec![("x".into(), col(0, "A", DTy::I32))],
    };
    verify(&respelled, &cat()).unwrap();
}

#[test]
fn verifier_rejects_non_bool_filter() {
    let p = Rel::Filter {
        input: Box::new(Rel::Scan { table: "t".into() }),
        pred: col(0, "a", DTy::I32),
    };
    assert!(matches!(verify(&p, &cat()), Err(DialectError::Internal(_))));
}

#[test]
fn unknown_table_is_bind_not_internal() {
    let p = Rel::Scan {
        table: "nope".into(),
    };
    assert!(matches!(verify(&p, &cat()), Err(DialectError::Bind(_))));
}

#[test]
fn derivation_follows_the_pins() {
    let i = || col(0, "a", DTy::I32);
    // strings-operators.json: typeof(1/2) = DOUBLE, typeof(1//2) = INTEGER.
    let fdiv = Expr::Bin {
        op: BinOp::FDiv,
        l: Box::new(i()),
        r: Box::new(i()),
    };
    assert_eq!(fdiv.ty().unwrap(), DTy::F64);
    let idiv = Expr::Bin {
        op: BinOp::IDiv,
        l: Box::new(i()),
        r: Box::new(i()),
    };
    assert_eq!(idiv.ty().unwrap(), DTy::I32);
    // Named refusals, not guesses: decimal arithmetic (lattice phase 5)…
    let dec = Expr::Bin {
        op: BinOp::Add,
        l: Box::new(lit(DTy::Dec(3, 1), "1.5")),
        r: Box::new(lit(DTy::Dec(3, 1), "0.5")),
    };
    assert!(matches!(dec.ty(), Err(DialectError::Unsupported(_))));
    // …but decimal comparison is fine.
    let dec_cmp = Expr::Bin {
        op: BinOp::Lt,
        l: Box::new(lit(DTy::Dec(3, 1), "1.5")),
        r: Box::new(lit(DTy::Dec(3, 1), "0.5")),
    };
    assert_eq!(dec_cmp.ty().unwrap(), DTy::Bool);
    // Type mismatches DuckDB would coerce refuse as Unsupported — Bind is
    // reserved for what the oracle itself rejects (unknown names).
    let coerced = Expr::Bin {
        op: BinOp::Add,
        l: Box::new(lit(DTy::Bool, "true")),
        r: Box::new(i()),
    };
    assert!(matches!(coerced.ty(), Err(DialectError::Unsupported(_))));
}

#[test]
fn ty_short_names_round_trip() {
    let all = vec![
        DTy::Bool,
        DTy::I8,
        DTy::I16,
        DTy::I32,
        DTy::I64,
        DTy::I128,
        DTy::U8,
        DTy::U16,
        DTy::U32,
        DTy::U64,
        DTy::U128,
        DTy::F32,
        DTy::F64,
        DTy::Dec(38, 9),
        DTy::Str,
        DTy::Blob,
        DTy::Date,
        DTy::Time,
        DTy::TimeTz,
        DTy::TsS,
        DTy::TsMs,
        DTy::TsUs,
        DTy::TsNs,
        DTy::TsTz,
        DTy::Interval,
        DTy::Uuid,
        DTy::List(Box::new(DTy::Dec(18, 3))),
        DTy::Struct(vec![
            ("a".into(), DTy::I32),
            ("b".into(), DTy::List(Box::new(DTy::Str))),
        ]),
    ];
    for t in all {
        assert_eq!(
            DTy::parse(&t.name()),
            Some(t.clone()),
            "short name: {}",
            t.name()
        );
    }
}

#[test]
fn duckdb_type_names_ingest_and_print() {
    for (duck, ty) in [
        ("INTEGER", DTy::I32),
        ("DECIMAL(18,3)", DTy::Dec(18, 3)),
        ("TIMESTAMP WITH TIME ZONE", DTy::TsTz),
        ("VARCHAR", DTy::Str),
        ("INTEGER[]", DTy::List(Box::new(DTy::I32))),
    ] {
        assert_eq!(DTy::from_duckdb(duck).unwrap(), ty, "{duck}");
    }
    // Round-trip through the DuckDB spelling where both directions exist.
    assert_eq!(
        DTy::from_duckdb(&DTy::Dec(38, 9).duckdb_name().unwrap()).unwrap(),
        DTy::Dec(38, 9)
    );
    // Not-yet-carried catalog types refuse by name.
    assert!(matches!(
        DTy::from_duckdb("MAP(INTEGER, INTEGER)"),
        Err(DialectError::Unsupported(_))
    ));
}

// --- DuckDB frontend + printer (L1 on SQL-derived plans) --------------------

#[test]
fn frontend_binds_and_round_trips() {
    let c = cat();
    let sql = "SELECT a, t.b AS label, (a + 1) * 2 AS scaled, CASE WHEN a IS NOT NULL THEN 'y' ELSE b END AS f FROM t WHERE c > 1.5e3";
    let p = super::duckdb::parse_sql(sql, &c).unwrap();
    // L1 through the plan text.
    assert_eq!(super::text::parse(&super::text::print(&p)).unwrap(), p);
    // The printed SQL parses back to the SAME plan (frontend∘printer fixpoint).
    let printed = super::duckdb::print_sql(&p, &c).unwrap();
    let reparsed = super::duckdb::parse_sql(&printed, &c).unwrap();
    // On ribbon shapes the fixpoint is exact: the same plan comes back.
    assert_eq!(reparsed, p, "printed: {printed}");
}

#[test]
fn frontend_star_and_spelling() {
    let c = cat();
    let p = super::duckdb::parse_sql("SELECT *, A FROM t", &c).unwrap();
    let schema = p.schema(&c).unwrap();
    // * expands to catalog spellings; bare `A` preserves the QUERY spelling.
    let names: Vec<&str> = schema.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, vec!["a", "b", "c", "A"]);
}

#[test]
fn frontend_named_refusals() {
    let c = cat();
    for (sql, needle) in [
        ("SELECT a FROM t ORDER BY a", "ORDER BY"),
        ("SELECT count(a) FROM t", "function: count"),
        ("SELECT 1e3 FROM t", "auto-name"),
        // sqlparser reads ASOF as Snowflake syntax and cleanly fails the
        // parse (never misparsing it as a table alias).
        ("SELECT a FROM t ASOF JOIN t AS u ON t.a >= u.a", "MATCH_CONDITION"),
        ("SELECT DISTINCT a FROM t", "DISTINCT"),
        ("SELECT NULL AS n FROM t", "NULL literal"),
    ] {
        match super::duckdb::parse_sql(sql, &c) {
            Err(DialectError::Unsupported(m)) => {
                assert!(m.contains(needle), "{sql}: {m}");
            }
            other => panic!("{sql}: expected Unsupported, got {other:?}"),
        }
    }
    assert!(matches!(
        super::duckdb::parse_sql("SELECT zzz FROM t", &c),
        Err(DialectError::Bind(_))
    ));
}

#[test]
fn printer_quotes_and_parenthesizes() {
    let c = cat();
    let p = super::duckdb::parse_sql("SELECT a // 2 + 1 AS q FROM t WHERE NOT (b = 'x''y')", &c)
        .unwrap();
    let printed = super::duckdb::print_sql(&p, &c).unwrap();
    assert!(printed.contains("\"a\""), "{printed}");
    assert!(printed.contains("'x''y'"), "{printed}");
    assert!(printed.contains("//"), "{printed}");
}

#[test]
fn literal_typing_follows_the_lattice() {
    // typeof(1)=INTEGER, typeof(3000000000)=BIGINT, decimal by digits,
    // exponent = DOUBLE (2026-08-11-duckdb-type-lattice-design.md +
    // pins-dialect).
    let c = cat();
    let sql = "SELECT 1 AS a, 3000000000 AS b, 1.50 AS c, 0.1 AS d, 1e3 AS e FROM t";
    let p = super::duckdb::parse_sql(sql, &c).unwrap();
    let tys: Vec<DTy> = p.schema(&c).unwrap().into_iter().map(|(_, t)| t).collect();
    assert_eq!(
        tys,
        // 0.1 is DECIMAL(2,1): a bare zero integer part still counts one
        // digit (review-confirmed measurement; typeof(0.5) = DECIMAL(2,1)).
        vec![DTy::I32, DTy::I64, DTy::Dec(3, 2), DTy::Dec(2, 1), DTy::F64]
    );
}

// --- Join (TASK-104; 2026-08-13-dialect-join-node-design.md) ----------------

fn cat2() -> Catalog {
    let t = |name: &str, cols: Vec<(&str, DTy)>| Table {
        name: name.into(),
        cols: cols
            .into_iter()
            .map(|(n, ty)| ColDef {
                name: n.into(),
                ty,
                nullable: true,
            })
            .collect(),
    };
    Catalog {
        tables: vec![
            t("t1", vec![("a", DTy::I32), ("b", DTy::Str)]),
            t("t2", vec![("a", DTy::I32), ("c", DTy::F64)]),
            t("t3", vec![("d", DTy::I32)]),
        ],
    }
}

/// parse → plan-text round-trip → print → reparse must reach the SAME plan.
fn assert_join_fixpoint(sql: &str, c: &Catalog) -> Rel {
    let p = super::duckdb::parse_sql(sql, c).unwrap_or_else(|e| panic!("{sql}: {e:?}"));
    assert_eq!(
        super::text::parse(&super::text::print(&p)).unwrap(),
        p,
        "plan-text round trip: {sql}"
    );
    let printed = super::duckdb::print_sql(&p, c).unwrap();
    assert_eq!(
        super::duckdb::parse_sql(&printed, c).unwrap(),
        p,
        "fixpoint: {sql} -> {printed}"
    );
    p
}

#[test]
fn join_on_inner_left_bind_and_fixpoint() {
    let c = cat2();
    let p = assert_join_fixpoint(
        "SELECT t1.a AS x, t2.c AS y FROM t1 INNER JOIN t2 ON t1.a = t2.a WHERE t2.c > 1e0",
        &c,
    );
    assert_eq!(
        p.schema(&c).unwrap(),
        vec![("x".into(), DTy::I32), ("y".into(), DTy::F64)]
    );
    assert_join_fixpoint(
        "SELECT t1.b AS b FROM t1 LEFT JOIN t2 ON t1.a IS NOT DISTINCT FROM t2.a",
        &c,
    );
    // General ON predicates are admitted, not just equality conjunctions.
    assert_join_fixpoint("SELECT t1.b AS b FROM t1 JOIN t2 ON t1.a > t2.a", &c);
    assert_join_fixpoint(
        "SELECT t1.b AS b FROM t1 JOIN t2 ON t1.a = t2.a AND t2.c < 5e0",
        &c,
    );
}

#[test]
fn join_star_keeps_both_sides_on_on_joins() {
    let c = cat2();
    let p = assert_join_fixpoint("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a", &c);
    let names: Vec<String> = p.schema(&c).unwrap().into_iter().map(|(n, _)| n).collect();
    // ON joins keep both key columns — duplicate `a` included, printable
    // through qualified refs.
    assert_eq!(names, vec!["a", "b", "a", "c"]);
}

#[test]
fn join_right_full_cross_comma_and_chains() {
    let c = cat2();
    assert_join_fixpoint("SELECT t2.c AS c FROM t1 RIGHT JOIN t2 ON t1.a = t2.a", &c);
    assert_join_fixpoint("SELECT t1.a AS x, t2.a AS y FROM t1 FULL JOIN t2 ON t1.a = t2.a", &c);
    assert_join_fixpoint("SELECT t1.a AS x, t3.d AS d FROM t1 CROSS JOIN t3", &c);
    // Comma-joins are CROSS with keys staying in WHERE (author structure).
    let p = assert_join_fixpoint("SELECT t1.b AS b FROM t1, t2 WHERE t1.a = t2.a", &c);
    assert!(
        matches!(&p, Rel::Project { input, .. } if matches!(input.as_ref(), Rel::Filter { .. })),
        "comma-join keys stay in the Filter"
    );
    // Chains fold left-nested; later ON sees every earlier source.
    assert_join_fixpoint(
        "SELECT t3.d AS d FROM t1 JOIN t2 ON t1.a = t2.a JOIN t3 ON t3.d = t1.a",
        &c,
    );
    assert_join_fixpoint(
        "SELECT t3.d AS d FROM t1, t3 JOIN t2 ON t1.a = t2.a",
        &c,
    );
}

#[test]
fn join_using_and_natural_follow_the_probe() {
    // Measured 2026-08-13 (probe in the TASK-104 commit): USING merges the
    // key column at its LEFT-side position; remaining left, then right.
    let c = cat2();
    let p = assert_join_fixpoint("SELECT * FROM t1 JOIN t2 USING (a)", &c);
    let names: Vec<String> = p.schema(&c).unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, vec!["a", "b", "c"]);
    // The merged name resolves unqualified even though both sides carry it.
    assert_join_fixpoint("SELECT a AS k FROM t1 JOIN t2 USING (a) WHERE a > 0", &c);
    // RIGHT: merged column is the right side's; LEFT/INNER: the left's.
    assert_join_fixpoint("SELECT a AS k FROM t1 RIGHT JOIN t2 USING (a)", &c);
    // FULL: merged column is the null-preferring CASE over both sides.
    let p = assert_join_fixpoint("SELECT a AS k FROM t1 FULL JOIN t2 USING (a)", &c);
    let text = super::text::print(&p);
    assert!(text.contains("(case"), "FULL USING merge is a CASE:\n{text}");
    // Qualified access to both underlying columns survives USING.
    assert_join_fixpoint("SELECT t1.a AS l, t2.a AS r FROM t1 FULL JOIN t2 USING (a)", &c);
    // NATURAL = USING(common columns).
    let p = assert_join_fixpoint("SELECT * FROM t1 NATURAL JOIN t2", &c);
    let names: Vec<String> = p.schema(&c).unwrap().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, vec!["a", "b", "c"]);
    // NATURAL with no common columns is DuckDB's own binder error -> Bind.
    assert!(matches!(
        super::duckdb::parse_sql("SELECT 1 AS one FROM t1 NATURAL JOIN t3", &c),
        Err(DialectError::Bind(_))
    ));
}

#[test]
fn join_named_refusals_and_bind_errors() {
    let c = cat2();
    for (sql, needle) in [
        // sqlparser reads ASOF as Snowflake syntax and fails the parse
        // cleanly rather than misparsing it as a table alias.
        (
            "SELECT t1.a AS x FROM t1 ASOF JOIN t2 ON t1.a >= t2.a",
            "MATCH_CONDITION",
        ),
        ("SELECT t1.a AS x FROM t1 SEMI JOIN t2 ON t1.a = t2.a", "SEMI"),
        ("SELECT t1.a AS x FROM t1 POSITIONAL JOIN t2", "POSITIONAL"),
        ("SELECT 1 AS one FROM t1 JOIN t2", "join without ON"),
    ] {
        match super::duckdb::parse_sql(sql, &c) {
            Err(DialectError::Unsupported(m)) => {
                assert!(m.contains(needle), "{sql}: {m}");
            }
            other => panic!("{sql}: expected Unsupported, got {other:?}"),
        }
    }
    // DuckDB itself rejects these — Bind, not Unsupported.
    for sql in [
        // `a` lives on both sides of an ON join: ambiguous.
        "SELECT a FROM t1 JOIN t2 ON t1.a = t2.a",
        // duplicate table alias
        "SELECT 1 AS one FROM t1 AS x JOIN t2 AS x ON true",
        // unknown qualifier
        "SELECT zz.a FROM t1 JOIN t2 ON t1.a = t2.a",
    ] {
        assert!(
            matches!(super::duckdb::parse_sql(sql, &c), Err(DialectError::Bind(_))),
            "{sql}: expected Bind"
        );
    }
}

#[test]
fn join_prints_on_spark_with_portable_spellings() {
    let c = cat2();
    let p = super::duckdb::parse_sql(
        "SELECT t1.b AS b FROM t1 LEFT JOIN t2 ON t1.a IS NOT DISTINCT FROM t2.a",
        &c,
    )
    .unwrap();
    let spark = super::spark::print_sql(&p, &c).unwrap();
    assert!(
        spark.contains("LEFT JOIN") && spark.contains("IS NOT DISTINCT FROM"),
        "{spark}"
    );
}

// --- BigQuery printer (documented-semantics; phase-4 remote gate owed) ------

#[test]
fn bigquery_prints_the_forced_spellings() {
    let c = cat();
    // a is i32: arithmetic computes at i32 (trap at 2^31) — refuses by name.
    let p = super::duckdb::parse_sql("SELECT a // 2 AS q FROM t", &c).unwrap();
    assert!(matches!(
        super::bigquery::print_sql(&p, &c),
        Err(DialectError::Unsupported(ref m)) if m.contains("INT64")
    ));

    // Widened to i64 the computation matches BigQuery's overflow class.
    let sql = "SELECT CAST(a AS BIGINT) // 2 AS q, 1.5 / c AS d, b || 'x\\y' AS s FROM t WHERE b IS NOT DISTINCT FROM 'o''k'";
    let p = super::duckdb::parse_sql(sql, &c).unwrap();
    let printed = super::bigquery::print_sql(&p, &c).unwrap();
    // DIV inside the zero-divisor guard; / as IEEE_DIVIDE; backslash
    // string escaping; IS NOT DISTINCT FROM on non-floats prints direct.
    assert!(
        printed.contains("DIV(CAST(`a` AS INT64), 2)")
            && printed.contains("WHEN 2 = 0 THEN CAST(NULL AS INT64)")
            && printed.contains("IEEE_DIVIDE(CAST(NUMERIC '1.5' AS FLOAT64), `c`)")
            && printed.contains("(`b` || 'x\\\\y')")
            && printed.contains("(`b` IS NOT DISTINCT FROM 'o\\'k')"),
        "{printed}"
    );
}

#[test]
fn bigquery_type_landing_zones() {
    use super::bigquery;
    let c = cat();
    for (sql, needle) in [
        (
            "SELECT TRY_CAST(a AS DOUBLE) AS x FROM t",
            "SAFE_CAST(`a` AS FLOAT64)",
        ),
        (
            "SELECT CAST(a AS DECIMAL(18,3)) AS x FROM t",
            "CAST(`a` AS NUMERIC(18,3))",
        ),
        (
            "SELECT CAST(a AS DECIMAL(38,20)) AS x FROM t",
            "CAST(`a` AS BIGNUMERIC(38,20))",
        ),
    ] {
        let p = super::duckdb::parse_sql(sql, &c).unwrap();
        let printed = bigquery::print_sql(&p, &c).unwrap();
        assert!(printed.contains(needle), "{sql}: {printed}");
    }
    // Named refusals: narrow-int CAST targets stay unforced.
    let p = super::duckdb::parse_sql("SELECT CAST(a AS INTEGER) AS x FROM t", &c).unwrap();
    assert!(matches!(
        bigquery::print_sql(&p, &c),
        Err(DialectError::Unsupported(_))
    ));
}

// --- Spark printer (pinned config; live gate in test_dialect_cross_engine_gate.py) --

#[test]
fn spark_prints_the_forced_spellings() {
    let c = cat();
    // Narrow ints are native in Spark: i32 arithmetic prints plainly.
    let p = super::duckdb::parse_sql("SELECT a + 1 AS s FROM t", &c).unwrap();
    assert_eq!(
        super::spark::print_sql(&p, &c).unwrap(),
        "SELECT (`a` + 1) AS `s` FROM `t`"
    );
    // // re-narrows div's BIGINT through a checked CAST inside the
    // zero-divisor guard (DuckDB: NULL on zero; trap class forced).
    let p = super::duckdb::parse_sql("SELECT a // 2 AS q FROM t", &c).unwrap();
    let printed = super::spark::print_sql(&p, &c).unwrap();
    assert!(
        printed.contains("WHEN 2 = 0 THEN CAST(NULL AS INT)")
            && printed.contains("CAST((`a` div 2) AS INT)"),
        "{printed}"
    );
    // / carries the IEEE zero-divisor CASE and casts decimal operands.
    let p = super::duckdb::parse_sql("SELECT 1.5 / c AS d FROM t", &c).unwrap();
    let printed = super::spark::print_sql(&p, &c).unwrap();
    assert!(
        printed.contains("CAST(1.5 AS DOUBLE) / `c`")
            && printed.contains("CAST('NaN' AS DOUBLE)")
            && printed.contains("CAST('-Infinity' AS DOUBLE)"),
        "{printed}"
    );
    // % guards zero and the INT_MIN % -1 trap.
    let p = super::duckdb::parse_sql("SELECT CAST(a AS BIGINT) % 3 AS r FROM t", &c).unwrap();
    let printed = super::spark::print_sql(&p, &c).unwrap();
    assert!(
        printed.contains("WHEN 3 = 0 THEN CAST(NULL AS BIGINT)")
            && printed.contains("(-9223372036854775807 - 1)"),
        "{printed}"
    );
    // Cast forcing: DOUBLE->int via rint (half-even, DuckDB's rule);
    // string sources refuse by name.
    let p = super::duckdb::parse_sql("SELECT CAST(c AS INTEGER) AS i FROM t", &c).unwrap();
    assert!(super::spark::print_sql(&p, &c)
        .unwrap()
        .contains("CAST(rint(`c`) AS INT)"));
    let p = super::duckdb::parse_sql("SELECT TRY_CAST(b AS INTEGER) AS i FROM t", &c).unwrap();
    assert!(matches!(
        super::spark::print_sql(&p, &c),
        Err(DialectError::Unsupported(ref m)) if m.contains("conversion domain")
    ));
    // Unbought landing zones refuse by name.
    let p = super::duckdb::parse_sql("SELECT CAST(b AS UUID) AS u FROM t", &c).unwrap();
    assert!(matches!(
        super::spark::print_sql(&p, &c),
        Err(DialectError::Unsupported(_))
    ));
}
