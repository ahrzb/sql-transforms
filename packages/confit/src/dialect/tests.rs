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
    // Bind errors are Bind, not Unsupported: bool arithmetic is WRONG.
    let bad = Expr::Bin {
        op: BinOp::Add,
        l: Box::new(lit(DTy::Bool, "true")),
        r: Box::new(i()),
    };
    assert!(matches!(bad.ty(), Err(DialectError::Bind(_))));
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
