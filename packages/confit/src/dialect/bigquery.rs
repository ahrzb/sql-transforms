//! The BigQuery printer: plan → GoogleSQL, forcing the plan's DuckDB-pinned
//! semantics in BigQuery's syntax (design D1/D3).
//!
//! STATUS: documented-semantics, unprobed. Every spelling below follows
//! BigQuery's published GoogleSQL reference; none has run against the real
//! service yet — that is the design's phase-4 remote gate, still owed.
//! Until it runs, the refusal set stays conservative: anything whose
//! BigQuery behavior could diverge from the pinned DuckDB semantics in a
//! way documentation cannot settle refuses by name. There is no BigQuery
//! frontend; this dialect is print-only (the pushdown direction).
//!
//! The load-bearing decisions, each traceable to the design's type table
//! or a pin:
//!
//! * **Narrow-int arithmetic refuses.** BigQuery has only INT64; DuckDB's
//!   i8/i16/i32 operators trap at their own width (pinned error class), and
//!   widening erases the trap threshold — a value-vs-error divergence, not
//!   an ε. INT64 arithmetic is fine: both engines error on 64-bit overflow.
//!   Narrow-int columns may still be selected, compared, filtered — no trap
//!   exists on those paths. Guard expressions via `ERROR()` are the named
//!   phase-4 upgrade path.
//! * **`/` forces FLOAT64.** DuckDB `/` is double division whatever the
//!   operands (pinned); BigQuery `NUMERIC / NUMERIC` stays NUMERIC, so
//!   decimal operands are cast to FLOAT64 explicitly. INT64/INT64 already
//!   yields FLOAT64 in BigQuery.
//! * **`//` → `DIV()`, `%` → `MOD()`,** both INT64-only: BigQuery's DIV and
//!   MOD truncate toward zero with sign-of-dividend remainders — the same
//!   observed values as DuckDB's pinned ((-7)//2, (-7)%2, 7//(-2), 7%(-2))
//!   = (-3, -1, -3, 1).
//! * **Decimal literals print typed** (`NUMERIC '1.5'` / `BIGNUMERIC`):
//!   a bare decimal-pointed literal is FLOAT64 in BigQuery but DECIMAL(p,s)
//!   in DuckDB — printing the lexeme bare would silently change its type.
//! * **Strings escape with backslashes:** `''` is not an escape in
//!   GoogleSQL.

use super::plan::{BinOp, Catalog, Expr, Rel, UnOp};
use super::ty::DTy;
use super::verify::verify;
use super::{unsup, DialectError};

/// Print a verified plan as BigQuery GoogleSQL. `cat` supplies scan schemas.
pub fn print_sql(rel: &Rel, cat: &Catalog) -> Result<String, DialectError> {
    verify(rel, cat)?;
    let (sql, _) = print_query(rel, cat, 0)?;
    Ok(sql)
}

fn quote_ident(name: &str) -> String {
    format!("`{}`", name.replace('\\', "\\\\").replace('`', "\\`"))
}

/// The type's BigQuery landing zone (design table, "to pin by probe"):
/// used for CAST targets and typed literals. Refusals are the design's
/// rows, verbatim.
fn bq_name(ty: &DTy) -> Result<String, DialectError> {
    Ok(match ty {
        DTy::Bool => "BOOL".into(),
        DTy::I64 => "INT64".into(),
        DTy::I8 | DTy::I16 | DTy::I32 => {
            return Err(unsup(format!(
                "bigquery: {} CAST target (INT64-only; width traps not forced)",
                ty.name()
            )));
        }
        DTy::F64 => "FLOAT64".into(),
        DTy::Dec(p, s) => {
            // NUMERIC: scale <= 9, integer digits <= 29. BIGNUMERIC:
            // scale <= 38, integer digits <= 38 — every DuckDB DECIMAL fits.
            if *s <= 9 && p - s <= 29 {
                format!("NUMERIC({p},{s})")
            } else {
                format!("BIGNUMERIC({p},{s})")
            }
        }
        DTy::Str => "STRING".into(),
        DTy::Blob => "BYTES".into(),
        DTy::Date => "DATE".into(),
        DTy::Time => "TIME".into(),
        DTy::TsUs => "DATETIME".into(),
        DTy::TsTz => "TIMESTAMP".into(),
        t => {
            return Err(unsup(format!(
                "bigquery: no landing zone bought for {} yet",
                t.name()
            )));
        }
    })
}

fn print_query(
    rel: &Rel,
    cat: &Catalog,
    depth: usize,
) -> Result<(String, Vec<(String, DTy)>), DialectError> {
    // Same ribbon flattening as the DuckDB printer.
    match rel {
        Rel::Project { input, items } => {
            if let Some((table, pred)) = as_ribbon(input) {
                let in_schema = Rel::Scan {
                    table: table.clone(),
                }
                .schema(cat)?;
                let mut outs = Vec::new();
                for (name, e) in items {
                    outs.push(format!(
                        "{} AS {}",
                        print_expr(e, &in_schema)?,
                        quote_ident(name)
                    ));
                }
                let where_ = match pred {
                    Some(p) => format!(" WHERE {}", print_expr(p, &in_schema)?),
                    None => String::new(),
                };
                return Ok((
                    format!(
                        "SELECT {} FROM {}{where_}",
                        outs.join(", "),
                        quote_ident(&table)
                    ),
                    rel.schema(cat)?,
                ));
            }
        }
        Rel::Filter { input, pred } => {
            if let Rel::Scan { table } = input.as_ref() {
                let in_schema = input.schema(cat)?;
                let cols: Vec<String> = in_schema.iter().map(|(n, _)| quote_ident(n)).collect();
                return Ok((
                    format!(
                        "SELECT {} FROM {} WHERE {}",
                        cols.join(", "),
                        quote_ident(table),
                        print_expr(pred, &in_schema)?
                    ),
                    rel.schema(cat)?,
                ));
            }
        }
        Rel::Scan { .. } => {}
    }
    match rel {
        Rel::Scan { table } => {
            let schema = rel.schema(cat)?;
            let cols: Vec<String> = schema.iter().map(|(n, _)| quote_ident(n)).collect();
            Ok((
                format!("SELECT {} FROM {}", cols.join(", "), quote_ident(table)),
                schema,
            ))
        }
        Rel::Filter { input, pred } => {
            let (inner, in_schema) = print_query(input, cat, depth + 1)?;
            let pred = print_expr(pred, &in_schema)?;
            let schema = rel.schema(cat)?;
            let cols: Vec<String> = in_schema.iter().map(|(n, _)| quote_ident(n)).collect();
            Ok((
                format!(
                    "SELECT {} FROM ({inner}) AS {} WHERE {pred}",
                    cols.join(", "),
                    quote_ident(&format!("__cf_q{depth}"))
                ),
                schema,
            ))
        }
        Rel::Project { input, items } => {
            let (inner, in_schema) = print_query(input, cat, depth + 1)?;
            let mut outs = Vec::new();
            for (name, e) in items {
                outs.push(format!(
                    "{} AS {}",
                    print_expr(e, &in_schema)?,
                    quote_ident(name)
                ));
            }
            let schema = rel.schema(cat)?;
            Ok((
                format!(
                    "SELECT {} FROM ({inner}) AS {}",
                    outs.join(", "),
                    quote_ident(&format!("__cf_q{depth}"))
                ),
                schema,
            ))
        }
    }
}

fn as_ribbon(input: &Rel) -> Option<(String, Option<&Expr>)> {
    match input {
        Rel::Scan { table } => Some((table.clone(), None)),
        Rel::Filter { input, pred } => match input.as_ref() {
            Rel::Scan { table } => Some((table.clone(), Some(pred))),
            _ => None,
        },
        _ => None,
    }
}

/// Integer ops must COMPUTE at i64 to preserve DuckDB's trap class —
/// DuckDB promotes to the wider operand width before computing, so an i32
/// literal beside an i64 column is an i64 computation (fine), while
/// i32-with-i32 traps at 2^31 and BigQuery's INT64 would not.
fn require_i64_computation(op: &str, derived: &DTy) -> Result<(), DialectError> {
    if *derived == DTy::I64 {
        return Ok(());
    }
    Err(unsup(format!(
        "bigquery: {op} computing at {} (only INT64 preserves DuckDB's trap \
         class; narrow widths need phase-4 guard expressions)",
        derived.name()
    )))
}

fn print_expr(e: &Expr, input: &[(String, DTy)]) -> Result<String, DialectError> {
    Ok(match e {
        Expr::Col { ordinal, name, .. } => {
            let dups = input
                .iter()
                .filter(|(n, _)| n.eq_ignore_ascii_case(name))
                .count();
            if dups != 1 {
                return Err(unsup(format!(
                    "printing a column reference over duplicate upstream names: {name}"
                )));
            }
            let (bound_name, _) = &input[*ordinal];
            quote_ident(bound_name)
        }
        Expr::Lit { lexeme, ty } => match ty {
            DTy::Str => format!("'{}'", escape_str(lexeme)),
            DTy::Bool | DTy::F64 | DTy::I32 | DTy::I64 => lexeme.clone(),
            // A bare decimal-pointed literal is FLOAT64 in BigQuery; type it.
            DTy::Dec(p, s) => {
                let head = if *s <= 9 && p - s <= 29 {
                    "NUMERIC"
                } else {
                    "BIGNUMERIC"
                };
                format!("{head} '{lexeme}'")
            }
            t => {
                return Err(unsup(format!(
                    "bigquery: {} literal not printed yet",
                    t.name()
                )));
            }
        },
        Expr::Bin { op, l, r } => {
            let (lt, rt) = (l.ty()?, r.ty()?);
            let node_ty = e.ty()?;
            let (ls, rs) = (print_expr(l, input)?, print_expr(r, input)?);
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    // F64 arithmetic is IEEE on both engines; INT64 shares
                    // the overflow-error class. Everything narrower refuses.
                    let sym = match op {
                        BinOp::Add => "+",
                        BinOp::Sub => "-",
                        _ => "*",
                    };
                    if node_ty != DTy::F64 {
                        require_i64_computation(sym, &node_ty)?;
                    }
                    format!("({ls} {sym} {rs})")
                }
                BinOp::FDiv => {
                    // Pinned DuckDB: / is double division, whatever the
                    // operands. INT64/INT64 is already FLOAT64 in BigQuery;
                    // decimals must be forced.
                    let force = |t: &DTy, s: String| -> String {
                        if matches!(t, DTy::Dec(..)) {
                            format!("CAST({s} AS FLOAT64)")
                        } else {
                            s
                        }
                    };
                    format!("({} / {})", force(&lt, ls), force(&rt, rs))
                }
                BinOp::IDiv => {
                    require_i64_computation("//", &node_ty)?;
                    format!("DIV({ls}, {rs})")
                }
                BinOp::Rem => {
                    require_i64_computation("%", &node_ty)?;
                    format!("MOD({ls}, {rs})")
                }
                BinOp::Concat => format!("({ls} || {rs})"),
                BinOp::And => format!("({ls} AND {rs})"),
                BinOp::Or => format!("({ls} OR {rs})"),
                BinOp::Eq => format!("({ls} = {rs})"),
                BinOp::Neq => format!("({ls} <> {rs})"),
                BinOp::Lt => format!("({ls} < {rs})"),
                BinOp::Lte => format!("({ls} <= {rs})"),
                BinOp::Gt => format!("({ls} > {rs})"),
                BinOp::Gte => format!("({ls} >= {rs})"),
            }
        }
        Expr::Un { op, e } => match op {
            UnOp::Neg => {
                let t = e.ty()?;
                if t != DTy::I64 && t != DTy::F64 && !matches!(t, DTy::Dec(..)) {
                    return Err(unsup(format!(
                        "bigquery: unary - over {} (narrow-width trap not forced)",
                        t.name()
                    )));
                }
                format!("(- {})", print_expr(e, input)?)
            }
            UnOp::Not => format!("(NOT {})", print_expr(e, input)?),
        },
        Expr::Cast { strict, e, target } => format!(
            "{}({} AS {})",
            if *strict { "CAST" } else { "SAFE_CAST" },
            print_expr(e, input)?,
            bq_name(target)?
        ),
        Expr::Case { whens, else_ } => {
            let mut s = String::from("(CASE");
            for (c, v) in whens {
                s.push_str(&format!(
                    " WHEN {} THEN {}",
                    print_expr(c, input)?,
                    print_expr(v, input)?
                ));
            }
            if let Some(el) = else_ {
                s.push_str(&format!(" ELSE {}", print_expr(el, input)?));
            }
            s.push_str(" END)");
            s
        }
        Expr::IsNull { negated, e } => format!(
            "({} IS {}NULL)",
            print_expr(e, input)?,
            if *negated { "NOT " } else { "" }
        ),
        Expr::IsDistinct { negated, l, r } => format!(
            "({} IS {}DISTINCT FROM {})",
            print_expr(l, input)?,
            if *negated { "NOT " } else { "" },
            print_expr(r, input)?
        ),
    })
}

/// GoogleSQL string escaping: backslash escapes, `''` is NOT a quote escape.
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}
