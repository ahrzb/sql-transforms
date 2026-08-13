//! The BigQuery printer: plan → GoogleSQL, forcing the plan's DuckDB-pinned
//! semantics in BigQuery's syntax (design D1/D3).
//!
//! STATUS: documented-semantics, unprobed. Every spelling below follows
//! BigQuery's published GoogleSQL reference; none has run against the real
//! service yet — that is the design's phase-4 remote gate, still owed
//! (tests/test_dialect_cross_engine_gate.py carries the credential-gated
//! seam). Until it runs, the refusal set stays conservative: anything whose
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
//! * **`/` prints as `IEEE_DIVIDE`** over FLOAT64-cast operands: DuckDB `/`
//!   is IEEE double division INCLUDING zero divisors (1/0 = inf, 0/0 =
//!   NaN — review-confirmed), and BigQuery's bare `/` errors there while
//!   IEEE_DIVIDE reproduces the IEEE answers exactly.
//! * **`//` → `DIV()`, `%` → `MOD()`,** both INT64-only: BigQuery's DIV and
//!   MOD truncate toward zero with sign-of-dividend remainders — the same
//!   observed values as DuckDB's pinned ((-7)//2, (-7)%2, 7//(-2), 7%(-2))
//!   = (-3, -1, -3, 1). Zero divisors return NULL in DuckDB and error in
//!   BigQuery — guarded with a CASE; `INT64_MIN % -1` traps in DuckDB and
//!   is forced back into an error through DIV, which overflows there.
//! * **Float comparisons are NaN-forced.** DuckDB orders floats totally
//!   (NaN equals NaN and exceeds everything); BigQuery comparisons are
//!   IEEE. Every comparison and IS [NOT] DISTINCT FROM over FLOAT64
//!   prints a CASE on IS_NAN reproducing the total order.
//! * **CAST pairs are an allow-list.** DECIMAL→INT64 rounds half-away on
//!   both engines (documented) and prints bare; FLOAT64→INT64 rounds
//!   half-even in DuckDB but half-away in BigQuery and REFUSES until a
//!   forcing lands; string sources refuse (parse domains differ).
//! * **Known unforced divergence, phase 4:** FLOAT64 `+ - *` that overflow
//!   a finite operand pair to non-finite return inf in DuckDB but error in
//!   BigQuery. No cheap post-hoc guard exists (the operation itself
//!   errors); magnitude pre-guards via `ERROR()` are the named upgrade.
//!   Until then this divergence exists only beyond ±1.8e308 intermediate
//!   results and the module documents rather than hides it.
//! * **Decimal literals print typed** (`NUMERIC '1.5'` / `BIGNUMERIC`):
//!   a bare decimal-pointed literal is FLOAT64 in BigQuery but DECIMAL(p,s)
//!   in DuckDB — printing the lexeme bare would silently change its type.
//! * **Strings escape with backslashes:** `''` is not an escape in
//!   GoogleSQL.

use super::plan::{BinOp, Catalog, Expr, Rel, UnOp};
use super::printer::{col_ref, query, ExprPrinter};
use super::ty::DTy;
use super::verify::verify;
use super::{unsup, DialectError};

/// Print a verified plan as BigQuery GoogleSQL. `cat` supplies scan schemas.
pub fn print_sql(rel: &Rel, cat: &Catalog) -> Result<String, DialectError> {
    verify(rel, cat)?;
    let (sql, _) = query(&BigQuery, rel, cat, 0)?;
    Ok(sql)
}

struct BigQuery;

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
            if *s <= 9 && p.saturating_sub(*s) <= 29 {
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

impl ExprPrinter for BigQuery {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('\\', "\\\\").replace('`', "\\`"))
    }

    fn expr(&self, e: &Expr, input: &[(String, DTy)]) -> Result<String, DialectError> {
        Ok(match e {
            Expr::Col { ordinal, name, .. } => col_ref(self, *ordinal, name, input)?,
            Expr::Lit { lexeme, ty } => match ty {
                DTy::Str => format!("'{}'", escape_str(lexeme)),
                DTy::Bool | DTy::F64 | DTy::I32 | DTy::I64 => lexeme.clone(),
                // A bare decimal-pointed literal is FLOAT64 in BigQuery;
                // type it.
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
                let (ls, rs) = (self.expr(l, input)?, self.expr(r, input)?);
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul => {
                        // F64 arithmetic is IEEE on both engines; INT64
                        // shares the overflow-error class. Everything
                        // narrower refuses.
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
                        // Pinned DuckDB: / is IEEE double division whatever
                        // the operands, zero divisors included. IEEE_DIVIDE
                        // reproduces that; bare / would error on zero.
                        let force = |t: &DTy, s: String| -> String {
                            if matches!(t, DTy::F64) {
                                s
                            } else {
                                format!("CAST({s} AS FLOAT64)")
                            }
                        };
                        format!("IEEE_DIVIDE({}, {})", force(&lt, ls), force(&rt, rs))
                    }
                    BinOp::IDiv => {
                        require_i64_computation("//", &node_ty)?;
                        // DuckDB: zero divisor -> NULL; BigQuery DIV errors.
                        format!(
                            "(CASE WHEN {rs} = 0 THEN CAST(NULL AS INT64) ELSE DIV({ls}, {rs}) END)"
                        )
                    }
                    BinOp::Rem => {
                        require_i64_computation("%", &node_ty)?;
                        // Zero divisor -> NULL (DuckDB); INT64_MIN % -1
                        // traps in DuckDB but MOD returns 0 - DIV overflows
                        // there and forces the matching error class.
                        format!(
                            "(CASE WHEN {rs} = 0 THEN CAST(NULL AS INT64)                              WHEN {ls} = (-9223372036854775807 - 1) AND {rs} = -1 THEN DIV({ls}, {rs})                              ELSE MOD({ls}, {rs}) END)"
                        )
                    }
                    BinOp::Concat => format!("({ls} || {rs})"),
                    BinOp::And => format!("({ls} AND {rs})"),
                    BinOp::Or => format!("({ls} OR {rs})"),
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte => {
                        nan_forced_compare(*op, &lt, &rt, &ls, &rs)
                    }
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
                    format!("(- {})", self.expr(e, input)?)
                }
                UnOp::Not => format!("(NOT {})", self.expr(e, input)?),
            },
            Expr::Cast { strict, e, target } => {
                let src = e.ty()?;
                let inner = self.expr(e, input)?;
                bq_cast(*strict, &src, target, inner)?
            }
            Expr::Case { whens, else_ } => {
                let mut s = String::from("(CASE");
                for (c, v) in whens {
                    s.push_str(&format!(
                        " WHEN {} THEN {}",
                        self.expr(c, input)?,
                        self.expr(v, input)?
                    ));
                }
                if let Some(el) = else_ {
                    s.push_str(&format!(" ELSE {}", self.expr(el, input)?));
                }
                s.push_str(" END)");
                s
            }
            Expr::IsNull { negated, e } => format!(
                "({} IS {}NULL)",
                self.expr(e, input)?,
                if *negated { "NOT " } else { "" }
            ),
            Expr::Call { func, .. } => {
                return Err(unsup(format!(
                    "bigquery: function {} (scalar calls await the phase-4 remote gate)",
                    func.name()
                )));
            }
            Expr::IsDistinct { negated, l, r } => {
                let (lt, rt) = (l.ty()?, r.ty()?);
                let (ls, rs) = (self.expr(l, input)?, self.expr(r, input)?);
                if involves_float(&lt, &rt) {
                    // DuckDB's total order: NaN IS NOT DISTINCT FROM NaN is
                    // true; BigQuery's IS DISTINCT uses IEEE equality.
                    let both_null = format!("({ls} IS NULL AND {rs} IS NULL)");
                    let both_present_equal = format!(
                        "({ls} IS NOT NULL AND {rs} IS NOT NULL AND                          (CASE WHEN IS_NAN({ls}) THEN IS_NAN({rs})                          WHEN IS_NAN({rs}) THEN FALSE ELSE {ls} = {rs} END))"
                    );
                    if *negated {
                        format!("({both_null} OR {both_present_equal})")
                    } else {
                        format!("(NOT ({both_null} OR {both_present_equal}))")
                    }
                } else {
                    format!(
                        "({ls} IS {}DISTINCT FROM {rs})",
                        if *negated { "NOT " } else { "" }
                    )
                }
            }
        })
    }
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

fn involves_float(l: &DTy, r: &DTy) -> bool {
    matches!(l, DTy::F64) || matches!(r, DTy::F64)
}

/// DuckDB orders floats TOTALLY: NaN equals NaN and exceeds every value.
/// BigQuery comparisons are IEEE (everything with NaN is false except <>).
/// Force the total order with IS_NAN cases; NULL stays NULL.
fn nan_forced_compare(op: BinOp, lt: &DTy, rt: &DTy, ls: &str, rs: &str) -> String {
    let sym = match op {
        BinOp::Eq => "=",
        BinOp::Neq => "<>",
        BinOp::Lt => "<",
        BinOp::Lte => "<=",
        BinOp::Gt => ">",
        BinOp::Gte => ">=",
        _ => unreachable!("comparison ops only"),
    };
    if !involves_float(lt, rt) {
        return format!("({ls} {sym} {rs})");
    }
    // Truth table for l NaN / r NaN under DuckDB's total order.
    let (l_nan, r_nan) = match op {
        BinOp::Eq => ("IS_NAN({r})", "FALSE"),
        BinOp::Neq => ("NOT IS_NAN({r})", "TRUE"),
        BinOp::Lt => ("FALSE", "TRUE"),
        BinOp::Lte => ("IS_NAN({r})", "TRUE"),
        BinOp::Gt => ("NOT IS_NAN({r})", "FALSE"),
        BinOp::Gte => ("TRUE", "FALSE"),
        _ => unreachable!(),
    };
    let l_nan = l_nan.replace("{r}", rs);
    let r_nan = r_nan.replace("{r}", rs);
    format!(
        "(CASE WHEN {ls} IS NULL OR {rs} IS NULL THEN CAST(NULL AS BOOL)          WHEN IS_NAN({ls}) THEN {l_nan}          WHEN IS_NAN({rs}) THEN {r_nan}          ELSE {ls} {sym} {rs} END)"
    )
}

/// CAST pairs the BigQuery printer has bought. DECIMAL→INT64 rounds
/// half-away on both engines (documented) and prints bare; FLOAT64→int
/// rounds half-even in DuckDB but half-away in BigQuery — REFUSED until a
/// forcing lands; string sources refuse (parse domains differ, and
/// SAFE_CAST/TRY_CAST NULL domains differ with them).
fn bq_cast(strict: bool, src: &DTy, target: &DTy, inner: String) -> Result<String, DialectError> {
    let kw = if strict { "CAST" } else { "SAFE_CAST" };
    let t = bq_name(target)?;
    let refuse = || {
        Err(unsup(format!(
            "bigquery: CAST {} -> {} (conversion domain not pinned)",
            src.name(),
            target.name()
        )))
    };
    if src == target {
        return Ok(format!("{kw}({inner} AS {t})"));
    }
    // Widening a signed int is value-exact everywhere; that source class
    // is safe into every bought numeric/string landing zone.
    let signed_int = matches!(src, DTy::I8 | DTy::I16 | DTy::I32 | DTy::I64);
    Ok(match target {
        DTy::I64 => match src {
            _ if signed_int => format!("{kw}({inner} AS {t})"),
            DTy::Dec(..) => format!("{kw}({inner} AS {t})"),
            _ => return refuse(),
        },
        DTy::F64 => match src {
            _ if signed_int => format!("{kw}({inner} AS {t})"),
            DTy::Dec(..) => format!("{kw}({inner} AS {t})"),
            _ => return refuse(),
        },
        DTy::Dec(..) => match src {
            _ if signed_int => format!("{kw}({inner} AS {t})"),
            _ => return refuse(),
        },
        DTy::Str => match src {
            _ if signed_int => format!("{kw}({inner} AS {t})"),
            _ => return refuse(),
        },
        _ => return refuse(),
    })
}
