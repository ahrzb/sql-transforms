//! The Spark printer: plan → Spark SQL under the PINNED configuration
//! (design D6, pins-dialect/spark-ansi.json): `spark.sql.ansi.enabled=true`,
//! `spark.sql.session.timeZone=UTC`. Output is only valid under that
//! config — ANSI off changes overflow, cast, and division semantics
//! wholesale, i.e. it is a different dialect this printer does not speak.
//!
//! Unlike BigQuery, Spark HAS DuckDB's narrow integer widths
//! (TINYINT/SMALLINT/INT/BIGINT), and under ANSI their operators error on
//! overflow at their own width — the same trap class DuckDB pins. So
//! narrow-int arithmetic prints natively here, and the two engines'
//! literal-width rules agree (32-bit if it fits, else 64): the printed
//! computation happens at the same width DuckDB computed at.
//!
//! The forced spellings, each traceable to a pin (spark-ansi.json unless
//! noted):
//!
//! * `/` is double division on both engines (pinned both sides); Spark
//!   DECIMAL/DECIMAL stays DECIMAL, so decimal operands cast to DOUBLE.
//! * `//` → `div`, which always computes at BIGINT in Spark (pinned:
//!   typeof(1 div 2) = bigint), so a narrow-width result re-narrows
//!   through a checked CAST: DuckDB's `int32min // -1` trap becomes
//!   Spark's ANSI CAST_OVERFLOW — same error class, forced. `%` keeps the
//!   operand width and cannot overflow; both engines pin the same
//!   sign-of-dividend values.
//! * `<=>` exists, but IS [NOT] DISTINCT FROM is accepted too (pinned) —
//!   printed in the portable spelling.
//! * try_cast / named-error CAST match DuckDB's strict|try split (pinned
//!   error classes).
//! * Strings escape with backslashes; identifiers quote with backticks
//!   (`` ` `` doubles inside).
//! * Decimal literals print bare: Spark types them DECIMAL by digit count
//!   like DuckDB (pinned literal typing class), and the v0 plan admits
//!   decimals only into comparisons and CASE, where value semantics —
//!   not precision propagation — decide.
//!
//! Print-only, like BigQuery: the "from Spark" frontend is design phase 5.

use super::plan::{BinOp, Catalog, Expr, Rel, UnOp};
use super::printer::{col_ref, query, ExprPrinter};
use super::ty::DTy;
use super::verify::verify;
use super::{unsup, DialectError};

/// Print a verified plan as Spark SQL (ANSI, UTC — see module doc).
pub fn print_sql(rel: &Rel, cat: &Catalog) -> Result<String, DialectError> {
    verify(rel, cat)?;
    let (sql, _) = query(&Spark, rel, cat, 0)?;
    Ok(sql)
}

struct Spark;

/// The type's Spark landing zone: CAST targets. Narrow ints are native.
fn spark_name(ty: &DTy) -> Result<String, DialectError> {
    Ok(match ty {
        DTy::Bool => "BOOLEAN".into(),
        DTy::I8 => "TINYINT".into(),
        DTy::I16 => "SMALLINT".into(),
        DTy::I32 => "INT".into(),
        DTy::I64 => "BIGINT".into(),
        DTy::F32 => "FLOAT".into(),
        DTy::F64 => "DOUBLE".into(),
        DTy::Dec(p, s) => format!("DECIMAL({p},{s})"),
        DTy::Str => "STRING".into(),
        DTy::Blob => "BINARY".into(),
        DTy::Date => "DATE".into(),
        // Wall-clock lands on TIMESTAMP_NTZ; the instant type on TIMESTAMP,
        // which is session-zone-relative — pinned UTC makes it the instant.
        DTy::TsUs => "TIMESTAMP_NTZ".into(),
        DTy::TsTz => "TIMESTAMP".into(),
        t => {
            return Err(unsup(format!(
                "spark: no landing zone bought for {} yet",
                t.name()
            )));
        }
    })
}

impl ExprPrinter for Spark {
    fn quote_ident(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn expr(&self, e: &Expr, input: &[(String, DTy)]) -> Result<String, DialectError> {
        Ok(match e {
            Expr::Col { ordinal, name, .. } => col_ref(self, *ordinal, name, input)?,
            Expr::Lit { lexeme, ty } => match ty {
                DTy::Str => format!("'{}'", escape_str(lexeme)),
                // Integer literal width rules agree (32-bit if it fits,
                // else 64); decimal literals type by digit count on both;
                // F64 lexemes carry an exponent and are DOUBLE on both.
                DTy::Bool | DTy::F64 | DTy::I32 | DTy::I64 | DTy::Dec(..) => lexeme.clone(),
                t => {
                    return Err(unsup(format!(
                        "spark: {} literal not printed yet",
                        t.name()
                    )));
                }
            },
            Expr::Bin { op, l, r } => {
                let (lt, rt) = (l.ty()?, r.ty()?);
                let node_ty = e.ty()?;
                let (ls, rs) = (self.expr(l, input)?, self.expr(r, input)?);
                match op {
                    // Native widths: ANSI overflow errors at the computation
                    // width — DuckDB's trap class, no forcing needed.
                    BinOp::Add => format!("({ls} + {rs})"),
                    BinOp::Sub => format!("({ls} - {rs})"),
                    BinOp::Mul => format!("({ls} * {rs})"),
                    BinOp::FDiv => {
                        let force = |t: &DTy, s: String| -> String {
                            if matches!(t, DTy::Dec(..)) {
                                format!("CAST({s} AS DOUBLE)")
                            } else {
                                s
                            }
                        };
                        format!("({} / {})", force(&lt, ls), force(&rt, rs))
                    }
                    BinOp::IDiv => {
                        // div computes at BIGINT (pinned); re-narrow through
                        // a checked CAST so the width's trap class holds.
                        if node_ty == DTy::I64 {
                            format!("({ls} div {rs})")
                        } else {
                            format!("CAST(({ls} div {rs}) AS {})", spark_name(&node_ty)?)
                        }
                    }
                    BinOp::Rem => format!("({ls} % {rs})"),
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
                UnOp::Neg => format!("(- {})", self.expr(e, input)?),
                UnOp::Not => format!("(NOT {})", self.expr(e, input)?),
            },
            Expr::Cast { strict, e, target } => format!(
                "{}({} AS {})",
                if *strict { "CAST" } else { "try_cast" },
                self.expr(e, input)?,
                spark_name(target)?
            ),
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
            Expr::IsDistinct { negated, l, r } => format!(
                "({} IS {}DISTINCT FROM {})",
                self.expr(l, input)?,
                if *negated { "NOT " } else { "" },
                self.expr(r, input)?
            ),
        })
    }
}

/// Spark string escaping under the pinned config: backslash escapes
/// (`''` doubling is not relied on).
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
