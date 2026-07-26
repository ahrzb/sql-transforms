//! Frontend: SQL text -> bound, typed relational IR. Parsing is sqlparser's
//! DuckDB dialect; binding and type derivation follow DuckDB semantics as
//! measured (see plan.rs notes and the pins in exec/interp.rs).
//!
//! Error discipline (the corpus three-outcome contract depends on it):
//! * [`PrepareError::Unsupported`] — the construct is real SQL we don't do
//!   YET; the message names it. Corpus replay counts these as clean.
//! * [`PrepareError::Bind`] — the query is wrong against this schema
//!   (unknown column, type mismatch). Never used for missing features.
//!
//! Identifier semantics: DuckDB matches case-insensitively and preserves
//! spelling — `SELECT AGE` binds a column named `age` and the output column
//! is spelled `AGE`.
//!
//! Known v0 divergence, deliberate: DuckDB types `1.5` as DECIMAL(2,1); we
//! map decimal literals to f64 (same collapse the existing engines make).

use sqlparser::ast::{
    BinaryOperator, Expr as SqlExpr, SelectItem, SetExpr, Statement, TableFactor, UnaryOperator,
    Value as SqlValue,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use super::ir::{CmpPred, Col, Lit, Ty};
use super::plan::{ArithOp, Rel, SExpr, SKind};

#[derive(Debug, PartialEq, Eq)]
pub enum PrepareError {
    Parse(String),
    /// Real SQL, not lowered yet — names the construct (clean-unsupported).
    Unsupported(String),
    /// Wrong against this schema/type system.
    Bind(String),
    /// Lowering produced unverifiable IR — always a bug in the specializer.
    Internal(String),
}

impl std::fmt::Display for PrepareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrepareError::Parse(m) => write!(f, "parse error: {m}"),
            PrepareError::Unsupported(m) => write!(f, "unsupported: {m}"),
            PrepareError::Bind(m) => write!(f, "bind error: {m}"),
            PrepareError::Internal(m) => write!(f, "internal specializer bug: {m}"),
        }
    }
}

fn unsup(what: impl Into<String>) -> PrepareError {
    PrepareError::Unsupported(what.into())
}

/// SQL text + the dynamic table's schema -> bound relational tree plus the
/// derived output schema.
pub fn frontend(sql: &str, in_cols: &[Col]) -> Result<(Rel, Vec<Col>), PrepareError> {
    let statements = Parser::parse_sql(&DuckDbDialect {}, sql)
        .map_err(|e| PrepareError::Parse(e.to_string()))?;
    let [statement] = statements.as_slice() else {
        return Err(unsup("multiple SQL statements"));
    };
    let query = match statement {
        Statement::Query(q) => q,
        other => return Err(unsup(format!("statement kind: {other}"))),
    };
    if query.with.is_some() {
        return Err(unsup("WITH / common table expressions"));
    }
    if !query.order_by.is_none() {
        return Err(unsup("ORDER BY"));
    }
    if query.limit_clause.is_some() {
        return Err(unsup("LIMIT/OFFSET"));
    }
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        SetExpr::SetOperation { .. } => return Err(unsup("UNION/INTERSECT/EXCEPT")),
        other => return Err(unsup(format!("query body: {other}"))),
    };
    if select.distinct.is_some() {
        return Err(unsup("DISTINCT"));
    }
    let grouped = match &select.group_by {
        sqlparser::ast::GroupByExpr::Expressions(exprs, modifiers) => {
            !exprs.is_empty() || !modifiers.is_empty()
        }
        sqlparser::ast::GroupByExpr::All(_) => true,
    };
    if grouped || select.having.is_some() {
        return Err(unsup("GROUP BY / HAVING / aggregation"));
    }

    check_from(select)?;

    let binder = Binder { in_cols };
    let mut rel = Rel::Scan;
    if let Some(pred) = &select.selection {
        let pred = binder.expr(pred)?;
        if pred.ty != Ty::I1 {
            return Err(PrepareError::Bind(format!(
                "WHERE predicate must be BOOLEAN, got {}",
                pred.ty.name()
            )));
        }
        rel = Rel::Filter { input: Box::new(rel), pred };
    }

    let mut out_cols = Vec::new();
    let mut exprs = Vec::new();
    for item in &select.projection {
        let (name, e) = match item {
            SelectItem::UnnamedExpr(e) => (default_name(e), binder.expr(e)?),
            SelectItem::ExprWithAlias { expr, alias } => (alias.value.clone(), binder.expr(expr)?),
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..) => {
                return Err(unsup("SELECT * (star expansion)"))
            }
            SelectItem::ExprWithAliases { .. } => {
                return Err(unsup("multi-alias SELECT item"))
            }
        };
        if out_cols.iter().any(|c: &Col| c.name == name) {
            // DuckDB allows duplicate output names; our IR requires unique
            // columns. Rare in real queries — punt cleanly for now.
            return Err(unsup(format!("duplicate output column name '{name}'")));
        }
        out_cols.push(Col {
            name,
            ty: super::ir::ColTy { ty: e.ty, nullable: e.nullable },
        });
        exprs.push(e);
    }
    if exprs.is_empty() {
        return Err(PrepareError::Bind("SELECT list is empty".to_string()));
    }

    let named = out_cols
        .iter()
        .map(|c| c.name.clone())
        .zip(exprs)
        .collect::<Vec<_>>();
    Ok((Rel::Project { input: Box::new(rel), exprs: named }, out_cols))
}

/// v0: exactly `FROM __THIS__` (case-insensitive), no alias, no joins.
fn check_from(select: &sqlparser::ast::Select) -> Result<(), PrepareError> {
    let [table] = select.from.as_slice() else {
        return Err(match select.from.len() {
            0 => unsup("FROM-less SELECT"),
            _ => unsup("multiple FROM relations"),
        });
    };
    if !table.joins.is_empty() {
        return Err(unsup("JOIN (arrives with binding-time analysis)"));
    }
    match &table.relation {
        TableFactor::Table { name, alias, .. } => {
            if alias.is_some() {
                return Err(unsup("FROM alias"));
            }
            let n = name.to_string();
            if !n.eq_ignore_ascii_case("__THIS__") {
                return Err(unsup(format!(
                    "table '{n}' (only the dynamic table __THIS__ until static tables land)"
                )));
            }
            Ok(())
        }
        other => Err(unsup(format!("FROM {other}"))),
    }
}

/// DuckDB names an unaliased projection after the identifier it selects
/// (spelling preserved), else after the expression's text.
fn default_name(e: &SqlExpr) -> String {
    match e {
        SqlExpr::Identifier(ident) => ident.value.clone(),
        SqlExpr::CompoundIdentifier(parts) if !parts.is_empty() => {
            parts.last().unwrap().value.clone()
        }
        other => other.to_string(),
    }
}

struct Binder<'a> {
    in_cols: &'a [Col],
}

impl Binder<'_> {
    fn expr(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        match e {
            SqlExpr::Identifier(ident) => self.column(&ident.value),
            SqlExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [table, col] if table.value.eq_ignore_ascii_case("__THIS__") => {
                    self.column(&col.value)
                }
                [table, _] => Err(PrepareError::Bind(format!("unknown table '{}'", table.value))),
                _ => Err(unsup("nested field access")),
            },
            SqlExpr::Nested(inner) => self.expr(inner),
            SqlExpr::Value(v) => literal(&v.value),
            SqlExpr::BinaryOp { left, op, right } => self.binary(op, left, right),
            SqlExpr::UnaryOp { op: UnaryOperator::Minus, expr } => {
                // DuckDB `-x`: lower as 0 - x, reusing Sub's promotion.
                let zero = SExpr { kind: SKind::Lit(Lit::I64(0)), ty: Ty::I64, nullable: false };
                self.arith(ArithOp::Sub, zero, self.expr(expr)?)
            }
            SqlExpr::UnaryOp { op: UnaryOperator::Plus, expr } => self.expr(expr),
            SqlExpr::UnaryOp { op, .. } => Err(unsup(format!("unary operator {op:?}"))),
            SqlExpr::Function(f) => Err(unsup(format!(
                "function {} (catalogue arrives after the lowering spine)",
                f.name
            ))),
            SqlExpr::Case { .. } => Err(unsup("CASE (arrives with 3VL lowering)")),
            SqlExpr::Cast { .. } => Err(unsup("CAST (arrives with 3VL lowering)")),
            SqlExpr::IsNull(_) | SqlExpr::IsNotNull(_) => {
                Err(unsup("IS [NOT] NULL (arrives with 3VL lowering)"))
            }
            other => Err(unsup(format!("expression: {other}"))),
        }
    }

    /// Case-insensitive, spelling-preserving column bind (DuckDB semantics).
    fn column(&self, name: &str) -> Result<SExpr, PrepareError> {
        let mut hit = None;
        for (i, c) in self.in_cols.iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                if hit.is_some() {
                    return Err(PrepareError::Bind(format!("ambiguous column '{name}'")));
                }
                hit = Some((i, c));
            }
        }
        let (i, c) = hit.ok_or_else(|| {
            PrepareError::Bind(format!("column '{name}' does not exist in __THIS__"))
        })?;
        Ok(SExpr { kind: SKind::Col(i as u32), ty: c.ty.ty, nullable: c.ty.nullable })
    }

    fn binary(
        &self,
        op: &BinaryOperator,
        left: &SqlExpr,
        right: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let a = self.expr(left)?;
        let b = self.expr(right)?;
        match op {
            BinaryOperator::Plus => self.arith(ArithOp::Add, a, b),
            BinaryOperator::Minus => self.arith(ArithOp::Sub, a, b),
            BinaryOperator::Multiply => self.arith(ArithOp::Mul, a, b),
            BinaryOperator::Divide => self.arith(ArithOp::Div, a, b),
            BinaryOperator::Modulo => self.arith(ArithOp::Rem, a, b),
            BinaryOperator::Eq => self.cmp(CmpPred::Eq, a, b),
            BinaryOperator::NotEq => self.cmp(CmpPred::Ne, a, b),
            BinaryOperator::Lt => self.cmp(CmpPred::Lt, a, b),
            BinaryOperator::LtEq => self.cmp(CmpPred::Le, a, b),
            BinaryOperator::Gt => self.cmp(CmpPred::Gt, a, b),
            BinaryOperator::GtEq => self.cmp(CmpPred::Ge, a, b),
            BinaryOperator::And | BinaryOperator::Or => {
                Err(unsup("AND/OR (Kleene logic arrives with 3VL lowering)"))
            }
            BinaryOperator::StringConcat => Err(unsup("|| (arrives with the string ops)")),
            other => Err(unsup(format!("operator {other}"))),
        }
    }

    fn arith(&self, op: ArithOp, a: SExpr, b: SExpr) -> Result<SExpr, PrepareError> {
        let (a, b, ty) = numeric_promote(op, a, b)?;
        let nullable = a.nullable || b.nullable;
        Ok(SExpr {
            kind: SKind::Arith { op, a: Box::new(a), b: Box::new(b) },
            ty,
            nullable,
        })
    }

    fn cmp(&self, pred: CmpPred, a: SExpr, b: SExpr) -> Result<SExpr, PrepareError> {
        let (a, b) = match (a.ty, b.ty) {
            (x, y) if x == y => (a, b),
            (Ty::I64, Ty::F64) => (promote_f64(a), b),
            (Ty::F64, Ty::I64) => (a, promote_f64(b)),
            (x, y) => {
                return Err(PrepareError::Bind(format!(
                    "cannot compare {} with {}",
                    x.name(),
                    y.name()
                )))
            }
        };
        if a.ty == Ty::I1 {
            return Err(unsup("comparison on BOOLEAN"));
        }
        let nullable = a.nullable || b.nullable;
        Ok(SExpr {
            kind: SKind::Cmp { pred, a: Box::new(a), b: Box::new(b) },
            ty: Ty::I1,
            nullable,
        })
    }
}

fn literal(v: &SqlValue) -> Result<SExpr, PrepareError> {
    let (lit, ty) = match v {
        SqlValue::Number(text, _) => {
            if text.contains('.') || text.to_ascii_lowercase().contains('e') {
                // DuckDB would type this DECIMAL; v0 collapses to f64.
                let f = text
                    .parse::<f64>()
                    .map_err(|_| PrepareError::Bind(format!("bad numeric literal '{text}'")))?;
                (Lit::F64(f), Ty::F64)
            } else {
                let i = text
                    .parse::<i64>()
                    .map_err(|_| PrepareError::Bind(format!("bad integer literal '{text}'")))?;
                (Lit::I64(i), Ty::I64)
            }
        }
        SqlValue::SingleQuotedString(s) => (Lit::Str(s.clone()), Ty::Str),
        SqlValue::Boolean(b) => (Lit::I1(*b), Ty::I1),
        SqlValue::Null => return Err(unsup("bare NULL literal (arrives with 3VL lowering)")),
        other => return Err(unsup(format!("literal {other}"))),
    };
    Ok(SExpr { kind: SKind::Lit(lit), ty, nullable: false })
}

/// DuckDB numeric promotion for the v0 type lattice: `/` is always float
/// division; otherwise int op int stays int, anything touching f64 is f64.
fn numeric_promote(op: ArithOp, a: SExpr, b: SExpr) -> Result<(SExpr, SExpr, Ty), PrepareError> {
    let numeric = |e: &SExpr| matches!(e.ty, Ty::I64 | Ty::F64);
    if !numeric(&a) || !numeric(&b) {
        return Err(PrepareError::Bind(format!(
            "arithmetic needs numeric operands, got {} and {}",
            a.ty.name(),
            b.ty.name()
        )));
    }
    if op == ArithOp::Div {
        return Ok((promote_f64(a), promote_f64(b), Ty::F64));
    }
    match (a.ty, b.ty) {
        (Ty::I64, Ty::I64) => Ok((a, b, Ty::I64)),
        (Ty::F64, Ty::F64) => Ok((a, b, Ty::F64)),
        (Ty::I64, Ty::F64) => Ok((promote_f64(a), b, Ty::F64)),
        (Ty::F64, Ty::I64) => Ok((a, promote_f64(b), Ty::F64)),
        _ => unreachable!("guarded numeric above"),
    }
}

fn promote_f64(e: SExpr) -> SExpr {
    if e.ty == Ty::F64 {
        return e;
    }
    let nullable = e.nullable;
    SExpr { kind: SKind::IntToFloat(Box::new(e)), ty: Ty::F64, nullable }
}
