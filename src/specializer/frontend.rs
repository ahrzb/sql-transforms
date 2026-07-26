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
//! NULL literal: typed by context (the other operand, the CASE unification,
//! the CAST target). A bare `SELECT NULL` has no context and stays
//! unsupported.
//!
//! Known v0 divergences, deliberate: DuckDB types `1.5` as DECIMAL(2,1); we
//! map decimal literals to f64. Integer-ish CAST targets (including HUGEINT)
//! all collapse to i64.

use sqlparser::ast::{
    BinaryOperator, CastKind, Expr as SqlExpr, JoinConstraint, JoinOperator, SelectItem, SetExpr,
    Statement, TableFactor, UnaryOperator, Value as SqlValue,
};
use sqlparser::dialect::DuckDbDialect;
use sqlparser::parser::Parser;

use super::fold::fold;
use super::ir::{CmpPred, Col, Lit, Ty};
use super::plan::{ArithOp, JoinKind, JoinSpec, Rel, SExpr, SKind, StaticTable};

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

/// SQL text + the dynamic table's name/schema + the static-table catalog ->
/// bound relational tree, the equi-joins in FROM order, and the derived
/// output schema.
pub fn frontend(
    sql: &str,
    this_name: &str,
    in_cols: &[Col],
    statics: &[StaticTable],
) -> Result<(Rel, Vec<JoinSpec>, Vec<Col>), PrepareError> {
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
    if query.order_by.is_some() {
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

    let (binder, joins) = bind_from(select, this_name, in_cols, statics)?;

    let mut rel = Rel::Scan;
    if let Some(pred) = &select.selection {
        let pred = fold(binder.expr(pred)?);
        if pred.ty != Ty::I1 {
            return Err(PrepareError::Bind(format!(
                "WHERE predicate must be BOOLEAN, got {}",
                pred.ty.name()
            )));
        }
        rel = Rel::Filter {
            input: Box::new(rel),
            pred,
        };
    }

    let mut out_cols = Vec::new();
    let mut exprs = Vec::new();
    for item in &select.projection {
        let (name, e) = match item {
            SelectItem::UnnamedExpr(e) => (default_name(e), fold(binder.expr(e)?)),
            SelectItem::ExprWithAlias { expr, alias } => {
                (alias.value.clone(), fold(binder.expr(expr)?))
            }
            SelectItem::Wildcard(_) | SelectItem::QualifiedWildcard(..) => {
                return Err(unsup("SELECT * (star expansion)"))
            }
            SelectItem::ExprWithAliases { .. } => return Err(unsup("multi-alias SELECT item")),
        };
        if out_cols.iter().any(|c: &Col| c.name == name) {
            // DuckDB allows duplicate output names; our IR requires unique
            // columns. Rare in real queries — punt cleanly for now.
            return Err(unsup(format!("duplicate output column name '{name}'")));
        }
        out_cols.push(Col {
            name,
            ty: super::ir::ColTy {
                ty: e.ty,
                nullable: e.nullable,
            },
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
    Ok((
        Rel::Project {
            input: Box::new(rel),
            exprs: named,
        },
        joins,
        out_cols,
    ))
}

/// Parse and bind the FROM clause: the dynamic table, then zero or more
/// equi-joins to static tables. Returns the fully-scoped binder (every join
/// visible) and the join specs in FROM order.
fn bind_from<'a>(
    select: &sqlparser::ast::Select,
    this_name: &str,
    in_cols: &'a [Col],
    statics: &'a [StaticTable],
) -> Result<(Binder<'a>, Vec<JoinSpec>), PrepareError> {
    let [table] = select.from.as_slice() else {
        return Err(match select.from.len() {
            0 => unsup("FROM-less SELECT"),
            _ => unsup("multiple FROM relations (comma join)"),
        });
    };
    let dyn_name = match &table.relation {
        TableFactor::Table { name, alias, .. } => {
            if alias.is_some() {
                return Err(unsup("alias on the dynamic table"));
            }
            let n = name.to_string();
            if !n.eq_ignore_ascii_case(this_name) {
                return Err(unsup(format!(
                    "table '{n}' as the driving relation (must be the dynamic table '{this_name}')"
                )));
            }
            n
        }
        other => return Err(unsup(format!("FROM {other}"))),
    };

    let mut binder = Binder {
        this_name: dyn_name,
        in_cols,
        joins: Vec::new(),
    };
    let mut specs: Vec<JoinSpec> = Vec::new();

    for join in &table.joins {
        let (kind, constraint) = match &join.join_operator {
            JoinOperator::Join(c) | JoinOperator::Inner(c) => (JoinKind::Inner, c),
            JoinOperator::Left(c) | JoinOperator::LeftOuter(c) => (JoinKind::Left, c),
            other => return Err(unsup(format!("join type {other:?}"))),
        };
        let on = match constraint {
            JoinConstraint::On(e) => e,
            JoinConstraint::Using(_) => return Err(unsup("JOIN USING")),
            JoinConstraint::Natural => return Err(unsup("NATURAL JOIN")),
            JoinConstraint::None => return Err(unsup("JOIN without ON (cross join)")),
        };
        let (raw_name, scope_name) = match &join.relation {
            TableFactor::Table { name, alias, .. } => {
                let n = name.to_string();
                let s = alias
                    .as_ref()
                    .map(|a| a.name.value.clone())
                    .unwrap_or_else(|| n.clone());
                (n, s)
            }
            other => return Err(unsup(format!("JOIN {other}"))),
        };
        if raw_name.eq_ignore_ascii_case(this_name) {
            return Err(unsup("joining the dynamic table to itself"));
        }
        let mut table_idx = None;
        for (i, st) in statics.iter().enumerate() {
            if st.name.eq_ignore_ascii_case(&raw_name) {
                if table_idx.is_some() {
                    return Err(PrepareError::Bind(format!(
                        "ambiguous static table '{raw_name}'"
                    )));
                }
                table_idx = Some(i);
            }
        }
        let Some(table_idx) = table_idx else {
            return Err(PrepareError::Bind(format!(
                "table '{raw_name}' was not provided as a static table"
            )));
        };
        if binder.this_name.eq_ignore_ascii_case(&scope_name)
            || binder
                .joins
                .iter()
                .any(|j| j.name.eq_ignore_ascii_case(&scope_name))
        {
            return Err(PrepareError::Bind(format!(
                "duplicate table name '{scope_name}' in FROM"
            )));
        }

        let st = &statics[table_idx];
        let (keys, key_cols) = bind_on(&binder, st, &scope_name, on)?;
        let val_cols: Vec<u32> = (0..st.cols.len() as u32)
            .filter(|c| !key_cols.contains(c))
            .collect();
        if val_cols.is_empty() {
            return Err(unsup(format!(
                "join to '{raw_name}' where every column is a key (no value columns)"
            )));
        }

        binder.joins.push(ScopeJoin {
            name: scope_name,
            table: st,
            kind,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
        });
        specs.push(JoinSpec {
            table: table_idx,
            kind,
            keys,
            key_cols,
            val_cols,
        });
    }
    Ok((binder, specs))
}

/// Bind a JOIN ... ON condition: a conjunction of `<expr> = <static col>`
/// equalities. Returns the dynamic-side key expressions (promoted to the
/// map's key types) and the static key columns they match, aligned.
fn bind_on(
    binder: &Binder<'_>,
    st: &StaticTable,
    scope_name: &str,
    on: &SqlExpr,
) -> Result<(Vec<SExpr>, Vec<u32>), PrepareError> {
    let mut conjuncts = Vec::new();
    collect_conjuncts(on, &mut conjuncts);
    let mut keys = Vec::new();
    let mut key_cols = Vec::new();
    for c in conjuncts {
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = c
        else {
            return Err(unsup(format!(
                "JOIN ON condition '{c}' (only AND-ed equalities are supported)"
            )));
        };
        let l = static_col_of(left, st, scope_name)?;
        let r = static_col_of(right, st, scope_name)?;
        let (col, dyn_side, static_side) = match (l, r) {
            (Some(c), None) => (c, right.as_ref(), left.as_ref()),
            (None, Some(c)) => (c, left.as_ref(), right.as_ref()),
            (Some(_), Some(_)) => {
                return Err(unsup(format!(
                    "JOIN ON '{c}': both sides are columns of '{scope_name}'"
                )))
            }
            (None, None) => {
                return Err(unsup(format!(
                    "JOIN ON '{c}': neither side is a column of '{scope_name}'"
                )))
            }
        };
        // A bare identifier on the static side that also binds in the outer
        // scope is ambiguous (DuckDB rejects it too).
        if let SqlExpr::Identifier(id) = static_side {
            if binder.column(&id.value).is_ok() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{}' in JOIN ON (qualify it)",
                    id.value
                )));
            }
        }
        let key = fold(binder.expr(dyn_side)?);
        let col_ty = st.cols[col as usize].ty.ty;
        let key = match (key.ty, col_ty) {
            (a, b) if a == b => key,
            (Ty::I64, Ty::F64) => promote_f64(key),
            // Static-side ints promote at materialization: the map key type
            // (the key expression's type) becomes F64 and the build side is
            // converted while the probe table is built.
            (Ty::F64, Ty::I64) => key,
            (a, b) => {
                return Err(PrepareError::Bind(format!(
                    "cannot join {} with {} (ON '{}')",
                    a.name(),
                    b.name(),
                    st.cols[col as usize].name
                )))
            }
        };
        keys.push(key);
        key_cols.push(col);
    }
    Ok((keys, key_cols))
}

fn collect_conjuncts<'e>(e: &'e SqlExpr, out: &mut Vec<&'e SqlExpr>) {
    match e {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_conjuncts(left, out);
            collect_conjuncts(right, out);
        }
        SqlExpr::Nested(inner) => collect_conjuncts(inner, out),
        other => out.push(other),
    }
}

/// Does `e` name a column of the static table being joined? Qualified form
/// matches on the join's scope name; a bare identifier matches if the table
/// has that column.
fn static_col_of(
    e: &SqlExpr,
    st: &StaticTable,
    scope_name: &str,
) -> Result<Option<u32>, PrepareError> {
    let name = match e {
        SqlExpr::Identifier(id) => &id.value,
        SqlExpr::CompoundIdentifier(parts) => match parts.as_slice() {
            [t, c] if t.value.eq_ignore_ascii_case(scope_name) => &c.value,
            _ => return Ok(None),
        },
        SqlExpr::Nested(inner) => return static_col_of(inner, st, scope_name),
        _ => return Ok(None),
    };
    let mut hit = None;
    for (i, c) in st.cols.iter().enumerate() {
        if c.name.eq_ignore_ascii_case(name) {
            if hit.is_some() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{name}' in static table '{}'",
                    st.name
                )));
            }
            hit = Some(i as u32);
        }
    }
    // Qualified misses are errors; bare misses just mean "not the static
    // side" — the caller will try binding it dynamically.
    if hit.is_none() {
        if let SqlExpr::CompoundIdentifier(_) = e {
            return Err(PrepareError::Bind(format!(
                "column '{name}' does not exist in '{scope_name}'"
            )));
        }
    }
    Ok(hit)
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

/// One joined static table in scope: how it is named, which of its columns
/// are probe values (bindable) vs keys (ON-clause only).
struct ScopeJoin<'a> {
    name: String,
    table: &'a StaticTable,
    kind: JoinKind,
    key_cols: Vec<u32>,
    val_cols: Vec<u32>,
}

struct Binder<'a> {
    /// The dynamic table's name as spelled in FROM.
    this_name: String,
    in_cols: &'a [Col],
    joins: Vec<ScopeJoin<'a>>,
}

impl Binder<'_> {
    /// Bind an expression that must have a definite type on its own — a bare
    /// NULL literal here is unsupported (no context to type it).
    fn expr(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        self.expr_or_null(e)?
            .ok_or_else(|| unsup("bare NULL literal without a typing context"))
    }

    /// Like `expr`, but a bare NULL literal comes back as `None` for the
    /// caller to type from context.
    fn expr_or_null(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        match e {
            SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => Ok(None),
            SqlExpr::Nested(inner) => self.expr_or_null(inner),
            other => self.bind(other).map(Some),
        }
    }

    fn bind(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        match e {
            SqlExpr::Identifier(ident) => self.column(&ident.value),
            SqlExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [table, col] => self.qualified(&table.value, &col.value),
                _ => Err(unsup("nested field access")),
            },
            SqlExpr::Nested(inner) => self.expr(inner),
            SqlExpr::Value(v) => literal(&v.value),
            SqlExpr::BinaryOp { left, op, right } => self.binary(op, left, right),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
                // DuckDB `-x`: lower as 0 - x, reusing Sub's promotion.
                let zero = SExpr {
                    kind: SKind::Lit(Lit::I64(0)),
                    ty: Ty::I64,
                    nullable: false,
                };
                self.arith(ArithOp::Sub, zero, self.expr(expr)?)
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => self.expr(expr),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
            } => {
                let inner = self.expr(expr)?;
                if inner.ty != Ty::I1 {
                    return Err(PrepareError::Bind(format!(
                        "NOT requires BOOLEAN, got {}",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Not(Box::new(inner)),
                    ty: Ty::I1,
                    nullable,
                })
            }
            SqlExpr::UnaryOp { op, .. } => Err(unsup(format!("unary operator {op:?}"))),
            SqlExpr::IsNull(inner) => self.is_null(inner, false),
            SqlExpr::IsNotNull(inner) => self.is_null(inner, true),
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } => self.case(operand.as_deref(), conditions, else_result.as_deref()),
            SqlExpr::Cast {
                kind,
                expr,
                data_type,
                ..
            } => {
                let trying = match kind {
                    CastKind::Cast | CastKind::DoubleColon => false,
                    CastKind::TryCast | CastKind::SafeCast => true,
                };
                self.cast(expr, data_type, trying)
            }
            SqlExpr::Function(f) => Err(unsup(format!(
                "function {} (catalogue arrives after the lowering spine)",
                f.name
            ))),
            SqlExpr::Between { .. } => Err(unsup("BETWEEN")),
            SqlExpr::InList { .. } => Err(unsup("IN (...)")),
            SqlExpr::Like { .. } => Err(unsup("LIKE")),
            other => Err(unsup(format!("expression: {other}"))),
        }
    }

    /// The lane of value column `pos` of join `j`: NULL-able exactly when
    /// the join is LEFT (a miss makes it NULL); INNER misses never reach an
    /// expression (the row was skipped).
    fn static_lane(&self, j: usize, pos: usize) -> SExpr {
        let sj = &self.joins[j];
        let col = &sj.table.cols[sj.val_cols[pos] as usize];
        SExpr {
            kind: SKind::StaticCol {
                join: j as u32,
                col: pos as u32,
            },
            ty: col.ty.ty,
            nullable: sj.kind == JoinKind::Left,
        }
    }

    /// Case-insensitive, spelling-preserving bare-column bind over the whole
    /// scope: the dynamic table plus every joined static table's value
    /// columns (DuckDB semantics; ambiguity is an error).
    fn column(&self, name: &str) -> Result<SExpr, PrepareError> {
        let mut hits: Vec<SExpr> = Vec::new();
        for (i, c) in self.in_cols.iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                hits.push(SExpr {
                    kind: SKind::Col(i as u32),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                });
            }
        }
        let mut key_only = false;
        for (j, sj) in self.joins.iter().enumerate() {
            for pos in 0..sj.val_cols.len() {
                if sj.table.cols[sj.val_cols[pos] as usize]
                    .name
                    .eq_ignore_ascii_case(name)
                {
                    hits.push(self.static_lane(j, pos));
                }
            }
            key_only |= sj
                .key_cols
                .iter()
                .any(|&ci| sj.table.cols[ci as usize].name.eq_ignore_ascii_case(name));
        }
        match hits.len() {
            1 => Ok(hits.pop().expect("len checked")),
            0 if key_only => Err(unsup(format!(
                "referencing join key column '{name}' outside its ON clause"
            ))),
            0 => Err(PrepareError::Bind(format!(
                "column '{name}' does not exist in scope"
            ))),
            _ => Err(PrepareError::Bind(format!("ambiguous column '{name}'"))),
        }
    }

    /// `table.col` bind: the dynamic table by its FROM spelling, a joined
    /// static table by its alias (or name).
    fn qualified(&self, table: &str, name: &str) -> Result<SExpr, PrepareError> {
        if table.eq_ignore_ascii_case(&self.this_name) {
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
                PrepareError::Bind(format!("column '{name}' does not exist in '{table}'"))
            })?;
            return Ok(SExpr {
                kind: SKind::Col(i as u32),
                ty: c.ty.ty,
                nullable: c.ty.nullable,
            });
        }
        for (j, sj) in self.joins.iter().enumerate() {
            if !sj.name.eq_ignore_ascii_case(table) {
                continue;
            }
            let mut hit = None;
            for pos in 0..sj.val_cols.len() {
                if sj.table.cols[sj.val_cols[pos] as usize]
                    .name
                    .eq_ignore_ascii_case(name)
                {
                    if hit.is_some() {
                        return Err(PrepareError::Bind(format!("ambiguous column '{name}'")));
                    }
                    hit = Some(pos);
                }
            }
            if let Some(pos) = hit {
                return Ok(self.static_lane(j, pos));
            }
            if sj
                .key_cols
                .iter()
                .any(|&ci| sj.table.cols[ci as usize].name.eq_ignore_ascii_case(name))
            {
                return Err(unsup(format!(
                    "referencing join key column '{table}.{name}' outside its ON clause"
                )));
            }
            return Err(PrepareError::Bind(format!(
                "column '{name}' does not exist in '{table}'"
            )));
        }
        Err(PrepareError::Bind(format!("unknown table '{table}'")))
    }

    fn binary(
        &self,
        op: &BinaryOperator,
        left: &SqlExpr,
        right: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let a = self.expr_or_null(left)?;
        let b = self.expr_or_null(right)?;
        // A NULL literal adopts the other side's type; the op itself is not
        // folded (NULL AND FALSE is FALSE, so folding would be wrong).
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                let n = null_of(null_context_ty(op, a.ty));
                (a, n)
            }
            (None, Some(b)) => {
                let n = null_of(null_context_ty(op, b.ty));
                (n, b)
            }
            (None, None) => return Err(unsup("NULL <op> NULL without a typing context")),
        };
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
                for side in [&a, &b] {
                    if side.ty != Ty::I1 {
                        return Err(PrepareError::Bind(format!(
                            "AND/OR requires BOOLEAN, got {}",
                            side.ty.name()
                        )));
                    }
                }
                let nullable = a.nullable || b.nullable;
                let (a, b) = (Box::new(a), Box::new(b));
                let kind = if matches!(op, BinaryOperator::And) {
                    SKind::And { a, b }
                } else {
                    SKind::Or { a, b }
                };
                Ok(SExpr {
                    kind,
                    ty: Ty::I1,
                    nullable,
                })
            }
            BinaryOperator::StringConcat => Err(unsup("|| (arrives with the string ops)")),
            other => Err(unsup(format!("operator {other}"))),
        }
    }

    fn is_null(&self, inner: &SqlExpr, negated: bool) -> Result<SExpr, PrepareError> {
        // NULL IS NULL is legal and constant; type the literal as i64
        // arbitrarily (only its flag matters).
        let inner = match self.expr_or_null(inner)? {
            Some(e) => e,
            None => null_of(Ty::I64),
        };
        Ok(SExpr {
            kind: SKind::IsNull {
                negated,
                inner: Box::new(inner),
            },
            ty: Ty::I1,
            nullable: false,
        })
    }

    fn case(
        &self,
        operand: Option<&SqlExpr>,
        conditions: &[sqlparser::ast::CaseWhen],
        else_result: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        if conditions.is_empty() {
            return Err(PrepareError::Bind("CASE with no WHEN arms".to_string()));
        }
        // Bind conditions: searched form directly; simple form desugars to
        // `operand = value` per arm (operand re-bound per arm via clone —
        // pure re-evaluation, same result).
        let bound_operand = operand.map(|op| self.expr(op)).transpose()?;
        let mut conds = Vec::with_capacity(conditions.len());
        for when in conditions {
            let c = match &bound_operand {
                Some(op) => {
                    let v = match self.expr_or_null(&when.condition)? {
                        Some(v) => v,
                        None => null_of(op.ty),
                    };
                    self.cmp(CmpPred::Eq, op.clone(), v)?
                }
                None => {
                    let c = match self.expr_or_null(&when.condition)? {
                        Some(c) => c,
                        None => null_of(Ty::I1),
                    };
                    if c.ty != Ty::I1 {
                        return Err(PrepareError::Bind(format!(
                            "CASE WHEN condition must be BOOLEAN, got {}",
                            c.ty.name()
                        )));
                    }
                    c
                }
            };
            conds.push(c);
        }

        // Bind results (NULL allowed), then unify their types.
        let mut results: Vec<Option<SExpr>> = Vec::with_capacity(conditions.len());
        for when in conditions {
            results.push(self.expr_or_null(&when.result)?);
        }
        let else_bound: Option<Option<SExpr>> =
            else_result.map(|e| self.expr_or_null(e)).transpose()?;

        let mut unified: Option<Ty> = None;
        for r in results.iter().chain(else_bound.iter()).flatten() {
            unified = Some(match unified {
                None => r.ty,
                Some(u) if u == r.ty => u,
                Some(Ty::I64) if r.ty == Ty::F64 => Ty::F64,
                Some(Ty::F64) if r.ty == Ty::I64 => Ty::F64,
                Some(u) => {
                    return Err(PrepareError::Bind(format!(
                        "CASE branches disagree: {} vs {}",
                        u.name(),
                        r.ty.name()
                    )))
                }
            });
        }
        let Some(unified) = unified else {
            return Err(unsup("CASE where every branch is NULL"));
        };

        let coerce = |r: Option<SExpr>| -> SExpr {
            match r {
                None => null_of(unified),
                Some(e) if e.ty == Ty::I64 && unified == Ty::F64 => promote_f64(e),
                Some(e) => e,
            }
        };
        let results: Vec<SExpr> = results.into_iter().map(coerce).collect();
        let default = else_bound.map(coerce);

        let nullable = default.is_none()
            || results.iter().any(|r| r.nullable)
            || default.as_ref().is_some_and(|d| d.nullable);
        let arms = conds.into_iter().zip(results).collect();
        Ok(SExpr {
            kind: SKind::Case {
                arms,
                default: default.map(Box::new),
            },
            ty: unified,
            nullable,
        })
    }

    fn cast(
        &self,
        expr: &SqlExpr,
        data_type: &sqlparser::ast::DataType,
        trying: bool,
    ) -> Result<SExpr, PrepareError> {
        let to = cast_target(data_type)?;
        let inner = match self.expr_or_null(expr)? {
            Some(e) => e,
            // CAST(NULL AS T) is just a typed NULL, both forms.
            None => return Ok(null_of(to)),
        };
        if inner.ty == to && !trying {
            return Ok(inner);
        }
        if inner.ty == Ty::Str && to == Ty::I1 {
            return Err(unsup("CAST VARCHAR -> BOOLEAN"));
        }
        let nullable = trying || inner.nullable;
        Ok(SExpr {
            kind: SKind::Cast {
                inner: Box::new(inner),
                trying,
            },
            ty: to,
            nullable,
        })
    }

    fn arith(&self, op: ArithOp, a: SExpr, b: SExpr) -> Result<SExpr, PrepareError> {
        let (a, b, ty) = numeric_promote(op, a, b)?;
        let nullable = a.nullable || b.nullable;
        Ok(SExpr {
            kind: SKind::Arith {
                op,
                a: Box::new(a),
                b: Box::new(b),
            },
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
            kind: SKind::Cmp {
                pred,
                a: Box::new(a),
                b: Box::new(b),
            },
            ty: Ty::I1,
            nullable,
        })
    }
}

fn null_of(ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::NullOf,
        ty,
        nullable: true,
    }
}

/// The type a bare NULL adopts next to a typed operand.
fn null_context_ty(op: &BinaryOperator, other: Ty) -> Ty {
    match op {
        BinaryOperator::And | BinaryOperator::Or => Ty::I1,
        _ => other,
    }
}

fn cast_target(dt: &sqlparser::ast::DataType) -> Result<Ty, PrepareError> {
    let name = dt.to_string().to_uppercase();
    if name.contains("INT") {
        // BIGINT/INTEGER/SMALLINT/TINYINT/HUGEINT/U* all collapse to i64
        // (range divergence for HUGEINT noted in the module docs).
        Ok(Ty::I64)
    } else if name.starts_with("DOUBLE")
        || name.starts_with("FLOAT")
        || name.starts_with("REAL")
        || name.starts_with("DECIMAL")
        || name.starts_with("NUMERIC")
    {
        Ok(Ty::F64)
    } else if name.starts_with("VARCHAR")
        || name.starts_with("TEXT")
        || name.starts_with("STRING")
        || name.starts_with("CHAR")
    {
        Ok(Ty::Str)
    } else if name.starts_with("BOOL") {
        Ok(Ty::I1)
    } else {
        Err(unsup(format!("CAST target type {name}")))
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
        SqlValue::Null => unreachable!("NULL handled by expr_or_null"),
        other => return Err(unsup(format!("literal {other}"))),
    };
    Ok(SExpr {
        kind: SKind::Lit(lit),
        ty,
        nullable: false,
    })
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
    if op == ArithOp::Rem && (a.ty == Ty::F64 || b.ty == Ty::F64) {
        return Err(unsup("% on DOUBLE (needs an frem instruction)"));
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
    // A typed NULL promotes by retyping — no conversion node needed.
    if matches!(e.kind, SKind::NullOf) {
        return SExpr {
            kind: SKind::NullOf,
            ty: Ty::F64,
            nullable: true,
        };
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::IntToFloat(Box::new(e)),
        ty: Ty::F64,
        nullable,
    }
}
