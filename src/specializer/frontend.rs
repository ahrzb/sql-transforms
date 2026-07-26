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
use super::ir::{CmpPred, Col, Lit, TrimSide, Ty};
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
        let pred = fold(bool_context(binder.expr(pred)?, "WHERE predicate")?);
        rel = Rel::Filter {
            input: Box::new(rel),
            pred,
        };
    }

    let mut out_cols = Vec::new();
    let mut exprs = Vec::new();
    let push_item = |out_cols: &mut Vec<Col>,
                     exprs: &mut Vec<SExpr>,
                     name: String,
                     e: SExpr|
     -> Result<(), PrepareError> {
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
        Ok(())
    };
    for item in &select.projection {
        match item {
            SelectItem::UnnamedExpr(e) => push_item(
                &mut out_cols,
                &mut exprs,
                default_name(e),
                fold(binder.expr(e)?),
            )?,
            SelectItem::ExprWithAlias { expr, alias } => push_item(
                &mut out_cols,
                &mut exprs,
                alias.value.clone(),
                fold(binder.expr(expr)?),
            )?,
            SelectItem::Wildcard(opts) => {
                for (name, e) in binder.expand_star(None, opts)? {
                    push_item(&mut out_cols, &mut exprs, name, e)?;
                }
            }
            SelectItem::QualifiedWildcard(kind, opts) => {
                let table = match kind {
                    sqlparser::ast::SelectItemQualifiedWildcardKind::ObjectName(n) => n.to_string(),
                    sqlparser::ast::SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(unsup("expression.* wildcard"))
                    }
                };
                for (name, e) in binder.expand_star(Some(&table), opts)? {
                    push_item(&mut out_cols, &mut exprs, name, e)?;
                }
            }
            SelectItem::ExprWithAliases { .. } => return Err(unsup("multi-alias SELECT item")),
        };
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
        select_aliases: select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.clone()),
                _ => None,
            })
            .collect(),
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
    /// SELECT-list aliases, for cleanly rejecting DuckDB's lateral-alias
    /// extension (an alias referenced inside WHERE) as unsupported.
    select_aliases: Vec<String>,
}

/// AST constructors for the BETWEEN/IN desugars.
fn ast_bin(op: BinaryOperator, l: SqlExpr, r: SqlExpr) -> SqlExpr {
    SqlExpr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

fn ast_not_if(negated: bool, e: SqlExpr) -> SqlExpr {
    if negated {
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(e),
        }
    } else {
        e
    }
}

impl Binder<'_> {
    /// DuckDB unifies BETWEEN/IN across the WHOLE construct (wave-1 pins):
    /// one common type for the subject and every bound/element, so a single
    /// f64 side promotes all sides. Numeric-with-string/bool mixing has
    /// exec-time cast semantics we don't model — clean-unsupported.
    fn unify_family(&self, exprs: &[&SqlExpr]) -> Result<Vec<SqlExpr>, PrepareError> {
        let (mut any_f64, mut any_num, mut any_other) = (false, false, false);
        for e in exprs {
            if let Some(b) = self.expr_or_null(e)? {
                match b.ty {
                    Ty::F64 => (any_f64, any_num) = (true, true),
                    Ty::I64 => any_num = true,
                    Ty::Str | Ty::I1 => any_other = true,
                }
            }
        }
        if any_num && any_other {
            return Err(unsup(
                "BETWEEN/IN mixing strings or booleans with numbers (exec-time cast semantics)",
            ));
        }
        Ok(exprs
            .iter()
            .map(|e| {
                let e = (*e).clone();
                if any_f64 {
                    // CAST is a no-op on already-f64 sides and types NULL
                    // literals from context; exactly DuckDB's unification.
                    SqlExpr::Cast {
                        kind: CastKind::Cast,
                        expr: Box::new(e),
                        data_type: sqlparser::ast::DataType::Double(
                            sqlparser::ast::ExactNumberInfo::None,
                        ),
                        format: None,
                        array: false,
                    }
                } else {
                    e
                }
            })
            .collect())
    }

    /// Expand `*` / `tbl.*` per DuckDB's measured semantics (1.5.5): FROM
    /// order, declared column order within a table, EXCLUDE filtered
    /// case-insensitively. A star item covering a joined static table is
    /// unsupported: DuckDB's expansion includes the join-key column there
    /// (and emits duplicate output names for the shared key), neither of
    /// which the engine models — rejected by name, not silently narrowed.
    fn expand_star(
        &self,
        qualifier: Option<&str>,
        opts: &sqlparser::ast::WildcardAdditionalOptions,
    ) -> Result<Vec<(String, SExpr)>, PrepareError> {
        use sqlparser::ast::ExcludeSelectItem;
        if opts.opt_ilike.is_some() {
            return Err(unsup("SELECT * ILIKE (COLUMNS filter)"));
        }
        if opts.opt_except.is_some() {
            return Err(unsup("SELECT * EXCEPT"));
        }
        if opts.opt_replace.is_some() {
            return Err(unsup("SELECT * REPLACE"));
        }
        if opts.opt_rename.is_some() {
            return Err(unsup("SELECT * RENAME"));
        }
        fn exclude_name(n: &sqlparser::ast::ObjectName) -> Result<&str, PrepareError> {
            match n.0.as_slice() {
                [part] => part
                    .as_ident()
                    .map(|i| i.value.as_str())
                    .ok_or_else(|| unsup("EXCLUDE list entry form")),
                _ => Err(unsup("qualified name in EXCLUDE list")),
            }
        }
        let exclude: Vec<&str> = match &opts.opt_exclude {
            None => Vec::new(),
            Some(ExcludeSelectItem::Single(id)) => vec![exclude_name(id)?],
            Some(ExcludeSelectItem::Multiple(ids)) => ids
                .iter()
                .map(exclude_name)
                .collect::<Result<Vec<_>, _>>()?,
        };
        for (i, a) in exclude.iter().enumerate() {
            if exclude[..i].iter().any(|b| b.eq_ignore_ascii_case(a)) {
                // DuckDB rejects this at parse; ours surfaces at bind.
                return Err(PrepareError::Bind(format!(
                    "duplicate entry \"{a}\" in EXCLUDE list"
                )));
            }
        }

        let mut cols: Vec<(String, SExpr)> = Vec::new();
        let mut matched = false;
        if qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&self.this_name)) {
            matched = true;
            for (i, c) in self.in_cols.iter().enumerate() {
                cols.push((
                    c.name.clone(),
                    SExpr {
                        kind: SKind::Col(i as u32),
                        ty: c.ty.ty,
                        nullable: c.ty.nullable,
                    },
                ));
            }
        }
        for sj in &self.joins {
            if qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&sj.name)) {
                let key = &sj.table.cols[sj.key_cols[0] as usize].name;
                return Err(unsup(format!(
                    "star expansion over joined table '{}' (includes join key column '{key}')",
                    sj.name
                )));
            }
        }
        if !matched {
            return Err(PrepareError::Bind(format!(
                "table '{}' in wildcard does not exist in FROM",
                qualifier.unwrap_or("?")
            )));
        }

        for ex in &exclude {
            if !cols.iter().any(|(n, _)| n.eq_ignore_ascii_case(ex)) {
                return Err(PrepareError::Bind(format!(
                    "column \"{ex}\" in EXCLUDE list not found in FROM clause"
                )));
            }
        }
        cols.retain(|(n, _)| !exclude.iter().any(|ex| ex.eq_ignore_ascii_case(n)));
        Ok(cols)
    }

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
                let inner = bool_context(self.expr(expr)?, "NOT operand")?;
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
            SqlExpr::Function(f) => self.function(f),
            SqlExpr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => {
                let side = match trim_where {
                    None | Some(sqlparser::ast::TrimWhereField::Both) => TrimSide::Both,
                    Some(sqlparser::ast::TrimWhereField::Leading) => TrimSide::Lead,
                    Some(sqlparser::ast::TrimWhereField::Trailing) => TrimSide::Trail,
                };
                let chars: Option<&SqlExpr> = match (trim_what, trim_characters) {
                    (Some(w), _) => Some(w),
                    (None, Some(cs)) if cs.len() == 1 => Some(&cs[0]),
                    (None, Some(cs)) if cs.is_empty() => None,
                    (None, Some(_)) => return Err(unsup("TRIM with multiple character args")),
                    (None, None) => None,
                };
                self.trim_node(side, expr, chars)
            }
            SqlExpr::Substring {
                expr,
                substring_from,
                substring_for,
                ..
            } => self.substr_node(expr, substring_from.as_deref(), substring_for.as_deref()),
            // BETWEEN and IN are exact K3 desugars (wave-1 pins): DuckDB's
            // truth tables over NULL/NaN fall out of Kleene AND/OR of the
            // duck_fcmp comparisons with zero special cases. DuckDB unifies
            // types across the WHOLE construct (one common type for the
            // subject and every bound/element), so any f64 side promotes
            // all sides before the pairwise desugar.
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let mut u = self.unify_family(&[expr, low, high])?;
                let (e, lo, hi) = (u.remove(0), u.remove(0), u.remove(0));
                let both = ast_bin(
                    BinaryOperator::And,
                    ast_bin(BinaryOperator::GtEq, e.clone(), lo),
                    ast_bin(BinaryOperator::LtEq, e, hi),
                );
                self.bind(&ast_not_if(*negated, both))
            }
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let mut family: Vec<&SqlExpr> = vec![expr];
                family.extend(list.iter());
                let mut unified = self.unify_family(&family)?;
                let subject = unified.remove(0);
                let mut chain: Option<SqlExpr> = None;
                for item in unified {
                    let eq = ast_bin(BinaryOperator::Eq, subject.clone(), item);
                    chain = Some(match chain {
                        None => eq,
                        Some(prev) => ast_bin(BinaryOperator::Or, prev, eq),
                    });
                }
                let chain = chain.ok_or_else(|| unsup("empty IN list"))?;
                self.bind(&ast_not_if(*negated, chain))
            }
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
            // Real DuckDB features we don't model reject cleanly, not as
            // bind errors: the rowid pseudo-column, and DuckDB's lateral
            // alias extension (a SELECT alias visible inside WHERE).
            0 if name.eq_ignore_ascii_case("rowid") => Err(unsup("rowid pseudo-column")),
            0 if self
                .select_aliases
                .iter()
                .any(|a| a.eq_ignore_ascii_case(name)) =>
            {
                Err(unsup(format!(
                    "SELECT alias '{name}' referenced outside the SELECT list (lateral alias)"
                )))
            }
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
                let a = bool_context(a, "AND/OR operand")?;
                let b = bool_context(b, "AND/OR operand")?;
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
            BinaryOperator::StringConcat => {
                // DuckDB: || is ALWAYS string concat (1 || 2 = '12',
                // true || true = 'truetrue'), NULL-propagating; operands
                // implicitly cast to VARCHAR.
                let (a, b) = (to_varchar(a), to_varchar(b));
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Concat {
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
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
                None => match self.expr_or_null(&when.condition)? {
                    Some(c) => bool_context(c, "CASE WHEN condition")?,
                    None => null_of(Ty::I1),
                },
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
        // DuckDB pin (2026-07-26): integer % by zero is NULL, not an error —
        // guard with a CASE unless the divisor is a provably non-zero
        // literal. The irem trap stays reachable only for MIN % -1, where
        // DuckDB traps too. Float % is IEEE (x % 0.0 = NaN), no guard.
        if op == ArithOp::Rem
            && ty == Ty::I64
            && !matches!(b.kind, SKind::Lit(Lit::I64(n)) if n != 0)
        {
            let zero = SExpr {
                kind: SKind::Lit(Lit::I64(0)),
                ty: Ty::I64,
                nullable: false,
            };
            // The guard must fire for a NULL divisor too: `b = 0` alone is
            // NULL there (arm not taken) and the irem would run on the
            // garbage payload. TRUE OR NULL = TRUE makes IS NULL the shield.
            let is_zero = self.cmp(CmpPred::Eq, b.clone(), zero)?;
            let cond = if b.nullable {
                let is_null = SExpr {
                    kind: SKind::IsNull {
                        negated: false,
                        inner: Box::new(b.clone()),
                    },
                    ty: Ty::I1,
                    nullable: false,
                };
                SExpr {
                    kind: SKind::Or {
                        a: Box::new(is_null),
                        b: Box::new(is_zero),
                    },
                    ty: Ty::I1,
                    nullable: true,
                }
            } else {
                is_zero
            };
            let rem = SExpr {
                kind: SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                },
                ty,
                nullable,
            };
            return Ok(SExpr {
                kind: SKind::Case {
                    arms: vec![(cond, null_of(Ty::I64))],
                    default: Some(Box::new(rem)),
                },
                ty: Ty::I64,
                nullable: true,
            });
        }
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

    /// The v0 builtin catalogue. Everything here follows the measured pins
    /// in docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md; names
    /// not listed reject as clean unsupported.
    fn function(&self, f: &sqlparser::ast::Function) -> Result<SExpr, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let name = f.name.to_string().to_lowercase();
        let FunctionArguments::List(list) = &f.args else {
            return Err(unsup(format!(
                "function {} without an argument list",
                f.name
            )));
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Err(unsup(format!("function {} argument clauses", f.name)));
        }
        let mut args: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => args.push(e),
                _ => return Err(unsup(format!("function {} argument form", f.name))),
            }
        }
        match name.as_str() {
            "upper" | "lower" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 1 argument"
                    )));
                };
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::Str));
                };
                if inner.ty != Ty::Str {
                    // DuckDB has no implicit numeric->VARCHAR coercion here.
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({})",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StrCase {
                        upper: name == "upper",
                        a: Box::new(inner),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            "ltrim" | "rtrim" => {
                let side = if name == "ltrim" {
                    TrimSide::Lead
                } else {
                    TrimSide::Trail
                };
                match args[..] {
                    [s] => self.trim_node(side, s, None),
                    [s, c] => self.trim_node(side, s, Some(c)),
                    _ => Err(PrepareError::Bind(format!("{name} takes 1 or 2 arguments"))),
                }
            }
            "abs" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(
                        "abs takes exactly 1 argument".to_string(),
                    ));
                };
                // abs(NULL) binds to abs(BIGINT) in DuckDB.
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::I64));
                };
                if !matches!(inner.ty, Ty::I64 | Ty::F64) {
                    return Err(PrepareError::Bind(format!(
                        "no function matches abs({})",
                        inner.ty.name()
                    )));
                }
                let (ty, nullable) = (inner.ty, inner.nullable);
                Ok(SExpr {
                    kind: SKind::Abs(Box::new(inner)),
                    ty,
                    nullable,
                })
            }
            "round" => match args[..] {
                [arg] => {
                    let Some(inner) = self.expr_or_null(arg)? else {
                        return Ok(null_of(Ty::I64));
                    };
                    match inner.ty {
                        // Measured: integer round is identity, type preserved.
                        Ty::I64 => Ok(inner),
                        Ty::F64 => {
                            let nullable = inner.nullable;
                            Ok(SExpr {
                                kind: SKind::Round(Box::new(inner)),
                                ty: Ty::F64,
                                nullable,
                            })
                        }
                        other => Err(PrepareError::Bind(format!(
                            "no function matches round({})",
                            other.name()
                        ))),
                    }
                }
                [_, _] => Err(unsup("round with digits (scale-then-round algorithm)")),
                _ => Err(PrepareError::Bind(
                    "round takes 1 or 2 arguments".to_string(),
                )),
            },
            "concat" => {
                if args.is_empty() {
                    return Err(PrepareError::Bind(
                        "concat needs at least 1 argument".to_string(),
                    ));
                }
                // CONCAT skips NULLs (measured): a literal NULL contributes
                // nothing, a nullable arg becomes CASE WHEN x IS NULL THEN ''
                // ELSE x END, and the all-NULL call is ''.
                let mut acc: Option<SExpr> = None;
                for arg in &args {
                    let Some(e) = self.expr_or_null(arg)? else {
                        continue;
                    };
                    let e = to_varchar(e);
                    let piece = if e.nullable {
                        let cond = SExpr {
                            kind: SKind::IsNull {
                                negated: false,
                                inner: Box::new(e.clone()),
                            },
                            ty: Ty::I1,
                            nullable: false,
                        };
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(cond, lit_str(""))],
                                default: Some(Box::new(e)),
                            },
                            ty: Ty::Str,
                            // Never NULL: either arm produces a value. The
                            // default's flag is provably true on its path.
                            nullable: false,
                        }
                    } else {
                        e
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(p) => SExpr {
                            kind: SKind::Concat {
                                a: Box::new(p),
                                b: Box::new(piece),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        },
                    });
                }
                Ok(acc.unwrap_or_else(|| lit_str("")))
            }
            "coalesce" => {
                // Lazy per-row (measured: untaken erroring arms don't fire) —
                // guaranteed here because CASE branches run only when taken.
                let mut bound = Vec::with_capacity(args.len());
                for arg in &args {
                    if let Some(e) = self.expr_or_null(arg)? {
                        bound.push(e);
                    } // literal NULL args never produce a value: drop them
                }
                if bound.is_empty() {
                    return Err(unsup("COALESCE of only NULL literals"));
                }
                let mut unified = bound[0].ty;
                for e in &bound[1..] {
                    unified = match (unified, e.ty) {
                        (u, t) if u == t => u,
                        (Ty::I64, Ty::F64) | (Ty::F64, Ty::I64) => Ty::F64,
                        (u, t) => {
                            return Err(PrepareError::Bind(format!(
                                "COALESCE arguments disagree: {} vs {}",
                                u.name(),
                                t.name()
                            )))
                        }
                    };
                }
                let mut bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|e| {
                        if e.ty == Ty::I64 && unified == Ty::F64 {
                            promote_f64(e)
                        } else {
                            e
                        }
                    })
                    .collect();
                // Args after the first non-nullable one are unreachable.
                if let Some(stop) = bound.iter().position(|e| !e.nullable) {
                    bound.truncate(stop + 1);
                }
                let mut it = bound.into_iter().rev();
                let mut acc = it.next().expect("non-empty");
                for a in it {
                    let nullable = a.nullable && acc.nullable;
                    let cond = SExpr {
                        kind: SKind::IsNull {
                            negated: true,
                            inner: Box::new(a.clone()),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    };
                    acc = SExpr {
                        kind: SKind::Case {
                            arms: vec![(cond, a)],
                            default: Some(Box::new(acc)),
                        },
                        ty: unified,
                        nullable,
                    };
                }
                Ok(acc)
            }
            "nullif" => {
                let [a, b] = args[..] else {
                    return Err(PrepareError::Bind(
                        "nullif takes exactly 2 arguments".to_string(),
                    ));
                };
                match (self.expr_or_null(a)?, self.expr_or_null(b)?) {
                    (None, Some(b)) => Ok(null_of(b.ty)),
                    // a = NULL is never TRUE, so nullif(a, NULL) is a.
                    (Some(a), None) => Ok(a),
                    (None, None) => Err(unsup("NULLIF(NULL, NULL)")),
                    (Some(a), Some(b)) => {
                        // Comparison at the promoted type; result keeps a's
                        // ORIGINAL type (measured: nullif(1, 1.0) -> INTEGER).
                        let cond = self.cmp(CmpPred::Eq, a.clone(), b)?;
                        let ty = a.ty;
                        Ok(SExpr {
                            kind: SKind::Case {
                                arms: vec![(cond, null_of(ty))],
                                default: Some(Box::new(a)),
                            },
                            ty,
                            nullable: true,
                        })
                    }
                }
            }
            _ => Err(unsup(format!(
                "function {} (not in the v0 catalogue)",
                f.name
            ))),
        }
    }

    /// All TRIM forms plus ltrim/rtrim. `chars` is the optional trim-set
    /// expression; absent means DuckDB's default — the single space (only
    /// 0x20 is trimmed, never tabs/newlines).
    fn trim_node(
        &self,
        side: TrimSide,
        s: &SqlExpr,
        chars: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        let Some(s) = self.expr_or_null(s)? else {
            return Ok(null_of(Ty::Str));
        };
        if s.ty != Ty::Str {
            return Err(PrepareError::Bind(format!(
                "trim needs VARCHAR, got {}",
                s.ty.name()
            )));
        }
        let chars = match chars {
            Some(c) => match self.expr_or_null(c)? {
                // A NULL trim-set propagates NULL (measured).
                None => return Ok(null_of(Ty::Str)),
                Some(c) if c.ty == Ty::Str => c,
                Some(c) => {
                    return Err(PrepareError::Bind(format!(
                        "trim characters must be VARCHAR, got {}",
                        c.ty.name()
                    )))
                }
            },
            None => lit_str(ZS_SPACES),
        };
        let nullable = s.nullable || chars.nullable;
        Ok(SExpr {
            kind: SKind::Trim {
                side,
                a: Box::new(s),
                chars: Box::new(chars),
            },
            ty: Ty::Str,
            nullable,
        })
    }

    /// SUBSTR / SUBSTRING (both syntaxes). Missing start means 1; missing
    /// length means i64::MAX ("rest of the string" under the saturating
    /// window arithmetic in the interpreter).
    fn substr_node(
        &self,
        s: &SqlExpr,
        from: Option<&SqlExpr>,
        for_: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        let Some(s) = self.expr_or_null(s)? else {
            return Ok(null_of(Ty::Str));
        };
        if s.ty != Ty::Str {
            return Err(PrepareError::Bind(format!(
                "substr needs VARCHAR, got {}",
                s.ty.name()
            )));
        }
        // Ok(None) = a literal NULL argument: the whole call is NULL.
        let num = |e: &SqlExpr| -> Result<Option<SExpr>, PrepareError> {
            match self.expr_or_null(e)? {
                None => Ok(None),
                Some(x) if x.ty == Ty::I64 => Ok(Some(x)),
                Some(x) => Err(PrepareError::Bind(format!(
                    "substr position/length must be INTEGER, got {}",
                    x.ty.name()
                ))),
            }
        };
        let start = match from {
            Some(e) => match num(e)? {
                Some(x) => x,
                None => return Ok(null_of(Ty::Str)),
            },
            None => lit_i64(1),
        };
        let len = match for_ {
            Some(e) => match num(e)? {
                Some(x) => Some(Box::new(x)),
                None => return Ok(null_of(Ty::Str)),
            },
            None => None,
        };
        let nullable = s.nullable || start.nullable || len.as_ref().is_some_and(|l| l.nullable);
        Ok(SExpr {
            kind: SKind::Substr {
                a: Box::new(s),
                start: Box::new(start),
                len,
            },
            ty: Ty::Str,
            nullable,
        })
    }
}

/// DuckDB's default trim set (adversarial census, 1.5.5): exactly the
/// Unicode Zs space separators — NOT tab/newline/ZWSP/BOM/LS/PS/NEL.
const ZS_SPACES: &str = "\u{20}\u{A0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\
                         \u{2006}\u{2007}\u{2008}\u{2009}\u{200A}\u{202F}\u{205F}\u{3000}";

fn lit_str(s: &str) -> SExpr {
    SExpr {
        kind: SKind::Lit(Lit::Str(s.to_string())),
        ty: Ty::Str,
        nullable: false,
    }
}

fn lit_i64(n: i64) -> SExpr {
    SExpr {
        kind: SKind::Lit(Lit::I64(n)),
        ty: Ty::I64,
        nullable: false,
    }
}

/// DuckDB coerces numeric values to BOOLEAN in conditional contexts —
/// WHERE, AND/OR/NOT operands, CASE WHEN conditions (measured 1.5.5:
/// nonzero -> true including NaN, 0 and -0.0 -> false, NULL -> NULL).
/// Strings stay a bind error (DuckDB errors at runtime; such queries never
/// mine into the corpus).
fn bool_context(e: SExpr, what: &str) -> Result<SExpr, PrepareError> {
    match e.ty {
        Ty::I1 => Ok(e),
        Ty::I64 | Ty::F64 => {
            if matches!(e.kind, SKind::NullOf) {
                return Ok(null_of(Ty::I1));
            }
            let nullable = e.nullable;
            Ok(SExpr {
                kind: SKind::Cast {
                    inner: Box::new(e),
                    trying: false,
                },
                ty: Ty::I1,
                nullable,
            })
        }
        other => Err(PrepareError::Bind(format!(
            "{what} must be BOOLEAN, got {}",
            other.name()
        ))),
    }
}

/// DuckDB's implicit VARCHAR coercion for concatenation: ints, floats and
/// bools all render through the same conversion CAST uses.
fn to_varchar(e: SExpr) -> SExpr {
    if e.ty == Ty::Str {
        return e;
    }
    if matches!(e.kind, SKind::NullOf) {
        return null_of(Ty::Str);
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::Cast {
            inner: Box::new(e),
            trying: false,
        },
        ty: Ty::Str,
        nullable,
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
