//! Frontend: SQL text -> bound, typed relational IR. Parsing is sqlparser's
//! GenericDialect (a measured superset of DuckDbDialect for the forms we
//! serve — pins-wave5/sqlparser-spike.json); binding and type derivation
//! follow DuckDB semantics as measured (see plan.rs notes and the pins in
//! exec/interp.rs).
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
    AccessExpr, BinaryOperator, CastKind, Expr as SqlExpr, JoinConstraint, JoinOperator,
    SelectItem, SetExpr, Statement, Subscript, TableFactor, UnaryOperator, Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use super::fold::fold;
use super::ir::{BinOp, CmpPred, Col, Lit, NumOp1, StrOp2, StrOp2i, StrOp3, TrimSide, Ty};
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
    // GenericDialect, not DuckDbDialect: measured as a strict superset for
    // the forms we serve (adds ^@, * ILIKE, * RENAME) and matches the oracle
    // path in datafusion/plan.rs — see pins-wave5/sqlparser-spike.json.
    // DuckDB-only surface forms sqlparser can't represent are token-rewritten
    // first (rewrite.rs).
    if sql.to_ascii_lowercase().contains("__glob_pat") {
        // Reserved for the GLOB rewrite marker — never valid user SQL.
        return Err(unsup("reserved identifier __glob_pat"));
    }
    let dialect = GenericDialect {};
    let tokens = sqlparser::tokenizer::Tokenizer::new(&dialect, sql)
        .tokenize()
        .map_err(|e| PrepareError::Parse(e.to_string()))?;
    let tokens = super::rewrite::rewrite_glob(super::rewrite::rewrite_colon_aliases(tokens));
    let statements = Parser::new(&dialect)
        .with_tokens(tokens)
        .parse_statements()
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

    let (binder, joins, leftover_where) = bind_from(select, this_name, in_cols, statics)?;

    let mut rel = Rel::Scan;
    if let Some(pred) = &leftover_where {
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
) -> Result<(Binder<'a>, Vec<JoinSpec>, Option<SqlExpr>), PrepareError> {
    let Some((table, comma_rels)) = select.from.split_first() else {
        return Err(unsup("FROM-less SELECT"));
    };
    let dyn_name = match &table.relation {
        TableFactor::Table { name, alias, .. } => {
            let n = name.to_string();
            if !n.eq_ignore_ascii_case(this_name) {
                return Err(unsup(format!(
                    "table '{n}' as the driving relation (must be the dynamic table '{this_name}')"
                )));
            }
            match alias {
                // Measured: an alias REPLACES the original name entirely
                // (qualified refs through the original are binder errors in
                // DuckDB) — making the alias the binder's this_name gives
                // exactly that scoping.
                Some(a) if a.columns.is_empty() => a.name.value.clone(),
                Some(_) => return Err(unsup("column-renaming table alias t(a, b, ...)")),
                None => n,
            }
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
        let table_idx = resolve_static(statics, &raw_name)?;
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
        let (keys, key_cols, residual_raw, using) = match constraint {
            JoinConstraint::On(e) => {
                let (keys, key_cols, res) = bind_on(&binder, st, &scope_name, e)?;
                (keys, key_cols, res, false)
            }
            // USING desugar (wave-4 pins): each column pairs the LEFT
            // scope's binding with this table's column; duplicates in the
            // list dedupe silently; ambiguity in the left scope (e.g.
            // after a prior ON join) errors exactly like DuckDB.
            JoinConstraint::Using(cols) => {
                let mut keys = Vec::new();
                let mut key_cols = Vec::new();
                for obj in cols {
                    let [part] = obj.0.as_slice() else {
                        return Err(unsup("qualified name in JOIN USING"));
                    };
                    let name = part
                        .as_ident()
                        .map(|i| i.value.clone())
                        .ok_or_else(|| unsup("JOIN USING entry form"))?;
                    let mut col = None;
                    for (i, c) in st.cols.iter().enumerate() {
                        if c.name.eq_ignore_ascii_case(&name) {
                            col = Some(i as u32);
                        }
                    }
                    let Some(col) = col else {
                        return Err(PrepareError::Bind(format!(
                            "column \"{name}\" does not exist on right side of join!"
                        )));
                    };
                    if key_cols.contains(&col) {
                        continue; // USING (a, a) dedupes silently (measured)
                    }
                    let key = fold(binder.column(&name)?);
                    keys.push(promote_key(key, st, col)?);
                    key_cols.push(col);
                }
                (keys, key_cols, Vec::new(), true)
            }
            JoinConstraint::Natural => return Err(unsup("NATURAL JOIN")),
            JoinConstraint::None => return Err(unsup("JOIN without ON (cross join)")),
        };
        let val_cols: Vec<u32> = (0..st.cols.len() as u32)
            .filter(|c| !key_cols.contains(c))
            .collect();

        binder.joins.push(ScopeJoin {
            name: scope_name,
            table: st,
            kind,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
            keys: keys.clone(),
            using,
        });
        // Residual conjuncts bind with THIS join in scope.
        let j = (binder.joins.len() - 1) as u32;
        let residual = bind_residual(&binder, j, &residual_raw)?;
        specs.push(JoinSpec {
            table: table_idx,
            kind,
            keys,
            key_cols,
            val_cols,
            residual,
        });
    }

    // Comma relations (measured: FROM t, u WHERE t.k = u.k is bit-identical
    // to the INNER probe, star order included; residual WHERE placement is
    // free under INNER). Equi conjuncts pairing the current scope with a
    // comma table's column are consumed as its probe keys; everything else
    // stays WHERE. A keyless comma table is a cross join — correct for a
    // 1-row static via the empty-key map (the duplicate-key check enforces
    // single-entry-ness at compile; a 0-row static annihilates, also
    // measured).
    let mut conjuncts: Vec<&SqlExpr> = Vec::new();
    if let Some(sel) = &select.selection {
        collect_conjuncts(sel, &mut conjuncts);
    }
    let mut consumed = vec![false; conjuncts.len()];
    for rel in comma_rels {
        if !rel.joins.is_empty() {
            return Err(unsup("JOIN attached to a comma-joined relation"));
        }
        let (raw_name, scope_name) = match &rel.relation {
            TableFactor::Table { name, alias, .. } => {
                let n = name.to_string();
                let s = alias
                    .as_ref()
                    .map(|a| a.name.value.clone())
                    .unwrap_or_else(|| n.clone());
                (n, s)
            }
            other => return Err(unsup(format!("FROM {other}"))),
        };
        if raw_name.eq_ignore_ascii_case(this_name) {
            return Err(unsup("joining the dynamic table to itself"));
        }
        // Unresolvable comma tables (schema-qualified names, table
        // functions we didn't get as statics) stay CLEAN.
        let table_idx = resolve_static(statics, &raw_name).map_err(|_| {
            unsup(format!(
                "comma-joined table '{raw_name}' is not a provided static table"
            ))
        })?;
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
        let mut keys = Vec::new();
        let mut key_cols = Vec::new();
        for (ci, c) in conjuncts.iter().enumerate() {
            if consumed[ci] {
                continue;
            }
            let SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } = c
            else {
                continue;
            };
            let l = static_col_of(left, st, &scope_name)?;
            let r = static_col_of(right, st, &scope_name)?;
            let (col, dyn_side, static_side) = match (l, r) {
                (Some(c), None) => (c, right.as_ref(), left.as_ref()),
                (None, Some(c)) => (c, left.as_ref(), right.as_ref()),
                _ => continue, // stays WHERE
            };
            if let SqlExpr::Identifier(id) = static_side {
                if binder.column(&id.value).is_ok() {
                    return Err(PrepareError::Bind(format!(
                        "ambiguous column '{}' in WHERE (qualify it)",
                        id.value
                    )));
                }
            }
            // The dynamic side must bind in the scope BEFORE this table —
            // if it references this or a later comma table, leave the
            // conjunct in WHERE (a later table may consume it).
            let Ok(key) = binder.expr(dyn_side) else {
                continue;
            };
            keys.push(promote_key(fold(key), st, col)?);
            key_cols.push(col);
            consumed[ci] = true;
        }
        let val_cols: Vec<u32> = (0..st.cols.len() as u32)
            .filter(|c| !key_cols.contains(c))
            .collect();
        binder.joins.push(ScopeJoin {
            name: scope_name,
            table: st,
            kind: JoinKind::Inner,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
            keys: keys.clone(),
            using: false,
        });
        specs.push(JoinSpec {
            table: table_idx,
            kind: JoinKind::Inner,
            keys,
            key_cols,
            val_cols,
            residual: None,
        });
    }

    // Rebuild the WHERE from unconsumed conjuncts (identity when nothing
    // was consumed — the single-relation path always takes this shape).
    let leftover = if comma_rels.is_empty() {
        select.selection.clone()
    } else {
        let mut acc: Option<SqlExpr> = None;
        for (ci, c) in conjuncts.iter().enumerate() {
            if consumed[ci] {
                continue;
            }
            acc = Some(match acc {
                None => (*c).clone(),
                Some(p) => ast_bin(BinaryOperator::And, p, (*c).clone()),
            });
        }
        acc
    };
    Ok((binder, specs, leftover))
}

fn resolve_static(statics: &[StaticTable], raw_name: &str) -> Result<usize, PrepareError> {
    let mut table_idx = None;
    for (i, st) in statics.iter().enumerate() {
        if st.name.eq_ignore_ascii_case(raw_name) {
            if table_idx.is_some() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous static table '{raw_name}'"
                )));
            }
            table_idx = Some(i);
        }
    }
    table_idx.ok_or_else(|| {
        PrepareError::Bind(format!(
            "table '{raw_name}' was not provided as a static table"
        ))
    })
}

/// Bind the non-key ON conjuncts of join `j` and AND them into the spec's
/// residual, enforcing the wave-4 evaluation-order rule: single-side
/// residuals must be conservatively trap-free (DuckDB scan-pushes them —
/// different error timing); both-sides residuals may trap (DuckDB
/// evaluates them per candidate pair, exactly our hit-guarded lowering).
fn bind_residual(
    binder: &Binder<'_>,
    j: u32,
    raw: &[&SqlExpr],
) -> Result<Option<SExpr>, PrepareError> {
    let mut acc: Option<SExpr> = None;
    for c in raw {
        let bound = bool_context(fold(binder.expr(c)?), "JOIN ON condition")?;
        let (mut right, mut left, mut total, mut known) = (false, false, true, true);
        scan_residual(&bound, j, &mut right, &mut left, &mut total, &mut known);
        if !(total || (left && right && known)) {
            return Err(unsup(format!(
                "JOIN ON condition '{c}' (single-side residual with trapping \
                 ops: DuckDB's scan-pushed evaluation order differs)"
            )));
        }
        acc = Some(match acc {
            None => bound,
            Some(p) => {
                let nullable = p.nullable || bound.nullable;
                SExpr {
                    kind: SKind::And {
                        a: Box::new(p),
                        b: Box::new(bound),
                    },
                    ty: Ty::I1,
                    nullable,
                }
            }
        });
    }
    Ok(acc)
}

/// Bind a JOIN ... ON condition. Equalities pairing a dynamic-side
/// expression with a static column become probe keys; every OTHER conjunct
/// (non-equalities, constant equalities, both-sides-static equalities) is
/// returned raw for residual binding once the join is in scope (wave-4:
/// `match = key_hit AND residual`).
fn bind_on<'e>(
    binder: &Binder<'_>,
    st: &StaticTable,
    scope_name: &str,
    on: &'e SqlExpr,
) -> Result<(Vec<SExpr>, Vec<u32>, Vec<&'e SqlExpr>), PrepareError> {
    let mut conjuncts = Vec::new();
    collect_conjuncts(on, &mut conjuncts);
    let mut keys = Vec::new();
    let mut key_cols = Vec::new();
    let mut residual = Vec::new();
    for c in conjuncts {
        let SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } = c
        else {
            residual.push(c);
            continue;
        };
        let l = static_col_of(left, st, scope_name)?;
        let r = static_col_of(right, st, scope_name)?;
        let (col, dyn_side, static_side) = match (l, r) {
            (Some(c), None) => (c, right.as_ref(), left.as_ref()),
            (None, Some(c)) => (c, left.as_ref(), right.as_ref()),
            // Both sides this table (r.a = r.b) or neither (test.b = 2,
            // NULL = 2): a residual match condition, not a key (measured —
            // DuckDB binds these fine and they filter matches).
            _ => {
                residual.push(c);
                continue;
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
        let key = promote_key(key, st, col)?;
        keys.push(key);
        key_cols.push(col);
    }
    Ok((keys, key_cols, residual))
}

/// Promote a dynamic-side key expression to the map's key type.
fn promote_key(key: SExpr, st: &StaticTable, col: u32) -> Result<SExpr, PrepareError> {
    let col_ty = st.cols[col as usize].ty.ty;
    match (key.ty, col_ty) {
        (a, b) if a == b => Ok(key),
        (Ty::I64, Ty::F64) => Ok(promote_f64(key)),
        // Static-side ints promote at materialization: the map key type
        // (the key expression's type) becomes F64 and the build side is
        // converted while the probe table is built.
        (Ty::F64, Ty::I64) => Ok(key),
        (a, b) => Err(PrepareError::Bind(format!(
            "cannot join {} with {} (ON '{}')",
            a.name(),
            b.name(),
            st.cols[col as usize].name
        ))),
    }
}

/// One traversal answering the wave-4 residual questions about a bound ON
/// residual: `right`/`left` — does it reference THIS join's columns / any
/// other scope; `total` — is every node in the conservative trap-free
/// allowlist (columns, literals, comparisons, IS NULL, logic); `known` —
/// was every node classifiable at all. Acceptance rule at the call site:
/// `total || (left && right && known)` — measured: DuckDB scan-pushes
/// single-side residuals (eager trap timing) but evaluates both-sides
/// residuals per candidate pair, which our hit-guarded lowering matches.
fn scan_residual(
    e: &SExpr,
    j: u32,
    right: &mut bool,
    left: &mut bool,
    total: &mut bool,
    known: &mut bool,
) {
    match &e.kind {
        SKind::StaticCol { join, .. } | SKind::JoinHit(join) => {
            if *join == j {
                *right = true;
            } else {
                *left = true;
            }
        }
        SKind::Col(_) => *left = true,
        SKind::Lit(_) | SKind::NullOf => {}
        SKind::Cmp { a, b, .. } | SKind::And { a, b } | SKind::Or { a, b } => {
            scan_residual(a, j, right, left, total, known);
            scan_residual(b, j, right, left, total, known);
        }
        SKind::Not(a) | SKind::IsNull { inner: a, .. } | SKind::IntToFloat(a) => {
            scan_residual(a, j, right, left, total, known);
        }
        SKind::Arith { a, b, .. } => {
            *total = false;
            scan_residual(a, j, right, left, total, known);
            scan_residual(b, j, right, left, total, known);
        }
        SKind::Case { arms, default } => {
            *total = false;
            for (c, r) in arms {
                scan_residual(c, j, right, left, total, known);
                scan_residual(r, j, right, left, total, known);
            }
            if let Some(d) = default {
                scan_residual(d, j, right, left, total, known);
            }
        }
        // Anything else: not classifiable — the caller must reject rather
        // than risk the permissive both-sides path on a wrong guess.
        _ => {
            *total = false;
            *known = false;
        }
    }
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
/// are probe values (bindable directly) vs keys (reconstructed from the
/// dynamic side: `r.id` ≡ CASE match THEN dyn-key ELSE NULL — wave-4).
struct ScopeJoin<'a> {
    name: String,
    table: &'a StaticTable,
    kind: JoinKind,
    key_cols: Vec<u32>,
    val_cols: Vec<u32>,
    /// Dynamic-side key expressions aligned with `key_cols` — the material
    /// for key-column reconstruction.
    keys: Vec<SExpr>,
    /// USING join: the static side's using (key) columns are merged into
    /// the left occurrence — hidden from bare-name binds and star
    /// expansion (measured: merged col sits at the LEFT position with the
    /// LEFT value; `t2.a` stays addressable and is NULL on a LEFT miss).
    using: bool,
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

fn math1_node(op: NumOp1, inner: SExpr) -> SExpr {
    let nullable = inner.nullable;
    SExpr {
        kind: SKind::MathF1 {
            op,
            a: Box::new(inner),
        },
        ty: Ty::F64,
        nullable,
    }
}

/// Find the `__glob_pat(...)` identity marker (rewrite.rs) in a LIKE
/// pattern tree and return the tree with the marker unwrapped; None means
/// "no marker — a plain LIKE".
fn strip_glob_marker(e: &SqlExpr) -> Option<SqlExpr> {
    fn marker_arg(e: &SqlExpr) -> Option<&SqlExpr> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let SqlExpr::Function(f) = e else { return None };
        if !f.name.to_string().eq_ignore_ascii_case("__glob_pat") {
            return None;
        }
        let FunctionArguments::List(list) = &f.args else {
            return None;
        };
        let [FunctionArg::Unnamed(FunctionArgExpr::Expr(inner))] = &list.args[..] else {
            return None;
        };
        Some(inner)
    }
    if let Some(inner) = marker_arg(e) {
        return Some(inner.clone());
    }
    match e {
        SqlExpr::BinaryOp { left, op, right } => {
            if let Some(l) = strip_glob_marker(left) {
                Some(SqlExpr::BinaryOp {
                    left: Box::new(l),
                    op: op.clone(),
                    right: right.clone(),
                })
            } else {
                strip_glob_marker(right).map(|r| SqlExpr::BinaryOp {
                    left: left.clone(),
                    op: op.clone(),
                    right: Box::new(r),
                })
            }
        }
        SqlExpr::Nested(inner) => strip_glob_marker(inner).map(|i| SqlExpr::Nested(Box::new(i))),
        _ => None,
    }
}

fn is_flat_bitop(op: &BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::PGBitwiseShiftLeft
            | BinaryOperator::PGBitwiseShiftRight
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseOr
    )
}

fn flat_bitop(op: &BinaryOperator) -> ArithOp {
    match op {
        BinaryOperator::PGBitwiseShiftLeft => ArithOp::Shl,
        BinaryOperator::PGBitwiseShiftRight => ArithOp::Shr,
        BinaryOperator::BitwiseAnd => ArithOp::BitAnd,
        _ => ArithOp::BitOr,
    }
}

/// In-order collect of a maximal run of flat-tier bit operators: yields the
/// operands and operators in SOURCE order regardless of how sqlparser
/// grouped them.
fn flatten_bitops<'e>(
    e: &'e SqlExpr,
    ops: &mut Vec<&'e BinaryOperator>,
    operands: &mut Vec<&'e SqlExpr>,
) {
    match e {
        SqlExpr::BinaryOp { left, op, right } if is_flat_bitop(op) => {
            flatten_bitops(left, ops, operands);
            ops.push(op);
            flatten_bitops(right, ops, operands);
        }
        other => operands.push(other),
    }
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
        // Joined tables expand in FROM order, columns in DECLARED order
        // (measured): value columns as probe lanes, key columns via the
        // dynamic-side reconstruction, USING keys suppressed (merged into
        // the left occurrence). Duplicate output names across the star are
        // caught by the existing duplicate-name check — DuckDB emits them
        // verbatim, our typed output model cannot (documented constraint).
        for (j, sj) in self.joins.iter().enumerate() {
            if !qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&sj.name)) {
                continue;
            }
            matched = true;
            for (ci, c) in sj.table.cols.iter().enumerate() {
                let ci = ci as u32;
                if let Some(pos) = sj.val_cols.iter().position(|&v| v == ci) {
                    cols.push((c.name.clone(), self.static_lane(j, pos)));
                } else {
                    let kp = sj
                        .key_cols
                        .iter()
                        .position(|&k| k == ci)
                        .expect("column is key or value");
                    if !sj.using {
                        cols.push((c.name.clone(), self.key_lane(j, kp)));
                    }
                }
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
            // DuckDB puts << >> & | in ONE flat left-associative tier;
            // sqlparser tiers them (& above << >> above |), so 4|1&1 would
            // silently parse as 4|(1&1)=5 where DuckDB computes (4|1)&1=1.
            // Re-associate: in-order traversal of the parsed run recovers
            // source order, then left-fold. User parens are Nested nodes,
            // which the flatten treats as leaves.
            SqlExpr::BinaryOp { op, .. } if is_flat_bitop(op) => {
                let (mut ops, mut operands) = (Vec::new(), Vec::new());
                flatten_bitops(e, &mut ops, &mut operands);
                let mut acc = self.expr_or_null(operands[0])?;
                for (o, rhs) in ops.iter().zip(&operands[1..]) {
                    let b = self.expr_or_null(rhs)?;
                    let (av, bv) = match (acc, b) {
                        (Some(x), Some(y)) => (x, y),
                        (Some(x), None) => {
                            let n = null_of(x.ty);
                            (x, n)
                        }
                        (None, Some(y)) => {
                            let n = null_of(y.ty);
                            (n, y)
                        }
                        (None, None) => {
                            return Err(unsup("NULL <op> NULL without a typing context"))
                        }
                    };
                    acc = Some(self.arith(flat_bitop(o), av, bv)?);
                }
                Ok(acc.expect("a flat-bitop run has at least one operator"))
            }
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
            // SQL-standard position(needle IN haystack) — needle-first,
            // same op as instr/strpos (measured).
            SqlExpr::Position { expr, r#in } => self.str2("position", StrOp2::Find, r#in, expr),
            // sqlparser gives FLOOR/CEIL dedicated AST nodes, not Function
            // calls; the datetime `CEIL(x TO field)` form rejects by name.
            SqlExpr::Floor { expr, field } => match field {
                sqlparser::ast::CeilFloorKind::Scale(_) => Err(unsup("floor with scale argument")),
                sqlparser::ast::CeilFloorKind::DateTimeField(
                    sqlparser::ast::DateTimeField::NoDateTime,
                ) => self.math1("floor", NumOp1::Ffloor, expr),
                _ => Err(unsup("FLOOR(x TO datetime-field)")),
            },
            SqlExpr::Ceil { expr, field } => match field {
                sqlparser::ast::CeilFloorKind::Scale(_) => Err(unsup("ceil with scale argument")),
                sqlparser::ast::CeilFloorKind::DateTimeField(
                    sqlparser::ast::DateTimeField::NoDateTime,
                ) => self.math1("ceil", NumOp1::Fceil, expr),
                _ => Err(unsup("CEIL(x TO datetime-field)")),
            },
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
            SqlExpr::Like {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            }
            | SqlExpr::ILike {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => {
                let ci = matches!(e, SqlExpr::ILike { .. });
                if *any {
                    return Err(unsup("LIKE ANY"));
                }
                // GLOB arrives as LIKE with the pattern wrapped in the
                // __glob_pat identity marker (rewrite.rs); unwrap anywhere
                // in the pattern tree so `s GLOB 'a' || x` still binds the
                // full concat as the pattern.
                if let Some(pat) = strip_glob_marker(pattern) {
                    if ci || *negated || escape_char.is_some() {
                        return Err(unsup("GLOB in a LIKE-variant position"));
                    }
                    let (ba, bp) = (self.expr_or_null(expr)?, self.expr_or_null(&pat)?);
                    let (Some(ba), Some(bp)) = (ba, bp) else {
                        return Ok(null_of(Ty::I1));
                    };
                    for side in [&ba, &bp] {
                        if side.ty != Ty::Str {
                            // GLOB has NO implicit casts (wave-5 pins;
                            // DuckDB's scalar name for it is ~~~).
                            return Err(PrepareError::Bind(format!(
                                "no function matches ~~~({}, {})",
                                ba.ty.name(),
                                bp.ty.name()
                            )));
                        }
                    }
                    let nullable = ba.nullable || bp.nullable;
                    return Ok(SExpr {
                        kind: SKind::Str2 {
                            op: StrOp2::Glob,
                            a: Box::new(ba),
                            b: Box::new(bp),
                        },
                        ty: Ty::I1,
                        nullable,
                    });
                }
                let (ba, bp) = (self.expr_or_null(expr)?, self.expr_or_null(pattern)?);
                let (Some(ba), Some(bp)) = (ba, bp) else {
                    // NULL on either side is NULL before any validation
                    // (even a bad ESCAPE never raises on NULL rows).
                    return Ok(null_of(Ty::I1));
                };
                for side in [&ba, &bp] {
                    if side.ty != Ty::Str {
                        return Err(PrepareError::Bind(format!(
                            "no function matches {}({})",
                            if ci { "ilike" } else { "like" },
                            side.ty.name()
                        )));
                    }
                }
                let esc = match escape_char {
                    None => None,
                    Some(v) => match &v.value {
                        SqlValue::SingleQuotedString(s) => Some(Box::new(lit_str(s))),
                        SqlValue::Null => return Ok(null_of(Ty::I1)),
                        other => return Err(unsup(format!("ESCAPE {other} (non-string escape)"))),
                    },
                };
                let nullable = ba.nullable || bp.nullable;
                let like = SExpr {
                    kind: SKind::Like {
                        ci,
                        a: Box::new(ba),
                        p: Box::new(bp),
                        esc,
                    },
                    ty: Ty::I1,
                    nullable,
                };
                Ok(if *negated {
                    SExpr {
                        kind: SKind::Not(Box::new(like)),
                        ty: Ty::I1,
                        nullable,
                    }
                } else {
                    like
                })
            }
            SqlExpr::SimilarTo { .. } => Err(unsup(
                "SIMILAR TO (DuckDB binds it to regexp_full_match, not SQL wildcards)",
            )),
            // Bracket syntax s[i] / s[a:b] — exactly array_extract /
            // array_slice in DuckDB (one shared implementation, measured:
            // pins-wave5/{subscripts-extended,slices}.json).
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                let mut cur = match self.expr_or_null(root)? {
                    Some(b) => b,
                    None => null_of(Ty::Str),
                };
                for acc in access_chain {
                    let AccessExpr::Subscript(sub) = acc else {
                        return Err(unsup("struct field access in a subscript chain"));
                    };
                    cur = match sub {
                        Subscript::Index { index } => {
                            self.apply_extract("array_extract", cur, index)?
                        }
                        Subscript::Slice {
                            lower_bound,
                            upper_bound,
                            stride,
                        } => {
                            if stride.is_some() {
                                // DuckDB rejects step slicing on VARCHAR for
                                // EVERY step value, including 1 (measured).
                                return Err(unsup(
                                    "slice with step (DuckDB: not implemented for string types)",
                                ));
                            }
                            self.apply_slice(
                                "array_slice",
                                cur,
                                lower_bound.as_ref(),
                                upper_bound.as_ref(),
                            )?
                        }
                    };
                }
                Ok(cur)
            }
            other => Err(unsup(format!("expression: {other}"))),
        }
    }

    /// s[i] / array_extract / list_extract on a bound VARCHAR subject:
    /// exec handles negatives (len+1+i), 0/out-of-range -> '' and the
    /// runtime +-2^32 offset trap (pins-wave5/subscripts-extended.json).
    fn apply_extract(
        &self,
        name: &str,
        bs: SExpr,
        n: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        if bs.ty != Ty::Str {
            // The LIST overload has different out-of-range semantics
            // (NULL, not '') — only the VARCHAR path ships in v0.
            return Err(unsup(format!(
                "{name} on {} (only VARCHAR subscripts in v0)",
                bs.ty.name()
            )));
        }
        let Some(bn) = self.expr_or_null(n)? else {
            return Ok(null_of(Ty::Str));
        };
        if bn.ty != Ty::I64 {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}(str, {})",
                bn.ty.name()
            )));
        }
        let nullable = bs.nullable || bn.nullable;
        Ok(SExpr {
            kind: SKind::Str2i {
                op: StrOp2i::Extract,
                a: Box::new(bs),
                n: Box::new(bn),
            },
            ty: Ty::Str,
            nullable,
        })
    }

    /// s[a:b] / array_slice / list_slice on a bound VARCHAR subject. Open
    /// bounds are pure syntax ([:b] == [1:b], [a:] == [a:-1]); a NULL bound
    /// is NOT open — it nulls the result (pins-wave5/slices.json).
    fn apply_slice(
        &self,
        name: &str,
        bs: SExpr,
        lo: Option<&SqlExpr>,
        hi: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        if bs.ty != Ty::Str {
            return Err(unsup(format!(
                "{name} on {} (only VARCHAR subscripts in v0)",
                bs.ty.name()
            )));
        }
        let mut bind_bound = |e: Option<&SqlExpr>, open: i64| -> Result<Option<SExpr>, PrepareError> {
            match e {
                None => Ok(Some(lit_i64(open))),
                Some(e) => self.expr_or_null(e),
            }
        };
        let (blo, bhi) = (bind_bound(lo, 1)?, bind_bound(hi, -1)?);
        let (Some(blo), Some(bhi)) = (blo, bhi) else {
            return Ok(null_of(Ty::Str));
        };
        for e in [&blo, &bhi] {
            if e.ty != Ty::I64 {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}(str, {}, {})",
                    blo.ty.name(),
                    bhi.ty.name()
                )));
            }
        }
        let nullable = bs.nullable || blo.nullable || bhi.nullable;
        Ok(SExpr {
            kind: SKind::Sslice {
                a: Box::new(bs),
                lo: Box::new(blo),
                hi: Box::new(bhi),
            },
            ty: Ty::Str,
            nullable,
        })
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

    /// KEY column `key_pos` of join `j`, reconstructed from the dynamic
    /// side (measured: on a match the static key equals the probe key;
    /// on a LEFT miss it is NULL): INNER rows all matched, so the key
    /// expression itself is exact; LEFT wraps it in CASE match THEN key
    /// ELSE NULL.
    fn key_lane(&self, j: usize, key_pos: usize) -> SExpr {
        let sj = &self.joins[j];
        let key = sj.keys[key_pos].clone();
        if sj.kind == JoinKind::Inner {
            return key;
        }
        let ty = key.ty;
        let hit = SExpr {
            kind: SKind::JoinHit(j as u32),
            ty: Ty::I1,
            nullable: false,
        };
        SExpr {
            kind: SKind::Case {
                arms: vec![(hit, key)],
                default: None,
            },
            ty,
            nullable: true,
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
        for (j, sj) in self.joins.iter().enumerate() {
            for pos in 0..sj.val_cols.len() {
                if sj.table.cols[sj.val_cols[pos] as usize]
                    .name
                    .eq_ignore_ascii_case(name)
                {
                    hits.push(self.static_lane(j, pos));
                }
            }
            // Key columns resolve via reconstruction. A USING join's key
            // is MERGED into the left occurrence (measured) — the static
            // side contributes no separate binding; an ON join's key
            // contributes one, so a bare shared key name is ambiguous,
            // exactly like DuckDB.
            if !sj.using {
                for (kp, &ci) in sj.key_cols.iter().enumerate() {
                    if sj.table.cols[ci as usize].name.eq_ignore_ascii_case(name) {
                        hits.push(self.key_lane(j, kp));
                    }
                }
            }
        }
        match hits.len() {
            1 => Ok(hits.pop().expect("len checked")),
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
            // Qualified key access reconstructs from the dynamic side —
            // measured to stay addressable even after USING (NULL on a
            // LEFT miss, never coalesced).
            for (kp, &ci) in sj.key_cols.iter().enumerate() {
                if sj.table.cols[ci as usize].name.eq_ignore_ascii_case(name) {
                    return Ok(self.key_lane(j, kp));
                }
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
            BinaryOperator::DuckIntegerDivide => self.arith(ArithOp::IDiv, a, b),
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
            // s ^@ p is exactly starts_with(s, p): byte-prefix compare,
            // VARCHAR-only with no implicit casts (wave-5 pins).
            BinaryOperator::PGStartsWith => {
                for side in [&a, &b] {
                    if side.ty != Ty::Str {
                        return Err(PrepareError::Bind(format!(
                            "no function matches ^@({}, {})",
                            a.ty.name(),
                            b.ty.name()
                        )));
                    }
                }
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op: StrOp2::Starts,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::I1,
                    nullable,
                })
            }
            // DuckDB's ^ IS pow — but sqlparser parses ^ BELOW * while
            // DuckDB binds it above (measured: duck 2*x^y = 2*(x^y),
            // sqlparser tree = (2*x)^y). Mapping it would silently compute
            // the wrong tree, so the operator stays cleanly unsupported;
            // pow()/power() cover the semantics.
            BinaryOperator::BitwiseXor => Err(unsup(
                "operator ^ (sqlparser precedence differs from DuckDB pow)",
            )),
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
        if matches!(
            op,
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitAnd | ArithOp::BitOr | ArithOp::BitXor
        ) {
            // Bitwise is BIGINT-only (wave-5 pins: non-integer operands are
            // binder errors). Computing in i64 matches DuckDB whenever
            // either operand is BIGINT — row-model ints always are; narrow
            // CASTs are already unsupported upstream.
            for e in [&a, &b] {
                if e.ty != Ty::I64 {
                    return Err(PrepareError::Bind(format!(
                        "no function matches bitwise op on ({}, {})",
                        a.ty.name(),
                        b.ty.name()
                    )));
                }
            }
            let nullable = a.nullable || b.nullable;
            return Ok(SExpr {
                kind: SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                },
                ty: Ty::I64,
                nullable,
            });
        }
        let (a, b, ty) = numeric_promote(op, a, b)?;
        let nullable = a.nullable || b.nullable;
        // DuckDB pins (2026-07-26, waves 1+3): integer % by zero is NULL,
        // and `//`/divide() by zero is NULL on BOTH ints and doubles —
        // guard with a CASE unless the divisor is a provably non-zero
        // literal. The idiv/irem traps stay reachable only for MIN op -1,
        // where DuckDB traps too. Float % is IEEE (x % 0.0 = NaN), no guard.
        let needs_guard = match (op, ty) {
            (ArithOp::Rem, Ty::I64) => true,
            (ArithOp::IDiv, _) => true,
            _ => false,
        };
        let nonzero_lit = matches!(b.kind, SKind::Lit(Lit::I64(n)) if n != 0)
            || matches!(b.kind, SKind::Lit(Lit::F64(x)) if x != 0.0);
        if needs_guard && !nonzero_lit {
            let zero = SExpr {
                kind: SKind::Lit(if ty == Ty::F64 {
                    Lit::F64(0.0)
                } else {
                    Lit::I64(0)
                }),
                ty,
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
                    arms: vec![(cond, null_of(ty))],
                    default: Some(Box::new(rem)),
                },
                ty,
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
            // ucase/lcase are alias-identical to upper/lower (wave-3 pins:
            // exhaustive all-codepoint sweep, zero mismatches).
            "upper" | "lower" | "ucase" | "lcase" => {
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
                        upper: matches!(name.as_str(), "upper" | "ucase"),
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
            // Wave-1 string search (pins): instr/strpos/2-arg position are
            // one op with (haystack, needle) order; prefix/suffix alias
            // starts_with/ends_with; positions are 1-based codepoints.
            "instr" | "strpos" | "position" | "contains" | "starts_with" | "prefix"
            | "ends_with" | "suffix" => {
                let op = match name.as_str() {
                    "instr" | "strpos" | "position" => StrOp2::Find,
                    "contains" => StrOp2::Contains,
                    "starts_with" | "prefix" => StrOp2::Starts,
                    _ => StrOp2::Ends,
                };
                let [h, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                self.str2(&name, op, h, n)
            }
            "length" | "len" | "char_length" | "character_length" | "strlen" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 1 argument"
                    )));
                };
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::I64));
                };
                if inner.ty != Ty::Str {
                    // No implicit numeric->VARCHAR casts here (measured).
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({})",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::SLen {
                        bytes: name == "strlen",
                        a: Box::new(inner),
                    },
                    ty: Ty::I64,
                    nullable,
                })
            }
            // Wave-1 f64 unary math (pins: 2026-07-26-wave1-builtin-pins.md).
            // 1-arg log IS base 10 in DuckDB — handled under "log" below.
            "ln" | "log2" | "log10" | "exp" | "sqrt" | "cbrt" | "sin" | "cos" | "tan" | "floor"
            | "ceil" | "ceiling" => {
                let op = match name.as_str() {
                    "ln" => NumOp1::Ln,
                    "log2" => NumOp1::Log2,
                    "log10" => NumOp1::Log10,
                    "exp" => NumOp1::Fexp,
                    "sqrt" => NumOp1::Fsqrt,
                    "cbrt" => NumOp1::Fcbrt,
                    "sin" => NumOp1::Fsin,
                    "cos" => NumOp1::Fcos,
                    "tan" => NumOp1::Ftan,
                    "floor" => NumOp1::Ffloor,
                    _ => NumOp1::Fceil,
                };
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 1 argument"
                    )));
                };
                self.math1(&name, op, arg)
            }
            "log" => match args[..] {
                [x] => self.math1("log", NumOp1::Log10, x),
                [b, x] => self.math2("log", BinOp::Flogb, b, x),
                _ => Err(PrepareError::Bind("log takes 1 or 2 arguments".to_string())),
            },
            "pow" | "power" => match args[..] {
                [x, y] => self.math2(&name, BinOp::Fpow, x, y),
                _ => Err(PrepareError::Bind(format!(
                    "{name} takes exactly 2 arguments"
                ))),
            },
            "pi" => {
                if !args.is_empty() {
                    return Err(PrepareError::Bind("pi takes no arguments".to_string()));
                }
                // Bit-equal to DuckDB's pi() (measured 0x400921FB54442D18).
                Ok(SExpr {
                    kind: SKind::Lit(Lit::F64(std::f64::consts::PI)),
                    ty: Ty::F64,
                    nullable: false,
                })
            }
            "trunc" => match args[..] {
                [arg] => {
                    let Some(inner) = self.expr_or_null(arg)? else {
                        return Ok(null_of(Ty::I64));
                    };
                    match inner.ty {
                        // Measured: integer trunc is identity, type preserved.
                        Ty::I64 => Ok(inner),
                        Ty::F64 => Ok(math1_node(NumOp1::Ftrunc, inner)),
                        other => Err(PrepareError::Bind(format!(
                            "no function matches trunc({})",
                            other.name()
                        ))),
                    }
                }
                [x, n] => self.round2(true, x, n),
                _ => Err(PrepareError::Bind(
                    "trunc takes 1 or 2 arguments".to_string(),
                )),
            },
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
                [x, n] => self.round2(false, x, n),
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
            // least/greatest: NULL-IGNORING (result NULL only when every
            // arg is), ties return the FIRST argument, NaN sorts above
            // +inf — all of which the CASE + duck-order-cmp composition
            // reproduces exactly (wave-1 pins), so no IR op exists.
            "least" | "greatest" => {
                if args.is_empty() {
                    return Err(PrepareError::Bind(format!(
                        "{name} needs at least 1 argument"
                    )));
                }
                let mut bound = Vec::new();
                for arg in &args {
                    // Literal NULL args contribute nothing (NULL-ignoring).
                    if let Some(e) = self.expr_or_null(arg)? {
                        bound.push(e);
                    }
                }
                if bound.is_empty() {
                    return Err(unsup(format!("{name} of only NULL literals")));
                }
                let mut unified = bound[0].ty;
                for e in &bound[1..] {
                    unified = match (unified, e.ty) {
                        (u, t) if u == t => u,
                        (Ty::I64, Ty::F64) | (Ty::F64, Ty::I64) => Ty::F64,
                        (u, t) => {
                            return Err(PrepareError::Bind(format!(
                                "{name} arguments disagree: {} vs {}",
                                u.name(),
                                t.name()
                            )))
                        }
                    };
                }
                let bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|e| {
                        if e.ty == Ty::I64 && unified == Ty::F64 {
                            promote_f64(e)
                        } else {
                            e
                        }
                    })
                    .collect();
                let pred = if name == "greatest" {
                    CmpPred::Ge
                } else {
                    CmpPred::Le
                };
                let mut it = bound.into_iter();
                let mut acc = it.next().expect("non-empty");
                for b in it {
                    let cmp = self.cmp(pred, acc.clone(), b.clone())?;
                    let is_null = |e: &SExpr| SExpr {
                        kind: SKind::IsNull {
                            negated: false,
                            inner: Box::new(e.clone()),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    };
                    let nullable = acc.nullable && b.nullable;
                    acc = SExpr {
                        kind: SKind::Case {
                            arms: vec![
                                (is_null(&acc), b.clone()),
                                (is_null(&b), acc.clone()),
                                (cmp, acc),
                            ],
                            default: Some(Box::new(b)),
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
            // Wave-3 similarity: all raw UTF-8 BYTE-based (measured);
            // editdist3 == levenshtein and mismatches == hamming exactly.
            "levenshtein" | "editdist3" | "damerau_levenshtein" | "jaccard" | "hamming"
            | "mismatches" => {
                let op = match name.as_str() {
                    "levenshtein" | "editdist3" => StrOp2::Levenshtein,
                    "damerau_levenshtein" => StrOp2::Damerau,
                    "jaccard" => StrOp2::Jaccard,
                    _ => StrOp2::Hamming,
                };
                let [a, b] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                self.str2(&name, op, a, b)
            }
            "repeat" => {
                let [s, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                let (bs, bn) = (self.expr_or_null(s)?, self.expr_or_null(n)?);
                let (Some(bs), Some(bn)) = (bs, bn) else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str {
                    // No implicit numeric->VARCHAR cast (measured).
                    return Err(PrepareError::Bind(format!(
                        "no function matches repeat({})",
                        bs.ty.name()
                    )));
                }
                if bn.ty != Ty::I64 {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}(str, {})",
                        bn.ty.name()
                    )));
                }
                let nullable = bs.nullable || bn.nullable;
                Ok(SExpr {
                    kind: SKind::Str2i {
                        op: StrOp2i::Repeat,
                        a: Box::new(bs),
                        n: Box::new(bn),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            "array_extract" | "list_extract" => {
                let [s, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                self.apply_extract(&name, bs, n)
            }
            "array_slice" | "list_slice" => {
                if args.len() == 4 {
                    // DuckDB rejects step slicing on VARCHAR for EVERY step
                    // value, including 1 (measured).
                    return Err(unsup(
                        "slice with step (DuckDB: not implemented for string types)",
                    ));
                }
                let [s, lo, hi] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 3 arguments"
                    )));
                };
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                self.apply_slice(&name, bs, Some(lo), Some(hi))
            }
            "lpad" | "rpad" => {
                let [s, l, pad] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 3 arguments"
                    )));
                };
                let (bs, bl, bp) = (
                    self.expr_or_null(s)?,
                    self.expr_or_null(l)?,
                    self.expr_or_null(pad)?,
                );
                let (Some(bs), Some(bl), Some(bp)) = (bs, bl, bp) else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str || bp.ty != Ty::Str || bl.ty != Ty::I64 {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({}, {}, {})",
                        bs.ty.name(),
                        bl.ty.name(),
                        bp.ty.name()
                    )));
                }
                let nullable = bs.nullable || bl.nullable || bp.nullable;
                Ok(SExpr {
                    kind: SKind::Spad {
                        left: name == "lpad",
                        a: Box::new(bs),
                        len: Box::new(bl),
                        pad: Box::new(bp),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            "replace" | "translate" => {
                let op = if name == "replace" {
                    StrOp3::Replace
                } else {
                    StrOp3::Translate
                };
                let [s, x, y] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 3 arguments"
                    )));
                };
                let (bs, bx, by) = (
                    self.expr_or_null(s)?,
                    self.expr_or_null(x)?,
                    self.expr_or_null(y)?,
                );
                let (Some(bs), Some(bx), Some(by)) = (bs, bx, by) else {
                    return Ok(null_of(Ty::Str));
                };
                for e in [&bs, &bx, &by] {
                    if e.ty != Ty::Str {
                        return Err(PrepareError::Bind(format!(
                            "no function matches {name}({})",
                            e.ty.name()
                        )));
                    }
                }
                let nullable = bs.nullable || bx.nullable || by.nullable;
                Ok(SExpr {
                    kind: SKind::Str3 {
                        op,
                        a: Box::new(bs),
                        b: Box::new(bx),
                        c: Box::new(by),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            // unicode('') = ord('') = -1, but ascii('') = 0 — the measured
            // sole divergence; all return the FIRST codepoint otherwise.
            "unicode" | "ord" | "ascii" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 1 argument"
                    )));
                };
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::I64));
                };
                if inner.ty != Ty::Str {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({})",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Sord {
                        empty_zero: name == "ascii",
                        a: Box::new(inner),
                    },
                    ty: Ty::I64,
                    nullable,
                })
            }
            // bit_length = 8 * strlen exactly (measured) — pure desugar.
            "bit_length" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(
                        "bit_length takes exactly 1 argument".to_string(),
                    ));
                };
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::I64));
                };
                if inner.ty != Ty::Str {
                    return Err(PrepareError::Bind(format!(
                        "no function matches bit_length({})",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                let slen = SExpr {
                    kind: SKind::SLen {
                        bytes: true,
                        a: Box::new(inner),
                    },
                    ty: Ty::I64,
                    nullable,
                };
                let eight = SExpr {
                    kind: SKind::Lit(Lit::I64(8)),
                    ty: Ty::I64,
                    nullable: false,
                };
                self.arith(ArithOp::Mul, eight, slen)
            }
            "strip_accents" => {
                let [arg] = args[..] else {
                    return Err(PrepareError::Bind(
                        "strip_accents takes exactly 1 argument".to_string(),
                    ));
                };
                let Some(inner) = self.expr_or_null(arg)? else {
                    return Ok(null_of(Ty::Str));
                };
                if inner.ty != Ty::Str {
                    return Err(PrepareError::Bind(format!(
                        "no function matches strip_accents({})",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StripAccents(Box::new(inner)),
                    ty: Ty::Str,
                    nullable,
                })
            }
            // concat_ws: NULL args are SKIPPED with their separator; NULL
            // sep -> NULL; all-args-NULL -> '' (measured). Desugars onto
            // Case/Or/Concat — the separator appears before arg i iff some
            // earlier arg was non-NULL.
            "concat_ws" => {
                if args.len() < 2 {
                    return Err(PrepareError::Bind(
                        "concat_ws needs a separator and at least 1 argument".to_string(),
                    ));
                }
                let sep = match self.expr_or_null(args[0])? {
                    // NULL separator -> NULL result, regardless of args.
                    None => return Ok(null_of(Ty::Str)),
                    Some(e) => e,
                };
                if sep.ty != Ty::Str {
                    // The separator does NOT implicitly cast (measured —
                    // unlike the value args).
                    return Err(PrepareError::Bind(format!(
                        "no function matches concat_ws({}, ...)",
                        sep.ty.name()
                    )));
                }
                let is_null = |e: &SExpr| SExpr {
                    kind: SKind::IsNull {
                        negated: false,
                        inner: Box::new(e.clone()),
                    },
                    ty: Ty::I1,
                    nullable: false,
                };
                let sconcat = |a: SExpr, b: SExpr| SExpr {
                    kind: SKind::Concat {
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::Str,
                    nullable: false,
                };
                // The body only evaluates when sep is non-NULL (the outer
                // CASE guards it), so pieces use a provably-non-null view
                // of the separator — the concat() precedent shape.
                let sep_body = if sep.nullable {
                    SExpr {
                        kind: SKind::Case {
                            arms: vec![(is_null(&sep), lit_str(""))],
                            default: Some(Box::new(sep.clone())),
                        },
                        ty: Ty::Str,
                        nullable: false,
                    }
                } else {
                    sep.clone()
                };
                // prior_nullable: IS-NOT-NULL exprs of earlier nullable
                // args; prior_sure: an earlier arg is provably non-NULL.
                let mut prior_nullable: Vec<SExpr> = Vec::new();
                let mut prior_sure = false;
                let mut acc: Option<SExpr> = None;
                for arg in &args[1..] {
                    let Some(e) = self.expr_or_null(arg)? else {
                        continue; // literal NULL: skipped entirely
                    };
                    let e = to_varchar(e);
                    let joined = sconcat(sep_body.clone(), e.clone());
                    let with_sep = if prior_sure {
                        joined
                    } else if prior_nullable.is_empty() {
                        e.clone()
                    } else {
                        let mut it = prior_nullable.iter();
                        let mut some_prior = SExpr {
                            kind: SKind::IsNull {
                                negated: true,
                                inner: Box::new(it.next().expect("non-empty").clone()),
                            },
                            ty: Ty::I1,
                            nullable: false,
                        };
                        for p in it {
                            let not_null = SExpr {
                                kind: SKind::IsNull {
                                    negated: true,
                                    inner: Box::new(p.clone()),
                                },
                                ty: Ty::I1,
                                nullable: false,
                            };
                            some_prior = SExpr {
                                kind: SKind::Or {
                                    a: Box::new(some_prior),
                                    b: Box::new(not_null),
                                },
                                ty: Ty::I1,
                                nullable: false,
                            };
                        }
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(some_prior, joined)],
                                default: Some(Box::new(e.clone())),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        }
                    };
                    let piece = if e.nullable {
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(is_null(&e), lit_str(""))],
                                default: Some(Box::new(with_sep)),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        }
                    } else {
                        with_sep
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(p) => sconcat(p, piece),
                    });
                    if e.nullable {
                        prior_nullable.push(e);
                    } else {
                        prior_sure = true;
                    }
                }
                let body = acc.unwrap_or_else(|| lit_str(""));
                if !sep.nullable {
                    return Ok(body);
                }
                // NULL separator -> NULL result (measured), even though
                // every piece is individually total.
                Ok(SExpr {
                    kind: SKind::Case {
                        arms: vec![(is_null(&sep), null_of(Ty::Str))],
                        default: Some(Box::new(body)),
                    },
                    ty: Ty::Str,
                    nullable: true,
                })
            }
            // Wave-3 math tail: add/subtract/multiply/divide/mod are EXACT
            // aliases of + - * // % (measured: same values, types, and
            // error texts); fdiv/fmod are the FLOOR pair (always DOUBLE);
            // nextafter is C nextafter, total.
            "add" | "subtract" | "multiply" | "divide" | "mod" | "xor" => {
                let [x, y] = args[..] else {
                    return Err(unsup(format!("{name} with {} arguments", args.len())));
                };
                let op = match name.as_str() {
                    "add" => ArithOp::Add,
                    "subtract" => ArithOp::Sub,
                    "multiply" => ArithOp::Mul,
                    "divide" => ArithOp::IDiv,
                    // xor is FUNCTION-only in DuckDB; `#`/`^` are not it
                    // (wave-5 pins — `^` is pow and stays unsupported).
                    "xor" => ArithOp::BitXor,
                    _ => ArithOp::Rem,
                };
                let (bx, by) = (self.expr_or_null(x)?, self.expr_or_null(y)?);
                let (bx, by) = match (bx, by) {
                    (Some(a), Some(b)) => (a, b),
                    (Some(a), None) => {
                        let n = null_of(a.ty);
                        (a, n)
                    }
                    (None, Some(b)) => {
                        let n = null_of(b.ty);
                        (n, b)
                    }
                    (None, None) => (null_of(Ty::I64), null_of(Ty::I64)),
                };
                self.arith(op, bx, by)
            }
            "fdiv" => match args[..] {
                [x, y] => self.math2("fdiv", BinOp::Ffloordiv, x, y),
                _ => Err(PrepareError::Bind(
                    "fdiv takes exactly 2 arguments".to_string(),
                )),
            },
            "fmod" => match args[..] {
                [x, y] => self.math2("fmod", BinOp::Ffloormod, x, y),
                _ => Err(PrepareError::Bind(
                    "fmod takes exactly 2 arguments".to_string(),
                )),
            },
            "nextafter" => match args[..] {
                [x, y] => self.math2("nextafter", BinOp::Fnextafter, x, y),
                _ => Err(PrepareError::Bind(
                    "nextafter takes exactly 2 arguments".to_string(),
                )),
            },
            // Named rejects (wave-3 AC #3): each states WHY, not just what.
            "sum" | "count" | "avg" | "min" | "max" | "geomean" | "product" | "string_agg"
            | "first" | "last" | "any_value" => Err(unsup(format!(
                "aggregate function {name} (no aggregation in v0)"
            ))),
            "regexp_matches" | "regexp_extract" | "regexp_full_match" | "regexp_replace"
            | "regexp_split_to_array" => Err(unsup(format!(
                "function {name} (RE2 regex semantics, not in v0)"
            ))),
            "reverse" => Err(unsup(
                "function reverse (grapheme-cluster semantics — measured UAX-29 \
                 incl. regional-indicator pairing — not modeled in v0)",
            )),
            _ => Err(unsup(format!(
                "function {} (not in the v0 catalogue)",
                f.name
            ))),
        }
    }

    /// Wave-1 string search: both args must be Str (no implicit numeric
    /// casts — measured binder errors). A literal NULL binds to the typed
    /// NULL result for every member EXCEPT contains, where DuckDB's
    /// overloads (MAP/LIST) make a bare NULL a binder error — mirrored.
    fn str2(
        &self,
        name: &str,
        op: StrOp2,
        h: &SqlExpr,
        n: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let (bh, bn) = (self.expr_or_null(h)?, self.expr_or_null(n)?);
        // contains has MAP/LIST overloads; a NULL literal NEEDLE binds only
        // when a NON-literal Str haystack anchors resolution (measured:
        // contains(s, NULL) and contains(NULL, 'o') work, contains('abc',
        // NULL) and contains(NULL, NULL) are binder errors — the corpus
        // refuted the fleet's blanket-error pin, so this mirrors exactly).
        if name == "contains" && bn.is_none() {
            let anchored = matches!(&bh, Some(e) if !matches!(e.kind, SKind::Lit(_)));
            if !anchored {
                return Err(PrepareError::Bind(
                    "contains with a NULL literal is ambiguous (VARCHAR/MAP/LIST overloads)"
                        .to_string(),
                ));
            }
        }
        let (Some(bh), Some(bn)) = (bh, bn) else {
            return Ok(null_of(op.result_ty()));
        };
        for e in [&bh, &bn] {
            if e.ty != Ty::Str {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    e.ty.name()
                )));
            }
        }
        let nullable = bh.nullable || bn.nullable;
        Ok(SExpr {
            kind: SKind::Str2 {
                op,
                a: Box::new(bh),
                b: Box::new(bn),
            },
            ty: op.result_ty(),
            nullable,
        })
    }

    /// round(x, n) / trunc(x, n): result type == subject type; digits must
    /// be integer-typed. Total on both types (i64 wraps — pinned).
    fn round2(&self, trunc: bool, x: &SqlExpr, n: &SqlExpr) -> Result<SExpr, PrepareError> {
        let name = if trunc { "trunc" } else { "round" };
        let Some(subject) = self.expr_or_null(x)? else {
            return Ok(null_of(Ty::I64));
        };
        if !matches!(subject.ty, Ty::I64 | Ty::F64) {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}({}, digits)",
                subject.ty.name()
            )));
        }
        let ty = subject.ty;
        let Some(digits) = self.expr_or_null(n)? else {
            return Ok(null_of(ty));
        };
        if digits.ty != Ty::I64 {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}({}, {})",
                ty.name(),
                digits.ty.name()
            )));
        }
        let nullable = subject.nullable || digits.nullable;
        Ok(SExpr {
            kind: SKind::Round2 {
                trunc,
                a: Box::new(subject),
                n: Box::new(digits),
            },
            ty,
            nullable,
        })
    }

    /// Wave-1 unary f64 math: numeric args promote to DOUBLE, VARCHAR and
    /// BOOLEAN columns are binder errors (no implicit cast — measured), a
    /// literal NULL binds to the DOUBLE overload.
    fn math1(&self, name: &str, op: NumOp1, arg: &SqlExpr) -> Result<SExpr, PrepareError> {
        let Some(inner) = self.expr_or_null(arg)? else {
            return Ok(null_of(Ty::F64));
        };
        let inner = match inner.ty {
            Ty::F64 => inner,
            Ty::I64 => promote_f64(inner),
            other => {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    other.name()
                )))
            }
        };
        Ok(math1_node(op, inner))
    }

    /// Wave-1 binary f64 math (pow / log(base, x)); same argument typing
    /// rules as `math1`. A literal NULL in either slot pre-empts every
    /// domain check (measured: log(-2.0, NULL) is NULL, not an error).
    fn math2(
        &self,
        name: &str,
        op: BinOp,
        a: &SqlExpr,
        b: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let (ba, bb) = (self.expr_or_null(a)?, self.expr_or_null(b)?);
        let (Some(ba), Some(bb)) = (ba, bb) else {
            return Ok(null_of(Ty::F64));
        };
        let promote = |e: SExpr| -> Result<SExpr, PrepareError> {
            match e.ty {
                Ty::F64 => Ok(e),
                Ty::I64 => Ok(promote_f64(e)),
                other => Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    other.name()
                ))),
            }
        };
        let (ba, bb) = (promote(ba)?, promote(bb)?);
        let nullable = ba.nullable || bb.nullable;
        Ok(SExpr {
            kind: SKind::MathF2 {
                op,
                a: Box::new(ba),
                b: Box::new(bb),
            },
            ty: Ty::F64,
            nullable,
        })
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
