//! The correlated scalar subquery with aggregates over one static table,
//! `(SELECT sum(s.v) FROM s WHERE s.k = t.k AND s.v > 5)`. It binds as a
//! hidden LEFT join to a static table that confit computes at build
//! ([`AggTable`]): the `s` rows on which every WHERE term that reads only
//! `s` is TRUE (its filter, [`plan::AggFilter`]), in groups with equal keys,
//! one row per group holding the group's aggregate. A request reads the row
//! of the group it matches, and a request that matches no group reads the
//! aggregate of no rows: 0 for `count`, NULL for the others. An aggregate
//! can read each distinct value of its column once, `count(DISTINCT s.v)`.
//! If the SELECT item is an expression over aggregates, `max(s.v) + 1`, the
//! row holds the expression over the group's aggregates instead
//! ([`plan::AggPost`]), and a request that matches no group reads the
//! expression over the aggregates of no rows, which the bind evaluates.
//! `EXISTS (SELECT * FROM s WHERE ...)` binds as the count of the same rows
//! above 0 ([`Binder::exists_subquery`]).
//!
//! DuckDB (1.5.5, optimizer off, measured 2026-10-14 and 2026-10-15)
//! evaluates the subquery's WHERE over (distinct correlated values) x
//! (static rows), in written order, the aggregate's argument over the
//! matched rows only, and the item once for each distinct correlated value,
//! over the aggregates of its matched rows (of no rows if none matches).
//! It plans the subquery below the operator that holds it: in the SELECT
//! list it runs over the rows that the WHERE passes, in the WHERE over the
//! rows that reach the WHERE, under a CASE too. Only one thing served here
//! can trap: the outer side of the first key ([`Binder::guarded_key`]),
//! which runs where DuckDB runs it. The inner side of each key is a bare
//! column, no key casts a VARCHAR, the outer side of a later key cannot trap
//! (`can_trap`), no filter term can trap, and the aggregates and the item
//! run before any request, the item over each group and over no rows, where
//! a trap or a sum that could overflow refuses. So the filter passes the
//! same rows in any order, and the hidden join runs after the FROM joins of
//! its level, before the WHERE when the WHERE reads it and after it
//! otherwise ([`plan::JoinSpec::after_where`]).

use super::*;
use super::super::plan::{self, Agg, AggFunc, AggTable};
use sqlparser::ast::{
    DuplicateTreatment, FunctionArg, FunctionArgExpr, FunctionArguments, WildcardAdditionalOptions,
};

/// The subquery's one static table, as its WHERE and its aggregate see it.
struct Inner<'s> {
    st: &'s StaticTable,
    /// The name it is in scope under: its alias, else its bare name.
    name: String,
    /// The schema it was named through; empty when aliased.
    schema: String,
}

impl Inner<'_> {
    /// Whether a bare name binds in this table: the inner scope binds first
    /// (measured: `WHERE s.k = v` reads `s.v` when both tables have `v`).
    fn has(&self, name: &str) -> bool {
        let st = self.st;
        st.cols
            .iter()
            .enumerate()
            .any(|(i, c)| !st.is_leaf_lane(i as u32) && c.name.eq_ignore_ascii_case(name))
            || st.structs.iter().any(|s| s.name.eq_ignore_ascii_case(name))
            || st.opaque.iter().any(|(c, _)| c.eq_ignore_ascii_case(name))
    }

    /// The plain lane of column `name`, or the refusal of a column this
    /// engine cannot read here.
    fn lane(&self, name: &str) -> Result<u32, PrepareError> {
        let st = self.st;
        let mut hit = None;
        for (i, c) in st.cols.iter().enumerate() {
            if !st.is_leaf_lane(i as u32) && c.name.eq_ignore_ascii_case(name) {
                if hit.is_some() {
                    return Err(PrepareError::Bind(format!(
                        "ambiguous column '{name}' in static table '{}'",
                        st.name
                    )));
                }
                hit = Some(i as u32);
            }
        }
        if let Some(i) = hit {
            return Ok(i);
        }
        if let Some(err) = opaque_static_refusal(st, name, &self.name) {
            return Err(err);
        }
        Err(PrepareError::Bind(format!(
            "column '{name}' does not exist in '{}'",
            self.name
        )))
    }

    /// The lane `e` names when it is a column of this table, `None` when it
    /// reads the outer scope.
    fn col(&self, e: &SqlExpr) -> Result<Option<u32>, PrepareError> {
        match e {
            SqlExpr::Nested(x) => self.col(x),
            SqlExpr::Identifier(id) if self.has(&id.value) => self.lane(&id.value).map(Some),
            SqlExpr::CompoundIdentifier(parts) => {
                let n = parts.len();
                let is = |i: usize, s: &str| !s.is_empty() && parts[i].value.eq_ignore_ascii_case(s);
                let plain = match n {
                    2 => is(0, &self.name),
                    3 => is(0, &self.schema) && is(1, &self.name),
                    4 => is(0, "memory") && is(1, &self.schema) && is(2, &self.name),
                    _ => false,
                };
                if plain {
                    return self.lane(&parts[n - 1].value).map(Some);
                }
                if self.reads(e)? {
                    return Err(unsup(
                        "a struct field, or a column named through another schema, of a \
                         subquery's table",
                    ));
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Whether `e` reads this table anywhere. A form this walk does not
    /// know refuses, so an outer key never reads the inner table unseen.
    fn reads(&self, e: &SqlExpr) -> Result<bool, PrepareError> {
        Ok(match e {
            SqlExpr::Identifier(id) => self.has(&id.value),
            // `s.k`, `main.s.k`, and a struct field `c.f` of an inner
            // column `c`: DuckDB resolves a name in the inner scope first.
            SqlExpr::CompoundIdentifier(parts) => {
                parts[..parts.len() - 1]
                    .iter()
                    .any(|p| p.value.eq_ignore_ascii_case(&self.name))
                    || self.has(&parts[0].value)
            }
            SqlExpr::Value(_) => false,
            SqlExpr::Nested(x)
            | SqlExpr::UnaryOp { expr: x, .. }
            | SqlExpr::Cast { expr: x, .. }
            | SqlExpr::IsNull(x)
            | SqlExpr::IsNotNull(x) => self.reads(x)?,
            SqlExpr::BinaryOp { left, right, .. } => self.reads(left)? || self.reads(right)?,
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } => {
                let mut parts: Vec<&SqlExpr> = operand.iter().map(|x| x.as_ref()).collect();
                for w in conditions {
                    parts.push(&w.condition);
                    parts.push(&w.result);
                }
                parts.extend(else_result.iter().map(|x| x.as_ref()));
                let mut any = false;
                for p in parts {
                    any |= self.reads(p)?;
                }
                any
            }
            SqlExpr::Function(f)
                if f.filter.is_none()
                    && f.over.is_none()
                    && f.within_group.is_empty()
                    && matches!(f.parameters, FunctionArguments::None) =>
            {
                let args = match &f.args {
                    FunctionArguments::None => Vec::new(),
                    FunctionArguments::List(l) if l.clauses.is_empty() => l.args.iter().collect(),
                    _ => return Err(outer_form_refusal(e)),
                };
                let mut any = false;
                for a in args {
                    match a {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(x)) => any |= self.reads(x)?,
                        _ => return Err(outer_form_refusal(e)),
                    }
                }
                any
            }
            _ => return Err(outer_form_refusal(e)),
        })
    }
}

fn outer_form_refusal(e: &SqlExpr) -> PrepareError {
    unsup(format!(
        "expression {e} as the outer side of a subquery's key -- the keys \
         served are column references, operators, CASE and plain function calls"
    ))
}

/// Whether `e` holds a subquery anywhere, through the lets and the SQL
/// function bodies it reads.
pub(super) fn has_subquery(b: &Binder<'_>, e: &SqlExpr) -> Result<bool, PrepareError> {
    use super::resolve::Walk;
    let at = |x: &SqlExpr| match x {
        SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => Walk::Stop,
        _ => Walk::Into,
    };
    let none = b.walk_expr(e, &mut |x| {
        let Some((id, _)) = calls::marker_call(x) else {
            return Ok(at(x));
        };
        // Every field read of a call holds the same marker, arguments
        // included: walking it at each read made a struct of K fields,
        // read field by field, walk K² nodes (#456).
        let known = b.call_subquery.borrow().get(&id).copied();
        let has = match known {
            Some(has) => has,
            None => {
                let has = !b.walk_expr(x, &mut |y| Ok(at(y)))?;
                b.call_subquery.borrow_mut().insert(id, has);
                has
            }
        };
        Ok(if has { Walk::Stop } else { Walk::Over })
    })?;
    Ok(!none)
}

/// The refusal of a subquery form outside what this module serves.
fn form_refusal(what: &str) -> PrepareError {
    unsup(format!(
        "a subquery with {what} -- served: (SELECT count(*) | f(c) | f(DISTINCT c) FROM s \
         WHERE ...) with f one of count, min, max, sum and avg, or an expression over such \
         aggregates, and [NOT] EXISTS (SELECT * | constants | columns of s FROM s WHERE ...), \
         over one static table, each WHERE term s.k = <outer expression> or a term that reads \
         only s, and none that can trap but the outer expression of the first key"
    ))
}

/// Whether `e` is a literal, which DuckDB reads without evaluating
/// anything: a number, maybe signed, a string, a boolean or NULL.
fn is_constant(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Nested(x) => is_constant(x),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus | UnaryOperator::Plus,
            expr,
        } => matches!(expr.as_ref(), SqlExpr::Value(v) if matches!(v.value, SqlValue::Number(..))),
        SqlExpr::Value(v) => matches!(
            v.value,
            SqlValue::Number(..) | SqlValue::SingleQuotedString(_) | SqlValue::Boolean(_) | SqlValue::Null
        ),
        _ => false,
    }
}

/// Whether a `*` item has no EXCLUDE, REPLACE or other option.
pub(super) fn plain_star(o: &WildcardAdditionalOptions) -> bool {
    let WildcardAdditionalOptions {
        wildcard_token: _,
        opt_ilike,
        opt_exclude,
        opt_except,
        opt_replace,
        opt_rename,
        opt_alias,
    } = o;
    opt_ilike.is_none()
        && opt_exclude.is_none()
        && opt_except.is_none()
        && opt_replace.is_none()
        && opt_rename.is_none()
        && opt_alias.is_none()
}

/// The refusal of a WHERE term that is neither a key nor a filter.
fn term_refusal(t: &SqlExpr) -> PrepareError {
    form_refusal(&format!(
        "the WHERE term {t} (each term must be an equality between a column of its table \
         and an outer expression, or read only its table)"
    ))
}

impl Binder<'_> {
    /// Bind `(SELECT agg(..) FROM s WHERE s.k = outer AND ...)`: its value,
    /// read through a hidden LEFT join this level runs after its FROM
    /// joins. The same text bound again at this level reads the same join.
    pub(super) fn scalar_subquery(&self, query: &sqlparser::ast::Query) -> Result<SExpr, PrepareError> {
        if !self.subqueries.get() {
            return Err(unsup("a subquery in FROM or in a JOIN condition"));
        }
        self.bound_subquery.set(true);
        if let Some((_, e)) = self.agg_reads.borrow().iter().find(|(q, _)| q == query) {
            return Ok(e.clone());
        }
        if query.with.is_some() {
            return Err(form_refusal("WITH"));
        }
        if query.order_by.is_some() || query.limit_clause.is_some() {
            return Err(form_refusal("ORDER BY or LIMIT"));
        }
        refuse_unhandled_query(query)?;
        let select = match query.body.as_ref() {
            SetExpr::Select(s) => s.as_ref(),
            _ => return Err(form_refusal("a body that is not one SELECT")),
        };
        if select.distinct.is_some() {
            return Err(form_refusal("DISTINCT"));
        }
        refuse_unhandled_select(select)?;
        let grouped = match &select.group_by {
            sqlparser::ast::GroupByExpr::Expressions(exprs, modifiers) => {
                !exprs.is_empty() || !modifiers.is_empty()
            }
            sqlparser::ast::GroupByExpr::All(_) => true,
        };
        if grouped || select.having.is_some() {
            return Err(form_refusal("GROUP BY or HAVING"));
        }

        // FROM: one static table.
        let [twj] = select.from.as_slice() else {
            return Err(form_refusal(if select.from.is_empty() {
                "no FROM"
            } else {
                "more than one table in FROM"
            }));
        };
        if !twj.joins.is_empty() {
            return Err(form_refusal("a join"));
        }
        let Some((name, alias)) = plain_table(&twj.relation)? else {
            return Err(form_refusal("a FROM that is not a table"));
        };
        if name.bare().eq_ignore_ascii_case(&self.request_table) {
            return Err(unsup(
                "a subquery over the request table -- a subquery that reads the other \
                 requests of a batch is outside the serving model",
            ));
        }
        let source = resolve_static(self.statics, &name)?;
        let renamed = apply_column_alias(&self.statics[source], alias)?;
        let st = renamed.as_ref().unwrap_or(&self.statics[source]);
        let inner = Inner {
            st,
            name: alias.map_or_else(|| name.bare().to_string(), |a| a.name.value.clone()),
            schema: if alias.is_some() { String::new() } else { name.schema() },
        };

        // WHERE: each term reads only the table (a filter the build applies
        // to its rows), or is an equality between a column of the table and
        // an expression of the outer scope (a key). Only the outer side of
        // the first key can trap ([`Binder::guarded_key`]).
        let mut terms = Vec::new();
        if let Some(w) = &select.selection {
            collect_conjuncts(w, &mut terms);
        }
        let mut keys = Vec::new();
        let mut key_cols = Vec::new();
        let mut key_lanes = Vec::new();
        let mut filters = Vec::new();
        // When the outer side of the first key can trap: how many filter
        // terms are written before that key.
        let mut trap_key = None;
        for t in terms {
            let mut t = t;
            while let SqlExpr::Nested(x) = t {
                t = x;
            }
            if self.reads_only(&inner, t)? {
                filters.push(t);
                continue;
            }
            let (left, right, cmp) = match t {
                SqlExpr::BinaryOp {
                    left,
                    op: BinaryOperator::Eq,
                    right,
                } => (left, right, KeyCmp::Eq),
                SqlExpr::IsNotDistinctFrom(left, right) => (left, right, KeyCmp::NotDistinct),
                _ => return Err(term_refusal(t)),
            };
            let (col, outer) = match (inner.col(left)?, inner.col(right)?) {
                (Some(c), None) => (c, right.as_ref()),
                (None, Some(c)) => (c, left.as_ref()),
                _ => return Err(term_refusal(t)),
            };
            if inner.reads(outer)? {
                return Err(form_refusal(&format!(
                    "the WHERE term {t}, whose outer side reads the subquery's own table"
                )));
            }
            // A let read answers its value here, not `SKind::Let`: the key
            // runs at the join, before the projection computes its lets.
            let (let_reads, projection_lets) = (self.let_reads.get(), self.projection_lets.get());
            self.let_reads.set(false);
            self.projection_lets.set(false);
            let key = self.expr(outer);
            self.let_reads.set(let_reads);
            self.projection_lets.set(projection_lets);
            let key = fold(key?);
            if refs_join(&mut key.clone()) {
                return Err(form_refusal(&format!(
                    "the WHERE term {t}, whose outer side reads a joined table"
                )));
            }
            let col_ty = st.cols[col as usize].ty.ty;
            if (key.ty == Ty::Str) != (col_ty == Ty::Str) {
                return Err(form_refusal(&format!(
                    "the WHERE term {t}, which compares VARCHAR with another type (the cast \
                     can fail)"
                )));
            }
            let key = promote_key(key, st, col)?;
            if can_trap(&key) {
                if !keys.is_empty() {
                    return Err(form_refusal(&format!(
                        "the WHERE term {t}, whose outer side can trap and follows another key \
                         (DuckDB evaluates it only over the static rows that the keys before \
                         it match)"
                    )));
                }
                trap_key = Some(filters.len());
            }
            key_cols.push(JoinKey {
                src: KeySrc::Lane(key_lanes.len() as u32),
                cmp,
            });
            key_lanes.push(col);
            keys.push(key);
        }

        // SELECT: one aggregate over a column of the table, or count(*), or
        // an expression over such aggregates ([`plan::AggPost`]).
        let [item] = select.projection.as_slice() else {
            return Err(form_refusal("more than one SELECT item"));
        };
        let (SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. }) = item else {
            return Err(form_refusal("a star in its SELECT"));
        };
        let mut e = e;
        while let SqlExpr::Nested(x) = e {
            e = x;
        }
        let single = match e {
            SqlExpr::Function(f) => agg_func(f).map(|func| (f, func)),
            _ => None,
        };
        let (aggs, ty, post) = match single {
            Some((f, func)) => {
                let (agg, ty) = self.agg_call(&inner, f, func)?;
                (vec![agg], ty, None)
            }
            None => {
                let (aggs, post) = self.agg_post(query, select, source, &inner, e)?;
                let ty = post.project[0].1.ty;
                (aggs, ty, Some(post))
            }
        };

        // The grouped table: the key lanes as the source has them, then the
        // aggregate, or the expression over the aggregates.
        let mut cols: Vec<Col> = key_lanes.iter().map(|&c| st.cols[c as usize].clone()).collect();
        let counts = post.is_none() && aggs.iter().all(|a| counts_rows(a.func));
        cols.push(Col {
            name: e.to_string(),
            ty: super::super::ir::ColTy {
                ty,
                nullable: !counts,
            },
        });
        let n_keys = key_lanes.len() as u32;
        // What a request that matches no group reads: the expression over
        // the aggregates of no rows, which the interpreter evaluates here.
        // DuckDB evaluates it for each such request (measured 2026-10-15),
        // so one that traps refuses.
        let miss = match &post {
            Some(p) if n_keys > 0 => {
                let miss = over_no_rows(p.project[0].1.clone(), &aggs);
                Some(match eval_closed(&miss, p.regexes.clone()) {
                    Some(Some(v)) => scalar_lit(v, miss.ty),
                    Some(None) => null_of(miss.ty),
                    None => {
                        return Err(form_refusal(&format!(
                            "the SELECT item {e}, which traps over the aggregates of no rows \
                             (DuckDB evaluates it for each request that matches no group)"
                        )))
                    }
                })
            }
            _ => None,
        };
        let filter = match filters.as_slice() {
            [] => None,
            _ => Some(self.static_filter(query, select, source, &filters)?),
        };
        if let Some(n) = trap_key {
            let key = keys.remove(0);
            keys.insert(0, self.guarded_key(query, select, source, &filters[..n], key)?);
        }
        let mut tables = self.agg_tables.borrow_mut();
        let table = self.statics.len() + tables.len();
        tables.push(AggTable {
            table: StaticTable::all_scalar(self.statics[source].name.clone(), cols),
            source,
            keys: key_lanes,
            aggs,
            filter,
            post,
        });
        drop(tables);
        let mut joins = self.agg_joins.borrow_mut();
        let join = (self.joins.len() + joins.len()) as u32;
        joins.push(JoinSpec {
            table,
            batch: false,
            kind: JoinKind::Left,
            keys,
            key_cols,
            val_cols: vec![n_keys],
            residual: None,
            after_where: false,
        });
        drop(joins);
        if trap_key.is_some() {
            self.trap_keys.borrow_mut().push(join);
        }
        let read = SExpr {
            kind: SKind::StaticCol { join, col: 0 },
            ty,
            nullable: true,
        };
        // A request that matches no group counts 0 rows: COALESCE(read, 0),
        // the form `coalesce` binds to. Under an expression, the group's
        // value or the value over no rows: the expression can be NULL on a
        // match.
        let value = if let Some(miss) = miss {
            let hit = SExpr {
                kind: SKind::JoinHit(join),
                ty: Ty::I1,
                nullable: false,
            };
            SExpr {
                kind: SKind::Case {
                    arms: vec![(hit, read)],
                    default: match miss.kind {
                        SKind::NullOf => None,
                        _ => Some(Box::new(miss)),
                    },
                },
                ty,
                nullable: true,
            }
        } else if counts {
            SExpr {
                kind: SKind::Case {
                    arms: vec![(
                        SExpr {
                            kind: SKind::IsNull {
                                negated: true,
                                inner: Box::new(read.clone()),
                            },
                            ty: Ty::I1,
                            nullable: false,
                        },
                        read,
                    )],
                    default: Some(Box::new(SExpr {
                        kind: SKind::Lit(Lit::I64(0)),
                        ty: Ty::I64,
                        nullable: false,
                    })),
                },
                ty: Ty::I64,
                nullable: false,
            }
        } else {
            read
        };
        self.agg_reads.borrow_mut().push((query.clone(), value.clone()));
        Ok(value)
    }

    /// Bind `EXISTS (SELECT ... FROM s WHERE ...)` as the test
    /// `(SELECT count(*) FROM s WHERE ...) > 0`, which reads the same hidden
    /// join: DuckDB answers the same value for both, and traps at the same
    /// requests (measured 2026-10-15). DuckDB evaluates the SELECT items
    /// over the matched rows, so each must be one that cannot trap there:
    /// `*`, a constant or a column of the table. The parser writes
    /// `NOT EXISTS` as NOT over EXISTS (`precedence.rs`).
    pub(super) fn exists_subquery(
        &self,
        query: &sqlparser::ast::Query,
        negated: bool,
    ) -> Result<SExpr, PrepareError> {
        let SetExpr::Select(select) = query.body.as_ref() else {
            return Err(form_refusal("a body that is not one SELECT"));
        };
        if select.projection.is_empty() {
            return Err(form_refusal("no SELECT item"));
        }
        // The items read the count's table; a FROM that the count refuses
        // is left to it, so the two refuse it alike.
        let table = match select.from.as_slice() {
            [twj] if twj.joins.is_empty() => plain_table(&twj.relation)?
                .filter(|(name, _)| !name.bare().eq_ignore_ascii_case(&self.request_table))
                .and_then(|(name, alias)| {
                    Some((resolve_static(self.statics, &name).ok()?, name, alias))
                }),
            _ => None,
        };
        if let Some((source, name, alias)) = table {
            let renamed = apply_column_alias(&self.statics[source], alias)?;
            let inner = Inner {
                st: renamed.as_ref().unwrap_or(&self.statics[source]),
                name: alias.map_or_else(|| name.bare().to_string(), |a| a.name.value.clone()),
                schema: if alias.is_some() { String::new() } else { name.schema() },
            };
            for item in &select.projection {
                let served = match item {
                    SelectItem::Wildcard(o) => plain_star(o),
                    SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                        is_constant(e) || inner.col(e)?.is_some()
                    }
                    _ => false,
                };
                if !served {
                    return Err(form_refusal(&format!(
                        "the SELECT item {item} under EXISTS (each item must be *, a constant \
                         or a column of its table: DuckDB evaluates the items over the matched \
                         rows, where they can trap)"
                    )));
                }
            }
        }
        let count = Parser::new(&super::super::precedence::DuckDbPrecedence)
            .try_with_sql("count(*)")
            .and_then(|mut p| p.parse_expr())
            .map_err(|e| PrepareError::Internal(format!("count(*) does not parse: {e}")))?;
        let mut counted = query.clone();
        if let SetExpr::Select(s) = counted.body.as_mut() {
            s.projection = vec![SelectItem::UnnamedExpr(count)];
        }
        let count = self.scalar_subquery(&counted)?;
        Ok(SExpr {
            kind: SKind::Cmp {
                pred: if negated { CmpPred::Eq } else { CmpPred::Gt },
                a: Box::new(count),
                b: Box::new(lit_i64(0)),
            },
            ty: Ty::I1,
            nullable: false,
        })
    }

    /// One served aggregate call, `count(*)`, `count`, `min`, `max`, `sum`
    /// or `avg` over a column of the subquery's table (`func` names it), and
    /// the type of its answer. A call over a column may read each distinct
    /// value once, `count(DISTINCT v)`, and the answer keeps its type
    /// (measured 2026-10-15).
    fn agg_call(
        &self,
        inner: &Inner<'_>,
        f: &sqlparser::ast::Function,
        func: AggFunc,
    ) -> Result<(Agg, Ty), PrepareError> {
        let st = inner.st;
        if f.uses_odbc_syntax
            || !matches!(f.parameters, FunctionArguments::None)
            || f.filter.is_some()
            || f.null_treatment.is_some()
            || f.over.is_some()
            || !f.within_group.is_empty()
        {
            return Err(form_refusal(&format!("the aggregate call {f}")));
        }
        let FunctionArguments::List(list) = &f.args else {
            return Err(form_refusal(&format!("the aggregate call {f}")));
        };
        if !list.clauses.is_empty() {
            return Err(form_refusal(&format!("the aggregate call {f} (ORDER BY or another clause)")));
        }
        let distinct = list.duplicate_treatment == Some(DuplicateTreatment::Distinct);
        let [arg] = list.args.as_slice() else {
            return Err(form_refusal(&format!("the aggregate call {f}")));
        };
        let (func, arg) = match (func, arg) {
            (AggFunc::Count, FunctionArg::Unnamed(FunctionArgExpr::Wildcard)) if !distinct => {
                (AggFunc::CountStar, None)
            }
            // A constant that is not NULL counts every row: count(1) and
            // count('x') answer count(*) (measured).
            (AggFunc::Count, FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(v))))
                if !distinct
                    && matches!(
                        &v.value,
                        SqlValue::Number(..) | SqlValue::SingleQuotedString(_) | SqlValue::Boolean(_)
                    ) =>
            {
                (AggFunc::CountStar, None)
            }
            (_, FunctionArg::Unnamed(FunctionArgExpr::Expr(x))) => match inner.col(x)? {
                Some(c) => (func, Some(c)),
                None => {
                    return Err(form_refusal(&format!(
                        "the aggregate call {f} (its argument must be a column of its table)"
                    )))
                }
            },
            _ => return Err(form_refusal(&format!("the aggregate call {f}"))),
        };
        let arg_ty = arg.map(|c| st.cols[c as usize].ty.ty);
        let ty = match (func, arg_ty) {
            (AggFunc::CountStar | AggFunc::Count, _) => Ty::I64,
            (AggFunc::Min | AggFunc::Max, Some(t))
                if t.is_integer() || matches!(t, Ty::I1 | Ty::F64 | Ty::Str | Ty::Dec(..)) =>
            {
                t
            }
            (AggFunc::Sum, Some(t)) if t.is_integer() || t == Ty::I1 => Ty::I128,
            (AggFunc::Sum, Some(Ty::Dec(_, s))) => Ty::Dec(38, s),
            (AggFunc::Sum, Some(Ty::F64)) => {
                return Err(unsup(
                    "sum over a DOUBLE column in a scalar subquery -- the order of the \
                     additions changes the answer, and its comparison bound waits on a \
                     decision",
                ))
            }
            (AggFunc::Sum, Some(Ty::Str)) => {
                return Err(PrepareError::Bind(
                    "No function matches the given name and argument types 'sum(VARCHAR)'"
                        .into(),
                ))
            }
            (AggFunc::Avg, Some(t)) if t.is_integer() || matches!(t, Ty::Dec(..)) => {
                if plan::avg_in_long_double(t) && !plan::LONG_DOUBLE_IS_X87 {
                    return Err(unsup(format!(
                        "avg over {} in a scalar subquery on this platform -- DuckDB \
                         finalizes it in C long double, and confit computes only the x87 \
                         format of x86_64 Linux",
                        duck_ty_name(t)
                    )));
                }
                Ty::F64
            }
            (AggFunc::Avg, Some(Ty::F64)) => {
                return Err(unsup(
                    "avg over a DOUBLE column in a scalar subquery -- the order of the \
                     additions changes the answer, and its comparison bound waits on a \
                     decision",
                ))
            }
            (AggFunc::Avg, Some(t @ (Ty::I1 | Ty::Str))) => {
                return Err(PrepareError::Bind(format!(
                    "No function matches the given name and argument types 'avg({})'",
                    duck_ty_name(t)
                )))
            }
            (_, t) => {
                return Err(form_refusal(&format!(
                    "the aggregate call {f} over {}",
                    t.map_or("no column".into(), duck_ty_name)
                )))
            }
        };
        // The least or greatest of the distinct values is the least or
        // greatest of all of them.
        let distinct = distinct && !matches!(func, AggFunc::Min | AggFunc::Max);
        Ok((Agg { func, arg, distinct }, ty))
    }

    /// The SELECT item of a subquery that is an expression over its
    /// aggregates, such as `max(v) + 1` or `coalesce(sum(v), 0)`: each
    /// served aggregate call becomes an input of a query of its own, which
    /// the build runs over the aggregates of each group ([`plan::AggPost`]).
    /// DuckDB evaluates the item once per distinct outer value, over the
    /// aggregates of the rows that it matches (measured 2026-10-15), so the
    /// values are the same. A column of the table, or of the outer query,
    /// outside an aggregate call refuses, as does a call of a SQL function
    /// and a subquery.
    fn agg_post(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        inner: &Inner<'_>,
        e: &SqlExpr,
    ) -> Result<(Vec<Agg>, plan::AggPost), PrepareError> {
        use super::resolve::Walk;
        let mut aggs: Vec<Agg> = Vec::new();
        let mut in_cols: Vec<Col> = Vec::new();
        let mut inputs: Vec<(SqlExpr, u32)> = Vec::new();
        let mut outside = None;
        self.walk_expr(e, &mut |x| {
            if calls::marker_call(x).is_some() || lets::marker_let(x).is_some() {
                return Err(form_refusal(&format!(
                    "the SELECT item {e}, which calls a SQL function"
                )));
            }
            Ok(match x {
                SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => {
                    return Err(form_refusal(&format!("the SELECT item {e}, which holds a subquery")))
                }
                SqlExpr::Function(f) => match agg_func(f) {
                    Some(func) => {
                        let (agg, ty) = self.agg_call(inner, f, func)?;
                        let i = match aggs.iter().position(|a| *a == agg) {
                            Some(i) => i,
                            None => {
                                aggs.push(agg);
                                in_cols.push(Col {
                                    name: f.to_string(),
                                    ty: super::super::ir::ColTy {
                                        ty,
                                        nullable: !counts_rows(func),
                                    },
                                });
                                aggs.len() - 1
                            }
                        };
                        inputs.push((x.clone(), i as u32));
                        Walk::Over
                    }
                    None => Walk::Into,
                },
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    outside.get_or_insert_with(|| x.to_string());
                    Walk::Over
                }
                _ => Walk::Into,
            })
        })?;
        if aggs.is_empty() {
            return Err(match e {
                SqlExpr::Function(f) => form_refusal(&format!("the aggregate {}", f.name)),
                _ => form_refusal(&format!("the SELECT item {e} (one aggregate call)")),
            });
        }
        if let Some(x) = outside {
            return Err(form_refusal(&format!(
                "the SELECT item {e}, which reads {x} outside an aggregate call"
            )));
        }
        // The subquery's own FROM, without its alias: every name the item
        // reads is inside an aggregate call, which binds as its input.
        let mut sel = select.clone();
        sel.selection = None;
        sel.projection = vec![SelectItem::UnnamedExpr(e.clone())];
        if let Some(TableFactor::Table { alias, .. }) = sel.from.first_mut().map(|t| &mut t.relation) {
            *alias = None;
        }
        let mut q = query.clone();
        q.body = Box::new(SetExpr::Select(Box::new(sel)));
        let env = Env {
            this_name: &self.statics[source].name,
            in_cols: &in_cols,
            opaque: &[],
            structs: &[],
            statics: &[],
            many: false,
            udfs: self.udfs,
            models: self.models,
            bind_eval: self.bind_eval,
            agg_inputs: &inputs,
        };
        let bound = bind_query(&q, None, &env, &[])?;
        let [stage] = <[Stage; 1]>::try_from(bound.stages).map_err(|_| {
            PrepareError::Internal("a scalar subquery's SELECT item bound to more than one stage".into())
        })?;
        if stage.project.len() != 1
            || !bound.joins.is_empty()
            || stage.pred.is_some()
            || !stage.lets.is_empty()
            || !bound.wide_outs.is_empty()
        {
            return Err(form_refusal(&format!(
                "the SELECT item {e}, which does not bind one scalar value"
            )));
        }
        // The build runs the item with no UDF and no model.
        if calls_extern(&mut stage.project[0].1.clone()) {
            return Err(form_refusal(&format!(
                "the SELECT item {e}, which calls a UDF or a model"
            )));
        }
        Ok((
            aggs,
            plan::AggPost {
                in_cols,
                project: stage.project,
                out_cols: bound.out_cols,
                regexes: bound.ctx.regexes,
            },
        ))
    }

    /// Whether `t` reads only `inner`'s table: each column it names is one
    /// of the table's. A constant reads only the table too. A term that
    /// holds a subquery, or that reads only the table and calls a SQL
    /// function, refuses by name.
    fn reads_only(&self, inner: &Inner<'_>, t: &SqlExpr) -> Result<bool, PrepareError> {
        use super::resolve::Walk;
        let mut calls_fn = false;
        let only = self.walk_expr(t, &mut |x| {
            if calls::marker_call(x).is_some() || lets::marker_let(x).is_some() {
                calls_fn = true;
                return Ok(Walk::Into);
            }
            Ok(match x {
                SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => {
                    return Err(form_refusal(&format!("the WHERE term {t}, which holds a subquery")))
                }
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => match inner.col(x)? {
                    Some(_) => Walk::Over,
                    None => Walk::Stop,
                },
                _ => Walk::Into,
            })
        })?;
        if only && calls_fn {
            return Err(form_refusal(&format!(
                "the WHERE term {t}, which reads only its table and calls a SQL function"
            )));
        }
        Ok(only)
    }

    /// The WHERE terms that read only the static table at `source`, bound
    /// as the items of a query of their own over that table, each cast to
    /// BOOLEAN as DuckDB casts a WHERE term. The build keeps the rows on
    /// which every term is TRUE. It evaluates each term over every row,
    /// where DuckDB evaluates it over the rows that the terms before it
    /// pass: so a term that can trap refuses, and the others pass the same
    /// rows on both engines.
    fn static_filter(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        terms: &[&SqlExpr],
    ) -> Result<plan::AggFilter, PrepareError> {
        let st = &self.statics[source];
        // The scalar columns in declared order, each nullable: a NULL in a
        // column declared NOT NULL still reads as NULL. A struct, or a
        // column of a type this engine does not serve, is opaque at its
        // position, as a column-alias list counts it.
        let mut in_cols = Vec::new();
        let mut opaque = Vec::new();
        for (pos, c) in st.star.iter().enumerate() {
            match c {
                plan::StarCol::Real(i) => {
                    let mut col = st.cols[*i as usize].clone();
                    col.ty.nullable = true;
                    in_cols.push(col);
                }
                plan::StarCol::Opaque(name) => opaque.push((pos, name.clone())),
            }
        }
        // The subquery's own FROM, so its names bind as they did there.
        let mut sel = select.clone();
        sel.selection = None;
        sel.projection = terms
            .iter()
            .map(|t| {
                SelectItem::UnnamedExpr(SqlExpr::Cast {
                    kind: CastKind::Cast,
                    expr: Box::new((*t).clone()),
                    data_type: sqlparser::ast::DataType::Boolean,
                    array: false,
                    format: None,
                })
            })
            .collect();
        let mut q = query.clone();
        q.body = Box::new(SetExpr::Select(Box::new(sel)));
        let env = Env {
            this_name: &st.name,
            in_cols: &in_cols,
            opaque: &opaque,
            structs: &[],
            statics: &[],
            many: false,
            udfs: self.udfs,
            models: self.models,
            bind_eval: self.bind_eval,
            agg_inputs: &[],
        };
        let bound = bind_query(&q, None, &env, &[])?;
        let [stage] = <[Stage; 1]>::try_from(bound.stages).map_err(|_| {
            PrepareError::Internal("a scalar subquery's filter bound to more than one stage".into())
        })?;
        if stage.project.len() != terms.len() {
            return Err(form_refusal(&format!(
                "the WHERE terms {}, which do not bind one value each",
                terms.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" AND ")
            )));
        }
        if !bound.joins.is_empty()
            || stage.pred.is_some()
            || !stage.lets.is_empty()
            || !bound.wide_outs.is_empty()
        {
            return Err(PrepareError::Internal(
                "a scalar subquery's filter bound more than its terms".into(),
            ));
        }
        for ((_, e), t) in stage.project.iter().zip(terms) {
            if e.ty != Ty::I1 {
                return Err(PrepareError::Internal(format!(
                    "the WHERE term {t} bound to {}, not BOOLEAN",
                    e.ty.name()
                )));
            }
            if can_trap(e) {
                return Err(form_refusal(&format!(
                    "the WHERE term {t}, which reads only its table and can trap"
                )));
            }
        }
        Ok(plan::AggFilter {
            in_cols,
            project: stage.project,
            out_cols: bound.out_cols,
            regexes: bound.ctx.regexes,
        })
    }

    /// The first key of a subquery whose outer side can trap, guarded.
    /// DuckDB evaluates that side over the static rows that the WHERE terms
    /// before it (`pre`, all filters) pass, so it traps only if one such row
    /// exists (measured 2026-10-15). The build counts those rows into a
    /// keyless table of its own, read through one more hidden join, and the
    /// key becomes `CASE WHEN count > 0 THEN key END`. When the count is 0
    /// the grouped table is empty, so the NULL key misses as the key would.
    /// DuckDB's planner moves the upper half of `x BETWEEN lo AND hi` to the
    /// end of the WHERE, after every key (see `where_conjuncts`), so only
    /// the lower half of such a term counts here.
    fn guarded_key(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        pre: &[&SqlExpr],
        key: SExpr,
    ) -> Result<SExpr, PrepareError> {
        let filter = match pre {
            [] => None,
            _ => {
                let mut f = self.static_filter(query, select, source, pre)?;
                for (i, t) in pre.iter().enumerate() {
                    if !is_between(t) {
                        continue;
                    }
                    let lower = match &f.project[i].1.kind {
                        SKind::And { a, .. } => (**a).clone(),
                        _ => {
                            return Err(form_refusal(&format!(
                                "the WHERE term {t} before a key whose outer side can trap \
                                 (its lower bound does not bind as a term of its own)"
                            )))
                        }
                    };
                    f.out_cols[i].ty.nullable = lower.nullable;
                    f.project[i].1 = lower;
                }
                Some(f)
            }
        };
        let count = Col {
            name: "count_star()".into(),
            ty: super::super::ir::ColTy {
                ty: Ty::I64,
                nullable: false,
            },
        };
        let mut tables = self.agg_tables.borrow_mut();
        let table = self.statics.len() + tables.len();
        tables.push(AggTable {
            table: StaticTable::all_scalar(self.statics[source].name.clone(), vec![count]),
            source,
            keys: Vec::new(),
            aggs: vec![Agg {
                func: AggFunc::CountStar,
                arg: None,
                distinct: false,
            }],
            filter,
            post: None,
        });
        drop(tables);
        let mut joins = self.agg_joins.borrow_mut();
        let join = (self.joins.len() + joins.len()) as u32;
        joins.push(JoinSpec {
            table,
            batch: false,
            kind: JoinKind::Left,
            keys: Vec::new(),
            key_cols: Vec::new(),
            val_cols: vec![0],
            residual: None,
            after_where: false,
        });
        drop(joins);
        let read = SExpr {
            kind: SKind::StaticCol { join, col: 0 },
            ty: Ty::I64,
            nullable: true,
        };
        let any = SExpr {
            kind: SKind::Cmp {
                pred: CmpPred::Gt,
                a: Box::new(read),
                b: Box::new(lit_i64(0)),
            },
            ty: Ty::I1,
            nullable: true,
        };
        let ty = key.ty;
        Ok(SExpr {
            kind: SKind::Case {
                arms: vec![(any, key)],
                default: None,
            },
            ty,
            nullable: true,
        })
    }
}

/// The aggregate that a served name calls, or `None`.
fn agg_func(f: &sqlparser::ast::Function) -> Option<AggFunc> {
    let [p] = f.name.0.as_slice() else {
        return None;
    };
    Some(match p.as_ident()?.value.to_ascii_lowercase().as_str() {
        "count" => AggFunc::Count,
        "min" => AggFunc::Min,
        "max" => AggFunc::Max,
        "sum" => AggFunc::Sum,
        "avg" => AggFunc::Avg,
        _ => return None,
    })
}

/// Whether an aggregate counts rows: 0 over no rows, never NULL.
fn counts_rows(func: AggFunc) -> bool {
    matches!(func, AggFunc::CountStar | AggFunc::Count)
}

/// The SELECT item `e` of an [`plan::AggPost`] over the aggregates of no
/// rows: each count is 0, and each other aggregate is NULL.
fn over_no_rows(mut e: SExpr, aggs: &[Agg]) -> SExpr {
    fn walk(e: &mut SExpr, aggs: &[Agg]) {
        if let SKind::Col(i) = e.kind {
            *e = if counts_rows(aggs[i as usize].func) {
                lit_i64(0)
            } else {
                SExpr {
                    kind: SKind::NullOf,
                    ty: e.ty,
                    nullable: true,
                }
            };
            return;
        }
        for c in e.children_mut() {
            walk(c, aggs);
        }
    }
    walk(&mut e, aggs);
    e
}

/// Whether `e` calls a UDF or scores a model.
fn calls_extern(e: &mut SExpr) -> bool {
    matches!(e.kind, SKind::ExternCall { .. } | SKind::TreePredict { .. })
        || e.children_mut().into_iter().any(calls_extern)
}

/// Whether `e` reads join `j`: a value column or its match flag.
pub(super) fn reads_join(e: &mut SExpr, j: u32) -> bool {
    matches!(e.kind, SKind::StaticCol { join, .. } | SKind::JoinHit(join) if join == j)
        || e.children_mut().into_iter().any(|c| reads_join(c, j))
}

/// Whether `e` reads a joined table (a value column or a match flag), or a
/// value the projection computes (`SKind::Let`, `SKind::Shared`), which a
/// join key evaluated before the projection cannot read.
fn refs_join(e: &mut SExpr) -> bool {
    matches!(
        e.kind,
        SKind::StaticCol { .. } | SKind::JoinHit(_) | SKind::Let(_) | SKind::Shared(_)
    ) || e.children_mut().into_iter().any(refs_join)
}
