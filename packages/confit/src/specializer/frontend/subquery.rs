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
//! above 0 ([`Binder::exists_subquery`]), and `x IN (SELECT s.c FROM s WHERE
//! ...)` from three such counts, one with the IN test as one more key
//! ([`Binder::in_subquery`]). `x < ANY (SELECT s.c FROM s WHERE ...)`, and
//! ANY or ALL with another comparison, binds as IN with a test of `x`
//! against the extremes of `s.c` in place of the first count, the residual
//! of one more hidden join ([`Binder::quantified_subquery`]). A SELECT item
//! with no aggregate reads the row that a request matches,
//! `(SELECT s.v FROM s WHERE s.k = t.k)`: the build reads the one row of
//! each group and refuses a group of more than one row, where DuckDB traps
//! ([`plan::AggPost::row`]). A WHERE term that compares `s` with the outer
//! query other than by a key, `s.lo <= t.x`, binds as a scan for each
//! request ([`Binder::scan_value`]): one hidden join reads the rows of `s`
//! that pass the filter and share the request's keys, at most
//! [`plan::SCAN_LIMIT`] of them, which the build checks
//! ([`plan::ScanRows`]), and a second one folds the aggregate over the rows
//! on which every other term is TRUE ([`plan::Fold`]), where the query reads
//! its value.
//!
//! DuckDB (1.5.5, optimizer off, measured 2026-10-14 and 2026-10-15) evaluates
//! the subquery's WHERE over (distinct correlated values) x (static rows), in
//! written order, the aggregate's argument over the matched rows only, and the
//! item once for each distinct correlated value, over the aggregates of its
//! matched rows (of no rows if none matches). It plans the subquery below the
//! operator that holds it: in the SELECT list it runs over the rows that the
//! WHERE passes, in the WHERE over the rows that reach the WHERE, under a CASE
//! too. Only two things served here can trap, and both run where DuckDB runs
//! them: the outer side of the first key ([`Binder::guarded_key`]), and the
//! left side of ANY or ALL ([`Binder::quantified_residual`]). No term of a scan
//! can trap. The inner side of each key is a bare column, no key casts a
//! VARCHAR, the outer side of a later key cannot trap (`can_trap`), no filter
//! term can trap, and the aggregates and the item run before any request, the
//! item over each group and over no rows, where a trap or a sum that could
//! overflow refuses. So the filter passes the same rows in any order, and the
//! hidden join runs after the FROM joins of its level, before the WHERE when
//! the WHERE reads it and after it otherwise ([`plan::JoinSpec::after_where`]).

use super::*;
use super::super::plan::{self, Agg, AggFunc, AggTable};
use sqlparser::ast::{
    DuplicateTreatment, FunctionArg, FunctionArgExpr, FunctionArguments, LimitClause, OrderByKind,
    WildcardAdditionalOptions,
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
        SqlExpr::Subquery(_)
        | SqlExpr::Exists { .. }
        | SqlExpr::InSubquery { .. }
        | SqlExpr::AnyOp { .. }
        | SqlExpr::AllOp { .. } => Walk::Stop,
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
         aggregates, (SELECT an expression over the columns of s FROM s WHERE ... [ORDER BY \
         columns of s] [LIMIT n] [OFFSET m]) where a request reads at most one row, [NOT] \
         EXISTS (SELECT * | constants | columns of s FROM s WHERE ...), \
         x [NOT] IN (SELECT a column of s FROM s WHERE ...), and x op ANY | SOME | ALL (SELECT \
         a column of s FROM s WHERE ...) with op a comparison, over one static table, each \
         WHERE term s.k = <outer expression> or a term that reads only s, and none that can \
         trap but the outer expression of the first key, or x over a WHERE without a key; \
         other WHERE terms that compare s with the outer query serve by a scan of at most \
         100 rows of s for each request, in (SELECT count(*) | count(c) | sum(c) | min(c) | \
         max(c) FROM s WHERE ...) where no term can trap"
    ))
}

/// The refusal of a SELECT item under IN, ANY or ALL (`form`) that is not
/// a column of the table.
fn item_refusal(item: &dyn std::fmt::Display, form: &str) -> PrepareError {
    form_refusal(&format!(
        "the SELECT item {item} under {form} (the item must be a column of its table)"
    ))
}

/// The refusal of a left side `x` of IN, ANY or ALL (`form`) that can trap,
/// over a WHERE with a key: whether DuckDB evaluates it for a request
/// depends on the other requests of the batch (measured 2026-10-15).
fn left_side_trap_refusal(x: &SqlExpr, form: &str) -> PrepareError {
    form_refusal(&format!(
        "the left side {x} of {form}, which can trap, over a WHERE with a key -- whether \
         DuckDB evaluates it for a request depends on the other requests of the batch"
    ))
}

/// The extremes of the item that the comparison `x pred c` reads over a
/// group ([`Binder::quantified_residual`]): the greatest under < and <=,
/// the least under > and >=, and both under <>. None under =, which binds
/// as IN.
fn extremes(pred: CmpPred) -> &'static [AggFunc] {
    match pred {
        CmpPred::Lt | CmpPred::Le => &[AggFunc::Max],
        CmpPred::Gt | CmpPred::Ge => &[AggFunc::Min],
        CmpPred::Ne => &[AggFunc::Min, AggFunc::Max],
        CmpPred::Eq => &[],
    }
}

/// What [`Binder::subquery_value`] binds besides the subquery as written.
#[derive(Clone, Copy)]
enum Member<'e> {
    /// Nothing: the value of its SELECT item.
    Value,
    /// The IN test of [`Binder::in_subquery`]: one more key after every
    /// term of the WHERE, the column `c` of the table equal to the outer
    /// expression `x`. `form` names the test in a refusal.
    In {
        c: &'e SqlExpr,
        x: &'e SqlExpr,
        form: &'e str,
    },
    /// The comparison `x pred c` of [`Binder::quantified_subquery`], with
    /// `c` a column of the table: the value is whether a row of the group
    /// that a request matches passes it.
    Cmp {
        c: &'e SqlExpr,
        x: &'e SqlExpr,
        pred: CmpPred,
        form: &'e str,
    },
}

impl Member<'_> {
    /// What tells two bindings of one subquery apart in the reads that
    /// [`Binder::subquery_value`] keeps; the name of the form does not.
    fn key(self) -> Option<(SqlExpr, SqlExpr, Option<CmpPred>)> {
        match self {
            Member::Value => None,
            Member::In { c, x, .. } => Some((c.clone(), x.clone(), None)),
            Member::Cmp { c, x, pred, .. } => Some((c.clone(), x.clone(), Some(pred))),
        }
    }
}

/// `count(arg)`, parsed as the binder reads it.
fn count_of(arg: &str) -> Result<SqlExpr, PrepareError> {
    Parser::new(&super::super::precedence::DuckDbPrecedence)
        .try_with_sql(&format!("count({arg})"))
        .and_then(|mut p| p.parse_expr())
        .map_err(|e| PrepareError::Internal(format!("count({arg}) does not parse: {e}")))
}

/// `query` with `item` as its one SELECT item.
fn with_item(query: &sqlparser::ast::Query, item: SqlExpr) -> sqlparser::ast::Query {
    let mut q = query.clone();
    if let SetExpr::Select(s) = q.body.as_mut() {
        s.projection = vec![SelectItem::UnnamedExpr(item)];
    }
    q
}

/// The comparison `a pred b` of two counts, never NULL.
fn cmp_i64(pred: CmpPred, a: SExpr, b: SExpr) -> SExpr {
    SExpr {
        kind: SKind::Cmp {
            pred,
            a: Box::new(a),
            b: Box::new(b),
        },
        ty: Ty::I1,
        nullable: false,
    }
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

/// The refusal of a subquery whose WHERE term `t` compares its table with
/// the outer query other than by a key, where the scan that serves such a
/// term ([`Binder::scan_value`]) does not apply, for the reason `why`.
fn scan_refusal(t: &SqlExpr, why: &str) -> PrepareError {
    unsup(format!(
        "a subquery with the WHERE term {t}, which compares its table with the outer query \
         other than by a key, where {why} -- confit serves such a term by a scan of at most \
         {} rows of the table for each request, in (SELECT count(*) | count(c) | sum(c) | \
         min(c) | max(c) FROM s WHERE ...) with sum over an integer or BOOLEAN column or a \
         DECIMAL of at most 36 digits, and min and max over an integer, DECIMAL or VARCHAR \
         column, where no term can trap",
        plan::SCAN_LIMIT
    ))
}

/// The ORDER BY, LIMIT and OFFSET of a scalar subquery as written, before
/// the SELECT item binds ([`plan::RowOrder`]): each ORDER BY term with
/// whether it sorts descending and whether NULLs come first, and the values
/// of LIMIT and OFFSET.
struct OrderDraft<'q> {
    terms: Vec<(&'q SqlExpr, bool, bool)>,
    limit: Option<u64>,
    offset: u64,
}

/// The [`OrderDraft`] of `query`, `None` without ORDER BY, LIMIT and
/// OFFSET. A form that DuckDB reads otherwise, or that it stops with an
/// error, refuses by name: ORDER BY ALL, ClickHouse's WITH FILL,
/// INTERPOLATE and LIMIT BY, MySQL's `LIMIT m, n`, and a LIMIT or OFFSET
/// that is not a whole number or NULL.
fn order_draft(query: &sqlparser::ast::Query) -> Result<Option<OrderDraft<'_>>, PrepareError> {
    if query.order_by.is_none() && query.limit_clause.is_none() {
        return Ok(None);
    }
    let mut terms = Vec::new();
    if let Some(ob) = &query.order_by {
        if ob.interpolate.is_some() {
            return Err(form_refusal("INTERPOLATE"));
        }
        let exprs = match &ob.kind {
            OrderByKind::All(_) => return Err(form_refusal("ORDER BY ALL")),
            OrderByKind::Expressions(e) => e,
        };
        for o in exprs {
            if o.with_fill.is_some() {
                return Err(form_refusal("WITH FILL"));
            }
            terms.push((&o.expr, o.options.asc == Some(false), o.options.nulls_first == Some(true)));
        }
    }
    let (limit, offset) = match &query.limit_clause {
        None => (None, 0),
        Some(LimitClause::LimitOffset {
            limit,
            offset,
            limit_by,
        }) => {
            if !limit_by.is_empty() {
                return Err(form_refusal("LIMIT BY"));
            }
            let limit = match limit {
                Some(l) => Some(row_count(l, "LIMIT")?.unwrap_or(i64::MAX as u64)),
                None => None,
            };
            let offset = match offset {
                Some(o) => row_count(&o.value, "OFFSET")?.unwrap_or(0),
                None => 0,
            };
            (limit, offset)
        }
        Some(LimitClause::OffsetCommaLimit { .. }) => {
            return Err(form_refusal("LIMIT <offset>, <count>"))
        }
    };
    Ok(Some(OrderDraft {
        terms,
        limit,
        offset,
    }))
}

/// The value of a LIMIT or OFFSET (`what`): a whole number that a BIGINT
/// holds, or `None` for NULL, which DuckDB reads as the greatest BIGINT
/// under LIMIT and as 0 under OFFSET (measured 2026-10-15,
/// [`plan::RowOrder::limit`]). Any other value refuses by name.
fn row_count(e: &SqlExpr, what: &str) -> Result<Option<u64>, PrepareError> {
    match e {
        SqlExpr::Nested(x) => row_count(x, what),
        SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => Ok(None),
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(n, _) => match n.parse::<i64>() {
                Ok(n) if n >= 0 => Ok(Some(n as u64)),
                _ => Err(form_refusal(&format!("{what} {e}"))),
            },
            _ => Err(form_refusal(&format!("{what} {e}"))),
        },
        _ => Err(form_refusal(&format!("{what} {e}, which is not a number or NULL"))),
    }
}

impl Binder<'_> {
    /// Bind `(SELECT agg(..) FROM s WHERE s.k = outer AND ...)`: its value,
    /// read through a hidden LEFT join this level runs after its FROM
    /// joins. The same text bound again at this level reads the same join.
    pub(super) fn scalar_subquery(&self, query: &sqlparser::ast::Query) -> Result<SExpr, PrepareError> {
        self.subquery_value(query, Member::Value)
    }

    /// [`Binder::scalar_subquery`], with what `member` adds to the
    /// subquery as written ([`Member`]).
    fn subquery_value(
        &self,
        query: &sqlparser::ast::Query,
        member: Member<'_>,
    ) -> Result<SExpr, PrepareError> {
        if !self.subqueries.get() {
            return Err(unsup("a subquery in FROM or in a JOIN condition"));
        }
        self.bound_subquery.set(true);
        let member_key = member.key();
        if let Some((_, _, e)) =
            self.agg_reads.borrow().iter().find(|(q, m, _)| q == query && *m == member_key)
        {
            return Ok(e.clone());
        }
        if query.with.is_some() {
            return Err(form_refusal("WITH"));
        }
        // ORDER BY, LIMIT and OFFSET serve over an item that reads one row.
        let order = match member {
            Member::Value => order_draft(query)?,
            _ if query.order_by.is_some() || query.limit_clause.is_some() => {
                return Err(form_refusal("ORDER BY or LIMIT"))
            }
            _ => None,
        };
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
        // the first key can trap ([`Binder::guarded_key`]). Any other term
        // compares the table with the outer scope: the subquery is then a
        // scan for each request ([`Binder::scan_value`]).
        let mut terms = Vec::new();
        if let Some(w) = &select.selection {
            collect_conjuncts(w, &mut terms);
        }
        let mut keys = Vec::new();
        let mut key_cols = Vec::new();
        let mut key_lanes = Vec::new();
        let mut key_terms = Vec::new();
        let mut filters = Vec::new();
        let mut mixed = Vec::new();
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
                _ => {
                    mixed.push(t);
                    continue;
                }
            };
            let (col, outer) = match (inner.col(left)?, inner.col(right)?) {
                (Some(c), None) => (c, right.as_ref()),
                (None, Some(c)) => (c, left.as_ref()),
                _ => {
                    mixed.push(t);
                    continue;
                }
            };
            if inner.reads(outer)? {
                mixed.push(t);
                continue;
            }
            let key = self.key_side(outer)?;
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
            key_terms.push(t);
            keys.push(key);
        }
        if !mixed.is_empty() {
            return self.scan_value(
                query,
                select,
                source,
                &inner,
                member,
                order.is_some(),
                ScanWhere {
                    mixed,
                    filters,
                    keys,
                    key_cols,
                    key_lanes,
                    key_terms,
                    trap_key: trap_key.is_some(),
                },
            );
        }
        // The IN test: DuckDB evaluates `x` once the subquery has an item
        // that is not NULL (its hash table holds no NULL), so without
        // another key the guard of the first key holds it, over every term
        // of the WHERE (the IN test is not one of them), both bounds of
        // BETWEEN too.
        let mut guard = Guard::Key;
        if let Member::In { c, x, form } = member {
            let (col, key) = self.member_sides(&inner, c, x, form)?;
            let key = promote_key(key, st, col)?;
            if can_trap(&key) {
                if !keys.is_empty() {
                    return Err(left_side_trap_refusal(x, form));
                }
                trap_key = Some(filters.len());
                guard = Guard::InItem(col);
            }
            key_cols.push(JoinKey {
                src: KeySrc::Lane(key_lanes.len() as u32),
                cmp: KeyCmp::Eq,
            });
            key_lanes.push(col);
            keys.push(key);
        }
        // The comparison of ANY or ALL: `x` against the item, a column of
        // the table, which a residual of the join tests over the extremes
        // of the item in the group ([`Binder::quantified_residual`]). It is
        // bound here once to refuse before any table is added; the join's
        // index comes later.
        let compared = match member {
            Member::Cmp { c, x, pred, form } => {
                let (col, key) = self.member_sides(&inner, c, x, form)?;
                let item = st.cols[col as usize].clone();
                let (_, traps) = self.quantified_residual(u32::MAX, &key, item.ty.ty, pred, c, form)?;
                if traps && !keys.is_empty() {
                    return Err(left_side_trap_refusal(x, form));
                }
                Some((col, item, key, pred, c, form, traps))
            }
            _ => None,
        };

        // The grouped table: the key lanes as the source has them, then the
        // value lanes.
        let mut cols: Vec<Col> = key_lanes.iter().map(|&c| st.cols[c as usize].clone()).collect();
        let n_keys = key_lanes.len() as u32;
        let (aggs, ty, post, counts, miss) = match &compared {
            // The count of the group's rows, then the extremes of its item.
            Some((col, item, _, pred, ..)) => {
                cols.push(Col {
                    name: "count_star()".into(),
                    ty: super::super::ir::ColTy {
                        ty: Ty::I64,
                        nullable: false,
                    },
                });
                let mut aggs = vec![Agg {
                    func: AggFunc::CountStar,
                    arg: None,
                    distinct: false,
                    ties: false,
                }];
                for &func in extremes(*pred) {
                    let name = if func == AggFunc::Min { "min" } else { "max" };
                    cols.push(Col {
                        name: format!("{name}({})", item.name),
                        ty: super::super::ir::ColTy {
                            ty: item.ty.ty,
                            nullable: true,
                        },
                    });
                    aggs.push(Agg {
                        func,
                        arg: Some(*col),
                        distinct: false,
                        ties: true,
                    });
                }
                (aggs, Ty::I1, None, false, None)
            }
            // SELECT: one aggregate over a column of the table, or count(*),
            // or an expression over such aggregates ([`plan::AggPost`]).
            None => {
                let [item] = select.projection.as_slice() else {
                    return Err(form_refusal("more than one SELECT item"));
                };
                let (SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. }) = item
                else {
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
                let (aggs, ty, mut post) = match single {
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
                if let Some(d) = &order {
                    let Some(p) = post.as_mut().filter(|p| p.row) else {
                        return Err(form_refusal(
                            "ORDER BY or LIMIT, and an aggregate call in its SELECT item",
                        ));
                    };
                    let alias = match item {
                        SelectItem::ExprWithAlias { alias, .. } => Some(alias),
                        _ => None,
                    };
                    p.order = Some(self.row_order(&inner, alias, d)?);
                }
                let counts = post.is_none() && aggs.iter().all(|a| counts_rows(a.func));
                cols.push(Col {
                    name: e.to_string(),
                    ty: super::super::ir::ColTy {
                        ty,
                        nullable: !counts,
                    },
                });
                // What a request that matches no group reads: the expression
                // over the aggregates of no rows, which the interpreter
                // evaluates here. DuckDB evaluates it for each such request
                // (measured 2026-10-15), so one that traps refuses.
                let miss = match &post {
                    Some(p) if n_keys > 0 && !p.row => {
                        let miss = over_no_rows(p.project[0].1.clone(), &aggs);
                        Some(match eval_closed(&miss, p.regexes.clone()) {
                            Some(Some(v)) => scalar_lit(v, miss.ty),
                            Some(None) => null_of(miss.ty),
                            None => {
                                return Err(form_refusal(&format!(
                                    "the SELECT item {e}, which traps over the aggregates of no \
                                     rows (DuckDB evaluates it for each request that matches no \
                                     group)"
                                )))
                            }
                        })
                    }
                    _ => None,
                };
                (aggs, ty, post, counts, miss)
            }
        };
        let val_cols: Vec<u32> = (n_keys..cols.len() as u32).collect();
        let filter = match filters.as_slice() {
            [] => None,
            _ => Some(self.static_filter(query, select, source, &filters)?),
        };
        if let Some(n) = trap_key {
            let key = keys.remove(0);
            keys.insert(0, self.guarded_key(query, select, source, &filters[..n], key, guard)?);
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
        let join = (self.joins.len() + self.agg_joins.borrow().len()) as u32;
        let residual = match &compared {
            Some((_, item, key, pred, c, form, _)) => {
                Some(self.quantified_residual(join, key, item.ty.ty, *pred, c, form)?.0)
            }
            None => None,
        };
        self.agg_joins.borrow_mut().push(JoinSpec {
            table,
            batch: false,
            kind: JoinKind::Left,
            keys,
            key_cols,
            val_cols,
            residual,
            after_where: false,
            fold: None,
            scan: None,
        });
        // A probe that evaluates an outer expression that can trap: the
        // outer side of the first key, or the left side of ANY or ALL in
        // the residual.
        if trap_key.is_some() || compared.as_ref().is_some_and(|c| c.6) {
            self.trap_keys.borrow_mut().push(join);
        }
        if compared.is_some() {
            let hit = SExpr {
                kind: SKind::JoinHit(join),
                ty: Ty::I1,
                nullable: false,
            };
            self.agg_reads.borrow_mut().push((query.clone(), member_key, hit.clone()));
            return Ok(hit);
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
            count_or_zero(read)
        } else {
            read
        };
        self.agg_reads.borrow_mut().push((query.clone(), member_key, value.clone()));
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
        self.over_table(select, |inner| {
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
            Ok(())
        })?;
        if query.order_by.is_some() || query.limit_clause.is_some() {
            return Err(form_refusal("ORDER BY or LIMIT"));
        }
        let count = self.scalar_subquery(&with_item(query, count_of("*")?))?;
        Ok(cmp_i64(if negated { CmpPred::Eq } else { CmpPred::Gt }, count, lit_i64(0)))
    }

    /// Bind `x [NOT] IN (SELECT s.c FROM s WHERE ...)` from three counts of
    /// the same FROM and WHERE ([`Binder::subquery_value`]): `hit`, the
    /// rows whose `s.c` equals `x` (the IN test, one more key); `rows`, all
    /// of them; and `values`, those whose `s.c` is not NULL. As DuckDB
    /// answers (measured 2026-10-15), IN is TRUE if `hit > 0`, else FALSE if
    /// `rows = 0`, else NULL if `values < rows` or `x` is NULL, else FALSE
    /// ([`Binder::member_value`]). DuckDB evaluates `x` for each request
    /// once the subquery has an item that is not NULL. Over a WHERE without
    /// a key, the build knows whether it has one, and the CASE tests `x`
    /// only after `values < rows`, so only when `values > 0`; over a WHERE
    /// with a key, that depends on the other requests of the batch, so
    /// there an `x` that can trap refuses. `form` names the test in a
    /// refusal: IN, or `= ANY` and `<> ALL`, which DuckDB answers as IN and
    /// NOT IN.
    pub(super) fn in_subquery(
        &self,
        x: &SqlExpr,
        query: &sqlparser::ast::Query,
        negated: bool,
        form: &str,
    ) -> Result<SExpr, PrepareError> {
        let c = self.member_item(x, query, form)?;
        let rows_q = with_item(query, count_of("*")?);
        let hit = self.subquery_value(&rows_q, Member::In { c, x, form })?;
        self.member_value(x, query, c, cmp_i64(CmpPred::Gt, hit, lit_i64(0)), !negated)
    }

    /// Bind `x op ANY (SELECT s.c FROM s WHERE ...)`, SOME alike, and
    /// `x op ALL (...)`. DuckDB answers ALL as NOT over ANY with the
    /// opposite comparison, `x > ALL (q)` as `NOT (x <= ANY (q))`, and
    /// `= ANY` as IN ([`Binder::in_subquery`]), which it runs as a hash
    /// join. It runs ANY with another comparison as a nested loop join:
    /// `x op ANY (q)` is TRUE if `x op c` is TRUE for the item `c` of a row
    /// that passes the WHERE, else FALSE if no row passes it, else NULL if
    /// `x` or such an item is NULL, else FALSE, and it evaluates `x` once a
    /// row passes the whole WHERE, NULL items too (measured 2026-10-15).
    /// The first test is the residual of one more hidden join
    /// ([`Binder::quantified_residual`]), and the others are the counts of
    /// IN ([`Binder::member_value`]).
    pub(super) fn quantified_subquery(
        &self,
        x: &SqlExpr,
        op: &BinaryOperator,
        right: &SqlExpr,
        all: bool,
    ) -> Result<SExpr, PrepareError> {
        let mut right = right;
        while let SqlExpr::Nested(r) = right {
            right = r;
        }
        let SqlExpr::Subquery(query) = right else {
            return Err(unsup(format!(
                "expression ANY/ALL over {right} -- served: x op ANY (SELECT ...) and \
                 x op ALL (SELECT ...) over a static table"
            )));
        };
        let pred = match op {
            BinaryOperator::Eq => CmpPred::Eq,
            BinaryOperator::NotEq => CmpPred::Ne,
            BinaryOperator::Lt => CmpPred::Lt,
            BinaryOperator::LtEq => CmpPred::Le,
            BinaryOperator::Gt => CmpPred::Gt,
            BinaryOperator::GtEq => CmpPred::Ge,
            _ => return Err(unsup(format!("expression ANY/ALL with the operator {op}"))),
        };
        let form = format!("{op} {}", if all { "ALL" } else { "ANY" });
        // ALL is NOT over ANY with the opposite comparison.
        let pred = if all {
            match pred {
                CmpPred::Eq => CmpPred::Ne,
                CmpPred::Ne => CmpPred::Eq,
                CmpPred::Lt => CmpPred::Ge,
                CmpPred::Le => CmpPred::Gt,
                CmpPred::Gt => CmpPred::Le,
                CmpPred::Ge => CmpPred::Lt,
            }
        } else {
            pred
        };
        if pred == CmpPred::Eq {
            return self.in_subquery(x, query, all, &form);
        }
        let c = self.member_item(x, query, &form)?;
        let hit = self.subquery_value(
            query,
            Member::Cmp {
                c,
                x,
                pred,
                form: &form,
            },
        )?;
        self.member_value(x, query, c, hit, !all)
    }

    /// The item `c` of `x form (SELECT c FROM s ...)`, checked as IN, ANY
    /// and ALL need it: one item, a column of the table, and an `x` that
    /// holds no subquery.
    fn member_item<'q>(
        &self,
        x: &SqlExpr,
        query: &'q sqlparser::ast::Query,
        form: &str,
    ) -> Result<&'q SqlExpr, PrepareError> {
        let SetExpr::Select(select) = query.body.as_ref() else {
            return Err(form_refusal("a body that is not one SELECT"));
        };
        let c = match select.projection.as_slice() {
            [SelectItem::UnnamedExpr(c) | SelectItem::ExprWithAlias { expr: c, .. }] => c,
            [item] => return Err(item_refusal(item, form)),
            _ => return Err(form_refusal(&format!("more than one SELECT item under {form}"))),
        };
        // A FROM that the counts refuse is left to them.
        if self.over_table(select, |inner| inner.col(c))? == Some(None) {
            return Err(item_refusal(c, form));
        }
        if has_subquery(self, x)? {
            return Err(form_refusal(&format!(
                "the left side {x} of {form}, which holds a subquery"
            )));
        }
        Ok(c)
    }

    /// The value of IN, ANY or ALL from `hit`, whether a row of the
    /// subquery passes its test: `on_hit` if one does. Else it is the
    /// opposite if no row passes the WHERE, else NULL if the item `c` of
    /// such a row is NULL or `x` is NULL, else the opposite. Two counts of
    /// the same FROM and WHERE tell these apart: `rows`, all the rows, and
    /// `values`, those whose item is not NULL. The CASE tests whether `x`
    /// is NULL last, so it evaluates `x` only if `values > 0`
    /// ([`Binder::in_subquery`]).
    fn member_value(
        &self,
        x: &SqlExpr,
        query: &sqlparser::ast::Query,
        c: &SqlExpr,
        hit: SExpr,
        on_hit: bool,
    ) -> Result<SExpr, PrepareError> {
        let rows = self.scalar_subquery(&with_item(query, count_of("*")?))?;
        let values = self.scalar_subquery(&with_item(query, count_of(&c.to_string())?))?;
        let x = self.expr(x)?;
        let lit = |v: bool| SExpr {
            kind: SKind::Lit(Lit::I1(v)),
            ty: Ty::I1,
            nullable: false,
        };
        let x_null = SExpr {
            kind: SKind::IsNull {
                negated: false,
                inner: Box::new(x),
            },
            ty: Ty::I1,
            nullable: false,
        };
        Ok(SExpr {
            kind: SKind::Case {
                arms: vec![
                    (hit, lit(on_hit)),
                    (cmp_i64(CmpPred::Eq, rows.clone(), lit_i64(0)), lit(!on_hit)),
                    (cmp_i64(CmpPred::Lt, values, rows), null_of(Ty::I1)),
                    (x_null, null_of(Ty::I1)),
                ],
                default: Some(Box::new(lit(!on_hit))),
            },
            ty: Ty::I1,
            nullable: true,
        })
    }

    /// The item `c` of IN, ANY or ALL (`form`) as a lane of the table, and
    /// the left side `x` bound as an outer expression of a probe. An item
    /// that is not a column refuses, as do an `x` that reads a joined table
    /// and a VARCHAR against another type, which DuckDB stops with a binder
    /// error.
    fn member_sides(
        &self,
        inner: &Inner<'_>,
        c: &SqlExpr,
        x: &SqlExpr,
        form: &str,
    ) -> Result<(u32, SExpr), PrepareError> {
        let Some(col) = inner.col(c)? else {
            return Err(item_refusal(c, form));
        };
        let key = self.key_side(x)?;
        if refs_join(&mut key.clone()) {
            return Err(form_refusal(&format!(
                "the left side {x} of {form}, which reads a joined table"
            )));
        }
        if (key.ty == Ty::Str) != (inner.st.cols[col as usize].ty.ty == Ty::Str) {
            return Err(form_refusal(&format!(
                "the left side {x} of {form} against {c}, which compares VARCHAR with another \
                 type"
            )));
        }
        Ok((col, key))
    }

    /// The residual of the hidden join of ANY and ALL ([`Member::Cmp`]):
    /// whether a row of the matched group passes `x pred c`. Lane 0 of the
    /// join is the count of the group's rows, and lanes 1 and 2 the
    /// extremes of its item ([`extremes`]). A row passes `x < c` iff the
    /// greatest item does, `x > c` iff the least does, and `x <> c` iff the
    /// least or the greatest does: DuckDB's order is total (NaN above every
    /// number, -0.0 equal to 0.0), and the cast of a side to the type of
    /// the comparison keeps it. So a tie of the extremes with other bits
    /// answers alike ([`plan::Agg::ties`]). The CASE evaluates `x` only if
    /// the count is above 0: DuckDB evaluates it once a row passes the
    /// WHERE, NULL items too (measured 2026-10-15). With the residual,
    /// whether the side of `x` can trap. An item whose cast can trap
    /// refuses: DuckDB casts every item, not the extremes only.
    ///
    /// DuckDB casts both sides of ANY to their max type, which for two
    /// DECIMALs past 38 digits keeps the integer digits and drops scale
    /// ([`dec_common`]; EXPLAIN shows `CAST(x AS DECIMAL(38,0))` for a
    /// DECIMAL(9,4) against a DECIMAL(38,0), measured 2026-10-15). The
    /// comparison operator keeps the scale instead, so the sides take the
    /// max type here, before [`Binder::cmp`].
    fn quantified_residual(
        &self,
        join: u32,
        x: &SExpr,
        item: Ty,
        pred: CmpPred,
        c: &SqlExpr,
        form: &str,
    ) -> Result<(SExpr, bool), PrepareError> {
        let read = |col: u32, ty: Ty| SExpr {
            kind: SKind::StaticCol { join, col },
            ty,
            nullable: true,
        };
        let common = dec_common(x.ty, item);
        let side = |e: SExpr| match common {
            Some(t) => to_common(e, t),
            None => e,
        };
        let mut traps = false;
        let mut test = |col: u32| -> Result<SExpr, PrepareError> {
            let e = self.cmp(pred, side(x.clone()), side(read(col, item)))?;
            let SKind::Cmp { a, b, .. } = &e.kind else {
                return Err(PrepareError::Internal(format!(
                    "the comparison of {form} did not bind as one"
                )));
            };
            if can_trap(b) {
                return Err(form_refusal(&format!(
                    "the item {c} under {form}, whose cast to the type of the comparison can \
                     trap (DuckDB casts every item)"
                )));
            }
            traps |= can_trap(a);
            Ok(e)
        };
        let test = match extremes(pred) {
            [_] => test(1)?,
            [_, _] => {
                let (lo, hi) = (test(1)?, test(2)?);
                SExpr {
                    kind: SKind::Or {
                        a: Box::new(lo),
                        b: Box::new(hi),
                    },
                    ty: Ty::I1,
                    nullable: true,
                }
            }
            _ => return Err(PrepareError::Internal(format!("{form} has no residual"))),
        };
        let any = SExpr {
            kind: SKind::Cmp {
                pred: CmpPred::Gt,
                a: Box::new(read(0, Ty::I64)),
                b: Box::new(lit_i64(0)),
            },
            ty: Ty::I1,
            nullable: true,
        };
        let residual = SExpr {
            kind: SKind::Case {
                arms: vec![(any, test)],
                default: None,
            },
            ty: Ty::I1,
            nullable: true,
        };
        Ok((residual, traps))
    }

    /// Run `f` over the subquery's table when its FROM is one static table;
    /// `None` for another FROM, which [`Binder::subquery_value`] refuses.
    fn over_table<R>(
        &self,
        select: &sqlparser::ast::Select,
        f: impl FnOnce(&Inner<'_>) -> Result<R, PrepareError>,
    ) -> Result<Option<R>, PrepareError> {
        let table = match select.from.as_slice() {
            [twj] if twj.joins.is_empty() => plain_table(&twj.relation)?
                .filter(|(name, _)| !name.bare().eq_ignore_ascii_case(&self.request_table))
                .and_then(|(name, alias)| {
                    Some((resolve_static(self.statics, &name).ok()?, name, alias))
                }),
            _ => None,
        };
        let Some((source, name, alias)) = table else {
            return Ok(None);
        };
        let renamed = apply_column_alias(&self.statics[source], alias)?;
        let inner = Inner {
            st: renamed.as_ref().unwrap_or(&self.statics[source]),
            name: alias.map_or_else(|| name.bare().to_string(), |a| a.name.value.clone()),
            schema: if alias.is_some() { String::new() } else { name.schema() },
        };
        f(&inner).map(Some)
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
            // DuckDB adds DOUBLE values in an order that can change the last
            // bits of the answer. The build adds them in table order, and the
            // comparison contract compares the answer within a bound
            // (docs/oracle/05, claim: float-reduction-bound), so only a whole
            // output column may read it ([`crate::specializer::FloatBound`]).
            (AggFunc::Sum | AggFunc::Avg, Some(Ty::F64)) => Ty::F64,
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
        Ok((
            Agg {
                func,
                arg,
                distinct,
                ties: false,
            },
            ty,
        ))
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
    ///
    /// An item with no aggregate call reads the columns of the matched row,
    /// `(SELECT s.v * 2 FROM s WHERE s.k = t.k)` ([`plan::AggPost::row`]):
    /// each column becomes the input `min(col)`, which over a group of one
    /// row is its value, and the table counts the rows of each group last.
    /// DuckDB evaluates the item over each matched row, traps when a
    /// request matches more than one row, and answers NULL for a request
    /// that matches none (measured 2026-10-15). A column of the outer query
    /// refuses, since the build cannot evaluate it, as do the table's whole
    /// row, a field or an element of one of its columns, and an item that
    /// is a bare NULL, which DuckDB types as NULL.
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
        // The names the item reads outside an aggregate call.
        let mut names: Vec<SqlExpr> = Vec::new();
        // The table's own name alone reads its whole row, as a struct
        // (measured: even when the outer query has a column of that name).
        let is_row = |x: &SqlExpr| {
            matches!(x, SqlExpr::Identifier(id) if id.value.eq_ignore_ascii_case(&inner.name))
        };
        self.walk_expr(e, &mut |x| {
            if calls::marker_call(x).is_some() || lets::marker_let(x).is_some() {
                return Err(form_refusal(&format!(
                    "the SELECT item {e}, which calls a SQL function"
                )));
            }
            Ok(match x {
                SqlExpr::Subquery(_)
                | SqlExpr::Exists { .. }
                | SqlExpr::InSubquery { .. }
                | SqlExpr::AnyOp { .. }
                | SqlExpr::AllOp { .. } => {
                    return Err(form_refusal(&format!("the SELECT item {e}, which holds a subquery")))
                }
                SqlExpr::Function(f) => match agg_func(f) {
                    Some(func) => {
                        let (agg, ty) = self.agg_call(inner, f, func)?;
                        if adds_doubles(&agg, inner.st) {
                            return Err(unsup(format!(
                                "a subquery whose SELECT item {e} computes over {f}, a DOUBLE \
                                 sum or avg -- the order of the additions can change the last \
                                 bits of its answer, so confit serves it only as the whole \
                                 SELECT item of a subquery that is a whole output column"
                            )));
                        }
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
                    // Over the static rows, an aggregate that confit does not
                    // serve names the served ones, not the refusal of an
                    // aggregate over the request rows.
                    None => {
                        let name = f.name.to_string().to_lowercase();
                        if NAMED_AGGREGATES.contains(&name.as_str()) {
                            return Err(form_refusal(&format!(
                                "the SELECT item {e}, which calls the aggregate function {name}"
                            )));
                        }
                        Walk::Into
                    }
                },
                SqlExpr::CompoundFieldAccess { root, .. }
                    if matches!(**root, SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)) =>
                {
                    if is_row(root) || inner.reads(root)? {
                        return Err(form_refusal(&format!(
                            "the SELECT item {e}, which reads a field or an element of a column \
                             of {}",
                            inner.name
                        )));
                    }
                    names.push(x.clone());
                    Walk::Over
                }
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                    names.push(x.clone());
                    Walk::Over
                }
                _ => Walk::Into,
            })
        })?;
        let row = aggs.is_empty();
        if !row {
            if let Some(x) = names.first() {
                return Err(form_refusal(&format!(
                    "the SELECT item {e}, which reads {x} outside an aggregate call"
                )));
            }
        }
        // An item over one row: each column it reads is `min` of that
        // column over a group of one row.
        for x in names.iter().filter(|_| row) {
            let Some(lane) = inner.col(x)? else {
                if is_row(x) {
                    return Err(form_refusal(&format!(
                        "the SELECT item {e}, which reads the whole row of {}",
                        inner.name
                    )));
                }
                // A name that the outer query does not bind either answers
                // the binder's error.
                self.key_side(x)?;
                return Err(form_refusal(&format!(
                    "the SELECT item {e}, which reads {x} of the outer query"
                )));
            };
            let col = &inner.st.cols[lane as usize];
            let ty = col.ty.ty;
            if !(ty.is_integer() || matches!(ty, Ty::I1 | Ty::F64 | Ty::Str | Ty::Dec(..))) {
                return Err(form_refusal(&format!(
                    "the SELECT item {e}, which reads {x} of type {}",
                    duck_ty_name(ty)
                )));
            }
            let agg = Agg {
                func: AggFunc::Min,
                arg: Some(lane),
                distinct: false,
                ties: false,
            };
            let i = match aggs.iter().position(|a| *a == agg) {
                Some(i) => i,
                None => {
                    aggs.push(agg);
                    in_cols.push(Col {
                        name: col.name.clone(),
                        ty: super::super::ir::ColTy { ty, nullable: true },
                    });
                    aggs.len() - 1
                }
            };
            inputs.push((x.clone(), i as u32));
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
        // The item runs over every row: ORDER BY and LIMIT are the build's.
        q.order_by = None;
        q.limit_clause = None;
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
        if row && bound.null_cols.first() == Some(&true) {
            return Err(form_refusal(&format!(
                "the SELECT item {e}, which DuckDB types as NULL"
            )));
        }
        Ok((
            aggs,
            plan::AggPost {
                in_cols,
                project: stage.project,
                out_cols: bound.out_cols,
                regexes: bound.ctx.regexes,
                row,
                order: None,
            },
        ))
    }

    /// The ORDER BY, LIMIT and OFFSET of an item over one row
    /// ([`plan::RowOrder`]). Each ORDER BY term is a column of the table, or
    /// the item, named by its position 1 or by its alias: DuckDB binds a
    /// bare name to the alias before a column of that name, but not to the
    /// name of an item without an alias (measured 2026-10-15). A term that
    /// reads the outer query refuses, as does any other expression and a
    /// position or a literal that DuckDB's binder refuses.
    fn row_order(
        &self,
        inner: &Inner<'_>,
        alias: Option<&Ident>,
        d: &OrderDraft<'_>,
    ) -> Result<plan::RowOrder, PrepareError> {
        let mut keys = Vec::with_capacity(d.terms.len());
        for &(term, desc, nulls_first) in &d.terms {
            let mut x = term;
            while let SqlExpr::Nested(y) = x {
                x = y;
            }
            let lane = match x {
                SqlExpr::Value(v) => match &v.value {
                    SqlValue::Number(n, _) if n.parse::<i64>() == Ok(1) => None,
                    SqlValue::Number(n, _) if n.parse::<i64>().is_ok() => {
                        return Err(form_refusal(&format!(
                            "the ORDER BY position {term} (DuckDB's binder stops with 'ORDER \
                             term out of range')"
                        )))
                    }
                    _ => {
                        return Err(form_refusal(&format!(
                            "the ORDER BY term {term} (DuckDB's binder stops with 'ORDER BY \
                             non-integer literal has no effect')"
                        )))
                    }
                },
                SqlExpr::Identifier(id) if alias.is_some_and(|a| a.value.eq_ignore_ascii_case(&id.value)) => None,
                _ => match inner.col(x)? {
                    Some(lane) => Some(lane),
                    None if matches!(x, SqlExpr::Identifier(id) if id.value.eq_ignore_ascii_case(&inner.name)) => {
                        return Err(form_refusal(&format!(
                            "the ORDER BY term {term}, which reads the whole row of {}",
                            inner.name
                        )))
                    }
                    None if matches!(x, SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)) => {
                        // A name that the outer query does not bind either
                        // answers the binder's error.
                        self.key_side(x)?;
                        return Err(form_refusal(&format!(
                            "the ORDER BY term {term}, which reads the outer query"
                        )));
                    }
                    None => {
                        return Err(form_refusal(&format!(
                            "the ORDER BY term {term} (each term must be a column of {}, the \
                             position 1 or the alias of the SELECT item)",
                            inner.name
                        )))
                    }
                },
            };
            keys.push(plan::OrderKey {
                lane,
                desc,
                nulls_first,
            });
        }
        Ok(plan::RowOrder {
            keys,
            limit: d.limit,
            offset: d.offset,
        })
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
                SqlExpr::Subquery(_)
                | SqlExpr::Exists { .. }
                | SqlExpr::InSubquery { .. }
                | SqlExpr::AnyOp { .. }
                | SqlExpr::AllOp { .. } => {
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
        q.order_by = None;
        q.limit_clause = None;
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

    /// The outer side of a key, bound in this scope. A let read answers its
    /// value here, not `SKind::Let`: the key runs at the join, before the
    /// projection computes its lets.
    fn key_side(&self, outer: &SqlExpr) -> Result<SExpr, PrepareError> {
        let (let_reads, projection_lets) = (self.let_reads.get(), self.projection_lets.get());
        self.let_reads.set(false);
        self.projection_lets.set(false);
        let key = self.expr(outer);
        self.let_reads.set(let_reads);
        self.projection_lets.set(projection_lets);
        Ok(fold(key?))
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
    /// the lower half of such a term counts for a key. The IN test runs
    /// after the whole WHERE, so for it the whole term counts, and only the
    /// rows whose item is not NULL ([`Guard::InItem`]).
    fn guarded_key(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        pre: &[&SqlExpr],
        key: SExpr,
        guard: Guard,
    ) -> Result<SExpr, PrepareError> {
        let halves = matches!(guard, Guard::Key);
        let filter = match pre {
            [] => None,
            _ => {
                let mut f = self.static_filter(query, select, source, pre)?;
                for (i, t) in pre.iter().enumerate() {
                    if !halves || !is_between(t) {
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
        let (func, arg) = match guard {
            Guard::Key => (AggFunc::CountStar, None),
            Guard::InItem(c) => (AggFunc::Count, Some(c)),
        };
        let count = Col {
            name: match arg {
                None => "count_star()".into(),
                Some(c) => format!("count({})", self.statics[source].cols[c as usize].name),
            },
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
                func,
                arg,
                distinct: false,
                ties: false,
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
            fold: None,
            scan: None,
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

    /// [`Binder::subquery_value`] over a WHERE with a term that compares
    /// the table with the outer query other than by a key, such as
    /// `(SELECT sum(s.v) FROM s WHERE s.lo <= t.x AND t.x < s.hi)`: one
    /// aggregate over the rows that a scan keeps for each request
    /// ([`plan::Fold`]). The scan reads the rows of the table that pass the
    /// filter and whose keys equal the request's, at most
    /// [`plan::SCAN_LIMIT`] of them, which the build checks
    /// ([`plan::ScanRows`]), and keeps those on which every other term is
    /// TRUE. A key that compares with IS NOT DISTINCT FROM is such a term.
    ///
    /// DuckDB evaluates the WHERE over each pair of a static row and a
    /// distinct correlated value of the batch (measured 2026-10-15). When
    /// no term can trap, the rows that it keeps for a value do not depend on
    /// the batch, and neither does an aggregate whose answer does not depend
    /// on the order of the rows: a count, a sum over integers, and min and
    /// max over a type where two equal values are one value. A term must
    /// not tell apart two DOUBLE values that DuckDB counts as one
    /// correlated value ([`double_taint`]).
    #[allow(clippy::too_many_arguments)]
    fn scan_value(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        inner: &Inner<'_>,
        member: Member<'_>,
        ordered: bool,
        w: ScanWhere<'_>,
    ) -> Result<SExpr, PrepareError> {
        let ScanWhere {
            mut mixed,
            filters,
            keys,
            key_cols,
            key_lanes,
            key_terms,
            trap_key,
        } = w;
        let first = mixed[0];
        if !matches!(member, Member::Value) {
            return Err(scan_refusal(first, "the subquery is under IN, ANY or ALL"));
        }
        if ordered {
            return Err(scan_refusal(first, "the subquery has ORDER BY or LIMIT"));
        }
        if trap_key {
            return Err(scan_refusal(first, "the outer side of a key can trap"));
        }
        let outer = |x: &SExpr| matches!(x.kind, SKind::Col(_) | SKind::Slot(_));
        let mut scan_keys = Vec::new();
        let mut scan_lanes = Vec::new();
        for (((key, kc), lane), t) in keys.into_iter().zip(&key_cols).zip(key_lanes).zip(key_terms) {
            if kc.cmp == KeyCmp::NotDistinct {
                mixed.push(t);
                continue;
            }
            if double_taint(&mut key.clone(), &outer).is_none() {
                return Err(scan_refusal(first, &zeros_refusal(t)));
            }
            scan_keys.push(key);
            scan_lanes.push(lane);
        }

        // SELECT: one aggregate call over the kept rows.
        let item = match select.projection.as_slice() {
            [SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. }] => e,
            [_] => return Err(form_refusal("a star in its SELECT")),
            _ => return Err(form_refusal("more than one SELECT item")),
        };
        let mut e = item;
        while let SqlExpr::Nested(x) = e {
            e = x;
        }
        let call = match e {
            SqlExpr::Function(f) => agg_func(f).map(|func| (f, func)),
            _ => None,
        };
        let Some((f, func)) = call else {
            return Err(scan_refusal(
                first,
                &format!("its SELECT item {e} is not one aggregate call"),
            ));
        };
        let (agg, ty) = self.agg_call(inner, f, func)?;
        let arg_ty = agg.arg.map(|c| inner.st.cols[c as usize].ty.ty);
        let served = !agg.distinct
            && match (agg.func, arg_ty) {
                (AggFunc::CountStar | AggFunc::Count, _) => true,
                // At most SCAN_LIMIT values below 10^p add up below 10^38.
                (AggFunc::Sum, Some(Ty::Dec(p, _))) => 10u128
                    .pow(p as u32)
                    .checked_mul(plan::SCAN_LIMIT as u128)
                    .is_some_and(|m| m <= 10u128.pow(38)),
                (AggFunc::Sum, Some(t)) => t.lane() == Ty::I64 || matches!(t, Ty::U64 | Ty::I1),
                (AggFunc::Min | AggFunc::Max, Some(t)) => {
                    matches!(t.lane(), Ty::I64 | Ty::I128 | Ty::Str | Ty::Dec(..))
                }
                _ => false,
            };
        if !served {
            return Err(scan_refusal(first, &format!("its SELECT item is {f}")));
        }

        // The terms, over the row of the scan and the outer query.
        let mut preds = Vec::new();
        for t in &mixed {
            preds.push(self.scan_term(query, select, source, inner, t)?);
        }
        let mut lanes: Vec<u32> = Vec::new();
        for p in preds.iter_mut() {
            scan_reads(p, &mut lanes);
        }
        lanes.extend(agg.arg);
        lanes.sort_unstable();
        lanes.dedup();
        let filter = match filters.as_slice() {
            [] => None,
            _ => Some(self.static_filter(query, select, source, &filters)?),
        };

        // The join of the rows, and the join of the fold that reads them.
        let mut tables = self.agg_tables.borrow_mut();
        let table = self.statics.len() + tables.len();
        let counts = counts_rows(agg.func);
        tables.push(AggTable {
            table: StaticTable::all_scalar(
                self.statics[source].name.clone(),
                vec![Col {
                    name: e.to_string(),
                    ty: super::super::ir::ColTy {
                        ty,
                        nullable: !counts,
                    },
                }],
            ),
            source,
            keys: Vec::new(),
            aggs: vec![agg],
            filter: None,
            post: None,
        });
        drop(tables);
        let mut joins = self.agg_joins.borrow_mut();
        let rows = (self.joins.len() + joins.len()) as u32;
        for p in preds.iter_mut() {
            into_scan(p, rows, &lanes);
        }
        let pred = preds.into_iter().reduce(|a, b| {
            let nullable = a.nullable || b.nullable;
            SExpr {
                kind: SKind::And {
                    a: Box::new(a),
                    b: Box::new(b),
                },
                ty: Ty::I1,
                nullable,
            }
        });
        let arg = agg
            .arg
            .map(|l| lanes.iter().position(|&x| x == l).expect("the argument's lane is read") as u32);
        joins.push(JoinSpec {
            table: source,
            batch: false,
            kind: JoinKind::Left,
            keys: scan_keys,
            key_cols: scan_lanes
                .iter()
                .map(|&l| JoinKey {
                    src: KeySrc::Lane(l),
                    cmp: KeyCmp::Eq,
                })
                .collect(),
            val_cols: lanes,
            residual: None,
            after_where: false,
            fold: None,
            scan: Some(plan::ScanRows { filter }),
        });
        joins.push(JoinSpec {
            table,
            batch: false,
            kind: JoinKind::Left,
            keys: Vec::new(),
            key_cols: Vec::new(),
            val_cols: vec![0],
            residual: None,
            after_where: false,
            fold: Some(plan::Fold {
                rows,
                pred,
                aggs: vec![(agg.func, arg)],
            }),
            scan: None,
        });
        drop(joins);
        let read = SExpr {
            kind: SKind::StaticCol {
                join: rows + 1,
                col: 0,
            },
            ty,
            nullable: true,
        };
        let value = if counts { count_or_zero(read) } else { read };
        self.agg_reads
            .borrow_mut()
            .push((query.clone(), member.key(), value.clone()));
        Ok(value)
    }

    /// One WHERE term `t` of a scan ([`Binder::scan_value`]), bound as a
    /// BOOLEAN over the row of the scan and the outer query. It binds as
    /// the item of a query of its own over the table, as DuckDB binds a
    /// name of the subquery in the table before the outer query, where each
    /// name of the outer query is an input bound in the outer query. Each
    /// column of the table it reads comes back as a value of the join
    /// [`SCAN_ROW`], by its lane, which [`into_scan`] renumbers. A term
    /// that can trap refuses, as do a term that calls a SQL function, a UDF
    /// or a model, one that reads a joined table, and one that reads a
    /// DOUBLE of the outer query through an operation that tells -0.0 from
    /// 0.0 ([`double_taint`]).
    fn scan_term(
        &self,
        query: &sqlparser::ast::Query,
        select: &sqlparser::ast::Select,
        source: usize,
        inner: &Inner<'_>,
        t: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        use super::resolve::Walk;
        // The names of the outer query that the term reads.
        let mut names: Vec<SqlExpr> = Vec::new();
        let is_row = |x: &SqlExpr| {
            matches!(x, SqlExpr::Identifier(id) if id.value.eq_ignore_ascii_case(&inner.name))
        };
        self.walk_expr(t, &mut |x| {
            if calls::marker_call(x).is_some() || lets::marker_let(x).is_some() {
                return Err(scan_refusal(t, "the term calls a SQL function"));
            }
            Ok(match x {
                SqlExpr::Subquery(_)
                | SqlExpr::Exists { .. }
                | SqlExpr::InSubquery { .. }
                | SqlExpr::AnyOp { .. }
                | SqlExpr::AllOp { .. } => {
                    return Err(form_refusal(&format!("the WHERE term {t}, which holds a subquery")))
                }
                SqlExpr::CompoundFieldAccess { root, .. }
                    if matches!(**root, SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)) =>
                {
                    if is_row(root) || inner.reads(root)? {
                        return Err(form_refusal(&format!(
                            "the WHERE term {t}, which reads a field or an element of a column of {}",
                            inner.name
                        )));
                    }
                    if !names.contains(x) {
                        names.push(x.clone());
                    }
                    Walk::Over
                }
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => match inner.col(x)? {
                    Some(_) => Walk::Over,
                    None if is_row(x) => {
                        return Err(form_refusal(&format!(
                            "the WHERE term {t}, which reads the whole row of {}",
                            inner.name
                        )))
                    }
                    None => {
                        if !names.contains(x) {
                            names.push(x.clone());
                        }
                        Walk::Over
                    }
                },
                _ => Walk::Into,
            })
        })?;
        // The query's input: the table's scalar columns in declared order,
        // each nullable as in [`Binder::static_filter`], then one input per
        // name of the outer query.
        let st = &self.statics[source];
        let mut in_cols = Vec::new();
        let mut lanes = Vec::new();
        let mut opaque = Vec::new();
        for (pos, c) in st.star.iter().enumerate() {
            match c {
                plan::StarCol::Real(i) => {
                    let mut col = st.cols[*i as usize].clone();
                    col.ty.nullable = true;
                    in_cols.push(col);
                    lanes.push(*i);
                }
                plan::StarCol::Opaque(name) => opaque.push((pos, name.clone())),
            }
        }
        let n = in_cols.len() as u32;
        let mut inputs = Vec::new();
        let mut outer = Vec::new();
        for (i, x) in names.iter().enumerate() {
            let e = self.key_side(x)?;
            if refs_join(&mut e.clone()) {
                return Err(scan_refusal(t, &format!("{x} reads a joined table")));
            }
            in_cols.push(Col {
                name: format!("\u{0}outer {i}"),
                ty: super::super::ir::ColTy {
                    ty: e.ty,
                    nullable: e.nullable,
                },
            });
            inputs.push((x.clone(), n + i as u32));
            outer.push(e);
        }
        let mut sel = select.clone();
        sel.selection = None;
        sel.projection = vec![SelectItem::UnnamedExpr(SqlExpr::Cast {
            kind: CastKind::Cast,
            expr: Box::new(t.clone()),
            data_type: sqlparser::ast::DataType::Boolean,
            array: false,
            format: None,
        })];
        let mut q = query.clone();
        q.body = Box::new(SetExpr::Select(Box::new(sel)));
        q.order_by = None;
        q.limit_clause = None;
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
            agg_inputs: &inputs,
        };
        let bound = bind_query(&q, None, &env, &[])?;
        let [stage] = <[Stage; 1]>::try_from(bound.stages).map_err(|_| {
            PrepareError::Internal("a scan's WHERE term bound to more than one stage".into())
        })?;
        if stage.project.len() != 1
            || !bound.joins.is_empty()
            || stage.pred.is_some()
            || !stage.lets.is_empty()
            || !bound.wide_outs.is_empty()
        {
            return Err(scan_refusal(t, "the term does not bind one value"));
        }
        let mut e = stage.project.into_iter().next().expect("one item").1;
        if e.ty != Ty::I1 {
            return Err(PrepareError::Internal(format!(
                "the WHERE term {t} bound to {}, not BOOLEAN",
                e.ty.name()
            )));
        }
        if calls_extern(&mut e.clone()) {
            return Err(scan_refusal(t, "the term calls a UDF or a model"));
        }
        // The item's regular expressions join this query's.
        let base = self.regexes.borrow().len() as u32;
        self.regexes.borrow_mut().extend(bound.ctx.regexes);
        shift_regexes(&mut e, base);
        fn splice(e: &mut SExpr, n: u32, lanes: &[u32], outer: &[SExpr]) {
            if let SKind::Col(i) = e.kind {
                if i < n {
                    e.kind = SKind::StaticCol {
                        join: SCAN_ROW,
                        col: lanes[i as usize],
                    };
                } else {
                    *e = outer[(i - n) as usize].clone();
                }
                return;
            }
            for c in e.children_mut() {
                splice(c, n, lanes, outer);
            }
        }
        splice(&mut e, n, &lanes, &outer);
        // After the splice, so that it sees through a lateral alias to the
        // column that DuckDB correlates on.
        let leaf = |x: &SExpr| matches!(x.kind, SKind::Col(_) | SKind::Slot(_));
        if double_taint(&mut e.clone(), &leaf).is_none() {
            return Err(scan_refusal(t, &zeros_refusal(t)));
        }
        if can_trap(&e) {
            return Err(scan_refusal(t, "the term can trap"));
        }
        Ok(e)
    }
}

/// The WHERE of a subquery that confit serves by a scan
/// ([`Binder::scan_value`]), as [`Binder::subquery_value`] split it.
struct ScanWhere<'q> {
    /// The terms that compare the table with the outer query other than by
    /// a key, in written order.
    mixed: Vec<&'q SqlExpr>,
    /// The terms that read only the table.
    filters: Vec<&'q SqlExpr>,
    /// The keys, as [`Binder::subquery_value`] binds them for a lookup:
    /// each one's outer side, its comparison, its lane of the table and its
    /// term as written.
    keys: Vec<SExpr>,
    key_cols: Vec<JoinKey>,
    key_lanes: Vec<u32>,
    key_terms: Vec<&'q SqlExpr>,
    /// Whether the outer side of the first key can trap.
    trap_key: bool,
}

/// The join that a WHERE term of a scan reads its row through while it
/// binds ([`Binder::scan_term`]): the join of the rows has no index yet.
const SCAN_ROW: u32 = u32::MAX;

/// The lanes of the table that a WHERE term of a scan reads, added to
/// `out` ([`Binder::scan_term`]).
fn scan_reads(e: &mut SExpr, out: &mut Vec<u32>) {
    if let SKind::StaticCol {
        join: SCAN_ROW,
        col,
    } = e.kind
    {
        out.push(col);
        return;
    }
    for c in e.children_mut() {
        scan_reads(c, out);
    }
}

/// A WHERE term of a scan as the program reads it: each column of the row
/// as value lane `lanes.index(lane)` of the join `rows`.
fn into_scan(e: &mut SExpr, rows: u32, lanes: &[u32]) {
    if let SKind::StaticCol {
        join: SCAN_ROW,
        col,
    } = e.kind
    {
        let pos = lanes.iter().position(|&l| l == col).expect("a read lane is a value");
        e.kind = SKind::StaticCol {
            join: rows,
            col: pos as u32,
        };
        return;
    }
    for c in e.children_mut() {
        into_scan(c, rows, lanes);
    }
}

/// `e` with each regular expression it uses `base` places later in the
/// list: the list of a query of its own joins the end of another's.
fn shift_regexes(e: &mut SExpr, base: u32) {
    match &mut e.kind {
        SKind::ReMatch { re, .. } | SKind::ReExtract { re, .. } | SKind::ReReplace { re, .. } => {
            *re += base
        }
        _ => {}
    }
    for c in e.children_mut() {
        shift_regexes(c, base);
    }
}

/// Whether `e` is a DOUBLE that reads a value of the outer query (`outer`
/// tells which leaves are one), or `None` when `e` reads such a value
/// through an operation that can answer differently for two values that
/// DuckDB counts as one: -0.0 and 0.0, or two NaNs. DuckDB evaluates a
/// correlated subquery once for each distinct correlated value of the
/// batch, over the first such value it meets (measured 2026-10-15:
/// `t.x::VARCHAR = '-0.0'` holds for the 0.0 of a batch whose -0.0 comes
/// first, and fails for both when 0.0 comes first). So an operation that
/// tells -0.0 from 0.0, such as `1 / x` or a cast to VARCHAR, answers for
/// one request what another request's value gives. A comparison, IS NULL,
/// `+`, `-` and `*`, a dividend, a negation, abs, CASE, and greatest and
/// least answer alike for equal values.
fn double_taint(e: &mut SExpr, outer: &dyn Fn(&SExpr) -> bool) -> Option<bool> {
    let double = e.ty == Ty::F64;
    if outer(e) {
        return Some(double);
    }
    if let SKind::Arith { op, a, b } = &mut e.kind {
        let op = *op;
        let (ta, tb) = (double_taint(a, outer)?, double_taint(b, outer)?);
        return match op {
            ArithOp::Add | ArithOp::Sub | ArithOp::Mul => Some(double && (ta || tb)),
            ArithOp::Div | ArithOp::Rem if !tb => Some(double && ta),
            _ if !ta && !tb => Some(false),
            _ => None,
        };
    }
    let mut taint = false;
    for c in e.children_mut() {
        taint |= double_taint(c, outer)?;
    }
    if !taint {
        return Some(false);
    }
    match &e.kind {
        SKind::Cmp { .. } | SKind::IsNull { .. } => Some(false),
        SKind::MathF1 {
            op: NumOp1::Fneg | NumOp1::Fabs,
            ..
        }
        | SKind::Case { .. }
        | SKind::Extreme { .. } => Some(double),
        _ => None,
    }
}

/// Why a scan refuses the term `t` that [`double_taint`] refuses.
fn zeros_refusal(t: &SqlExpr) -> String {
    format!(
        "{t} reads a DOUBLE of the outer query through an operation that can tell -0.0 from \
         0.0, and DuckDB evaluates the subquery once for both"
    )
}

/// A count read through a hidden join: 0 when the join misses.
fn count_or_zero(read: SExpr) -> SExpr {
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
            default: Some(Box::new(lit_i64(0))),
        },
        ty: Ty::I64,
        nullable: false,
    }
}

/// What the guard of an outer expression that can trap counts
/// ([`Binder::guarded_key`]).
#[derive(Clone, Copy)]
enum Guard {
    /// The first key of the WHERE: the rows that the terms before it pass.
    Key,
    /// The left side of IN: the rows that the whole WHERE passes whose item,
    /// this lane of the table, is not NULL.
    InItem(u32),
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

/// Whether `agg` adds DOUBLE values, a `sum` or `avg` over a DOUBLE column of
/// `st`: the order of its additions can change its answer.
fn adds_doubles(agg: &Agg, st: &StaticTable) -> bool {
    matches!(agg.func, AggFunc::Sum | AggFunc::Avg)
        && agg.arg.is_some_and(|c| st.cols[c as usize].ty.ty == Ty::F64)
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
