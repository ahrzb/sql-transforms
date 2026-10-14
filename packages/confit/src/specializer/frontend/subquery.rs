//! The correlated scalar subquery with one aggregate over one static table,
//! `(SELECT sum(s.v) FROM s WHERE s.k = t.k)`. It binds as a hidden LEFT
//! join to a static table that confit computes at build ([`AggTable`]): one
//! row per group of `s` rows with equal keys, holding the group's aggregate.
//! A request reads the row of the group it matches, and a request that
//! matches no group reads the aggregate of no rows: 0 for `count`, NULL for
//! the others.
//!
//! DuckDB (1.5.5, optimizer off, measured 2026-10-14) evaluates the
//! subquery's WHERE over (distinct correlated values) x (static rows), in
//! written order, and the aggregate's argument over the matched rows only.
//! Nothing served here can trap: each key's outer side cannot
//! (`can_trap`), its inner side is a bare column, no key casts a VARCHAR,
//! and the aggregates run at build, where a sum that could overflow
//! refuses. So the timing of that evaluation cannot show, and the hidden
//! join runs after the FROM joins of its level.

use super::*;
use super::super::plan::{self, Agg, AggFunc, AggTable};
use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};

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
                         scalar subquery's table",
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
        "expression {e} as the outer side of a scalar subquery's key -- the keys \
         served are column references, operators, CASE and plain function calls"
    ))
}

/// Whether `e` holds a subquery anywhere, through the lets and the SQL
/// function bodies it reads.
pub(super) fn has_subquery(b: &Binder<'_>, e: &SqlExpr) -> Result<bool, PrepareError> {
    use super::resolve::Walk;
    let none = b.walk_expr(e, &mut |x| {
        Ok(match x {
            SqlExpr::Subquery(_) | SqlExpr::Exists { .. } | SqlExpr::InSubquery { .. } => Walk::Stop,
            _ => Walk::Into,
        })
    })?;
    Ok(!none)
}

/// The refusal of a subquery form outside what this module serves.
fn form_refusal(what: &str) -> PrepareError {
    unsup(format!(
        "a scalar subquery with {what} -- served: (SELECT count(*) | count(c) | min(c) | \
         max(c) | sum(c) | avg(c) FROM s WHERE s.k = <outer expression> AND ...) over one \
         static table"
    ))
}

impl Binder<'_> {
    /// Bind `(SELECT agg(..) FROM s WHERE s.k = outer AND ...)`: its value,
    /// read through a hidden LEFT join this level runs after its FROM
    /// joins. The same text bound again at this level reads the same join.
    pub(super) fn scalar_subquery(&self, query: &sqlparser::ast::Query) -> Result<SExpr, PrepareError> {
        if !self.subqueries.get() {
            return Err(unsup("a scalar subquery in FROM or in a JOIN condition"));
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
                "a scalar subquery over the request table -- an aggregate across the \
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

        // WHERE: each term an equality between a column of the table and an
        // expression of the outer scope that cannot trap.
        let mut terms = Vec::new();
        if let Some(w) = &select.selection {
            collect_conjuncts(w, &mut terms);
        }
        let mut keys = Vec::new();
        let mut key_cols = Vec::new();
        let mut key_lanes = Vec::new();
        for t in terms {
            let mut t = t;
            while let SqlExpr::Nested(x) = t {
                t = x;
            }
            let (left, right, cmp) = match t {
                SqlExpr::BinaryOp {
                    left,
                    op: BinaryOperator::Eq,
                    right,
                } => (left, right, KeyCmp::Eq),
                SqlExpr::IsNotDistinctFrom(left, right) => (left, right, KeyCmp::NotDistinct),
                _ => {
                    return Err(form_refusal(&format!(
                        "the WHERE term {t} (each term must be an equality between a column \
                         of its table and an outer expression)"
                    )))
                }
            };
            let (col, outer) = match (inner.col(left)?, inner.col(right)?) {
                (Some(c), None) => (c, right.as_ref()),
                (None, Some(c)) => (c, left.as_ref()),
                _ => {
                    return Err(form_refusal(&format!(
                        "the WHERE term {t} (each term must be an equality between a column \
                         of its table and an outer expression)"
                    )))
                }
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
                return Err(form_refusal(&format!(
                    "the WHERE term {t}, whose outer side can trap"
                )));
            }
            key_cols.push(JoinKey {
                src: KeySrc::Lane(key_lanes.len() as u32),
                cmp,
            });
            key_lanes.push(col);
            keys.push(key);
        }

        // SELECT: one aggregate over a column of the table, or count(*).
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
        let SqlExpr::Function(f) = e else {
            return Err(form_refusal(&format!("the SELECT item {e} (one aggregate call)")));
        };
        let fname = match f.name.0.as_slice() {
            [p] => p.as_ident().map(|i| i.value.to_ascii_lowercase()),
            _ => None,
        };
        let func = match fname.as_deref() {
            Some("count") => AggFunc::Count,
            Some("min") => AggFunc::Min,
            Some("max") => AggFunc::Max,
            Some("sum") => AggFunc::Sum,
            Some("avg") => AggFunc::Avg,
            _ => return Err(form_refusal(&format!("the aggregate {}", f.name))),
        };
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
        if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
            return Err(form_refusal(&format!("the aggregate call {f} (DISTINCT, ALL or ORDER BY)")));
        }
        let [arg] = list.args.as_slice() else {
            return Err(form_refusal(&format!("the aggregate call {f}")));
        };
        let (func, arg) = match (func, arg) {
            (AggFunc::Count, FunctionArg::Unnamed(FunctionArgExpr::Wildcard)) => (AggFunc::CountStar, None),
            // A constant that is not NULL counts every row: count(1) and
            // count('x') answer count(*) (measured).
            (AggFunc::Count, FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(v))))
                if matches!(
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

        // The grouped table: the key lanes as the source has them, then the
        // aggregate.
        let mut cols: Vec<Col> = key_lanes.iter().map(|&c| st.cols[c as usize].clone()).collect();
        let counts = matches!(func, AggFunc::CountStar | AggFunc::Count);
        cols.push(Col {
            name: f.to_string(),
            ty: super::super::ir::ColTy {
                ty,
                nullable: !counts,
            },
        });
        let n_keys = key_lanes.len() as u32;
        let mut tables = self.agg_tables.borrow_mut();
        let table = self.statics.len() + tables.len();
        tables.push(AggTable {
            table: StaticTable::all_scalar(self.statics[source].name.clone(), cols),
            source,
            keys: key_lanes,
            aggs: vec![Agg { func, arg }],
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
        });
        drop(joins);
        let read = SExpr {
            kind: SKind::StaticCol { join, col: 0 },
            ty,
            nullable: true,
        };
        // A request that matches no group counts 0 rows: COALESCE(read, 0),
        // the form `coalesce` binds to.
        let value = if counts {
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
