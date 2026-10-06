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
//! Decimal literals: DuckDB types `1.5` as DECIMAL(2,1), and so does this
//! frontend. CAST targets without a lane (UHUGEINT, FLOAT/REAL, INTERVAL,
//! ...) refuse by name.

use sqlparser::ast::{
    AccessExpr, BinaryOperator, CastKind, Expr as SqlExpr, Ident, JoinConstraint, JoinOperator,
    SelectItem, SetExpr, Statement, Subscript, TableAlias, TableFactor, UnaryOperator,
    Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use super::exec::{ExternImpl, ScalarVal};
use super::fold::fold;
use super::ir::{BinOp, CmpPred, Col, Lit, NumOp1, StrOp2, StrOp2i, StrOp3, TrimSide, Ty};
use super::sig::{self, ArgTy, NullArg, Ret, Sig};
use super::plan::{
    ArithOp, CompareGrid, JoinKey, JoinKind, JoinSpec, KeyCmp, KeySrc, Plan, SExpr, SKind, Stage,
    StaticTable, StructCol, StructField, StructNode, bind_foldable, can_trap, may_trap, trap_skeleton,
};

mod refusal;
mod from;
mod joins;
mod star;
mod expr;
mod resolve;
mod structs;
mod regexp;
mod udf;
mod functions;
mod typing;
mod decimal;
pub mod macros;
mod calls;
mod lets;
mod lists;
mod naming;
mod outputs;

use self::refusal::*;

/// Stack left before a recursive step moves to a fresh segment, and that
/// segment's size (`stacker::maybe_grow`).
pub(crate) const RED_ZONE: usize = 256 * 1024;
pub(crate) const STACK_SEGMENT: usize = 8 * 1024 * 1024;
use self::from::*;
use self::joins::*;
use self::star::*;
use self::typing::*;
use self::decimal::*;
use self::lists::*;
use self::outputs::*;

pub use self::functions::{is_builtin, BUILTIN_NAMES};

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

/// SQL text + the dynamic table's name/schema + the static-table catalog
/// (+ declared UDF externs) -> bound relational tree, the equi-joins in
/// FROM order, the derived output schema, and the width-k UDF output
/// fields.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
/// sqlparser's own nesting limit, raised from its default 50 to DuckDB's
/// `max_expression_depth`: a SQL function body nested 32 parentheses deep
/// (a Normalizer over 48 features) parses on DuckDB and must here.
const PARSE_DEPTH: usize = 1000;

/// Run a parse with [`PARSE_DEPTH`] on a stack big enough for it: the
/// parser recurses per level without growing its own stack.
fn parse_deep<T>(
    parse: impl FnOnce() -> Result<T, sqlparser::parser::ParserError>,
) -> Result<T, sqlparser::parser::ParserError> {
    stacker::grow(256 * 1024 * 1024, parse)
}

pub fn frontend(
    sql: &str,
    this_name: &str,
    in_cols: &[Col],
    opaque: &[(usize, String)],
    structs: &[super::plan::StructCol],
    statics: &[StaticTable],
    many: bool,
    udfs: &[super::ir::ExternSpec],
    models: &[super::plan::ModelTable],
    bind_eval: &[ExternImpl],
    macros: &[macros::SqlMacro],
) -> Result<
    (
        Plan,
        Vec<JoinSpec>,
        Vec<Col>,
        Vec<super::ir::ReSpec>,
        Vec<super::WideOut>,
        Vec<u32>,
        // Lanes the binder MINTED (struct-node presence keys). The caller
        // APPENDS these to the lane list before lowering; see
        // `Binder::minted_lanes` for the write-side invariant.
        Vec<super::plan::InputLane>,
    ),
    PrepareError,
> {
    // GenericDialect, not DuckDbDialect: measured as a strict superset for
    // the forms we serve (adds ^@, * ILIKE, * RENAME) and matches the oracle
    // path in datafusion/plan.rs — see pins-wave5/sqlparser-spike.json.
    // DuckDB-only surface forms sqlparser can't represent are token-rewritten
    // first (rewrite.rs).
    if sql.to_ascii_lowercase().contains("__glob_pat") {
        // Reserved for the GLOB rewrite marker — never valid user SQL.
        return Err(unsup("reserved identifier __glob_pat"));
    }
    if sql.to_ascii_lowercase().contains(structs::SEQ_MARKER) {
        // Reserved for the struct field-read marker.
        return Err(unsup(format!("reserved identifier {}", structs::SEQ_MARKER)));
    }
    if sql.to_ascii_lowercase().contains(macros::CALL_MARKER) {
        // Reserved for the SQL function call marker.
        return Err(unsup(format!("reserved identifier {}", macros::CALL_MARKER)));
    }
    if sql.to_ascii_lowercase().contains(macros::LET_MARKER) {
        // Reserved for the SQL function let marker.
        return Err(unsup(format!("reserved identifier {}", macros::LET_MARKER)));
    }
    if sql.contains('\u{1}') {
        // Reserved for the star-filter rewrite marker.
        return Err(unsup("control character U+0001 in SQL"));
    }
    let dialect = GenericDialect {};
    let tokens = sqlparser::tokenizer::Tokenizer::new(&dialect, sql)
        .tokenize()
        .map_err(|e| PrepareError::Parse(e.to_string()))?;
    // DuckDB names an unaliased item after its text as written, before the
    // SQL function calls in it expand (`sc(x)`, not the body): with SQL
    // functions declared, the query as written is parsed too, for the names.
    let as_written = (!macros.is_empty()).then(|| tokens.clone());
    // SQL functions first, so their bodies pass through the same rewrites
    // as the query text they land in.
    let macros::Expanded {
        tokens,
        calls,
        lets,
    } = macros::expand(tokens, macros)?;
    let rewrite = |tokens| {
        super::rewrite::rewrite_glob(super::rewrite::rewrite_star_filters(
            super::rewrite::rewrite_parenless_replace(super::rewrite::rewrite_from_colon_aliases(
                super::rewrite::rewrite_colon_aliases(tokens),
            )),
        ))
    };
    let tokens = rewrite(tokens);
    // Each call read by field parses once; its reads bind against it.
    let calls = calls
        .into_iter()
        .map(|c| {
            let tokens = rewrite(c.tokens);
            let body = parse_deep(|| Parser::new(&dialect).with_recursion_limit(PARSE_DEPTH).with_tokens(tokens).parse_expr())
                .map_err(|e| {
                    PrepareError::Bind(format!("sql function '{}': its body: {e}", c.name))
                })?;
            Ok((c.name, body))
        })
        .collect::<Result<Vec<_>, PrepareError>>()?;
    let _calls = calls::Installed::new(calls);
    // Each let parses once; its reads bind against it.
    let lets = lets
        .into_iter()
        .map(|l| {
            let tokens = rewrite(l.tokens.clone());
            let e = parse_deep(|| Parser::new(&dialect).with_recursion_limit(PARSE_DEPTH).with_tokens(tokens).parse_expr())
                .map_err(|e| {
                    PrepareError::Bind(format!("sql function '{}': a let: {e}", l.name))
                })?;
            Ok((l.name, l.tokens, e))
        })
        .collect::<Result<Vec<_>, PrepareError>>()?;
    let _lets = lets::Installed::new(lets::table(lets));
    let parse = |tokens| {
        parse_deep(|| Parser::new(&dialect).with_recursion_limit(PARSE_DEPTH).with_tokens(tokens).parse_statements())
            .map_err(|e| PrepareError::Parse(e.to_string()))
    };
    let statements = parse(tokens)?;
    let [statement] = statements.as_slice() else {
        return Err(unsup("multiple SQL statements"));
    };
    let query = match statement {
        Statement::Query(q) => q,
        other => return Err(unsup(format!("statement kind: {other}"))),
    };
    // Expansion puts one parenthesized expression where each call stood, so
    // the query as written has the same levels and items as the one bound.
    let as_written = as_written.map(|t| parse(rewrite(t))).transpose()?;
    let as_written = match as_written.as_deref() {
        None => None,
        Some([Statement::Query(q)]) => Some(q.as_ref()),
        Some(_) => {
            return Err(PrepareError::Internal(
                "the query as written is not the one query it expands to".into(),
            ))
        }
    };
    let env = Env {
        this_name,
        in_cols,
        opaque,
        structs,
        statics,
        many,
        udfs,
        models,
        bind_eval,
    };
    let q = bind_query(query, as_written, &env, &[]).map_err(lets::spell_out_error)?;
    Ok((
        Plan { stages: q.stages },
        q.joins,
        q.out_cols,
        q.ctx.regexes,
        q.wide_outs,
        q.ctx.model_refs,
        q.ctx.minted_lanes,
    ))
}

/// Everything a query level binds against that does not change between
/// levels.
struct Env<'a> {
    this_name: &'a str,
    in_cols: &'a [Col],
    opaque: &'a [(usize, String)],
    structs: &'a [super::plan::StructCol],
    statics: &'a [StaticTable],
    many: bool,
    udfs: &'a [super::ir::ExternSpec],
    models: &'a [super::plan::ModelTable],
    bind_eval: &'a [ExternImpl],
}

/// One bound SELECT: its joins (indices into the query-wide list are its
/// positions, see [`bind_query`]), its WHERE, its projection and output.
struct BoundSelect {
    joins: Vec<JoinSpec>,
    pred: Option<SExpr>,
    project: Vec<(String, SExpr)>,
    /// What the projection reads through `SKind::Let` (see `lets.rs`).
    lets: Vec<SExpr>,
    out_cols: Vec<Col>,
    wide_outs: Vec<super::WideOut>,
    ctx: QueryCtx,
}

/// A query bound to the stages that compute it, innermost first.
struct BoundQuery {
    stages: Vec<Stage>,
    joins: Vec<JoinSpec>,
    out_cols: Vec<Col>,
    /// Per output column: a constant NULL, or a whole item that reads such a
    /// column of the level below. DuckDB keeps a bare NULL typed SQLNULL
    /// through each query level, and a consumer binds against that
    /// type; confit does not model it, so the level above binds such a
    /// column only as a whole projection item and refuses it anywhere else
    /// ([`Binder::null_col`]). The refusal comes where the column binds,
    /// before any fold: a fold drops the arms a constant CASE condition
    /// does not take, and with them the column, while the type the column
    /// gave the CASE stays (nightly seed 4824388:
    /// `CASE WHEN TRUE THEN -87.375 ELSE i0 END` typed DECIMAL(13,3), where
    /// DuckDB answers DECIMAL(5,3)).
    null_cols: Vec<bool>,
    wide_outs: Vec<super::WideOut>,
    ctx: QueryCtx,
}

/// Bind `query` and, through its FROM, every level below it. A FROM
/// naming a derived table binds the subquery first, then this level over
/// the subquery's output columns, whose references become slots: the
/// outer level sees only those columns (the inner scope is closed), and
/// each is computed once per row reaching the inner stage, read or not
/// (docs/specs/2026-09-26-row-local-subqueries-design.md). `as_written` is
/// the same query before its SQL function calls expanded, which names the
/// unaliased items (see [`frontend`]).
fn bind_query<'q>(
    query: &'q sqlparser::ast::Query,
    as_written: Option<&'q sqlparser::ast::Query>,
    env: &Env<'_>,
    ctes: &[CteDef<'q>],
) -> Result<BoundQuery, PrepareError> {
    // The CTEs in scope here: the enclosing ones, then this WITH's, each
    // body seeing only those declared before it. A CTE nobody reads is never
    // bound, as on DuckDB (measured: an unused CTE naming an unknown column
    // or table, or dividing by zero, leaves the query serving).
    let mut scope: Vec<CteDef<'q>> = ctes.to_vec();
    if let Some(with) = &query.with {
        if with.recursive {
            return Err(unsup("WITH RECURSIVE"));
        }
        let written = as_written.and_then(|w| w.with.as_ref());
        for (i, cte) in with.cte_tables.iter().enumerate() {
            if cte.materialized.is_some() || cte.from.is_some() {
                return Err(unsup("a MATERIALIZED hint on a CTE"));
            }
            let def = CteDef {
                alias: &cte.alias,
                query: &cte.query,
                as_written: written
                    .and_then(|w| w.cte_tables.get(i))
                    .map(|c| c.query.as_ref()),
                scope: scope.clone(),
                reads: std::rc::Rc::new(std::cell::Cell::new(0)),
            };
            scope.push(def);
        }
    }
    let find = |name: &RelName| -> Option<&CteDef<'q>> {
        let [part] = name.0.as_slice() else {
            return None;
        };
        scope
            .iter()
            .rev()
            .find(|d| d.alias.name.value.eq_ignore_ascii_case(part))
    };
    let select = level_select(query)?;
    let from = cross_joins_as_commas(&select.from);
    let written_select = as_written.map(level_select).transpose()?;
    let written_from = written_select.map(|s| cross_joins_as_commas(&s.from));
    // A CTE anywhere but the driving position would join a relation the
    // engine computes per request against the row: not served yet.
    for (i, rel) in from.iter().enumerate() {
        let joined = rel.joins.iter().map(|j| &j.relation);
        for factor in (i > 0).then_some(&rel.relation).into_iter().chain(joined) {
            if let Some((name, _)) = plain_table(factor)? {
                if find(&name).is_some() {
                    return Err(unsup("a CTE joined beside another relation"));
                }
            }
        }
    }
    let driving = from.first().map(|t| &t.relation);
    // The driving relation as a subquery: a derived table, or a CTE read
    // here (its body, the scope it binds in, and the alias lists that name
    // its columns, applied in order).
    let (subquery, written, sub_scope, name, renames): (_, _, &[CteDef<'q>], String, Vec<&TableAlias>) =
        match driving {
            Some(TableFactor::Derived {
                lateral,
                subquery,
                alias,
                sample,
            }) => {
                if *lateral {
                    return Err(unsup("LATERAL subquery"));
                }
                if sample.is_some() {
                    return Err(unsup("TABLESAMPLE on a derived table"));
                }
                let name = alias
                    .as_ref()
                    .map_or("unnamed_subquery".to_string(), |a| a.name.value.clone());
                let written = written_from
                    .as_ref()
                    .and_then(|f| f.first())
                    .and_then(|t| match &t.relation {
                        TableFactor::Derived { subquery, .. } => Some(subquery.as_ref()),
                        _ => None,
                    });
                (subquery.as_ref(), written, &scope, name, alias.iter().collect())
            }
            Some(factor) => match plain_table(factor)? {
                Some((rel_name, alias)) if find(&rel_name).is_some() => {
                    let def = find(&rel_name).expect("found above");
                    def.reads.set(def.reads.get() + 1);
                    if def.reads.get() > 1 {
                        return Err(unsup("a CTE read more than once"));
                    }
                    let name = alias.map_or(def.alias.name.value.clone(), |a| a.name.value.clone());
                    let renames = std::iter::once(def.alias).chain(alias).collect();
                    (def.query, def.as_written, def.scope.as_slice(), name, renames)
                }
                _ => return bind_request_level(select, written_select, env),
            },
            None => return bind_request_level(select, written_select, env),
        };
    let inner = bind_query(subquery, written, env, sub_scope)?;
    if !inner.wide_outs.is_empty() {
        return Err(unsup("a struct- or list-valued column in a derived table"));
    }
    let mut cols = inner.out_cols;
    for a in renames {
        // A PARTIAL list renames a prefix; the rest keep their names; too
        // many names is DuckDB's bind error (measured, as for a table).
        if a.columns.len() > cols.len() {
            return Err(PrepareError::Bind(format!(
                "table \"{name}\" has {} columns available but {} columns specified",
                cols.len(),
                a.columns.len()
            )));
        }
        for (c, def) in cols.iter_mut().zip(&a.columns) {
            c.name = def.name.value.clone();
        }
    }
    let mut b = bind_select(
        select,
        written_select,
        env,
        &cols,
        &[],
        &[],
        Driving::Derived {
            name,
            null_cols: inner.null_cols.clone(),
        },
        inner.ctx,
    )?;
    // This level references the subquery only through its columns; its
    // joins were numbered from 0 and continue after the inner levels' in
    // the query-wide list (storage identity; this stage owns them).
    let off = inner.joins.len() as u32;
    let over_null = |e: &mut SExpr| refs_col(e, &|i| inner.null_cols[i as usize]);
    // A backstop: every read of such a column refused where it bound.
    let null_refusal = null_col_refusal;
    for (_, e) in b.project.iter_mut() {
        if !matches!(e.kind, SKind::Col(_)) && over_null(e) {
            return Err(null_refusal());
        }
    }
    for e in b.lets.iter_mut() {
        if over_null(e) {
            return Err(null_refusal());
        }
    }
    let mut level_exprs: Vec<&mut SExpr> = Vec::new();
    level_exprs.extend(b.pred.as_mut());
    for spec in b.joins.iter_mut() {
        level_exprs.extend(spec.keys.iter_mut());
        level_exprs.extend(spec.residual.as_mut());
    }
    for e in level_exprs {
        if over_null(e) {
            return Err(null_refusal());
        }
        into_level(e, off);
    }
    for (_, e) in b.project.iter_mut() {
        into_level(e, off);
    }
    for e in b.lets.iter_mut() {
        into_level(e, off);
    }
    // A whole item that reads a bare NULL column passes it up as one: on
    // DuckDB it is still SQLNULL a level higher.
    let null_cols = b
        .project
        .iter()
        .map(|(_, e)| match e.kind {
            SKind::NullOf => true,
            SKind::Slot(i) => inner.null_cols[i as usize],
            _ => false,
        })
        .collect();
    let mut stages = inner.stages;
    stages.push(Stage {
        joins: (off..off + b.joins.len() as u32).collect(),
        pred: b.pred,
        project: b.project,
        lets: b.lets,
    });
    let mut joins = inner.joins;
    joins.extend(b.joins);
    Ok(BoundQuery {
        stages,
        joins,
        out_cols: b.out_cols,
        null_cols,
        wide_outs: b.wide_outs,
        ctx: b.ctx,
    })
}

/// A level whose FROM starts with the request table.
fn bind_request_level(
    select: &sqlparser::ast::Select,
    as_written: Option<&sqlparser::ast::Select>,
    env: &Env<'_>,
) -> Result<BoundQuery, PrepareError> {
    let b = bind_select(
        select,
        as_written,
        env,
        env.in_cols,
        env.opaque,
        env.structs,
        Driving::Request,
        QueryCtx::default(),
    )?;
    let null_cols = b.project.iter().map(|(_, e)| matches!(e.kind, SKind::NullOf)).collect();
    Ok(BoundQuery {
        stages: vec![Stage {
            joins: (0..b.joins.len() as u32).collect(),
            pred: b.pred,
            project: b.project,
            lets: b.lets,
        }],
        joins: b.joins,
        out_cols: b.out_cols,
        null_cols,
        wide_outs: b.wide_outs,
        ctx: b.ctx,
    })
}

/// A CTE in scope: its name and column list, its body (and the body as
/// written, see [`bind_query`]), the CTEs that body may read (those declared
/// before it), and how often the query has read it.
#[derive(Clone)]
struct CteDef<'q> {
    alias: &'q TableAlias,
    query: &'q sqlparser::ast::Query,
    as_written: Option<&'q sqlparser::ast::Query>,
    scope: Vec<CteDef<'q>>,
    reads: std::rc::Rc<std::cell::Cell<u32>>,
}

/// The refusal of a bare NULL column of the level below read anywhere but
/// as a whole projection item (see [`BoundQuery::null_cols`]).
fn null_col_refusal() -> PrepareError {
    unsup("an expression over a bare NULL subquery column")
}

/// Whether a projection item is one column reference, parenthesized or
/// not.
fn is_column_ref(item: &SelectItem) -> bool {
    let (SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. }) = item else {
        return false;
    };
    let mut e = e;
    while let SqlExpr::Nested(x) = e {
        e = x;
    }
    matches!(e, SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_))
}

/// Whether `e` references an input column `pick` selects.
fn refs_col(e: &mut SExpr, pick: &dyn Fn(u32) -> bool) -> bool {
    if let SKind::Col(i) = e.kind {
        return pick(i);
    }
    e.children_mut().into_iter().any(|c| refs_col(c, pick))
}

/// Place an expression bound at a level over a derived table into the
/// query: its column references read the previous stage's slots, and its
/// join references move past the `off` joins of the levels below.
fn into_level(e: &mut SExpr, off: u32) {
    match &mut e.kind {
        SKind::Col(i) => e.kind = SKind::Slot(*i),
        SKind::StaticCol { join, .. } | SKind::JoinHit(join) => *join += off,
        _ => {}
    }
    for c in e.children_mut() {
        into_level(c, off);
    }
}

/// The one SELECT a query level may be: every clause confit does not serve
/// is refused here, by name, at every level alike.
fn level_select(query: &sqlparser::ast::Query) -> Result<&sqlparser::ast::Select, PrepareError> {
    // Refusals below are a CLASS, not a list of features someone got round to.
    // A clause sqlparser parses and we ignore is a wrong ANSWER — the contract
    // is match-DuckDB-or-refuse, and dropping QUALIFY silently emitted every
    // row. Both helpers walk every field of their AST node, so
    // adding a clause to sqlparser breaks the build rather than the answers.
    if query.order_by.is_some() {
        return Err(unsup("ORDER BY"));
    }
    if query.limit_clause.is_some() {
        return Err(unsup("LIMIT/OFFSET"));
    }
    refuse_unhandled_query(query)?;
    let select = match query.body.as_ref() {
        SetExpr::Select(s) => s.as_ref(),
        SetExpr::SetOperation { .. } => return Err(unsup("UNION/INTERSECT/EXCEPT")),
        other => return Err(unsup(format!("query body: {other}"))),
    };
    if select.distinct.is_some() {
        return Err(unsup("DISTINCT"));
    }
    refuse_unhandled_select(select)?;
    let grouped = match &select.group_by {
        sqlparser::ast::GroupByExpr::Expressions(exprs, modifiers) => {
            !exprs.is_empty() || !modifiers.is_empty()
        }
        sqlparser::ast::GroupByExpr::All(_) => true,
    };
    if grouped || select.having.is_some() {
        return Err(unsup("GROUP BY / HAVING / aggregation"));
    }

    Ok(select)
}

/// One SELECT bound over `in_cols`: the request table's lanes, or a derived
/// table's output columns (`driving`).
#[allow(clippy::too_many_arguments)]
fn bind_select(
    select: &sqlparser::ast::Select,
    as_written: Option<&sqlparser::ast::Select>,
    env: &Env<'_>,
    in_cols: &[Col],
    opaque: &[(usize, String)],
    structs: &[super::plan::StructCol],
    driving: Driving,
    ctx: QueryCtx,
) -> Result<BoundSelect, PrepareError> {
    let (binder, joins, leftover_where) = bind_from(
        select,
        env.this_name,
        in_cols,
        opaque,
        structs,
        env.statics,
        env.many,
        env.udfs,
        env.models,
        env.bind_eval,
        driving,
        ctx,
    )?;

    let mut out_cols = Vec::new();
    let mut exprs = Vec::new();
    let mut wide_outs: Vec<super::WideOut> = Vec::new();
    let push_item = |out_cols: &mut Vec<Col>,
                     exprs: &mut Vec<SExpr>,
                     name: String,
                     e: SExpr|
     -> Result<(), PrepareError> {
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
    // A list or struct item expands to its lanes, the validity first; the
    // WideOut records how the boundary reassembles them into ONE field —
    // `list | None` for an unnamed extern or a list literal, a struct for
    // the rest (see `WideShape`).
    let push_wide = |out_cols: &mut Vec<Col>,
                         exprs: &mut Vec<SExpr>,
                         wide_outs: &mut Vec<super::WideOut>,
                         base: String,
                         lanes: Vec<(String, SExpr)>,
                         shape: super::WideShape|
     -> Result<(), PrepareError> {
        let first = out_cols.len() as u32;
        debug_assert_eq!(lanes.len() as u32, shape.lanes(), "lanes and shape disagree");
        for (name, ex) in lanes {
            out_cols.push(Col {
                name,
                ty: super::ir::ColTy {
                    ty: ex.ty,
                    nullable: ex.nullable,
                },
            });
            exprs.push(ex);
        }
        wide_outs.push(super::WideOut {
            name: base,
            first,
            shape,
        });
        Ok(())
    };
    // One star or COLUMNS column: a scalar lane, or a struct's lanes.
    let push_val = |out_cols: &mut Vec<Col>,
                    exprs: &mut Vec<SExpr>,
                    wide_outs: &mut Vec<super::WideOut>,
                    name: String,
                    v: OutVal|
     -> Result<(), PrepareError> {
        match v {
            OutVal::Scalar(e) => push_item(out_cols, exprs, name, e),
            v => {
                let (lanes, shape) = v.into_lanes(&name);
                push_wide(out_cols, exprs, wide_outs, name, lanes, shape)
            }
        }
    };
    // Each item as written, which names it (see `frontend`); without SQL
    // functions, the item itself.
    let written = as_written.map_or(&select.projection, |w| &w.projection);
    if written.len() != select.projection.len() {
        return Err(PrepareError::Bind(
            "a sql function body that is not one expression".into(),
        ));
    }
    // DuckDB qualifies every item's columns before any item binds, the
    // arguments a body never reads included.
    if as_written.is_some() {
        for w in written {
            if let SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } = w {
                binder.qualify_written(e)?;
            }
        }
    }
    // A projection `share.rs` lowers reads lets as `SKind::Let`.
    binder.let_reads.set(!env.many);
    for (item, written) in select.projection.iter().zip(written) {
        binder.whole_item.set(is_column_ref(item));
        // An unaliased item is named after its text as written, once.
        let written = match written {
            SelectItem::UnnamedExpr(w) => Some(w),
            _ => None,
        };
        let mut named: Option<String> = None;
        let mut name_of =
            |e: &SqlExpr| named.get_or_insert_with(|| default_name(written.unwrap_or(e))).clone();
        // unnest(udf(...)) expands IN PLACE, before any other item
        // handling, and an alias on it is ignored — the oracle's own
        // expansion (measured).
        let unnest_expr = match item {
            SelectItem::UnnamedExpr(e) => Some(e),
            SelectItem::ExprWithAlias { expr, .. } => Some(expr),
            _ => None,
        };
        if let Some(e) = unnest_expr {
            if let Some(cols) = binder.unnest_extern_columns(e)? {
                for (name, ex) in cols {
                    push_item(&mut out_cols, &mut exprs, name, ex)?;
                }
                continue;
            }
            // A list or struct-valued item: a list literal, a struct column,
            // struct_pack, a struct literal or a named extern's output.
            let base = match item {
                SelectItem::ExprWithAlias { alias, .. } => alias.value.clone(),
                _ => name_of(e),
            };
            if let Some((lanes, shape)) = binder.wide_item(e, &base)? {
                push_wide(&mut out_cols, &mut exprs, &mut wide_outs, base, lanes, shape)?;
                continue;
            }
        }
        match item {
            SelectItem::UnnamedExpr(e) => {
                if let Some((lanes, names)) = binder.wide_extern_lanes(e, &name_of(e))? {
                    let shape = extern_shape(&lanes, names);
                    push_wide(
                        &mut out_cols,
                        &mut exprs,
                        &mut wide_outs,
                        name_of(e),
                        lanes,
                        shape,
                    )?;
                    continue;
                }
                // COLUMNS('re') expands like a filtered star, keeping the
                // bare column names (pins-waveB/).
                if let Some(cols) = binder.expand_columns_item(e)? {
                    for (name, v) in cols {
                        push_val(&mut out_cols, &mut exprs, &mut wide_outs, name, v)?;
                    }
                } else {
                    push_item(&mut out_cols, &mut exprs, name_of(e), fold(binder.expr(e)?))?
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                if let Some((lanes, names)) = binder.wide_extern_lanes(expr, &alias.value)? {
                    // No lateral-alias registration: the assembled field is
                    // a container, which no scalar expression can reference.
                    let shape = extern_shape(&lanes, names);
                    push_wide(
                        &mut out_cols,
                        &mut exprs,
                        &mut wide_outs,
                        alias.value.clone(),
                        lanes,
                        shape,
                    )?;
                    continue;
                }
                // An alias on COLUMNS stamps EVERY expansion (duplicates
                // feed the dedup rename — measured).
                if let Some(cols) = binder.expand_columns_item(expr)? {
                    for (_, v) in cols {
                        push_val(&mut out_cols, &mut exprs, &mut wide_outs, alias.value.clone(), v)?;
                    }
                    continue;
                }
                let e = fold(binder.expr(expr)?);
                // Lateral aliases (pins-wave5/): later items and WHERE may
                // reference this alias; the real column still wins.
                binder
                    .bound_aliases
                    .borrow_mut()
                    .push((alias.value.clone(), e.clone()));
                push_item(&mut out_cols, &mut exprs, alias.value.clone(), e)?
            }
            SelectItem::Wildcard(opts) => {
                for (name, v) in binder.expand_star(None, opts)? {
                    push_val(&mut out_cols, &mut exprs, &mut wide_outs, name, v)?;
                }
            }
            SelectItem::QualifiedWildcard(kind, opts) => {
                let table = match kind {
                    // The identifier VALUES: `"d".*` names the relation d
                    // (quoting keeps case-insensitivity in DuckDB too).
                    sqlparser::ast::SelectItemQualifiedWildcardKind::ObjectName(n) => n
                        .0
                        .iter()
                        .map(|p| match p.as_ident() {
                            Some(i) => i.value.clone(),
                            None => p.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join("."),
                    sqlparser::ast::SelectItemQualifiedWildcardKind::Expr(_) => {
                        return Err(unsup("expression.* wildcard"))
                    }
                };
                for (name, v) in binder.expand_star(Some(&table), opts)? {
                    push_val(&mut out_cols, &mut exprs, &mut wide_outs, name, v)?;
                }
            }
            SelectItem::ExprWithAliases { .. } => return Err(unsup("multi-alias SELECT item")),
        };
    }
    binder.let_reads.set(false);
    binder.whole_item.set(false);
    if exprs.is_empty() {
        // Pinned text: an EXCLUDE-all star that empties the projection.
        return Err(PrepareError::Bind(
            "SELECT list is empty after resolving * expressions!".to_string(),
        ));
    }
    // The output fields' names, then the lanes' (a list or struct field's
    // lanes are named after it, so two fields of one name would share them).
    dedup_field_names(&mut out_cols, &mut wide_outs);
    dedup_output_names(&mut out_cols);

    // WHERE binds AFTER the projection so DuckDB's lateral-alias extension
    // (an alias visible inside WHERE when no real column shares the name)
    // resolves; the plan shape is unchanged — Filter still sits under
    // Project on the scan.
    let mut filter = None;
    if let Some(pred) = &leftover_where {
        let pred = fold(where_conjuncts(&binder, pred)?);
        // NO statically-NULL-conjunct elision here, deliberately: the oracle
        // runs DuckDB optimizer-OFF. Optimizer-ON DuckDB proves
        // such a filter selects nothing and deletes it along with its
        // operands; the ORACLE evaluates it:
        //
        //   SELECT 1 FROM t WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL
        //   oracle: Conversion Error: Could not convert string 'abc' to DOUBLE
        //
        // so the filter keeps its operands and its traps.
        filter = Some(pred);
    }

    let project = out_cols
        .iter()
        .map(|c| c.name.clone())
        .zip(exprs)
        .collect::<Vec<_>>();
    // Only the projection reads lets. A WHERE reads their values: what it
    // bound itself does, and a lateral alias it reads carries the
    // projection's reads.
    let lets = binder.lets.take();
    if let Some(pred) = filter.as_mut() {
        lets::inline(pred, &lets, &mut binder.let_inlined.borrow_mut(), lets::Cost::Reads)?;
    }
    Ok(BoundSelect {
        joins,
        pred: filter,
        project,
        lets,
        out_cols,
        wide_outs,
        ctx: binder.into_ctx(),
    })
}

/// One joined static table in scope: how it is named, which of its columns
/// are probe values (bindable directly) vs keys (reconstructed from the
/// dynamic side: `r.id` ≡ CASE match THEN dyn-key ELSE NULL — pins-wave4/).
struct ScopeJoin<'a> {
    name: String,
    /// The schema the relation was named through (see [`RelName::schema`]).
    schema: String,
    table: std::borrow::Cow<'a, StaticTable>,
    kind: JoinKind,
    key_cols: Vec<JoinKey>,
    val_cols: Vec<u32>,
    /// Dynamic-side key expressions aligned with `key_cols` — the material
    /// for key-column reconstruction.
    keys: Vec<SExpr>,
    /// USING join: the static side's using (key) columns are merged into
    /// the left occurrence — hidden from bare-name binds and star
    /// expansion (measured: merged col sits at the LEFT position with the
    /// LEFT value; `t2.a` stays addressable and is NULL on a LEFT miss).
    using: bool,
    /// A USING/NATURAL SELF-join's merged names. The batch side has no probe
    /// keys (the equality is a residual), so the merge is recorded by name:
    /// these value columns are hidden from bare-name binds and the
    /// UNQUALIFIED star, and stay addressable qualified (`u.k`, `u.*`) --
    /// measured, including a LEFT miss answering NULL.
    merged: Vec<String>,
}

struct Binder<'a> {
    /// The dynamic table's name as spelled in FROM.
    this_name: String,
    /// The schema the driving relation was named through ([`RelName::schema`]).
    this_schema: String,
    /// The dynamic table's columns AS THE BINDER SEES THEM: borrowed
    /// normally; an owned renamed copy under `t AS u(x, y)` (pins-wave5/ —
    /// prefix rename, old names fully shadowed). Positions never change,
    /// so the lowered program still marshals by the ORIGINAL field names.
    in_cols: std::borrow::Cow<'a, [Col]>,
    /// Plain scalar columns are `in_cols[..n_plain]`; struct leaf lanes
    /// follow, addressable only through struct paths — never by bare name
    /// or star expansion.
    n_plain: usize,
    /// Row-model columns whose types have NO scalar lane, at their MODEL
    /// positions. They exist for resolution — referencing one, or a star
    /// expansion that keeps one, is the named unsupported error — but an
    /// EXCLUDEd / name-filtered / REPLACEd one costs nothing, so a query
    /// that never touches the column serves.
    opaque: &'a [(usize, String)],
    /// Struct row columns flattened to leaf lanes.
    structs: &'a [super::plan::StructCol],
    joins: Vec<ScopeJoin<'a>>,
    /// All SELECT-list aliases (pins-wave5/: DuckDB's lateral aliases — a
    /// later item or WHERE may reference an earlier alias; the REAL column
    /// wins on a name clash; a forward reference is the pinned bind error).
    select_aliases: Vec<String>,
    /// Aliases already bound this pass, in SELECT order (frontend() fills
    /// this as it walks the projection; RefCell keeps `expr(&self)` intact).
    bound_aliases: std::cell::RefCell<Vec<(String, SExpr)>>,
    /// Program regex table under construction; indices are baked
    /// into ReMatch/ReExtract/ReReplace nodes.
    regexes: std::cell::RefCell<Vec<super::ir::ReSpec>>,
    /// Declared UDF externs: an unknown function matching one
    /// binds as an opaque ecall instead of the named refusal.
    udfs: &'a [super::ir::ExternSpec],
    /// The callables themselves, decl-order-aligned with `udfs`, for the
    /// bind-time fold of pure externs over constant args. Empty
    /// (every pure-rust caller) disables the fold — specs alone cannot
    /// execute python.
    bind_eval: &'a [ExternImpl],
    /// Declared tree transforms, by name — the same call-site namespace as
    /// `udfs`, resolved first because they lower to the native kernel.
    models: &'a [super::plan::ModelTable],
    /// Tree transforms actually referenced, as catalog indices in first-use
    /// order. Lowering appends one `StaticTy::Model` per entry AFTER every
    /// join static, so no existing probe's `@N` shifts.
    model_refs: std::cell::RefCell<Vec<u32>>,
    /// Call-site counter: the k+1 lanes of one width-k call share a site
    /// so lowering executes the callable once per row.
    sites: std::cell::Cell<u32>,
    /// Textually identical extern calls under field access share one site
    /// — the confit twin of DuckDB's common-subexpression elimination,
    /// which is what keeps call counts equal on both paths.
    extern_sites: std::cell::RefCell<Vec<(sqlparser::ast::Function, u32)>>,
    /// Bind-fold results by call site: one call site executes its pure
    /// callable ONCE at bind, however many contexts consult the fold
    /// (the scalar bind, then the || operand fold).
    #[allow(clippy::type_complexity)]
    bind_folds: std::cell::RefCell<Vec<(u32, Option<Result<Option<Vec<Option<ScalarVal>>>, String>>)>>,
    /// Depth of CASE/COALESCE arms being bound. DuckDB's plan-time constant
    /// evaluation SKIPS guarded arms (the coalesce lazy-bind pin: an
    /// untaken `CAST('nope' AS BIGINT)` must not fire), so the
    /// trapping-constant refusals only apply at depth 0 — a guarded
    /// trapping constant stays a lazy runtime question on both engines.
    in_guarded: std::cell::Cell<u32>,
    /// Set only while [`joins::bind_residual`] re-binds a conjunct to read
    /// which sides it NAMES: a join key column then binds as a reference to
    /// its static table instead of its probe-side reconstruction.
    classify_keys: std::cell::Cell<bool>,
    /// The top-level column names of the static table whose JOIN ON key is
    /// binding. That table is not in `joins` until its ON has bound, yet
    /// DuckDB already sees it: a bare name both sides have is ambiguous
    /// anywhere in the key expression (`c0 + 1 = s0.c0`).
    beside: std::cell::RefCell<Vec<String>>,
    /// Lanes the binder MINTED. Struct-node presence is the only minted kind
    /// today; the seam is shaped to take a second, key-only kind without
    /// changing its contract. The caller APPENDS these to the lane list before
    /// lowering. Named the same on the way out, because it is the same list
    /// moved out of this `RefCell` — one list, one name, in both places.
    ///
    /// WRITE-SIDE INVARIANT, which [`super::plan::LaneKind`] does NOT
    /// enforce and which is therefore written here: ONE vector, APPEND-ONLY,
    /// DEDUPED BY PATH ACROSS KINDS, INDEXED BY POSITION, and OFFSET BY
    /// `in_cols.len()`. `present_key` mints
    /// `SKind::Col((self.in_cols.len() + idx) as u32)` where `idx` is a
    /// position in this vector, so a second minter that pushed into a SECOND
    /// vector would collide on `idx`, and a second minter that deduped only
    /// within its own kind would mint two lanes for one path. `LaneKind`
    /// makes the boundary's READERS exhaustive; it says nothing about the
    /// writer. That asymmetry is the honest limit of this seam.
    ///
    /// Minted LAZILY — only when a struct actually becomes a join key.
    /// Minting them at schema parse would widen `in_cols` for every query
    /// over a struct-carrying row model, joined or not, and one extra
    /// unreferenced input lane measures +22..26 ns/row at the marshalling
    /// boundary against a ~200 ns/row floor. Arrow schemas default to
    /// nullable, so that would be charged to essentially every struct
    /// column in the repo for a feature one query shape uses.
    minted_lanes: std::cell::RefCell<Vec<super::plan::InputLane>>,
    /// Per SQL function call read by field, and scope: its siblings' trap
    /// skeletons (see `calls`), bound once for every read.
    call_siblings: std::cell::RefCell<std::collections::HashMap<calls::ScopeKey, calls::Siblings>>,
    /// Per SQL function call whose expansion is a CASE, and scope: whether
    /// DuckDB folds it to NULL (`struct_folds_to_null`), for every read.
    call_null: std::cell::RefCell<std::collections::HashMap<calls::ScopeKey, bool>>,
    /// Per call: the identifier words of its arguments, for the scope key.
    call_words: std::cell::RefCell<
        std::collections::HashMap<usize, std::rc::Rc<std::collections::HashSet<String>>>,
    >,
    /// Per call: how many bound aliases its arguments name, counted up to
    /// how many aliases were bound (see `calls::aliases_named`).
    call_aliases: std::cell::RefCell<std::collections::HashMap<usize, (usize, usize)>>,
    /// The values this level reads through `SKind::Let` (see `lets.rs`).
    lets: std::cell::RefCell<Vec<SExpr>>,
    /// Per let and scope: what its reads answer.
    let_vals: std::cell::RefCell<std::collections::HashMap<lets::LetKey, lets::LetVal>>,
    /// Whether a let read answers `SKind::Let`: while binding the
    /// projection of a stage `share.rs` lowers. Anywhere else it answers
    /// the value.
    let_reads: std::cell::Cell<bool>,
    /// The values of `lets`, inlined, for the reads outside the projection.
    let_inlined: std::cell::RefCell<lets::Inlined>,
    /// Per input column: a bare NULL of the level below (see
    /// [`BoundQuery::null_cols`]). Empty over the request table.
    null_cols: Vec<bool>,
    /// Set while a projection item that is one column reference binds:
    /// the one place a column of `null_cols` binds.
    whole_item: std::cell::Cell<bool>,
}

/// Decrements `in_guarded` on scope exit, whatever the exit path.
/// What every query level's binder shares: the design's query context. IDs
/// and tables that must be unique across the whole query (the regex table,
/// model references, UDF call sites, minted input lanes) are handed from
/// one level's binder to the next, append-only, so a level never renumbers
/// what an earlier one allocated.
#[derive(Default)]
struct QueryCtx {
    regexes: Vec<super::ir::ReSpec>,
    model_refs: Vec<u32>,
    sites: u32,
    minted_lanes: Vec<super::plan::InputLane>,
}

/// What a query level reads.
enum Driving {
    /// The request table (FROM names it, maybe aliased).
    Request,
    /// A derived table in scope as `name`; the binder's `in_cols` are its
    /// output columns, and a column reference binds to a slot. `null_cols`
    /// marks those that are a bare NULL (see [`BoundQuery::null_cols`]).
    Derived { name: String, null_cols: Vec<bool> },
}

impl Binder<'_> {
    /// The query context this binder accumulated, for the next level.
    fn into_ctx(self) -> QueryCtx {
        QueryCtx {
            regexes: self.regexes.into_inner(),
            model_refs: self.model_refs.into_inner(),
            sites: self.sites.get(),
            minted_lanes: self.minted_lanes.into_inner(),
        }
    }
}

struct GuardScope<'x>(&'x std::cell::Cell<u32>);

impl Drop for GuardScope<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

/// A WHERE predicate in DuckDB's evaluation order. Its parser flattens an AND
/// chain into one conjunct list, evaluated left to right with short-circuit;
/// but its binder turns `x BETWEEN l AND u` (non-volatile `x`) into
/// `(x >= l) AND (x <= u)`, and `LogicalFilter::SplitPredicates` -- the
/// planner, not the optimizer -- keeps `x >= l` in place and APPENDS `x <= u`
/// to the end of the list. So `(a BETWEEN a AND 46) AND (m * k)` evaluates
/// `m * k` on rows the upper bound would have stopped, and traps there
/// (nightly seed 3510602).
fn where_conjuncts(binder: &Binder<'_>, pred: &SqlExpr) -> Result<SExpr, PrepareError> {
    fn flatten<'e>(e: &'e SqlExpr, out: &mut Vec<&'e SqlExpr>) {
        match e {
            SqlExpr::Nested(i) => flatten(i, out),
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
            } => {
                flatten(left, out);
                flatten(right, out);
            }
            other => out.push(other),
        }
    }
    let mut conj = Vec::new();
    flatten(pred, &mut conj);
    let between = |e: &SqlExpr| {
        let mut e = e;
        while let SqlExpr::Nested(i) = e {
            e = i;
        }
        matches!(e, SqlExpr::Between { negated: false, .. })
    };
    if !conj.iter().any(|c| between(c)) {
        return bool_context(binder.expr(pred)?, "WHERE predicate");
    }
    let (mut head, mut tail) = (Vec::new(), Vec::new());
    for c in conj {
        let bound = match binder.expr_or_null(c)? {
            Some(e) => bool_context(e, "WHERE predicate")?,
            None => null_of(Ty::I1),
        };
        match bound.kind {
            SKind::And { a, b } if between(c) => {
                head.push(*a);
                tail.push(*b);
            }
            kind => head.push(SExpr { kind, ..bound }),
        }
    }
    let mut it = head.into_iter().chain(tail);
    let first = it.next().expect("a WHERE has a conjunct");
    Ok(it.fold(first, |acc, c| {
        let nullable = acc.nullable || c.nullable;
        SExpr {
            kind: SKind::And {
                a: Box::new(acc),
                b: Box::new(c),
            },
            ty: Ty::I1,
            nullable,
        }
    }))
}
