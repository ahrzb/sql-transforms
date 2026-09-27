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
//! Decimal literals: DuckDB types `1.5` as DECIMAL(2,1); this frontend types
//! them f64. CAST targets without a lane (HUGEINT, the unsigned family,
//! FLOAT/REAL, DECIMAL/NUMERIC, ...) refuse by name.

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
    ArithOp, CompareGrid, JoinKey, JoinKind, JoinSpec, KeyCmp, KeySrc, Rel, SExpr, SKind,
    StaticTable, StructCol, StructField, StructNode, bind_foldable, may_trap,
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

use self::refusal::*;
use self::from::*;
use self::joins::*;
use self::star::*;
use self::typing::*;

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
) -> Result<
    (
        Rel,
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
    if sql.contains('\u{1}') {
        // Reserved for the star-filter rewrite marker.
        return Err(unsup("control character U+0001 in SQL"));
    }
    let dialect = GenericDialect {};
    let tokens = sqlparser::tokenizer::Tokenizer::new(&dialect, sql)
        .tokenize()
        .map_err(|e| PrepareError::Parse(e.to_string()))?;
    let tokens = super::rewrite::rewrite_glob(super::rewrite::rewrite_star_filters(
        super::rewrite::rewrite_parenless_replace(super::rewrite::rewrite_from_colon_aliases(
            super::rewrite::rewrite_colon_aliases(tokens),
        )),
    ));
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
    // Refusals below are a CLASS, not a list of features someone got round to.
    // A clause sqlparser parses and we ignore is a wrong ANSWER — the contract
    // is match-DuckDB-or-refuse, and dropping QUALIFY silently emitted every
    // row. Both helpers walk every field of their AST node, so
    // adding a clause to sqlparser breaks the build rather than the answers.
    if query.with.is_some() {
        return Err(unsup("WITH / common table expressions"));
    }
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

    let (binder, joins, leftover_where) = bind_from(
        select, this_name, in_cols, opaque, structs, statics, many, udfs, models, bind_eval,
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
    // A bare wide UDF item expands to its whole-validity lane plus k
    // component lanes; the WideOut records how the boundary reassembles
    // them into ONE field — `list | None` for an unnamed extern, a struct
    // keyed by the declared names for a named one.
    let push_wide = |out_cols: &mut Vec<Col>,
                         exprs: &mut Vec<SExpr>,
                         wide_outs: &mut Vec<super::WideOut>,
                         base: String,
                         lanes: Vec<(String, SExpr)>,
                         names: Vec<String>|
     -> Result<(), PrepareError> {
        let first = out_cols.len() as u32;
        let width = (lanes.len() - 1) as u32;
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
            width,
            names,
        });
        Ok(())
    };
    for item in &select.projection {
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
            // A struct-valued item (θ export): same wide-lane boundary as a
            // named extern's output struct.
            let base = match item {
                SelectItem::ExprWithAlias { alias, .. } => alias.value.clone(),
                _ => default_name(e),
            };
            if let Some((lanes, names)) = binder.struct_pack_lanes(e, &base)? {
                push_wide(&mut out_cols, &mut exprs, &mut wide_outs, base, lanes, names)?;
                continue;
            }
        }
        match item {
            SelectItem::UnnamedExpr(e) => {
                if let Some((lanes, names)) = binder.wide_extern_lanes(e, &default_name(e))? {
                    push_wide(
                        &mut out_cols,
                        &mut exprs,
                        &mut wide_outs,
                        default_name(e),
                        lanes,
                        names,
                    )?;
                    continue;
                }
                // COLUMNS('re') expands like a filtered star, keeping the
                // bare column names (pins-waveB/).
                if let Some(cols) = binder.expand_columns_item(e)? {
                    for (name, ex) in cols {
                        push_item(&mut out_cols, &mut exprs, name, ex)?;
                    }
                } else {
                    push_item(
                        &mut out_cols,
                        &mut exprs,
                        default_name(e),
                        fold(binder.expr(e)?),
                    )?
                }
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                if let Some((lanes, names)) = binder.wide_extern_lanes(expr, &alias.value)? {
                    // No lateral-alias registration: the assembled field is
                    // a container, which no scalar expression can reference.
                    push_wide(
                        &mut out_cols,
                        &mut exprs,
                        &mut wide_outs,
                        alias.value.clone(),
                        lanes,
                        names,
                    )?;
                    continue;
                }
                // An alias on COLUMNS stamps EVERY expansion (duplicates
                // feed the dedup rename — measured).
                if let Some(cols) = binder.expand_columns_item(expr)? {
                    for (_, ex) in cols {
                        push_item(&mut out_cols, &mut exprs, alias.value.clone(), ex)?;
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
        // Pinned text: an EXCLUDE-all star that empties the projection.
        return Err(PrepareError::Bind(
            "SELECT list is empty after resolving * expressions!".to_string(),
        ));
    }
    dedup_output_names(&mut out_cols);

    // WHERE binds AFTER the projection so DuckDB's lateral-alias extension
    // (an alias visible inside WHERE when no real column shares the name)
    // resolves; the plan shape is unchanged — Filter still sits under
    // Project on the scan.
    let mut rel = Rel::Scan;
    if let Some(pred) = &leftover_where {
        let bound = binder.expr(pred);
        let pred = fold(bool_context(bound?, "WHERE predicate")?);
        // NO statically-NULL-conjunct elision here, deliberately: the oracle
        // runs DuckDB optimizer-OFF. Optimizer-ON DuckDB proves
        // such a filter selects nothing and deletes it along with its
        // operands; the ORACLE evaluates it:
        //
        //   SELECT 1 FROM t WHERE CAST(s AS DOUBLE) BETWEEN 61.591 AND NULL
        //   oracle: Conversion Error: Could not convert string 'abc' to DOUBLE
        //
        // so the filter keeps its operands and its traps.
        rel = Rel::Filter {
            input: Box::new(rel),
            pred,
        };
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
        binder.regexes.into_inner(),
        wide_outs,
        binder.model_refs.into_inner(),
        binder.minted_lanes.into_inner(),
    ))
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
    /// Depth of CASE/COALESCE arms being bound. DuckDB's plan-time constant
    /// evaluation SKIPS guarded arms (the coalesce lazy-bind pin: an
    /// untaken `CAST('nope' AS BIGINT)` must not fire), so the
    /// trapping-constant refusals only apply at depth 0 — a guarded
    /// trapping constant stays a lazy runtime question on both engines.
    in_guarded: std::cell::Cell<u32>,
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
}

/// Decrements `in_guarded` on scope exit, whatever the exit path.
struct GuardScope<'x>(&'x std::cell::Cell<u32>);

impl Drop for GuardScope<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}
