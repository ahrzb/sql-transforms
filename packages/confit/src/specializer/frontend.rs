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

use super::exec::{ExternImpl, ScalarVal};
use super::fold::fold;
use super::ir::{BinOp, CmpPred, Col, Lit, NumOp1, StrOp2, StrOp2i, StrOp3, TrimSide, Ty};
use super::sig::{self, ArgTy, NullArg, Ret, Sig};
use super::plan::{
    ArithOp, CompareGrid, JoinKind, JoinSpec, Rel, SExpr, SKind, StaticTable, bind_foldable,
    may_trap,
};

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

/// What the TASK-92 resolution head hands a table-resolved arm.
enum SigArgs {
    /// A bare-NULL argument made the whole call NULL of the result type.
    Null(SExpr),
    /// Typed (checked, promoted) arguments plus the resolved result type.
    Bound(Vec<SExpr>, Ty),
}

/// Every name the builtin catalogue in [`Binder::function`] claims.
///
/// A declared UDF may not take one of these. `function()` matches the
/// catalogue on `name.as_str()` BEFORE it ever consults `find_tree` /
/// `find_udf` (they live in the `_` arm), so the builtin would win here
/// silently — while DuckDB, which lets a registered function shadow its own
/// builtin, binds the UDF. Same SQL, two engines, different answers: the
/// contract's forbidden third mode.
///
/// Refusing is the only resolution that is right at every arity. Matching
/// DuckDB by resolving UDFs first would fix the common case and break
/// another: DuckDB overload-resolves, so `least(a, b, c)` against a
/// two-argument UDF falls back to its builtin, where a UDF-first binder
/// refuses on arity. One divergence traded for another.
///
/// `builtin_names_match_the_catalogue` re-derives this list from the match
/// itself, so a new arm cannot silently escape the guard.
pub const BUILTIN_NAMES: &[&str] = &[
    "abs", "add", "any_value", "array_extract", "array_slice", "ascii", "avg",
    "bit_length", "cbrt", "ceil", "ceiling", "char_length", "character_length",
    "coalesce", "concat", "concat_ws", "contains", "cos", "count",
    "damerau_levenshtein", "divide", "editdist3", "ends_with", "exp", "fdiv",
    "first", "floor", "fmod", "geomean", "greatest", "hamming", "instr",
    "jaccard", "last", "lcase", "least", "len", "length", "levenshtein",
    "list_extract", "list_slice", "ln", "log", "log10", "log2", "lower",
    "lpad", "ltrim", "max", "min", "mismatches", "mod", "multiply",
    "nextafter", "nullif", "ord", "pi", "position", "pow", "power", "prefix",
    "product", "regexp_extract", "regexp_extract_all", "regexp_full_match",
    "regexp_matches", "regexp_replace", "regexp_split_to_array", "repeat",
    "replace", "reverse", "round", "rpad", "rtrim", "sin", "sqrt",
    "starts_with", "string_agg", "strip_accents", "strlen", "strpos",
    "struct_extract", "subtract", "suffix", "sum", "tan", "translate",
    "trunc", "ucase", "unicode", "upper", "xor",
];

/// Whether a call-site name is claimed by the builtin catalogue. Matching is
/// ASCII-case-insensitive, like `function()`'s own lowercasing of the name.
pub fn is_builtin(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    BUILTIN_NAMES.contains(&lower.as_str())
}

/// Refuse every `Query` clause this engine does not implement.
///
/// Destructured EXHAUSTIVELY on purpose — no `..` pattern. When sqlparser
/// grows a clause this stops compiling, instead of silently ignoring it. That
/// silence is how `FETCH FIRST n ROWS ONLY` came to be parsed and dropped
/// while its synonym `LIMIT n` was refused by name (TASK-69): an ignored
/// clause is a wrong ANSWER, not a missing feature.
fn refuse_unhandled_query(query: &sqlparser::ast::Query) -> Result<(), PrepareError> {
    let sqlparser::ast::Query {
        // checked by the caller
        with: _,
        body: _,
        order_by: _,
        limit_clause: _,
        // not implemented — each is a silent wrong answer if ignored
        fetch,
        locks,
        for_clause,
        settings,
        format_clause,
        pipe_operators,
    } = query;
    if fetch.is_some() {
        return Err(unsup("FETCH FIRST/NEXT (row limit)"));
    }
    if !locks.is_empty() {
        return Err(unsup("FOR UPDATE / FOR SHARE"));
    }
    if for_clause.is_some() {
        return Err(unsup("FOR XML / FOR JSON"));
    }
    if settings.is_some() {
        return Err(unsup("SETTINGS"));
    }
    if format_clause.is_some() {
        return Err(unsup("FORMAT"));
    }
    if !pipe_operators.is_empty() {
        return Err(unsup("pipe operators"));
    }
    Ok(())
}

/// Refuse every `Select` clause this engine does not implement. Exhaustively
/// destructured for the same reason as [`refuse_unhandled_query`] — `QUALIFY`
/// was parsed and dropped, so a dedupe-to-latest query emitted every row and
/// `shape='map'` still certified it (TASK-69).
fn refuse_unhandled_select(select: &sqlparser::ast::Select) -> Result<(), PrepareError> {
    let sqlparser::ast::Select {
        // checked by the caller
        distinct: _,
        projection: _,
        from: _,
        selection: _,
        group_by: _,
        having: _,
        // positional markers, meaningless without the clause they order
        select_token: _,
        top_before_distinct: _,
        window_before_qualify: _,
        // not implemented — each is a silent wrong answer if ignored
        optimizer_hints,
        select_modifiers,
        top,
        exclude,
        into,
        lateral_views,
        prewhere,
        connect_by,
        cluster_by,
        distribute_by,
        sort_by,
        named_window,
        qualify,
        value_table_mode,
        flavor,
    } = select;
    // FROM-first (`FROM t SELECT *`, `FROM t`) is only a spelling, but an
    // untested spelling: refuse by name rather than assume it binds the same.
    if !matches!(flavor, sqlparser::ast::SelectFlavor::Standard) {
        return Err(unsup("FROM-first SELECT"));
    }
    if qualify.is_some() {
        return Err(unsup("QUALIFY"));
    }
    if top.is_some() {
        return Err(unsup("SELECT TOP (row limit)"));
    }
    if prewhere.is_some() {
        return Err(unsup("PREWHERE"));
    }
    if exclude.is_some() {
        return Err(unsup("EXCLUDE"));
    }
    if into.is_some() {
        return Err(unsup("SELECT INTO"));
    }
    if !lateral_views.is_empty() {
        return Err(unsup("LATERAL VIEW"));
    }
    if !connect_by.is_empty() {
        return Err(unsup("CONNECT BY"));
    }
    if !cluster_by.is_empty() {
        return Err(unsup("CLUSTER BY"));
    }
    if !distribute_by.is_empty() {
        return Err(unsup("DISTRIBUTE BY"));
    }
    if !sort_by.is_empty() {
        return Err(unsup("SORT BY"));
    }
    if !named_window.is_empty() {
        return Err(unsup("WINDOW (named window definitions)"));
    }
    if !optimizer_hints.is_empty() {
        return Err(unsup("optimizer hints"));
    }
    if select_modifiers.is_some() {
        return Err(unsup("select modifiers"));
    }
    if value_table_mode.is_some() {
        return Err(unsup("AS STRUCT / AS VALUE"));
    }
    Ok(())
}

/// SQL text + the dynamic table's name/schema + the static-table catalog
/// (+ declared UDF externs) -> bound relational tree, the equi-joins in
/// FROM order, the derived output schema, and the width-k UDF output
/// fields (DRAFT-22).
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
    // row (TASK-69). Both helpers walk every field of their AST node, so
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
    // them into ONE field — `list | None` for an unnamed extern (DRAFT-22
    // step 2), a struct keyed by the declared names for a named one
    // (slice 5).
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
        // unnest(udf(...)) expands IN PLACE to one plain scalar column per
        // declared field, named by the field names, alias ignored — the
        // oracle's own expansion (measured). Every column reads a lane of
        // the ONE shared ecall site.
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
                // bare column names (wave-B pins).
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
                // Lateral aliases (wave-5 pins): later items and WHERE may
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
        binder.in_filter.set(true);
        let bound = binder.expr(pred);
        binder.in_filter.set(false);
        let pred = fold(bool_context(bound?, "WHERE predicate")?);
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
    ))
}

/// Parse and bind the FROM clause: the dynamic table, then zero or more
/// equi-joins to static tables. Returns the fully-scoped binder (every join
/// visible) and the join specs in FROM order.
#[allow(clippy::too_many_arguments)]
fn bind_from<'a>(
    select: &sqlparser::ast::Select,
    this_name: &str,
    in_cols: &'a [Col],
    opaque: &'a [(usize, String)],
    structs: &'a [super::plan::StructCol],
    statics: &'a [StaticTable],
    many: bool,
    udfs: &'a [super::ir::ExternSpec],
    models: &'a [super::plan::ModelTable],
    // TASK-101: installed at construction so JOIN keys and residuals,
    // which bind in here, fold pure externs exactly like the projection.
    bind_eval: &'a [ExternImpl],
) -> Result<(Binder<'a>, Vec<JoinSpec>, Option<SqlExpr>), PrepareError> {
    // Plain scalar columns occupy in_cols[..n_plain]; struct leaf lanes
    // follow and are addressable ONLY through their struct paths.
    let n_plain = in_cols.len() - structs.iter().map(|s| s.leaf_count()).sum::<usize>();
    let Some((table, comma_rels)) = select.from.split_first() else {
        return Err(unsup("FROM-less SELECT"));
    };
    let dyn_name = match &table.relation {
        TableFactor::Table { name, alias, .. } => {
            let n = name.to_string();
            // The engine's registry is SCHEMA-LESS: a single schema
            // qualifier is accepted when the table part matches the
            // registered bare name (TASK-55, amends the wave-5 main.-only
            // rule — DuckDB's schema-existence errors are unknowable to a
            // schema-less registry; documented in known-limitations.md §5).
            let bare = match n.rsplit_once('.') {
                Some((_, t)) => t,
                None => &n,
            };
            if !bare.eq_ignore_ascii_case(this_name) {
                return Err(unsup(format!(
                    "table '{n}' as the driving relation (must be the dynamic table '{this_name}')"
                )));
            }
            let n = bare.to_string();
            match alias {
                // Measured: an alias REPLACES the original name entirely
                // (qualified refs through the original are binder errors in
                // DuckDB) — making the alias the binder's this_name gives
                // exactly that scoping.
                Some(a) if a.columns.is_empty() => (a.name.value.clone(), None),
                // `t AS u(x, y)`: a PARTIAL list is legal (prefix rename,
                // remaining columns keep their names); too many names is
                // the pinned bind error; old names are fully shadowed
                // (wave-5 pins).
                Some(a) => {
                    let model_cols = n_plain + opaque.len() + structs.len();
                    if a.columns.len() > model_cols {
                        return Err(PrepareError::Bind(format!(
                            "table \"{n}\" has {model_cols} columns available but {} columns specified",
                            a.columns.len()
                        )));
                    }
                    // The rename is positional over the FULL model; a name
                    // landing on an opaque/struct column has no plain lane
                    // to rename.
                    if let Some((_, oname)) = opaque.iter().find(|(p, _)| *p < a.columns.len()) {
                        return Err(unsup(format!(
                            "row column '{oname}' has a non-scalar type"
                        )));
                    }
                    if let Some(sc) = structs.iter().find(|s| s.pos < a.columns.len()) {
                        return Err(unsup(format!(
                            "column-list alias over struct column '{}'",
                            sc.name
                        )));
                    }
                    let mut renamed = in_cols.to_vec();
                    for (c, def) in renamed.iter_mut().zip(&a.columns) {
                        c.name = def.name.value.clone();
                    }
                    (a.name.value.clone(), Some(renamed))
                }
                None => (n, None),
            }
        }
        other => return Err(unsup(format!("FROM {other}"))),
    };
    let (dyn_name, renamed_cols) = dyn_name;

    let mut binder = Binder {
        this_name: dyn_name,
        in_cols: match renamed_cols {
            Some(v) => std::borrow::Cow::Owned(v),
            None => std::borrow::Cow::Borrowed(in_cols),
        },
        n_plain,
        opaque,
        structs,
        joins: Vec::new(),
        select_aliases: select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.clone()),
                _ => None,
            })
            .collect(),
        bound_aliases: std::cell::RefCell::new(Vec::new()),
        regexes: std::cell::RefCell::new(Vec::new()),
        udfs,
        bind_eval,
        models,
        model_refs: std::cell::RefCell::new(Vec::new()),
        sites: std::cell::Cell::new(0),
        extern_sites: std::cell::RefCell::new(Vec::new()),
        in_filter: std::cell::Cell::new(false),
        in_guarded: std::cell::Cell::new(0),
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
            if !many {
                return Err(unsup("joining the dynamic table to itself"));
            }
            if !opaque.is_empty() || !structs.is_empty() {
                return Err(unsup(
                    "self-join over a row model with non-scalar columns",
                ));
            }
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
            // Stage-B self-join: the build side is the BATCH — a keyless
            // batchmap (built per call) with the WHOLE ON as residual.
            let on = match constraint {
                JoinConstraint::On(e) => Some(e),
                JoinConstraint::Using(_) | JoinConstraint::Natural => {
                    return Err(unsup(
                        "self-join USING/NATURAL (stage-B follow-up; use ON)",
                    ))
                }
                JoinConstraint::None => {
                    return Err(unsup("JOIN without ON (cross join)"))
                }
            };
            let n_batch = binder.n_plain as u32;
            binder.joins.push(ScopeJoin {
                name: scope_name.clone(),
                table: std::borrow::Cow::Owned(StaticTable {
                    name: scope_name,
                    cols: in_cols[..binder.n_plain].to_vec(),
                }),
                kind,
                key_cols: Vec::new(),
                val_cols: (0..n_batch).collect(),
                keys: Vec::new(),
                using: false,
            });
            let residual = match on {
                None => None,
                Some(e) => Some(fold(bool_context(binder.expr(e)?, "JOIN condition")?)),
            };
            specs.push(JoinSpec {
                table: 0,
                batch: true,
                kind,
                keys: Vec::new(),
                key_cols: Vec::new(),
                key_indf: Vec::new(),
                val_cols: (0..n_batch).collect(),
                residual,
            });
            continue;
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
        let (keys, key_cols, key_indf, residual_raw, using) = match constraint {
            JoinConstraint::On(e) => {
                let (keys, key_cols, key_indf, res) = bind_on(&binder, st, &scope_name, e)?;
                (keys, key_cols, key_indf, res, false)
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
                let n = key_cols.len();
                (keys, key_cols, vec![false; n], Vec::new(), true)
            }
            // NATURAL = USING(all common column names), case-insensitive,
            // merged output like USING with the LEFT spelling; NO common
            // columns is a hard error, never a cross product (wave-5 pins).
            JoinConstraint::Natural => {
                let mut keys = Vec::new();
                let mut key_cols = Vec::new();
                for (i, c) in st.cols.iter().enumerate() {
                    let Ok(key) = binder.column(&c.name) else {
                        continue;
                    };
                    keys.push(promote_key(fold(key), st, i as u32)?);
                    key_cols.push(i as u32);
                }
                if key_cols.is_empty() {
                    return Err(PrepareError::Bind(
                        "No columns found to join on in NATURAL JOIN.\n\
                         Use CROSS JOIN if you intended for this to be a cross-product."
                            .into(),
                    ));
                }
                let n = key_cols.len();
                (keys, key_cols, vec![false; n], Vec::new(), true)
            }
            JoinConstraint::None => return Err(unsup("JOIN without ON (cross join)")),
        };
        let val_cols: Vec<u32> = (0..st.cols.len() as u32)
            .filter(|c| !key_cols.contains(c))
            .collect();

        binder.joins.push(ScopeJoin {
            name: scope_name,
            table: std::borrow::Cow::Borrowed(st),
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
            batch: false,
            kind,
            keys,
            key_cols,
            key_indf,
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
            if !many {
                return Err(unsup("joining the dynamic table to itself"));
            }
            if !opaque.is_empty() || !structs.is_empty() {
                return Err(unsup(
                    "self-join over a row model with non-scalar columns",
                ));
            }
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
            // Comma self-join = pure cross against the batch; equi
            // conjuncts stay in WHERE (cross-then-filter is bit-identical
            // under multiplicity — measured, pins-stageB).
            let n_batch = binder.n_plain as u32;
            binder.joins.push(ScopeJoin {
                name: scope_name.clone(),
                table: std::borrow::Cow::Owned(StaticTable {
                    name: scope_name,
                    cols: in_cols[..binder.n_plain].to_vec(),
                }),
                kind: JoinKind::Inner,
                key_cols: Vec::new(),
                val_cols: (0..n_batch).collect(),
                keys: Vec::new(),
                using: false,
            });
            specs.push(JoinSpec {
                table: 0,
                batch: true,
                kind: JoinKind::Inner,
                keys: Vec::new(),
                key_cols: Vec::new(),
                key_indf: Vec::new(),
                val_cols: (0..n_batch).collect(),
                residual: None,
            });
            continue;
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
            table: std::borrow::Cow::Borrowed(st),
            kind: JoinKind::Inner,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
            keys: keys.clone(),
            using: false,
        });
        let n = key_cols.len();
        specs.push(JoinSpec {
            table: table_idx,
            batch: false,
            kind: JoinKind::Inner,
            keys,
            key_cols,
            key_indf: vec![false; n],
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
    // Schema-less registry (TASK-55): an exact registered-name match wins;
    // otherwise a single-qualifier SQL name (`s1.t1`) matches a registered
    // bare `t1`. Ambiguity stays an error.
    let bare = raw_name.rsplit_once('.').map(|(_, t)| t);
    let mut table_idx = None;
    for (i, st) in statics.iter().enumerate() {
        let hit = st.name.eq_ignore_ascii_case(raw_name)
            || bare.is_some_and(|b| st.name.eq_ignore_ascii_case(b));
        if hit {
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
        let (mut right, mut left, mut known) = (false, false, true);
        scan_residual(&bound, j, &mut right, &mut left, &mut known);
        let total = !may_trap(&bound);
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

/// Bind a JOIN ... ON condition. Equalities (and IS NOT DISTINCT FROM)
/// pairing a dynamic-side expression with a static column become probe
/// keys; every OTHER conjunct (non-equalities, constant equalities,
/// both-sides-static equalities) is returned raw for residual binding once
/// the join is in scope (wave-4: `match = key_hit AND residual`).
#[allow(clippy::type_complexity)]
fn bind_on<'e>(
    binder: &Binder<'_>,
    st: &StaticTable,
    scope_name: &str,
    on: &'e SqlExpr,
) -> Result<(Vec<SExpr>, Vec<u32>, Vec<bool>, Vec<&'e SqlExpr>), PrepareError> {
    let mut conjuncts = Vec::new();
    collect_conjuncts(on, &mut conjuncts);
    let mut keys = Vec::new();
    let mut key_cols = Vec::new();
    let mut key_indf = Vec::new();
    let mut residual = Vec::new();
    for c in conjuncts {
        let (left, right, indf) = match c {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => (left, right, false),
            // IS NOT DISTINCT FROM: NULL is an ordinary key value — NULL
            // joins NULL (DRAFT-22's params-join contract).
            SqlExpr::IsNotDistinctFrom(left, right) => (left, right, true),
            _ => {
                residual.push(c);
                continue;
            }
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
        key_indf.push(indf);
    }
    Ok((keys, key_cols, key_indf, residual))
}

/// Promote a dynamic-side key expression to the map's key type.
fn promote_key(key: SExpr, st: &StaticTable, col: u32) -> Result<SExpr, PrepareError> {
    let col_ty = st.cols[col as usize].ty.ty;
    match (key.ty, col_ty) {
        (a, b) if a == b => Ok(key),
        // Integer widths share the key lane; the map stores i64 bits.
        (a, b) if a.is_int() && b.is_int() => Ok(key),
        (a, Ty::F64) if a.is_int() => Ok(promote_f64(key)),
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

/// One traversal answering the wave-4 residual SCOPE questions about a bound
/// ON residual: `right`/`left` — does it reference THIS join's columns / any
/// other scope; `known` — was every node classifiable at all, since an
/// unrecognised node's children are never visited and so `left`/`right` are
/// not trustworthy past it.
///
/// Trap-freeness is deliberately NOT computed here — it is
/// [`plan::may_trap`], shared with Kleene lowering so the two cannot drift
/// (TASK-74/75). Acceptance rule at the call site:
/// `!may_trap(e) || (left && right && known)` — measured: DuckDB scan-pushes
/// single-side residuals (eager trap timing) but evaluates both-sides
/// residuals per candidate pair, which our hit-guarded lowering matches.
fn scan_residual(e: &SExpr, j: u32, right: &mut bool, left: &mut bool, known: &mut bool) {
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
        SKind::Cmp { a, b, .. }
        | SKind::And { a, b }
        | SKind::Or { a, b }
        | SKind::Arith { a, b, .. } => {
            scan_residual(a, j, right, left, known);
            scan_residual(b, j, right, left, known);
        }
        SKind::Not(a)
        | SKind::IsNull { inner: a, .. }
        | SKind::IntToFloat(a)
        | SKind::IntToFloat32(a) => {
            scan_residual(a, j, right, left, known);
        }
        SKind::Case { arms, default } => {
            for (c, r) in arms {
                scan_residual(c, j, right, left, known);
                scan_residual(r, j, right, left, known);
            }
            if let Some(d) = default {
                scan_residual(d, j, right, left, known);
            }
        }
        // Anything else: not classifiable — the caller must reject rather
        // than risk the permissive both-sides path on a wrong guess.
        _ => *known = false,
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
    table: std::borrow::Cow<'a, StaticTable>,
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
    /// The dynamic table's columns AS THE BINDER SEES THEM: borrowed
    /// normally; an owned renamed copy under `t AS u(x, y)` (wave-5 pins —
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
    /// Struct row columns flattened to leaf lanes (TASK-56).
    structs: &'a [super::plan::StructCol],
    joins: Vec<ScopeJoin<'a>>,
    /// All SELECT-list aliases (wave-5 pins: DuckDB's lateral aliases — a
    /// later item or WHERE may reference an earlier alias; the REAL column
    /// wins on a name clash; a forward reference is the pinned bind error).
    select_aliases: Vec<String>,
    /// Aliases already bound this pass, in SELECT order (frontend() fills
    /// this as it walks the projection; RefCell keeps `expr(&self)` intact).
    bound_aliases: std::cell::RefCell<Vec<(String, SExpr)>>,
    /// Program regex table under construction (wave-B); indices are baked
    /// into ReMatch/ReExtract/ReReplace nodes.
    regexes: std::cell::RefCell<Vec<super::ir::ReSpec>>,
    /// Declared UDF externs (DRAFT-22): an unknown function matching one
    /// binds as an opaque ecall instead of the named refusal.
    udfs: &'a [super::ir::ExternSpec],
    /// TASK-101: the callables themselves, decl-order-aligned with `udfs`,
    /// for the bind-time fold of pure externs over constant args. Empty
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
    /// (TASK-63) — the confit twin of DuckDB's common-subexpression
    /// elimination, which is what keeps call counts equal on both paths.
    extern_sites: std::cell::RefCell<Vec<(sqlparser::ast::Function, u32)>>,
    /// True while binding the WHERE predicate. DuckDB's FILTER optimizer
    /// folds a constant dead range (`BETWEEN lo AND hi`, lo > hi) to FALSE
    /// without evaluating the operand, but evaluates the same expression in
    /// a projection — measured both ways (TASK-87 face C), so the fold is
    /// context-gated on this flag.
    in_filter: std::cell::Cell<bool>,
    /// Depth of CASE/COALESCE arms being bound. DuckDB's plan-time constant
    /// evaluation SKIPS guarded arms (the coalesce lazy-bind pin: an
    /// untaken `CAST('nope' AS BIGINT)` must not fire), so the TASK-87
    /// trapping-constant refusals only apply at depth 0 — a guarded
    /// trapping constant stays a lazy runtime question on both engines.
    in_guarded: std::cell::Cell<u32>,
}

/// Decrements `in_guarded` on scope exit, whatever the exit path.
struct GuardScope<'x>(&'x std::cell::Cell<u32>);

impl Drop for GuardScope<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

/// The bound subject of a regex op must be VARCHAR (no implicit casts —
/// wave-B pins; DuckDB binder errors name the function).
fn str_only(name: &str, e: SExpr) -> Result<SExpr, PrepareError> {
    if e.ty != Ty::Str {
        return Err(PrepareError::Bind(format!(
            "no function matches {name}({})",
            e.ty.name()
        )));
    }
    Ok(e)
}

/// `''` for non-NULL subjects, NULL for NULL ones (the pinned NULL-group
/// result of regexp_extract).
fn empty_for_nonnull(subject: SExpr) -> SExpr {
    if !subject.nullable {
        return lit_str("");
    }
    let is_null = SExpr {
        kind: SKind::IsNull {
            negated: false,
            inner: Box::new(subject),
        },
        ty: Ty::I1,
        nullable: false,
    };
    SExpr {
        kind: SKind::Case {
            arms: vec![(is_null, null_of(Ty::Str))],
            default: Some(Box::new(lit_str(""))),
        },
        ty: Ty::Str,
        nullable: true,
    }
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

/// DuckDB's boundary rename for duplicate output names (wave-5 pins,
/// pins-wave5/dup-names-client-contract.json): left-to-right after star
/// expansion, first occurrence keeps its name, later ones get
/// `<own-original-case-name>_N` with the smallest free N; the collision
/// check is case-insensitive and covers generated candidates too
/// (id,ID -> id,ID_1; id,id,id_1 -> id,id_1,id_1_1). Identical to what
/// DuckDB itself does at every subquery/CTE/CTAS boundary and in .df().
fn dedup_output_names(cols: &mut [Col]) {
    let mut seen = std::collections::HashSet::new();
    for c in cols {
        if seen.insert(c.name.to_lowercase()) {
            continue;
        }
        let mut n = 1;
        loop {
            let cand = format!("{}_{}", c.name, n);
            if seen.insert(cand.to_lowercase()) {
                c.name = cand;
                break;
            }
            n += 1;
        }
    }
}

/// A star-expansion lane: a real scalar lane, or an opaque row-model
/// column (kept under its ORIGINAL name so EXCLUDE / REPLACE / name
/// filters can still remove it). One surviving to the output is the
/// named unsupported error — deferred so COLUMNS('re') can filter first.
enum StarLane {
    Real(SExpr),
    Opaque(String),
}

fn finalize_star(cols: Vec<(String, StarLane)>) -> Result<Vec<(String, SExpr)>, PrepareError> {
    cols.into_iter()
        .map(|(n, l)| match l {
            StarLane::Real(e) => Ok((n, e)),
            StarLane::Opaque(orig) => Err(unsup(format!(
                "row column '{orig}' has a non-scalar type"
            ))),
        })
        .collect()
}

/// Bind-time LIKE over column NAMES for star filters: `%`/`_` over
/// codepoints, no ESCAPE (an ESCAPE clause after a star filter does not
/// parse), ci = ILIKE's Unicode case fold.
fn like_match(s: &str, p: &str, ci: bool) -> bool {
    let norm = |x: &str| {
        if ci {
            x.to_lowercase()
        } else {
            x.to_string()
        }
    };
    let s: Vec<char> = norm(s).chars().collect();
    let p: Vec<char> = norm(p).chars().collect();
    let (mut si, mut pi) = (0usize, 0usize);
    let (mut bt_p, mut bt_s) = (usize::MAX, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == '%' {
            bt_p = pi;
            pi += 1;
            bt_s = si;
        } else if pi < p.len() && (p[pi] == '_' || p[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if bt_p != usize::MAX {
            bt_s += 1;
            si = bt_s;
            pi = bt_p + 1;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '%' {
        pi += 1;
    }
    pi == p.len()
}

/// A star name filter, decoded from the ILIKE slot (rewrite.rs encodes
/// LIKE / NOT LIKE / GLOB / NOT ILIKE there with a \u{1} marker; an
/// unmarked pattern is a genuine * ILIKE).
enum StarFilter {
    Like { ci: bool, neg: bool },
    Glob,
    /// Wave-B pins: positive = unanchored RE2 SEARCH over names; NOT =
    /// NOT full-match — independent predicates, never complements.
    Similar { neg: bool },
}

/// Decoded star filter + any EXCLUDE entries the rewrite absorbed into the
/// marker (sqlparser parses ILIKE and EXCLUDE as mutually exclusive).
struct DecodedFilter {
    op: StarFilter,
    pat: String,
    excludes: Vec<(Option<String>, String)>,
}

fn decode_star_filter(pattern: &str) -> DecodedFilter {
    if let Some(rest) = pattern.strip_prefix('\u{1}') {
        for (code, op) in [
            ("L:", StarFilter::Like { ci: false, neg: false }),
            ("NL:", StarFilter::Like { ci: false, neg: true }),
            ("NI:", StarFilter::Like { ci: true, neg: true }),
            ("G:", StarFilter::Glob),
            ("S:", StarFilter::Similar { neg: false }),
            ("NS:", StarFilter::Similar { neg: true }),
        ] {
            let Some(body) = rest.strip_prefix(code) else {
                continue;
            };
            let (exc, pat) = body.split_once('\u{2}').unwrap_or(("", body));
            let excludes = exc
                .split(',')
                .filter(|e| !e.is_empty())
                .map(|e| match e.split_once('.') {
                    Some((t, c)) => (Some(t.to_string()), c.to_string()),
                    None => (None, e.to_string()),
                })
                .collect();
            return DecodedFilter {
                op,
                pat: pat.to_string(),
                excludes,
            };
        }
    }
    DecodedFilter {
        op: StarFilter::Like { ci: true, neg: false },
        pat: pattern.to_string(),
        excludes: Vec::new(),
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
        let (mut any_f64, mut any_num) = (false, false);
        for e in exprs {
            if let Some(b) = self.expr_or_null(e)? {
                match b.ty {
                    Ty::F64 => (any_f64, any_num) = (true, true),
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => any_num = true,
                    Ty::Str | Ty::I1 => {}
                }
            }
        }
        // Wave-5 pins: mixing casts the string/bool side to the NUMERIC
        // side (strings numerically with half-away-from-zero rounding to
        // ints; bool -> 0/1; non-numeric strings are DuckDB Conversion
        // Errors). Only literals convert at bind — a string/bool COLUMN
        // would need runtime cast traps and stays unsupported.
        let mut owned: Vec<SqlExpr> = Vec::with_capacity(exprs.len());
        for e in exprs {
            let b = self.expr_or_null(e)?;
            let needs_cast =
                any_num && b.as_ref().is_some_and(|b| matches!(b.ty, Ty::Str | Ty::I1));
            if !needs_cast {
                owned.push((*e).clone());
                continue;
            }
            let lit = match b.map(|b| b.kind) {
                Some(SKind::Lit(Lit::Str(s))) => {
                    // Non-numeric strings convert (and fail) at EXECUTION
                    // time in DuckDB — an empty input succeeds — so a
                    // bind-time error would be over-eager; stay clean.
                    let Ok(x) = s.trim().parse::<f64>() else {
                        return Err(unsup(
                            "BETWEEN/IN mixing non-numeric string literals with numbers \
                             (exec-time conversion)",
                        ));
                    };
                    if any_f64 {
                        s.trim().to_string()
                    } else {
                        let r = if x >= 0.0 {
                            (x + 0.5).floor()
                        } else {
                            (x - 0.5).ceil()
                        };
                        if r < i64::MIN as f64 || r > i64::MAX as f64 {
                            return Err(unsup(
                                "BETWEEN/IN mixing out-of-range string literals with numbers \
                                 (exec-time conversion)",
                            ));
                        }
                        format!("{}", r as i64)
                    }
                }
                Some(SKind::Lit(Lit::I1(v))) => format!("{}", v as i64),
                _ => {
                    return Err(unsup(
                        "BETWEEN/IN mixing strings or booleans with numbers \
                         (non-literal side needs exec-time cast semantics)",
                    ))
                }
            };
            owned.push(SqlExpr::Value(sqlparser::ast::ValueWithSpan {
                value: SqlValue::Number(lit, false),
                span: sqlparser::tokenizer::Span::empty(),
            }));
        }
        Ok(owned
            .into_iter()
            .map(|e| {
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

    /// Expand `*` / `tbl.*` per DuckDB's measured semantics (1.5.5, wave-5
    /// pins): FROM order, declared column order within a table; grammar
    /// order EXCLUDE -> REPLACE -> RENAME with the name filter applying
    /// after EXCLUDE only. Duplicate output names across the star survive
    /// here and are renamed by [`dedup_output_names`] (DuckDB's own
    /// boundary-rename contract).
    fn expand_star(
        &self,
        qualifier: Option<&str>,
        opts: &sqlparser::ast::WildcardAdditionalOptions,
    ) -> Result<Vec<(String, SExpr)>, PrepareError> {
        finalize_star(self.expand_star_lanes(qualifier, opts)?)
    }

    fn expand_star_lanes(
        &self,
        qualifier: Option<&str>,
        opts: &sqlparser::ast::WildcardAdditionalOptions,
    ) -> Result<Vec<(String, StarLane)>, PrepareError> {
        use sqlparser::ast::{ExcludeSelectItem, RenameSelectItem};
        if opts.opt_except.is_some() {
            return Err(unsup("SELECT * EXCEPT"));
        }
        // EXCLUDE entries: (optional table qualifier, column name).
        fn exclude_name(
            n: &sqlparser::ast::ObjectName,
        ) -> Result<(Option<&str>, &str), PrepareError> {
            fn ident(p: &sqlparser::ast::ObjectNamePart) -> Option<&str> {
                p.as_ident().map(|i| i.value.as_str())
            }
            match n.0.as_slice() {
                [part] => ident(part)
                    .map(|c| (None, c))
                    .ok_or_else(|| unsup("EXCLUDE list entry form")),
                [t, part] => match (ident(t), ident(part)) {
                    (Some(t), Some(c)) => Ok((Some(t), c)),
                    _ => Err(unsup("EXCLUDE list entry form")),
                },
                _ => Err(unsup("EXCLUDE list entry form")),
            }
        }
        let mut exclude: Vec<(Option<String>, String)> = match &opts.opt_exclude {
            None => Vec::new(),
            Some(ExcludeSelectItem::Single(id)) => vec![exclude_name(id)?],
            Some(ExcludeSelectItem::Multiple(ids)) => ids
                .iter()
                .map(exclude_name)
                .collect::<Result<Vec<_>, _>>()?,
        }
        .into_iter()
        .map(|(t, c)| (t.map(str::to_string), c.to_string()))
        .collect();
        // Name filter, decoded from the ILIKE slot (rewrite.rs) — it may
        // carry EXCLUDE entries the rewrite absorbed.
        let filter = opts
            .opt_ilike
            .as_ref()
            .map(|il| decode_star_filter(&il.pattern));
        if let Some(f) = &filter {
            exclude.extend(f.excludes.iter().cloned());
        }
        for (i, (_, a)) in exclude.iter().enumerate() {
            if exclude[..i].iter().any(|(_, b)| b.eq_ignore_ascii_case(a)) {
                // DuckDB rejects this at parse; ours surfaces at bind.
                return Err(PrepareError::Bind(format!(
                    "duplicate entry \"{a}\" in EXCLUDE list"
                )));
            }
        }
        let excluded_lists_conflict = |list: &str, name: &str| -> Result<(), PrepareError> {
            if exclude.iter().any(|(_, e)| e.eq_ignore_ascii_case(name)) {
                // DuckDB: Parser Error — same clean class via Parse.
                return Err(PrepareError::Parse(format!(
                    "Column \"{name}\" cannot occur in both EXCLUDE and {list} list"
                )));
            }
            Ok(())
        };
        if filter.is_some() && opts.opt_replace.is_some() {
            return Err(PrepareError::Bind(
                "Replace list cannot be combined with a filtering operation".into(),
            ));
        }
        if filter.is_some() && opts.opt_rename.is_some() {
            return Err(PrepareError::Bind(
                "Rename list cannot be combined with a filtering operation".into(),
            ));
        }

        // (origin table, output name, lane) — the origin drives qualified
        // EXCLUDE; it is dropped on return.
        let mut cols: Vec<(String, String, StarLane)> = Vec::new();
        let mut matched = false;
        if qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&self.this_name)) {
            matched = true;
            // Interleave scalar lanes with opaque and struct columns back
            // into MODEL order (their positions are model positions;
            // scalars fill the rest in order). Struct columns expand like
            // opaque ones under a TABLE star: DuckDB would output the
            // whole struct — non-scalar unless EXCLUDEd/REPLACEd.
            let mut scalars = self.in_cols[..self.n_plain].iter().enumerate();
            for pos in 0..self.n_plain + self.opaque.len() + self.structs.len() {
                if let Some((_, oname)) = self.opaque.iter().find(|(p, _)| *p == pos) {
                    cols.push((
                        self.this_name.clone(),
                        oname.clone(),
                        StarLane::Opaque(oname.clone()),
                    ));
                } else if let Some(sc) = self.structs.iter().find(|s| s.pos == pos) {
                    cols.push((
                        self.this_name.clone(),
                        sc.name.clone(),
                        StarLane::Opaque(sc.name.clone()),
                    ));
                } else {
                    let (i, c) = scalars.next().expect("scalar count matches positions");
                    cols.push((
                        self.this_name.clone(),
                        c.name.clone(),
                        StarLane::Real(SExpr {
                            kind: SKind::Col(i as u32),
                            ty: c.ty.ty,
                            nullable: c.ty.nullable,
                        }),
                    ));
                }
            }
        }
        // Joined tables expand in FROM order, columns in DECLARED order
        // (measured): value columns as probe lanes, key columns via the
        // dynamic-side reconstruction, USING keys suppressed (merged into
        // the left occurrence).
        for (j, sj) in self.joins.iter().enumerate() {
            if !qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&sj.name)) {
                continue;
            }
            matched = true;
            for (ci, c) in sj.table.cols.iter().enumerate() {
                let ci = ci as u32;
                if let Some(pos) = sj.val_cols.iter().position(|&v| v == ci) {
                    cols.push((
                        sj.name.clone(),
                        c.name.clone(),
                        StarLane::Real(self.static_lane(j, pos)),
                    ));
                } else {
                    let kp = sj
                        .key_cols
                        .iter()
                        .position(|&k| k == ci)
                        .expect("column is key or value");
                    if !sj.using {
                        cols.push((
                            sj.name.clone(),
                            c.name.clone(),
                            StarLane::Real(self.key_lane(j, kp)),
                        ));
                    } else if exclude.iter().any(|(t, e)| {
                        t.as_deref()
                            .is_some_and(|t| t.eq_ignore_ascii_case(&sj.name))
                            && e.eq_ignore_ascii_case(&c.name)
                    }) {
                        // Measured: EXCLUDE (right.key) on a USING join
                        // UNMERGES the column (it reappears at the right
                        // table's position with right values) — not modeled.
                        return Err(unsup(
                            "EXCLUDE of a USING-merged column (DuckDB unmerges it)",
                        ));
                    }
                }
            }
        }
        // Struct-star `a.*` — checked AFTER tables: a table alias with the
        // same name WINS over the struct column (measured, silently).
        if !matched {
            if let Some(q) = qualifier {
                if let Some(sc) = self.structs.iter().find(|s| s.name.eq_ignore_ascii_case(q)) {
                    matched = true;
                    if filter.is_some() || opts.opt_rename.is_some() {
                        return Err(unsup(
                            "struct star with a name filter or RENAME (unpinned)",
                        ));
                    }
                    use super::plan::StructNode;
                    for f in &sc.fields {
                        let lane = match &f.node {
                            StructNode::Leaf(l) => {
                                let c = &self.in_cols[*l as usize];
                                StarLane::Real(SExpr {
                                    kind: SKind::Col(*l),
                                    ty: c.ty.ty,
                                    nullable: c.ty.nullable,
                                })
                            }
                            // Nested-struct / unmappable fields expand as
                            // non-scalar entries: EXCLUDE removes them,
                            // surviving is the named error.
                            _ => StarLane::Opaque(f.name.clone()),
                        };
                        cols.push((sc.name.clone(), f.name.clone(), lane));
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

        for (t, ex) in &exclude {
            let hit = cols.iter().any(|(ct, cn, _)| {
                cn.eq_ignore_ascii_case(ex)
                    && t.as_deref().is_none_or(|t| t.eq_ignore_ascii_case(ct))
            });
            if !hit {
                let disp = match t {
                    Some(t) => format!("{t}.{ex}"),
                    None => ex.to_string(),
                };
                return Err(PrepareError::Bind(format!(
                    "column \"{disp}\" in EXCLUDE list not found in FROM clause"
                )));
            }
        }
        // Unqualified EXCLUDE strips ALL same-named copies; qualified
        // strips one table's (measured).
        cols.retain(|(ct, cn, _)| {
            !exclude.iter().any(|(t, ex)| {
                cn.eq_ignore_ascii_case(ex)
                    && t.as_deref().is_none_or(|t| t.eq_ignore_ascii_case(ct))
            })
        });

        // REPLACE (expr AS col): position and name kept, type may change,
        // expr sees the full original scope (incl. EXCLUDEd columns).
        if let Some(rep) = &opts.opt_replace {
            for (i, it) in rep.items.iter().enumerate() {
                let name = &it.column_name.value;
                if rep.items[..i]
                    .iter()
                    .any(|p| p.column_name.value.eq_ignore_ascii_case(name))
                {
                    return Err(PrepareError::Parse(format!(
                        "Duplicate entry \"{name}\" in REPLACE list"
                    )));
                }
                excluded_lists_conflict("REPLACE", name)?;
                let hits: Vec<usize> = cols
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, cn, _))| cn.eq_ignore_ascii_case(name))
                    .map(|(i, _)| i)
                    .collect();
                match hits[..] {
                    [] => {
                        return Err(PrepareError::Bind(format!(
                            "column \"{name}\" in REPLACE list not found in FROM clause"
                        )))
                    }
                    [pos] => {
                        cols[pos].2 = StarLane::Real(fold(self.expr(&it.expr)?));
                        // The output name takes the REPLACE alias's exact
                        // case (measured on struct-star; the match itself
                        // stays case-insensitive).
                        cols[pos].1 = it.column_name.value.clone();
                    }
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "ambiguous reference to column name \"{name}\" in REPLACE list"
                        )))
                    }
                }
            }
        }

        // RENAME (a AS b): position kept; a NONEXISTENT target is silently
        // ignored and an ambiguous name renames ALL copies (measured);
        // collisions become duplicates for dedup_output_names.
        if let Some(ren) = &opts.opt_rename {
            let items: Vec<&sqlparser::ast::IdentWithAlias> = match ren {
                RenameSelectItem::Single(i) => vec![i],
                RenameSelectItem::Multiple(v) => v.iter().collect(),
            };
            for it in items {
                let name = &it.ident.value;
                excluded_lists_conflict("RENAME", name)?;
                if let Some(rep) = &opts.opt_replace {
                    if rep
                        .items
                        .iter()
                        .any(|p| p.column_name.value.eq_ignore_ascii_case(name))
                    {
                        return Err(PrepareError::Parse(format!(
                            "Column \"{name}\" cannot occur in both REPLACE and RENAME list"
                        )));
                    }
                }
                for c in cols.iter_mut() {
                    if c.1.eq_ignore_ascii_case(name) {
                        c.1 = it.alias.value.clone();
                    }
                }
            }
        }

        // Name filter LAST in our pipeline but semantically after EXCLUDE
        // only (REPLACE/RENAME + filter were rejected above).
        if let Some(DecodedFilter { op, pat, .. }) = &filter {
            let similar = match op {
                // Positive SIMILAR TO searches names UNANCHORED; the NOT
                // form negates a FULL match — measured to not be
                // complements ('a.*': "Weird Name" is in BOTH results).
                StarFilter::Similar { neg } => Some(self.name_regex(pat, *neg)?),
                _ => None,
            };
            cols.retain(|(_, cn, _)| match (op, &similar) {
                (StarFilter::Like { ci, neg }, _) => like_match(cn, pat, *ci) != *neg,
                (StarFilter::Glob, _) => super::exec::interp::duck_glob(cn, pat),
                (StarFilter::Similar { neg }, Some(rx)) => rx.is_match(cn) != *neg,
                (StarFilter::Similar { .. }, None) => unreachable!(),
            });
            if cols.is_empty() {
                return Err(PrepareError::Bind(format!(
                    "star expression with name filter '{pat}' resulted in an empty set of columns"
                )));
            }
        }
        Ok(cols.into_iter().map(|(_, n, l)| (n, l)).collect())
    }

    /// Bind an expression that must have a definite type on its own. A bare
    /// NULL with no adopting context takes DuckDB's SQLNULL default:
    /// INTEGER (m-8 phase 2 — before widths there was no int32 to give it).
    fn expr(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        Ok(self.expr_or_null(e)?.unwrap_or_else(|| null_of(Ty::I32)))
    }

    /// Like `expr`, but a bare NULL literal comes back as `None` for the
    /// caller to type from context.
    fn expr_or_null(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        match e {
            SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => Ok(None),
            SqlExpr::Nested(inner) => self.expr_or_null(inner),
            SqlExpr::Function(f) => {
                if self.nullif_sqlnull(f)? {
                    return Ok(None);
                }
                // TASK-93: struct_extract over struct_pack desugars HERE
                // too, so a bare-NULL field keeps its adopting context.
                if let Some(sub) = self.desugar_struct_extract(f)? {
                    return self.expr_or_null(&sub);
                }
                self.bind_or_sqlnull(e)
            }
            other => {
                if let Some(sub) = self.desugar_struct_field(other)? {
                    return self.expr_or_null(&sub);
                }
                self.bind_or_sqlnull(other)
            }
        }
    }

    /// Bind `e`; if the result is DuckDB's SQLNULL surface — the || collapse
    /// (TASK-102) or a pure-udf field fold to whole-call NULL (TASK-101) —
    /// answer the ADOPTABLE channel instead: `- ((udf(1, NULL)).f1)` is
    /// BIGINT on DuckDB, `abs(s || NULL)` BIGINT too, because SQLNULL
    /// re-promotes by signature. The gate is the SPELLING, not the bound
    /// type: a builtin's whole-call NULL is a COMMITTED int32 there
    /// (measured: `-(ascii(NULL))` is INTEGER and `upper(ascii(NULL))` a
    /// binder error), and within the gated shapes NullOf(int32) has no
    /// other producer (a udf cannot declare an int32 field; || otherwise
    /// types Str).
    fn bind_or_sqlnull(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        let bound = self.bind(e)?;
        let sqlnull = matches!(bound.kind, SKind::NullOf)
            && bound.ty == Ty::I32
            && self.sqlnull_capable_shape(e);
        Ok((!sqlnull).then_some(bound))
    }

    /// The spellings whose bind may yield the SQLNULL surface: `||`, and
    /// field access over a DECLARED udf (dot form and struct_extract).
    fn sqlnull_capable_shape(&self, e: &SqlExpr) -> bool {
        let over_udf = |root: &SqlExpr| {
            let mut b = root;
            while let SqlExpr::Nested(i) = b {
                b = i;
            }
            matches!(b, SqlExpr::Function(f) if self.find_udf(&f.name.to_string()).is_some())
        };
        match e {
            SqlExpr::BinaryOp {
                op: BinaryOperator::StringConcat,
                ..
            } => true,
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                matches!(access_chain.as_slice(), [AccessExpr::Dot(_)]) && over_udf(root)
            }
            SqlExpr::Function(f) => {
                f.name.to_string().eq_ignore_ascii_case("struct_extract")
            }
            _ => false,
        }
    }

    /// Whether `f` is `nullif(NULL, x)` — which propagates DuckDB's
    /// SQLNULL: the whole call is an ADOPTABLE bare NULL, not a committed
    /// int32 (`- nullif(NULL, 1)` is BIGINT there, `nullif(NULL, 1) *
    /// 1::SMALLINT` SMALLINT; fleet 2026-08-13). The second argument still
    /// binds so its own errors fire. Builtin names cannot be UDF-shadowed
    /// (TASK-89), so the name test is enough.
    fn nullif_sqlnull(
        &self,
        f: &sqlparser::ast::Function,
    ) -> Result<bool, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("nullif")
            || f.uses_odbc_syntax
            || !matches!(f.parameters, FunctionArguments::None)
            || f.filter.is_some()
            || f.null_treatment.is_some()
            || f.over.is_some()
            || !f.within_group.is_empty()
        {
            return Ok(false);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(false);
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Ok(false);
        }
        let plain: Vec<&SqlExpr> = list
            .args
            .iter()
            .filter_map(|a| match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e),
                _ => None,
            })
            .collect();
        if plain.len() != 2 || plain.len() != list.args.len() {
            return Ok(false);
        }
        if self.expr_or_null(plain[0])?.is_some() {
            return Ok(false);
        }
        let _second_binds = self.expr_or_null(plain[1])?;
        Ok(true)
    }

    fn bind(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        match e {
            SqlExpr::Identifier(ident) => self.column(&ident.value),
            SqlExpr::CompoundIdentifier(parts) => self.compound(parts),
            SqlExpr::Nested(inner) => self.expr(inner),
            SqlExpr::Value(v) => literal(&v.value),
            // DuckDB puts << >> & | in ONE flat left-associative tier;
            // sqlparser tiers them (& above << >> above |), so 4|1&1 would
            // silently parse as 4|(1&1)=5 where DuckDB computes (4|1)&1=1.
            // Re-associate: in-order traversal of the parsed run recovers
            // source order, then left-fold. User parens are Nested nodes,
            // which the flatten treats as leaves.
            // ~ / !~ are FULL regex match in DuckDB (measured: the binder
            // error names regexp_full_match — NOT the Postgres search).
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::PGRegexMatch,
                right,
            } => self.regex_full_predicate("~", left, right, false),
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::PGRegexNotMatch,
                right,
            } => self.regex_full_predicate("!~", left, right, true),
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
                    acc = Some(self.arith(flat_bitop(o), av, bv, (None, None))?);
                }
                Ok(acc.expect("a flat-bitop run has at least one operator"))
            }
            SqlExpr::BinaryOp { left, op, right } => self.binary(op, left, right),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
                // DuckDB `-x`: lower as (zero) - x, reusing Sub's promotion.
                // The zero must carry the SIGN BIT for floats: IEEE
                // 0.0 - 0.0 is +0.0, so subtracting from +0 erased negative
                // zero everywhere it could arise -- the literal -0.0e0, a
                // runtime negate at x = 0.0, and the sign of infinity after
                // dividing by the result (TASK-80). -0.0 - x is exact IEEE
                // negation for every double. Integers keep 0 - x and its
                // i64::MIN trap, which is DuckDB's own overflow behaviour.
                // sqlparser parses `-a % b` as `-(a % b)`; DuckDB binds
                // `(-a) % b` (its unary minus is tighter than mul/div/mod).
                // The minus distributes over these ops so VALUES agree, but
                // INT32_MIN's literal type doesn't — mirror DuckDB's tree.
                // Explicit parens arrive as Nested and are untouched.
                if let SqlExpr::BinaryOp { left, op, right } = &**expr {
                    if matches!(
                        op,
                        BinaryOperator::Multiply
                            | BinaryOperator::Divide
                            | BinaryOperator::Modulo
                            | BinaryOperator::DuckIntegerDivide
                    ) {
                        let rewritten = SqlExpr::BinaryOp {
                            left: Box::new(SqlExpr::UnaryOp {
                                op: UnaryOperator::Minus,
                                expr: left.clone(),
                            }),
                            op: op.clone(),
                            right: right.clone(),
                        };
                        return self.expr(&rewritten);
                    }
                }
                // Measured: unary +/- on a BARE NULL is BIGINT on DuckDB
                // (the SQLNULL INTEGER default does not survive negation);
                // a typed NULL — CAST(NULL AS INTEGER) — keeps its width.
                let Some(inner) = self.expr_or_null(expr)? else {
                    return Ok(null_of(Ty::I64));
                };
                let zero = if inner.ty == Ty::F64 {
                    SExpr {
                        kind: SKind::Lit(Lit::F64(-0.0)),
                        ty: Ty::F64,
                        nullable: false,
                    }
                } else {
                    SExpr {
                        kind: SKind::Lit(Lit::I64(0)),
                        // A zero literal's natural width; the value-fits
                        // promotion hands -x its operand's own width.
                        ty: Ty::I32,
                        nullable: false,
                    }
                };
                self.arith(
                    ArithOp::Sub,
                    zero,
                    inner,
                    (Some(0), ast_int_literal(expr)),
                )
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => match self.expr_or_null(expr)? {
                // Same BIGINT rule as unary minus (measured: +NULL).
                None => Ok(null_of(Ty::I64)),
                // DuckDB's + is a real unary function over numerics only:
                // +'a' / +TRUE are binder errors there (fleet 2026-08-13).
                Some(e) if e.ty.is_int() || e.ty == Ty::F64 => Ok(e),
                Some(e) => Err(PrepareError::Bind(format!(
                    "no function matches +({})",
                    e.ty.name()
                ))),
            },
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
                // TASK-87 face C (measured both ways): DuckDB's FILTER
                // optimizer folds a constant dead range (lo > hi) to FALSE
                // without ever evaluating the operand — while a PROJECTION
                // evaluates it and traps, on both engines alike. The probe
                // binds lo/hi only when they are literal-shaped, so no call
                // site is consumed twice (extern call-count parity).
                if self.in_filter.get()
                    && !*negated
                    && const_number_shaped(&lo)
                    && const_number_shaped(&hi)
                {
                    let (blo, bhi) = (fold(self.expr(&lo)?), fold(self.expr(&hi)?));
                    let dead = match (&blo.kind, &bhi.kind) {
                        (SKind::Lit(Lit::I64(a)), SKind::Lit(Lit::I64(b))) => a > b,
                        (SKind::Lit(Lit::F64(a)), SKind::Lit(Lit::F64(b))) => a > b,
                        (SKind::Lit(Lit::I64(a)), SKind::Lit(Lit::F64(b))) => {
                            (*a as f64) > *b
                        }
                        (SKind::Lit(Lit::F64(a)), SKind::Lit(Lit::I64(b))) => {
                            *a > (*b as f64)
                        }
                        _ => false,
                    };
                    if dead {
                        return Ok(SExpr {
                            kind: SKind::Lit(Lit::I1(false)),
                            ty: Ty::I1,
                            nullable: false,
                        });
                    }
                }
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
            // SIMILAR TO on VALUES is exactly regexp_full_match on the RAW
            // pattern — DuckDB translates NO wildcards ('h%o' is literal %,
            // 'h.llo' is a live regex dot). Wave-B pins.
            SqlExpr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => {
                if escape_char.is_some() {
                    return Err(unsup("Custom escape in SIMILAR TO (DuckDB: not implemented)"));
                }
                self.regex_full_predicate("SIMILAR TO", expr, pattern, *negated)
            }
            // Bracket syntax s[i] / s[a:b] — exactly array_extract /
            // array_slice in DuckDB (one shared implementation, measured:
            // pins-wave5/{subscripts-extended,slices}.json).
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                // Field read over struct_pack: the bind-time desugar
                // (TASK-93) — the field's own expression binds in place.
                if let Some(sub) = self.desugar_struct_field(e)? {
                    return self.expr(&sub);
                }
                // Field read over a declared wide extern: a lane off one
                // shared ecall (TASK-63).
                if let [AccessExpr::Dot(SqlExpr::Identifier(id))] = access_chain.as_slice() {
                    let mut base: &SqlExpr = root;
                    while let SqlExpr::Nested(i) = base {
                        base = i;
                    }
                    if let SqlExpr::Function(func) = base {
                        if let Some(lane) = self.extern_field_lane(func, &id.value)? {
                            return Ok(lane);
                        }
                    }
                }
                // audit 2026-08-13: a bare-NULL root types Str here and the
                // chain still applies; DuckDB agrees for (NULL)[2] (VARCHAR)
                // but types (NULL)[1:2] INTEGER (its SQLNULL fallback) —
                // value parity holds (NULL either way). Preserved.
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

    /// Compile a column-NAME regex for star filters / COLUMNS: unanchored
    /// for the positive search, `\A..\z`-wrapped for the NOT-full-match
    /// form. Bind-time only — never reaches the exec regex table.
    fn name_regex(&self, pat: &str, full: bool) -> Result<regex::Regex, PrepareError> {
        let translated = super::retrans::translate_pattern(pat)?;
        let pattern = if full {
            format!("\\A(?:{translated})\\z")
        } else {
            translated
        };
        regex::RegexBuilder::new(&pattern)
            .octal(true)
            .build()
            .map_err(|e| {
                PrepareError::Bind(format!("Failed to compile regex \"{pat}\": {e}"))
            })
    }

    /// Expand a `COLUMNS('re')` / `COLUMNS(*)` SELECT item (wave-B):
    /// unanchored RE2 search over declared-case names, table-declaration
    /// order. Returns None when `e` is not a COLUMNS call; expression
    /// forms (COLUMNS(..) + 1) stay unsupported upstream.
    fn expand_columns_item(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<Vec<(String, SExpr)>>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let SqlExpr::Function(f) = e else {
            return Ok(None);
        };
        if !f.name.to_string().eq_ignore_ascii_case("columns") {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        let all =
            self.expand_star_lanes(None, &sqlparser::ast::WildcardAdditionalOptions::default())?;
        match &list.args[..] {
            [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => Ok(Some(finalize_star(all)?)),
            // COLUMNS(* EXCLUDE/REPLACE/... ) — measured identical to the
            // bare `* <modifiers>` select item (names, order, values;
            // pins-waveA/columns-replace.json), so route through the same
            // star expansion.
            [FunctionArg::Unnamed(FunctionArgExpr::WildcardWithOptions(opts))] => {
                Ok(Some(self.expand_star(None, opts)?))
            }
            [FunctionArg::Unnamed(FunctionArgExpr::Expr(p))] => {
                let Some(bp) = self.expr_or_null(p)? else {
                    return Err(PrepareError::Bind(
                        "COLUMNS requires a constant pattern".into(),
                    ));
                };
                let SKind::Lit(Lit::Str(pat)) = bp.kind else {
                    return Err(unsup("COLUMNS with a non-constant or list argument"));
                };
                let rx = self.name_regex(&pat, false)?;
                // Filter BEFORE the opaque check: a regex that never
                // matches an opaque column must not reject the query.
                let cols: Vec<(String, StarLane)> =
                    all.into_iter().filter(|(n, _)| rx.is_match(n)).collect();
                if cols.is_empty() {
                    return Err(PrepareError::Bind(format!(
                        "No matching columns found that match regex \"{pat}\""
                    )));
                }
                Ok(Some(finalize_star(cols)?))
            }
            _ => Err(unsup("COLUMNS argument form")),
        }
    }

    /// `~` / `!~` / SIMILAR TO: FULL match on the raw pattern (measured —
    /// `~` is regexp_full_match in DuckDB, NOT the Postgres search).
    fn regex_full_predicate(
        &self,
        name: &str,
        subject: &SqlExpr,
        pattern: &SqlExpr,
        negated: bool,
    ) -> Result<SExpr, PrepareError> {
        let Some(bs) = self.expr_or_null(subject)? else {
            return Ok(null_of(Ty::I1));
        };
        let bs = str_only(name, bs)?;
        let Some(re) = self.regex_pattern(pattern, super::retrans::ReOptions::default(), true)?
        else {
            return Ok(null_of(Ty::I1));
        };
        let nullable = bs.nullable;
        let m = SExpr {
            kind: SKind::ReMatch {
                re,
                a: Box::new(bs),
            },
            ty: Ty::I1,
            nullable,
        };
        Ok(if negated {
            SExpr {
                kind: SKind::Not(Box::new(m)),
                ty: Ty::I1,
                nullable,
            }
        } else {
            m
        })
    }

    /// Bind a regex options argument: absent -> defaults; otherwise a
    /// non-NULL constant string ("must not be NULL" / "must be a constant"
    /// are the pinned texts).
    fn regex_options(
        &self,
        opts: Option<&SqlExpr>,
        allow_g: bool,
    ) -> Result<super::retrans::ReOptions, PrepareError> {
        match self.regex_options_raw(opts, allow_g)? {
            Some(o) => Ok(o),
            None => Err(PrepareError::Bind(
                "Regex options field must not be NULL".into(),
            )),
        }
    }

    /// regexp_replace's variant: a NULL options argument makes the whole
    /// RESULT NULL (pinned asymmetry) — None here means "return NULL".
    fn regex_options_nullable(
        &self,
        opts: Option<&SqlExpr>,
    ) -> Result<Option<super::retrans::ReOptions>, PrepareError> {
        self.regex_options_raw(opts, true)
    }

    fn regex_options_raw(
        &self,
        opts: Option<&SqlExpr>,
        allow_g: bool,
    ) -> Result<Option<super::retrans::ReOptions>, PrepareError> {
        let Some(o) = opts else {
            return Ok(Some(super::retrans::ReOptions::default()));
        };
        match self.expr_or_null(o)? {
            None => Ok(None),
            // CAST(NULL AS VARCHAR) options behave exactly like bare NULL
            // options (measured: same "must not be NULL" error class /
            // regexp_replace NULL result).
            Some(b) if matches!(b.kind, SKind::NullOf) && b.ty == Ty::Str => Ok(None),
            Some(b) => match b.kind {
                SKind::Lit(Lit::Str(s)) => {
                    super::retrans::parse_options(&s, allow_g).map(Some)
                }
                _ => Err(PrepareError::Bind(
                    "Regex options field must be a constant".into(),
                )),
            },
        }
    }

    /// Bind a constant regex pattern into the program regex table:
    /// translate (retrans), optionally full-match anchor, and COMPILE NOW
    /// so invalid patterns error at prepare (pinned bind-time eagerness).
    /// `Ok(None)` = the pattern was a NULL literal (result is NULL).
    fn regex_pattern(
        &self,
        p: &SqlExpr,
        o: super::retrans::ReOptions,
        full: bool,
    ) -> Result<Option<u32>, PrepareError> {
        Ok(self.regex_pattern_counted(p, o, full)?.map(|(re, _)| re))
    }

    fn regex_pattern_counted(
        &self,
        p: &SqlExpr,
        o: super::retrans::ReOptions,
        full: bool,
    ) -> Result<Option<(u32, usize)>, PrepareError> {
        let Some(bp) = self.expr_or_null(p)? else {
            return Ok(None);
        };
        if matches!(bp.kind, SKind::NullOf) && bp.ty == Ty::Str {
            // CAST(NULL AS VARCHAR) pattern — measured NULL result, same
            // as a bare NULL literal (pins-waveA/regex-null-pattern.json).
            return Ok(None);
        }
        let SKind::Lit(Lit::Str(raw)) = bp.kind else {
            if bp.ty != Ty::Str {
                return Err(PrepareError::Bind(format!(
                    "no function matches a regex with a {} pattern",
                    bp.ty.name()
                )));
            }
            // Column patterns compile per row in DuckDB; the engine model
            // is prepare-time compilation only.
            return Err(unsup("non-constant regex pattern (compiled at prepare in v0)"));
        };
        let translated = if o.literal {
            regex::escape(&raw)
        } else {
            super::retrans::translate_pattern(&raw)?
        };
        let pattern = if full {
            format!("\\A(?:{translated})\\z")
        } else {
            translated
        };
        let rx = regex::RegexBuilder::new(&pattern)
            .case_insensitive(o.case_insensitive)
            .dot_matches_new_line(o.dotall)
            .octal(true)
            .build()
            .map_err(|e| PrepareError::Bind(format!("Invalid Input Error: {e}")))?;
        let group_count = rx.captures_len() - 1;
        let spec = super::ir::ReSpec {
            pattern,
            ci: o.case_insensitive,
            dotall: o.dotall,
            rewrite: None,
        };
        let mut v = self.regexes.borrow_mut();
        // Reuse identical rewrite-less entries (star filters + repeated
        // predicates); replace ops mutate `rewrite` after, so only share
        // entries that still have none.
        if let Some(i) = v.iter().position(|r| *r == spec) {
            return Ok(Some((i as u32, group_count)));
        }
        v.push(spec);
        Ok(Some(((v.len() - 1) as u32, group_count)))
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
        if !bn.ty.is_int() {
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
        let bind_bound = |e: Option<&SqlExpr>, open: i64| -> Result<Option<SExpr>, PrepareError> {
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
            if !e.ty.is_int() {
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
            // NULL-able on a LEFT miss OR when the static column itself is
            // declared nullable (TASK-55: NULL values ride as validity+
            // payload pairs through the probe).
            nullable: sj.kind == JoinKind::Left || col.ty.nullable,
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

    /// n-part dotted reference (pins-waveA/struct-nested.json): qualifier
    /// prefixes longest-first — (schema.table).column, then
    /// (table|alias).column, then bare column — committing to the longest
    /// prefix whose COLUMN part binds, remaining parts becoming struct
    /// field extractions. A failed column bind BACKTRACKS to the next
    /// shorter interpretation; a failed FIELD walk after a bound column is
    /// a hard error (measured). The schema part follows the registry-noise
    /// rule (known-limitations §5).
    fn compound(&self, parts: &[sqlparser::ast::Ident]) -> Result<SExpr, PrepareError> {
        let not_a_struct = |field: &str, col: &str| {
            PrepareError::Bind(format!(
                "Cannot extract field '{field}' from expression \"{col}\" \
                 because it is not a struct"
            ))
        };
        // R1: schema.(this|join).column[.fields...]
        if parts.len() >= 3 {
            if parts[1].value.eq_ignore_ascii_case(&self.this_name) {
                if let Some(r) = self.this_col_with_fields(&parts[2].value, &parts[3..]) {
                    return r;
                }
            } else if self
                .joins
                .iter()
                .any(|sj| sj.name.eq_ignore_ascii_case(&parts[1].value))
            {
                let col = self.qualified(&parts[1].value, &parts[2].value);
                return match (col, parts.len()) {
                    (Ok(c), 3) => Ok(c),
                    // Statics have no struct columns.
                    (Ok(_), _) => Err(not_a_struct(&parts[3].value, &parts[2].value)),
                    (Err(e), _) => Err(e),
                };
            }
        }
        // R2: this.column[.fields...] / join.column
        if parts.len() >= 2 {
            if parts[0].value.eq_ignore_ascii_case(&self.this_name) {
                if let Some(r) = self.this_col_with_fields(&parts[1].value, &parts[2..]) {
                    return r;
                }
            } else if self
                .joins
                .iter()
                .any(|sj| sj.name.eq_ignore_ascii_case(&parts[0].value))
            {
                let col = self.qualified(&parts[0].value, &parts[1].value);
                return match (col, parts.len()) {
                    (Ok(c), 2) => Ok(c),
                    (Ok(_), _) => Err(not_a_struct(&parts[2].value, &parts[1].value)),
                    (Err(e), _) => Err(e),
                };
            }
        }
        // R3: bare column[.fields...]
        if parts.len() >= 2 {
            if let Some(r) = self.bare_col_with_fields(&parts[0].value, &parts[1..]) {
                return r;
            }
            // Nothing bound anywhere: reproduce the pre-struct error
            // shapes (unknown table / column does not exist).
            return match parts.len() {
                2 => self.qualified(&parts[0].value, &parts[1].value),
                3 => self.qualified(&parts[1].value, &parts[2].value),
                _ => Err(PrepareError::Bind(format!(
                    "Referenced table \"{}.{}\" not found",
                    parts[0].value, parts[1].value
                ))),
            };
        }
        self.column(&parts[0].value)
    }

    /// The driving table's column `name` followed by struct-field `fields`.
    /// `None` = no such column (the caller backtracks); `Some(Err)` = the
    /// column bound but the reference is an error (hard, per pins).
    fn this_col_with_fields(
        &self,
        name: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Option<Result<SExpr, PrepareError>> {
        for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                if let Some(f) = fields.first() {
                    return Some(Err(PrepareError::Bind(format!(
                        "Cannot extract field '{}' from expression \"{}\" \
                         because it is not a struct",
                        f.value, c.name
                    ))));
                }
                return Some(Ok(SExpr {
                    kind: SKind::Col(i as u32),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                }));
            }
        }
        if let Some((_, n)) = self
            .opaque
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
        {
            return Some(Err(unsup(format!(
                "row column '{n}' has a non-scalar type"
            ))));
        }
        let sc = self
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))?;
        Some(self.walk_struct(sc, fields))
    }

    /// A bare first part: the driving table's columns (incl. structs and
    /// opaque) — static value columns are scalars, so `x.field` never
    /// binds through them silently (a scalar hit with fields is the hard
    /// not-a-struct error, matching DuckDB).
    fn bare_col_with_fields(
        &self,
        name: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Option<Result<SExpr, PrepareError>> {
        if let Some(r) = self.this_col_with_fields(name, fields) {
            return Some(r);
        }
        for sj in &self.joins {
            for &ci in sj.val_cols.iter().chain(sj.key_cols.iter()) {
                if sj.table.cols[ci as usize].name.eq_ignore_ascii_case(name) {
                    return Some(Err(PrepareError::Bind(format!(
                        "Cannot extract field '{}' from expression \"{name}\" \
                         because it is not a struct",
                        fields[0].value
                    ))));
                }
            }
        }
        None
    }

    /// Walk `fields` down a struct column to a scalar leaf lane. Empty
    /// fields = the whole struct (non-scalar output, named rejection).
    fn walk_struct(
        &self,
        sc: &super::plan::StructCol,
        fields: &[sqlparser::ast::Ident],
    ) -> Result<SExpr, PrepareError> {
        use super::plan::StructNode;
        if fields.is_empty() {
            return Err(unsup(format!(
                "struct column '{}' as a whole value (project its fields instead)",
                sc.name
            )));
        }
        let mut cur = &sc.fields;
        for (k, f) in fields.iter().enumerate() {
            // Field matching is case-insensitive even when quoted
            // (measured — quoting does not opt into case sensitivity).
            let Some(sf) = cur.iter().find(|x| x.name.eq_ignore_ascii_case(&f.value)) else {
                return Err(PrepareError::Bind(format!(
                    "Could not find key \"{}\" in struct",
                    f.value
                )));
            };
            match &sf.node {
                StructNode::Leaf(lane) => {
                    if k + 1 == fields.len() {
                        let c = &self.in_cols[*lane as usize];
                        return Ok(SExpr {
                            kind: SKind::Col(*lane),
                            ty: c.ty.ty,
                            nullable: c.ty.nullable,
                        });
                    }
                    return Err(PrepareError::Bind(format!(
                        "Cannot extract field '{}' from expression \"{}\" \
                         because it is not a struct",
                        fields[k + 1].value, sf.name
                    )));
                }
                StructNode::Opaque => {
                    return Err(unsup(format!(
                        "struct field '{}' has a non-scalar type",
                        sf.name
                    )));
                }
                StructNode::Nested(n) => {
                    if k + 1 == fields.len() {
                        return Err(unsup(format!(
                            "struct field '{}' as a whole value (project its \
                             scalar leaves instead)",
                            sf.name
                        )));
                    }
                    cur = n;
                }
            }
        }
        unreachable!("loop returns on the last field")
    }

    /// Case-insensitive, spelling-preserving bare-column bind over the whole
    /// scope: the dynamic table plus every joined static table's value
    /// columns (DuckDB semantics; ambiguity is an error).
    fn column(&self, name: &str) -> Result<SExpr, PrepareError> {
        let mut hits: Vec<SExpr> = Vec::new();
        for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                hits.push(SExpr {
                    kind: SKind::Col(i as u32),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                });
            }
        }
        if let Some((_, n)) = self
            .opaque
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
        {
            // The column exists in the model — it just has no lane. Same
            // precedence as a real-column hit (beats lateral aliases).
            return Err(unsup(format!("row column '{n}' has a non-scalar type")));
        }
        if let Some(sc) = self
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
        {
            // A bare struct reference is its WHOLE value — non-scalar out.
            return Err(unsup(format!(
                "struct column '{}' as a whole value (project its fields instead)",
                sc.name
            )));
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
            // The REAL column wins over a same-named select alias (measured
            // in both SELECT and WHERE — wave-5 pins).
            1 => Ok(hits.pop().expect("len checked")),
            0 if name.eq_ignore_ascii_case("rowid") => Err(unsup("rowid pseudo-column")),
            0 => {
                // Lateral aliases: an already-bound alias resolves to its
                // expression; a known-but-later alias is the pinned
                // forward-reference error. A name defined MORE THAN ONCE
                // falls through to that same error: DuckDB resolves lateral
                // refs to the LAST definition (so a between-definitions ref
                // is its forward-reference error, measured 1.5.5), while
                // our per-occurrence binding would take the first — and a
                // shared extern site bound through a mutating alias would
                // silently freeze it (TASK-63 review). Refusal keeps
                // binding time-invariant for every accepted query.
                let dup = self
                    .select_aliases
                    .iter()
                    .filter(|a| a.eq_ignore_ascii_case(name))
                    .count()
                    > 1;
                if !dup {
                    if let Some((_, e)) = self
                        .bound_aliases
                        .borrow()
                        .iter()
                        .rev()
                        .find(|(a, _)| a.eq_ignore_ascii_case(name))
                    {
                        return Ok(e.clone());
                    }
                }
                if self
                    .select_aliases
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(name))
                {
                    return Err(PrepareError::Bind(format!(
                        "column \"{name}\" referenced that exists in the SELECT clause - \
                         but this column cannot be referenced before it is defined"
                    )));
                }
                Err(PrepareError::Bind(format!(
                    "column '{name}' does not exist in scope"
                )))
            }
            _ => Err(PrepareError::Bind(format!("ambiguous column '{name}'"))),
        }
    }

    /// `table.col` bind: the dynamic table by its FROM spelling, a joined
    /// static table by its alias (or name).
    fn qualified(&self, table: &str, name: &str) -> Result<SExpr, PrepareError> {
        if table.eq_ignore_ascii_case(&self.this_name) {
            let mut hit = None;
            for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
                if c.name.eq_ignore_ascii_case(name) {
                    if hit.is_some() {
                        return Err(PrepareError::Bind(format!("ambiguous column '{name}'")));
                    }
                    hit = Some((i, c));
                }
            }
            if let Some((_, n)) = self
                .opaque
                .iter()
                .find(|(_, n)| n.eq_ignore_ascii_case(name))
            {
                return Err(unsup(format!("row column '{n}' has a non-scalar type")));
            }
            if let Some(sc) = self
                .structs
                .iter()
                .find(|s| s.name.eq_ignore_ascii_case(name))
            {
                return Err(unsup(format!(
                    "struct column '{}' as a whole value (project its fields instead)",
                    sc.name
                )));
            }
            if hit.is_none() && name.eq_ignore_ascii_case("rowid") {
                return Err(unsup("rowid pseudo-column"));
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
            if name.eq_ignore_ascii_case("rowid") {
                return Err(unsup("rowid pseudo-column"));
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
        // TASK-84: DuckDB types integer literals INTEGER and computes their
        // arithmetic in 32 bits, so `-6 * (- 2147483647)` ERRORS there while
        // a single i64 width serves an answer. A literal-shaped integer
        // subtree is re-evaluated here in checked int32 — DuckDB's own
        // semantics — and refuses at build if any step would trap. A BIGINT
        // operand anywhere (column, cast, out-of-int32 literal) makes the
        // whole tree 64-bit on both engines and is untouched. The residual
        // (`CAST(k AS INTEGER) * 2` trapping data-dependently at row time)
        // needs the declared-width design and stays on TASK-79/84.
        if matches!(
            op,
            BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Multiply
                | BinaryOperator::Modulo
        ) {
            let probe = SqlExpr::BinaryOp {
                left: Box::new(left.clone()),
                op: op.clone(),
                right: Box::new(right.clone()),
            };
            if let I32Fold::Traps = eval_i32_literal(&probe) {
                return Err(PrepareError::Bind(format!(
                    "integer literal arithmetic overflows INTEGER on DuckDB \
                     ({} {op} {}) — int-literal math runs in 32 bits there; \
                     make an operand BIGINT (CAST(.. AS BIGINT)) for 64-bit \
                     arithmetic",
                    left, right
                )));
            }
        }
        // TASK-87 face B: DuckDB folds an all-literal integer COMPARISON
        // through wide range analysis — `x > (huge * huge)` answers without
        // ever overflowing — while binding the operand here trapped on the
        // i64 multiply. Fold the comparison in i128, DuckDB's answer.
        // Skipped when either side would trap in ITS int32 literal math
        // (unmeasured whether DuckDB's analysis rescues those; the TASK-84
        // refusal on the operand keeps that boundary loud instead).
        if matches!(
            op,
            BinaryOperator::Gt
                | BinaryOperator::Lt
                | BinaryOperator::GtEq
                | BinaryOperator::LtEq
                | BinaryOperator::Eq
                | BinaryOperator::NotEq
        ) && !matches!(eval_i32_literal(left), I32Fold::Traps)
            && !matches!(eval_i32_literal(right), I32Fold::Traps)
        {
            if let (Some(x), Some(y)) =
                (eval_i128_literal(left), eval_i128_literal(right))
            {
                let v = match op {
                    BinaryOperator::Gt => x > y,
                    BinaryOperator::Lt => x < y,
                    BinaryOperator::GtEq => x >= y,
                    BinaryOperator::LtEq => x <= y,
                    BinaryOperator::Eq => x == y,
                    _ => x != y,
                };
                return Ok(SExpr {
                    kind: SKind::Lit(Lit::I1(v)),
                    ty: Ty::I1,
                    nullable: false,
                });
            }
        }
        let a = self.expr_or_null(left)?;
        let b = self.expr_or_null(right)?;
        // DuckDB folds a strict op over a DECIMAL literal and a bare NULL
        // to SQLNULL — INTEGER — discarding the decimal (measured:
        // -2.681 + NULL and 2.5 * NULL are INTEGER; / stays DOUBLE). Our
        // decimal literals are f64 (the documented v0 narrowing), so
        // adoption would answer double.
        if matches!(
            op,
            BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Multiply
                | BinaryOperator::Modulo
        ) && ((a.is_none() && b.is_some() && ast_decimal_literal(right))
            || (b.is_none() && a.is_some() && ast_decimal_literal(left)))
        {
            return Ok(null_of(Ty::I32));
        }
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
            (None, None) => {
                // Wave-5 pins: NULL <op> NULL types by the operator —
                // + - * % -> BIGINT, / -> DOUBLE, comparisons -> BOOLEAN
                // (via I64 operands), AND/OR -> BOOLEAN.
                let ty = match op {
                    BinaryOperator::Divide => Ty::F64,
                    BinaryOperator::And | BinaryOperator::Or => Ty::I1,
                    BinaryOperator::PGStartsWith | BinaryOperator::StringConcat => Ty::Str,
                    _ => Ty::I64,
                };
                (null_of(ty), null_of(ty))
            }
        };
        let lits = (ast_int_literal(left), ast_int_literal(right));
        match op {
            BinaryOperator::Plus => self.arith(ArithOp::Add, a, b, lits),
            BinaryOperator::Minus => self.arith(ArithOp::Sub, a, b, lits),
            BinaryOperator::Multiply => self.arith(ArithOp::Mul, a, b, lits),
            BinaryOperator::Divide => self.arith(ArithOp::Div, a, b, lits),
            BinaryOperator::DuckIntegerDivide => self.arith(ArithOp::IDiv, a, b, lits),
            BinaryOperator::Modulo => self.arith(ArithOp::Rem, a, b, lits),
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
                // TASK-102: DuckDB's binder collapses || to an SQLNULL
                // constant (int32 at the boundary, ADOPTABLE upstream —
                // see expr_or_null's shape gate) when an operand its
                // binder can fold evaluates to NULL — any spelling, a
                // column on the OTHER side notwithstanding, since ||
                // propagates NULL to every row. Concat-specific: +, LIKE
                // and function calls keep their promoted type, and
                // concat() skips NULLs instead. The bind_foldable gate is
                // load-bearing: our own fold dead-arm-eliminates a CASE
                // whose column sits in an untaken arm, which DuckDB's
                // binder never folds — that spelling stays Str. Pure
                // extern operands TRY-fold by execution (TASK-101), a
                // folded VALUE baking in as a literal.
                let (a, a_null) = self.bind_fold_concat_operand(a);
                let (b, b_null) = self.bind_fold_concat_operand(b);
                if a_null || b_null {
                    return Ok(null_of(Ty::I32));
                }
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
        // CASE arms are guarded: plan-time trapping-constant refusals are
        // suspended inside (see `in_guarded`).
        self.in_guarded.set(self.in_guarded.get() + 1);
        let _guard = GuardScope(&self.in_guarded);
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

        // Width unification — DuckDB's fold (2026-08-13 fleet, 0 errors on
        // 19k probes): SEED from the ELSE (its syntactic-literal hint
        // intact); no ELSE — or ELSE NULL — seeds as an implicit non-
        // literal NULL. Then combine WHEN arms in order; every combine
        // makes the accumulator computed, so only the seed's literal-ness
        // ever survives. A NULL arm never widens (None; adopts below).
        let mut unified: Option<Ty> = None;
        let mut acc_lit: Option<i64> = None;
        if let Some(Some(r)) = &else_bound {
            unified = Some(r.ty);
            acc_lit = else_result.and_then(ast_int_literal);
        }
        for (r, when) in results.iter().zip(conditions) {
            let Some(r) = r else { continue };
            let new_lit = ast_int_literal(&when.result);
            unified = Some(match unified {
                None => r.ty,
                Some(u) if u == r.ty => u,
                Some(u) if u.is_int() && r.ty.is_int() => {
                    int_width_promote(u, acc_lit, r.ty, new_lit)
                }
                Some(u) if u.is_int() && r.ty == Ty::F64 => Ty::F64,
                Some(Ty::F64) if r.ty.is_int() => Ty::F64,
                Some(u) => {
                    return Err(PrepareError::Bind(format!(
                        "CASE branches disagree: {} vs {}",
                        u.name(),
                        r.ty.name()
                    )))
                }
            });
            acc_lit = None;
        }
        let Some(unified) = unified else {
            return Err(unsup("CASE where every branch is NULL"));
        };

        let coerce = |r: Option<SExpr>| -> SExpr {
            match r {
                None => null_of(unified),
                Some(e) if e.ty.is_int() && unified == Ty::F64 => promote_f64(e),
                Some(mut e) if e.ty.is_int() && unified.is_int() && e.ty != unified => {
                    // Width-only retype; the payload lane is shared.
                    e.ty = unified;
                    e
                }
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
        // TASK-87 face A: a constant cast that FAILS is a plan-time error
        // on DuckDB — measured to fire even over zero rows and under a
        // constant-false WHERE — while a row-driven engine never evaluates
        // it. Evaluate the constant here with the interpreter's EXACT parse
        // (trim_ascii + Rust parse; see Inst::StoiOpt / Inst::StofOpt) and
        // refuse a failure by name. TRY_CAST stays lazy: it yields NULL.
        let inner = fold(inner);
        if !trying && self.in_guarded.get() == 0 {
            if let SKind::Lit(Lit::Str(s)) = &inner.kind {
                let ok = match to {
                    t if t.is_int() => s
                        .trim_ascii()
                        .parse::<i64>()
                        .is_ok_and(|v| fits_width(t, v)),
                    Ty::F64 => s.trim_ascii().parse::<f64>().is_ok(),
                    _ => true,
                };
                if !ok {
                    return Err(PrepareError::Bind(format!(
                        "constant cast fails on every row: CAST('{s}' AS \
                         {}) — DuckDB errors at plan time; TRY_CAST is the \
                         NULL-yielding spelling",
                        duck_int_name(to)
                    )));
                }
            }
        }
        // A constant that misses a NARROW target's range: TRY_CAST is NULL;
        // CAST refuses — DuckDB's plan-time conversion error, and there is
        // no i32-range runtime trap to fall back on before m-8 phase 3, so
        // the refusal is NOT in_guarded-suspended (refusing a query DuckDB
        // could run lazily beats serving a value it would never produce).
        if let Some((lo, hi)) = to.int_range() {
            let const_out = match &inner.kind {
                SKind::Lit(Lit::I64(v)) => Some(!(lo..=hi).contains(v)),
                SKind::Lit(Lit::F64(f)) => {
                    let r = f.round_ties_even();
                    Some(!(lo as f64..=hi as f64).contains(&r))
                }
                _ => None,
            };
            match const_out {
                Some(true) if trying => return Ok(null_of(to)),
                Some(true) => {
                    return Err(PrepareError::Bind(format!(
                        "constant cast overflows {}: DuckDB errors at plan \
                         time; TRY_CAST is the NULL-yielding spelling",
                        duck_int_name(to)
                    )))
                }
                _ => {}
            }
            // TRY_CAST to a narrow width NULLs outside the range at row
            // time. No trap machinery: convert on the lane (NULL on
            // failure), then a range-guard CASE — pure re-evaluation
            // clones, like the %-by-zero guard.
            if trying && const_out.is_none() {
                let wide = if inner.ty.is_int() {
                    inner
                } else {
                    SExpr {
                        kind: SKind::Cast {
                            inner: Box::new(inner),
                            trying: true,
                        },
                        ty: Ty::I64,
                        nullable: true,
                    }
                };
                let bound = |n: i64| SExpr {
                    kind: SKind::Lit(Lit::I64(n)),
                    ty: Ty::I64,
                    nullable: false,
                };
                let ge = self.cmp(CmpPred::Ge, wide.clone(), bound(lo))?;
                let le = self.cmp(CmpPred::Le, wide.clone(), bound(hi))?;
                let nullable_cond = ge.nullable || le.nullable;
                let cond = SExpr {
                    kind: SKind::And {
                        a: Box::new(ge),
                        b: Box::new(le),
                    },
                    ty: Ty::I1,
                    nullable: nullable_cond,
                };
                let val = SExpr {
                    kind: SKind::Cast {
                        inner: Box::new(wide.clone()),
                        trying: false,
                    },
                    ty: to,
                    nullable: wide.nullable,
                };
                return Ok(SExpr {
                    kind: SKind::Case {
                        arms: vec![(cond, val)],
                        default: Some(Box::new(null_of(to))),
                    },
                    ty: to,
                    nullable: true,
                });
            }
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

    /// `lits` are the operands' SYNTACTIC-literal hints, computed by the
    /// caller from the SQL AST (`ast_int_literal`) — never from bound
    /// nodes (fleet 2026-08-13: every SExpr-shape heuristic leaks).
    fn arith(
        &self,
        op: ArithOp,
        a: SExpr,
        b: SExpr,
        lits: (Option<i64>, Option<i64>),
    ) -> Result<SExpr, PrepareError> {
        // TASK-85: a STRICT operator with a literal-NULL operand folds to
        // NULL at build, exactly as DuckDB's optimizer folds it — which
        // ELIMINATES the sibling subexpression, so a trapping ln/overflow/
        // giant-string under it never executes there and must not here.
        // Measured (2026-08-11): DuckDB's elision is literal-NULL only; a
        // runtime NULL does not spare the trap on either engine, so eager
        // per-row evaluation stays as it is. Checked BEFORE promotion —
        // promote_f64 wraps NullOf in a cast, hiding it. Type errors still
        // refuse first, below, exactly as DuckDB binder-errors before it
        // folds.
        // TASK-87 face D: folding the operands first makes a NULL PRODUCED
        // by constant folding (a constant-condition CASE landing on NULL)
        // visible to the strict-op check below, exactly as DuckDB's folder
        // sees it. fold() is pure and idempotent.
        let (a, b) = (fold(a), fold(b));
        let null_operand =
            matches!(a.kind, SKind::NullOf) || matches!(b.kind, SKind::NullOf);
        if matches!(
            op,
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitAnd | ArithOp::BitOr | ArithOp::BitXor
        ) {
            // Bitwise is integer-only (wave-5 pins: non-integer operands
            // are binder errors) and width-polymorphic (1 & 2 is INTEGER,
            // i & k is BIGINT — measured 2026-08-13); compute is i64 either
            // way.
            for e in [&a, &b] {
                if !e.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches bitwise op on ({}, {})",
                        a.ty.name(),
                        b.ty.name()
                    )));
                }
            }
            let ty = int_width_promote(a.ty, lits.0, b.ty, lits.1);
            if null_operand {
                return Ok(null_of(ty));
            }
            let nullable = a.nullable || b.nullable;
            return Ok(SExpr {
                kind: SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                },
                ty,
                nullable,
            });
        }
        let (a, b, ty) = numeric_promote(op, a, b, lits)?;
        if null_operand {
            return Ok(null_of(ty));
        }
        // TASK-87 faces A/B: DuckDB evaluates constants at plan time, so an
        // all-literal integer operation that TRAPS errors on every
        // execution there — even over zero rows — while a row-driven
        // engine would serve. Refuse by name. % and // are guarded below;
        // float arithmetic is IEEE and always folds.
        if let (SKind::Lit(Lit::I64(x)), SKind::Lit(Lit::I64(y))) =
            (&a.kind, &b.kind)
        {
            // Width-aware: the op traps in the RESULT's width (an i32 lane
            // overflows at ±2^31 on DuckDB, not ±2^63).
            let fits = |r: i64| fits_width(ty, r);
            let trapped = self.in_guarded.get() == 0
                && match op {
                    ArithOp::Add => !x.checked_add(*y).is_some_and(fits),
                    ArithOp::Sub => !x.checked_sub(*y).is_some_and(fits),
                    ArithOp::Mul => !x.checked_mul(*y).is_some_and(fits),
                    _ => false,
                };
            if trapped {
                return Err(PrepareError::Bind(format!(
                    "constant integer arithmetic overflows {} \
                     ({x} {op:?} {y}) — DuckDB evaluates constants at plan \
                     time and errors on every execution; this engine \
                     refuses instead of serving rows the oracle never would",
                    if ty == Ty::I64 { "BIGINT" } else { "INTEGER" }
                )));
            }
        }
        let nullable = a.nullable || b.nullable;
        // DuckDB pins (2026-07-26, waves 1+3): integer % by zero is NULL,
        // and `//`/divide() by zero is NULL on BOTH ints and doubles —
        // guard with a CASE unless the divisor is a provably non-zero
        // literal. The idiv/irem traps stay reachable only for MIN op -1,
        // where DuckDB traps too. Float % is IEEE (x % 0.0 = NaN), no guard.
        let needs_guard =
            (op == ArithOp::Rem && ty.is_int()) || op == ArithOp::IDiv;
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
        // TASK-92: the comparison result type is the operator table's rule.
        let Ret::Fixed(ret) = sig::op_ret(cmp_sym(pred)) else {
            unreachable!("comparisons are Fixed rows")
        };
        // TASK-87 face D — same reason as `arith`: a folded constant NULL
        // must reach the strict-op elision below.
        let (a, b) = (fold(a), fold(b));
        let (a, b) = match (a.ty, b.ty) {
            (x, y) if x == y => (a, b),
            // Mixed integer widths compare in the shared i64 lane.
            (x, y) if x.is_int() && y.is_int() => (a, b),
            (x, Ty::F64) if x.is_int() => (promote_f64(a), b),
            (Ty::F64, y) if y.is_int() => (a, promote_f64(b)),
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
        // TASK-85, same fold as `arith`: a comparison against a literal NULL
        // is NULL on DuckDB's optimizer BEFORE the sides evaluate, so a
        // trapping side must be eliminated here too. (promote_f64 retypes a
        // NullOf in place, so the plain match still sees it after promotion.)
        if matches!(a.kind, SKind::NullOf) || matches!(b.kind, SKind::NullOf) {
            return Ok(null_of(ret));
        }
        let nullable = a.nullable || b.nullable;
        Ok(SExpr {
            kind: SKind::Cmp {
                pred,
                a: Box::new(a),
                b: Box::new(b),
            },
            ty: ret,
            nullable,
        })
    }

    /// The v0 builtin catalogue. Everything here follows the measured pins
    /// in docs/superpowers/specs/2026-07-26-stretch4-builtin-pins.md; names
    /// not listed reject as clean unsupported.
    /// The declared UDF matching `name` (case-insensitive), if any.
    fn find_udf(&self, name: &str) -> Option<(u32, &super::ir::ExternSpec)> {
        self.udfs
            .iter()
            .enumerate()
            .find(|(_, u)| u.name.eq_ignore_ascii_case(name))
            .map(|(i, u)| (i as u32, u))
    }

    fn fresh_site(&self) -> u32 {
        let s = self.sites.get();
        self.sites.set(s + 1);
        s
    }

    /// The shared ecall site for a UDF call AST node: every mention of the
    /// SAME call — field reads and the whole item — binds one site
    /// (TASK-63 / P16 single-eval; the whole-item leg is the slice-5
    /// review round).
    fn site_for(&self, f: &sqlparser::ast::Function) -> u32 {
        let mut cache = self.extern_sites.borrow_mut();
        match cache.iter().find(|(k, _)| k == f) {
            Some((_, s)) => *s,
            None => {
                let s = self.fresh_site();
                cache.push((f.clone(), s));
                s
            }
        }
    }

    /// TASK-101: try to execute a pure extern at BIND, DuckDB's bind fold
    /// (spec 2026-08-13-bind-fold-alignment). `None` = not foldable here
    /// (side_effects declared, no evaluator, or a non-constant argument);
    /// `Some(Err(msg))` = the callable raised, and the CONTEXT decides
    /// (field access fails the build, || swallows — both measured);
    /// `Some(Ok(..))` = the folded result (`None` = whole-call NULL).
    #[allow(clippy::type_complexity)]
    fn try_extern_bind_fold(
        &self,
        ext: usize,
        spec: &super::ir::ExternSpec,
        args: &[SExpr],
    ) -> Option<Result<Option<Vec<Option<ScalarVal>>>, String>> {
        if spec.side_effects {
            return None;
        }
        let eval = self.bind_eval.get(ext)?;
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            if !bind_foldable(a) {
                return None;
            }
            vals.push(match fold(a.clone()).kind {
                SKind::NullOf => None,
                SKind::Lit(Lit::I1(v)) => Some(ScalarVal::I1(v)),
                SKind::Lit(Lit::I64(v)) => Some(ScalarVal::I64(v)),
                SKind::Lit(Lit::F64(v)) => Some(ScalarVal::F64(v)),
                SKind::Lit(Lit::Str(s)) => Some(ScalarVal::Str(s)),
                // A constant spelling our fold cannot finish (runtime-only
                // ops over literals) — DuckDB would fold; we pass. The
                // campaign owns finding any schema-visible residue.
                _ => return None,
            });
        }
        Some((eval.fun)(&vals))
    }

    /// TASK-102 gate, reading through TASK-101: bind-fold one || operand.
    /// `(_, true)` = the operand folds to NULL, so the whole || collapses
    /// to SQLNULL. Otherwise the (possibly rewritten) operand comes back:
    /// a pure extern's folded VALUE is baked as a literal — DuckDB
    /// executes once at bind, never per row, and a non-deterministic
    /// "pure" udf gets one baked sample there too — while a raising
    /// callable keeps the runtime call (DuckDB's fold swallows
    /// exceptions uniformly; review 2026-08-13).
    fn bind_fold_concat_operand(&self, e: SExpr) -> (SExpr, bool) {
        if bind_foldable(&e) && matches!(fold(e.clone()).kind, SKind::NullOf) {
            return (e, true);
        }
        // to_varchar wraps a non-Str operand in one Cast — look through.
        let (inner, wrap) = match &e.kind {
            SKind::Cast { inner, trying } => ((**inner).clone(), Some((e.ty, *trying))),
            _ => (e.clone(), None),
        };
        if let SKind::ExternCall {
            ext,
            ref args,
            whole: false,
            ..
        } = inner.kind
        {
            if let Some(spec) = self.udfs.get(ext as usize) {
                if spec.rets.len() == 1 {
                    match self.try_extern_bind_fold(ext as usize, spec, args) {
                        Some(Ok(None)) => return (e, true),
                        Some(Ok(Some(lanes))) => match lanes.into_iter().next() {
                            Some(Some(v)) => {
                                let lit = scalar_lit(v, inner.ty);
                                let rebuilt = match wrap {
                                    Some((ty, trying)) => SExpr {
                                        nullable: false,
                                        kind: SKind::Cast {
                                            inner: Box::new(lit),
                                            trying,
                                        },
                                        ty,
                                    },
                                    None => lit,
                                };
                                return (rebuilt, false);
                            }
                            // A NULL-valued fold result collapses exactly
                            // like any constant NULL operand (measured:
                            // s || CAST(NULL AS VARCHAR) is INTEGER).
                            _ => return (e, true),
                        },
                        _ => {}
                    }
                }
            }
        }
        (e, false)
    }

    /// TASK-93: field access over struct_pack is a pure bind-time desugar —
    /// extracting a field of a just-packed struct IS binding that field's
    /// expression. Handles the dot form `(struct_pack(a := e)).a` (chains
    /// peel one Dot per pass; re-entry desugars the rest) and returns the
    /// SUBSTITUTE AST. `Ok(None)` when the shape isn't
    /// field-access-over-struct-pack; the missing-key refusal uses DuckDB's
    /// wording. A bare-NULL field rides the adoptable-SQLNULL channel by
    /// construction — the substitute AST re-binds wherever the ORIGINAL
    /// stood (`- (struct_pack(a := NULL)).a` is BIGINT on DuckDB, the bare
    /// field INTEGER; measured 2026-08-13).
    fn desugar_struct_field(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<SqlExpr>, PrepareError> {
        let SqlExpr::CompoundFieldAccess { root, access_chain } = e else {
            return Ok(None);
        };
        let Some((AccessExpr::Dot(SqlExpr::Identifier(id)), rest)) =
            access_chain.split_first()
        else {
            return Ok(None);
        };
        let mut base: &SqlExpr = root;
        while let SqlExpr::Nested(i) = base {
            base = i;
        }
        let SqlExpr::Function(f) = base else {
            return Ok(None);
        };
        let Some(field) = self.struct_pack_field(f, &id.value)? else {
            return Ok(None);
        };
        if rest.is_empty() {
            return Ok(Some(field.clone()));
        }
        Ok(Some(SqlExpr::CompoundFieldAccess {
            root: Box::new(SqlExpr::Nested(Box::new(field.clone()))),
            access_chain: rest.to_vec(),
        }))
    }

    /// The struct_extract SPELLING of the same desugar (TASK-93):
    /// `struct_extract(struct_pack(a := e), 'a')` -> the field's AST.
    fn desugar_struct_extract(
        &self,
        f: &sqlparser::ast::Function,
    ) -> Result<Option<SqlExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("struct_extract") {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        let [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(target)),
            FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(v))),
        ] = list.args.as_slice()
        else {
            return Ok(None);
        };
        let SqlValue::SingleQuotedString(field) = &v.value else {
            return Ok(None);
        };
        let mut base: &SqlExpr = target;
        while let SqlExpr::Nested(i) = base {
            base = i;
        }
        let SqlExpr::Function(inner) = base else {
            return Ok(None);
        };
        Ok(self.struct_pack_field(inner, field)?.cloned())
    }

    /// The named field of a plain `struct_pack(...)` call, ASCII-case-
    /// insensitively (DuckDB's struct key matching). `Ok(None)` when `f`
    /// isn't a plain struct_pack; a struct_pack MISSING the key refuses
    /// with DuckDB's wording.
    fn struct_pack_field<'e>(
        &self,
        f: &'e sqlparser::ast::Function,
        field: &str,
    ) -> Result<Option<&'e SqlExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("struct_pack")
            || f.uses_odbc_syntax
            || !matches!(f.parameters, FunctionArguments::None)
            || f.filter.is_some()
            || f.null_treatment.is_some()
            || f.over.is_some()
        {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        for a in &list.args {
            if let FunctionArg::Named { name, arg, .. } = a {
                if name.value.eq_ignore_ascii_case(field) {
                    let FunctionArgExpr::Expr(v) = arg else {
                        return Ok(None);
                    };
                    return Ok(Some(v));
                }
            }
        }
        Err(PrepareError::Bind(format!(
            "Could not find key \"{field}\" in struct"
        )))
    }

    /// Field access over a declared width-k extern call: bind the named
    /// lane of ONE shared ecall (TASK-63). `Ok(None)` when this isn't
    /// that shape — callers fall through to their own handling.
    fn extern_field_lane(
        &self,
        f: &sqlparser::ast::Function,
        field: &str,
    ) -> Result<Option<SExpr>, PrepareError> {
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.ret_names.is_empty() {
            return Err(unsup(format!(
                "field access on udf '{}': no declared output field names",
                spec.name
            )));
        }
        let Some(ret) = spec
            .ret_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(field))
        else {
            return Err(PrepareError::Bind(format!(
                "udf '{}' has no output field '{}' (declared: {})",
                spec.name,
                field,
                spec.ret_names
                    .iter()
                    .map(|n| format!("'{n}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        let args = self.bind_udf_args(f, spec)?;
        // TASK-101: field access is a fold context — a pure udf with
        // constant args EXECUTES here at bind, special null handling
        // honored (the real result is used, never assumed). Whole-call
        // None is DuckDB's SQLNULL (surfaced int32, ADOPTED by consumers
        // via expr_or_null's shape gate); a real struct keeps its
        // declared field types, NULL fields included. A raised exception
        // is SWALLOWED and the runtime call stays — DuckDB's fold gives
        // up uniformly (review 2026-08-13: DESCRIBE succeeds, the error
        // fires at RUN with rows, a zero-row batch answers empty; an
        // earlier FROM-less probe that seemed to error at bind was eager
        // constant evaluation, not the binder).
        match self.try_extern_bind_fold(ext as usize, spec, &args) {
            Some(Err(_)) => {}
            Some(Ok(None)) => return Ok(Some(null_of(Ty::I32))),
            Some(Ok(Some(lanes))) => {
                let ty = spec.rets[ret];
                return Ok(Some(match lanes.into_iter().nth(ret) {
                    Some(Some(v)) => scalar_lit(v, ty),
                    _ => null_of(ty),
                }));
            }
            None => {}
        }
        let site = self.site_for(f);
        Ok(Some(SExpr {
            kind: SKind::ExternCall {
                site,
                ext,
                args,
                ret: ret as u32,
                whole: false,
            },
            ty: spec.rets[ret],
            nullable: true,
        }))
    }

    /// Bind and type-check a UDF call's arguments against its declared
    /// params. Bare NULLs adopt the param type; an i64 argument against a
    /// declared f64 promotes exactly like DuckDB's implicit cast; anything
    /// else refuses by name.
    fn bind_udf_args(
        &self,
        f: &sqlparser::ast::Function,
        spec: &super::ir::ExternSpec,
    ) -> Result<Vec<SExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let FunctionArguments::List(list) = &f.args else {
            return Err(unsup(format!(
                "function {} without an argument list",
                f.name
            )));
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Err(unsup(format!("function {} argument clauses", f.name)));
        }
        // Every caller of a declared UDF routes through here, so the
        // call-node modifiers the oracle rejects are screened once
        // (review round: IGNORE NULLS rode in on the INNER call of an
        // unnest item and expanded as if unadorned).
        if f.filter.is_some()
            || f.over.is_some()
            || f.null_treatment.is_some()
            || !f.within_group.is_empty()
        {
            return Err(unsup(format!(
                "modifier on udf call {} (FILTER, OVER, IGNORE NULLS and \
                 WITHIN GROUP apply to aggregates and window functions)",
                f.name
            )));
        }
        let mut raw: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => raw.push(e),
                _ => return Err(unsup(format!("function {} argument form", f.name))),
            }
        }
        if raw.len() != spec.params.len() {
            return Err(PrepareError::Bind(format!(
                "udf '{}' takes {} argument(s), the call passes {}",
                spec.name,
                spec.params.len(),
                raw.len()
            )));
        }
        let mut out = Vec::with_capacity(raw.len());
        for (i, (arg, &pt)) in raw.iter().zip(&spec.params).enumerate() {
            let bound = match self.expr_or_null(arg)? {
                None => null_of(pt),
                Some(e) => fold(e),
            };
            let bound = match (bound.ty, pt) {
                (a, b) if a == b => bound,
                // A narrow int upcasts into a declared int64 param (the
                // lane is shared) — DuckDB's implicit INTEGER -> BIGINT.
                (a, Ty::I64) if a.is_int() => {
                    let mut e = bound;
                    e.ty = Ty::I64;
                    e
                }
                (a, Ty::F64) if a.is_int() => promote_f64(bound),
                (a, b) => {
                    return Err(PrepareError::Bind(format!(
                        "udf '{}' argument {} is {}, declared {}",
                        spec.name,
                        i + 1,
                        a.name(),
                        b.name()
                    )))
                }
            };
            out.push(bound);
        }
        Ok(out)
    }

    /// A bare wide UDF call as a projection item expands to a whole-validity
    /// lane plus per-return nullable component lanes sharing one call site:
    /// a width-k (k >= 2) unnamed extern (the DRAFT-22 list boundary), or a
    /// NAMED extern at EVERY width (slice 5 — DuckDB registers named
    /// externs as STRUCT, so the boundary assembles a struct keyed by the
    /// returned declared names; empty names = list). `None` for anything
    /// else (width-1 unnamed calls stay ordinary scalar expressions). Lane
    /// names carry U+0001 (reserved at the SQL gate, so no user column can
    /// collide).
    /// A struct-VALUED projection item — `struct_pack(n := e, ...)`, or that
    /// guarded by `CASE WHEN g IS NULL THEN NULL ELSE ... END` (θ export,
    /// slice 6). Lowered to the same wide-lane shape a named extern uses: a
    /// whole-validity lane (false = the whole struct is NULL, distinct from
    /// a struct of NULLs) plus one component lane per field. `None` when
    /// this isn't that shape.
    ///
    /// audit 2026-08-13: stricter than DuckDB — unnamed args
    /// (`struct_pack(i)` infers the field name there) and
    /// leading-underscore fields bind on the oracle; the recognizer
    /// refuses both (projection-loop-only, pydantic model boundary).
    /// Preserved. Field ACCESS over struct_pack serves since TASK-93
    /// (`desugar_struct_field` — never reaches this recognizer).
    fn struct_pack_lanes(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, Vec<String>)>, PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        // The guard arm: exactly the shape the θ rewrite emits. Anything
        // else with a struct in it falls through and refuses by name.
        let (guard, packed) = match inner {
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } if operand.is_none() && conditions.len() == 1 => {
                let arm = &conditions[0];
                // The oracle's own serialization parenthesizes both arms.
                let mut res = &arm.result;
                while let SqlExpr::Nested(i) = res {
                    res = i;
                }
                let is_null_lit = matches!(
                    res,
                    SqlExpr::Value(v) if matches!(v.value, SqlValue::Null)
                );
                match (is_null_lit, else_result) {
                    (true, Some(alt)) => (Some(&arm.condition), &**alt),
                    _ => return Ok(None),
                }
            }
            other => (None, other),
        };
        let mut packed_inner = packed;
        while let SqlExpr::Nested(i) = packed_inner {
            packed_inner = i;
        }
        let SqlExpr::Function(f) = packed_inner else {
            return Ok(None);
        };
        if !f.name.to_string().eq_ignore_ascii_case("struct_pack") {
            return Ok(None);
        }
        use sqlparser::ast::{FunctionArg, FunctionArguments};
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        // Every field must be NAMED — an unnamed struct_pack arg is a
        // binder error in the oracle too.
        let mut names: Vec<String> = Vec::with_capacity(list.args.len());
        let mut values: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Named { name, arg, .. } => {
                    let sqlparser::ast::FunctionArgExpr::Expr(v) = arg else {
                        return Ok(None);
                    };
                    names.push(name.value.clone());
                    values.push(v);
                }
                _ => return Ok(None),
            }
        }
        if names.is_empty() {
            return Ok(None);
        }
        for (i, n) in names.iter().enumerate() {
            // Measured: DuckDB's binder rejects a duplicate struct entry
            // name, case-insensitively — never serve what batch cannot.
            if names[..i].iter().any(|m| m.eq_ignore_ascii_case(n)) {
                return Err(PrepareError::Bind(format!(
                    "duplicate struct entry name \"{n}\""
                )));
            }
            // A _-leading field becomes a pydantic private attribute, so
            // the row model would silently drop it while batch serves it.
            if n.starts_with('_') {
                return Err(unsup(format!(
                    "struct field '{n}' cannot cross the row-path model \
                     boundary (a leading underscore is private) — rename it"
                )));
            }
        }
        // Whole-struct validity: the guard's IS NOT NULL, else always-true.
        let valid = match guard {
            None => SExpr {
                kind: SKind::Lit(Lit::I1(true)),
                ty: Ty::I1,
                nullable: false,
            },
            Some(g) => {
                let mut guard_inner = g;
                while let SqlExpr::Nested(i) = guard_inner {
                    guard_inner = i;
                }
                let SqlExpr::IsNull(target) = guard_inner else {
                    return Ok(None);
                };
                let bound = match self.expr_or_null(target)? {
                    None => {
                        // A provably-NULL guard: the struct is always NULL.
                        SExpr {
                            kind: SKind::Lit(Lit::I1(false)),
                            ty: Ty::I1,
                            nullable: false,
                        }
                    }
                    Some(t) => SExpr {
                        kind: SKind::IsNull {
                            negated: true,
                            inner: Box::new(t),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    },
                };
                bound
            }
        };
        let mut lanes = Vec::with_capacity(1 + values.len());
        lanes.push((format!("{base}\u{1}valid"), valid));
        for (j, v) in values.iter().enumerate() {
            // struct_pack(a := NULL) is STRUCT(a INTEGER) on DuckDB —
            // SQLNULL's int32 home (m-8 phase 2).
            let bound = match self.expr_or_null(v)? {
                None => null_of(Ty::I32),
                Some(x) => fold(x),
            };
            lanes.push((format!("{base}\u{1}{j}"), bound));
        }
        Ok(Some((lanes, names)))
    }

    fn unnest_extern_columns(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<Vec<(String, SExpr)>>, PrepareError> {
        let mut outer = e;
        while let SqlExpr::Nested(i) = outer {
            outer = i;
        }
        let SqlExpr::Function(uf) = outer else {
            return Ok(None);
        };
        if !uf.name.to_string().eq_ignore_ascii_case("unnest") {
            return Ok(None);
        }
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let FunctionArguments::List(list) = &uf.args else {
            return Ok(None);
        };
        // Measured: the oracle rejects every modifier on UNNEST itself
        // (DISTINCT/FILTER/in-call ORDER BY "not applicable to UNNEST",
        // OVER a catalog error, IGNORE NULLS a parser error) — refuse by
        // name rather than expanding as if unadorned (review round).
        if !list.clauses.is_empty()
            || list.duplicate_treatment.is_some()
            || uf.filter.is_some()
            || uf.over.is_some()
            || uf.null_treatment.is_some()
            || !uf.within_group.is_empty()
        {
            return Err(unsup(
                "modifier on UNNEST (DISTINCT, FILTER, ORDER BY, OVER and \
                 IGNORE NULLS are not applicable to UNNEST)"
                    .to_string(),
            ));
        }
        let [FunctionArg::Unnamed(FunctionArgExpr::Expr(arg))] = &list.args[..] else {
            return Ok(None);
        };
        let mut inner = arg;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        let SqlExpr::Function(f) = inner else {
            return Ok(None);
        };
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.ret_names.is_empty() {
            return Err(unsup(format!(
                "unnest of udf '{}': no declared output field names",
                spec.name
            )));
        }
        let args = self.bind_udf_args(f, spec)?;
        let site = self.site_for(f);
        Ok(Some(
            spec.ret_names
                .iter()
                .zip(&spec.rets)
                .enumerate()
                .map(|(j, (n, &rt))| {
                    (
                        n.clone(),
                        SExpr {
                            kind: SKind::ExternCall {
                                site,
                                ext,
                                args: args.clone(),
                                ret: j as u32,
                                whole: false,
                            },
                            ty: rt,
                            nullable: true,
                        },
                    )
                })
                .collect(),
        ))
    }

    fn wide_extern_lanes(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, Vec<String>)>, PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        let SqlExpr::Function(f) = inner else {
            return Ok(None);
        };
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.rets.len() < 2 && spec.ret_names.is_empty() {
            return Ok(None);
        }
        let args = self.bind_udf_args(f, spec)?;
        let site = self.site_for(f);
        let mut lanes = Vec::with_capacity(1 + spec.rets.len());
        lanes.push((
            format!("{base}\u{1}valid"),
            SExpr {
                kind: SKind::ExternCall {
                    site,
                    ext,
                    args: args.clone(),
                    ret: 0,
                    whole: true,
                },
                ty: Ty::I1,
                nullable: false,
            },
        ));
        for (j, &rt) in spec.rets.iter().enumerate() {
            lanes.push((
                format!("{base}\u{1}{j}"),
                SExpr {
                    kind: SKind::ExternCall {
                        site,
                        ext,
                        args: args.clone(),
                        ret: j as u32,
                        whole: false,
                    },
                    ty: rt,
                    nullable: true,
                },
            ));
        }
        Ok(Some((lanes, spec.ret_names.clone())))
    }

    /// A declared tree transform, called `name(<i64 id>, feat, feat, ..)` —
    /// the same shape as any other transform (`PythonTransform`'s implicit
    /// leading instance id, then the features), the difference being that
    /// this one lowers to the native kernel instead of an ecall.
    ///
    /// Features bind by POSITION, in the order the transform declared its
    /// `takes`. A call site is free to name its columns anything.
    fn tree_call(
        &self,
        cat: usize,
        args: &[&SqlExpr],
    ) -> Result<SExpr, PrepareError> {
        let decl = &self.models[cat];
        let name = &decl.name;
        let Some((id, feats)) = args.split_first() else {
            return Err(PrepareError::Bind(format!(
                "udf '{name}' takes {} argument(s), the call passes 0",
                decl.takes.len() + 1
            )));
        };
        if feats.len() != decl.takes.len() {
            return Err(PrepareError::Bind(format!(
                "udf '{name}' takes {} argument(s), the call passes {}",
                decl.takes.len() + 1,
                args.len()
            )));
        }

        let mut bound: Vec<SExpr> = Vec::with_capacity(feats.len());
        for (i, fexpr) in feats.iter().enumerate() {
            let want = decl.takes[i];
            // A bare NULL feature is legal and means "missing" — the model
            // has an answer for that. It types as f64 like any other.
            let e = match self.expr_or_null(fexpr)? {
                None => null_of(Ty::F64),
                // The DECLARED type decides, not the argument's: DuckDB casts
                // the argument to the declaration before calling, so a BIGINT
                // column in a declared-DOUBLE lane reaches the model as
                // `float64(n)` and narrows from there. An i64 argument in a
                // declared-BIGINT lane is the only one that narrows in one
                // step, and only on a float32 grid.
                Some(e) => match (e.ty, want) {
                    (Ty::F64, Ty::F64) => e,
                    // DuckDB's implicit widening, exactly as `bind_udf_args`
                    // does it for every other UDF.
                    (Ty::I64, Ty::F64) => promote_f64(e),
                    // How an integer reaches the compare depends on the grid
                    // the model set declared.
                    //
                    // On a float32 grid (sklearn) it must narrow in ONE step:
                    // `_validate_X_predict` does `int64 -> float32`, whereas
                    // `promote_f64` would give `float32(float64(n))` — two
                    // roundings, a whole float32 ULP off above 2**53. Below
                    // 2**53 `float64(n)` is exact and the two agree, which is
                    // what makes the narrowing safe for every integer feature
                    // rather than only large ones (TASK-77).
                    //
                    // On a float64 grid the integer reaches the compare
                    // exactly, and narrowing it would throw away precision
                    // that library had every right to keep (TASK-77's
                    // follow-up: the grid is the PACKER's property, so it is
                    // declared, not assumed).
                    (Ty::I64, Ty::I64) => match decl.grid {
                        CompareGrid::F32 => narrow_f32(promote_f64(e)),
                        CompareGrid::F64 => promote_f64(e),
                    },
                    (a, b) => {
                        return Err(PrepareError::Bind(format!(
                            "udf '{name}' argument {} is {}, declared {}",
                            i + 2,
                            a.name(),
                            b.name()
                        )))
                    }
                },
            };
            bound.push(e);
        }

        // The instance id gates the RESULT: an unseen group has no model.
        // Feature nullability deliberately does not propagate.
        let Some(bid) = self.expr_or_null(id)? else {
            return Ok(null_of(Ty::F64));
        };
        let bid = match bid.ty {
            t if t.is_int() => bid,
            other => {
                return Err(PrepareError::Bind(format!(
                    "udf '{name}' argument 1 is {}, declared the instance id (i64)",
                    other.name()
                )))
            }
        };
        let nullable = bid.nullable;

        let mut refs = self.model_refs.borrow_mut();
        let model = match refs.iter().position(|r| *r == cat as u32) {
            Some(i) => i as u32,
            None => {
                refs.push(cat as u32);
                (refs.len() - 1) as u32
            }
        };
        Ok(SExpr {
            kind: SKind::TreePredict {
                model,
                id: Box::new(bid),
                feats: bound,
            },
            ty: Ty::F64,
            nullable,
        })
    }

    /// Index of a declared tree transform by call-site name, matched
    /// case-insensitively like [`Self::find_udf`] — they share a namespace.
    fn find_tree(&self, name: &str) -> Option<usize> {
        self.models
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// TASK-92 resolution head for `WholeCallNull` table rows: arity,
    /// eager argument binding, the bare-NULL whole-call short-circuit,
    /// per-arg type checks (byte-identical error strings), promotion into
    /// the f64 lane for the DOUBLE-returning math rows, and the result
    /// type. The arm then only builds its node.
    ///
    /// audit 2026-08-13: the NULL short-circuit running BEFORE the type
    /// checks is the audited dominant pattern — replace(NULL, 1, 2) binds
    /// NULL::VARCHAR and pow(s, NULL) binds NULL::DOUBLE (the latter is
    /// looser than DuckDB, which refuses the VARCHAR sibling). Preserved.
    fn sig_resolve(
        &self,
        name: &str,
        sig: &Sig,
        args: &[&SqlExpr],
    ) -> Result<SigArgs, PrepareError> {
        let n = sig.params.len();
        if (!sig.variadic && args.len() != n) || (sig.variadic && args.len() < n) {
            // audit 2026-08-13: reverse alone spells its arity error
            // "takes one argument"; every sibling says "exactly 1".
            return Err(PrepareError::Bind(if name == "reverse" {
                format!("{name} takes one argument")
            } else {
                match n {
                    0 => format!("{name} takes no arguments"),
                    1 => format!("{name} takes exactly 1 argument"),
                    _ => format!("{name} takes exactly {n} arguments"),
                }
            }));
        }
        let mut bound: Vec<Option<SExpr>> = Vec::with_capacity(args.len());
        for a in args {
            bound.push(self.expr_or_null(a)?);
        }
        if bound.iter().any(Option::is_none) {
            // A bare NULL adopts the result type; Arg(_) rows adopt BIGINT
            // (measured: abs(NULL) binds abs(BIGINT) in DuckDB).
            let ty = match sig.ret {
                Ret::Fixed(t) => t,
                _ => Ty::I64,
            };
            return Ok(SigArgs::Null(null_of(ty)));
        }
        let mut out = Vec::with_capacity(bound.len());
        for (p, e) in sig.params.iter().zip(bound) {
            let e = e.expect("checked above");
            if !sig::arg_ok(*p, e.ty) {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    e.ty.name()
                )));
            }
            // Num args feed the f64 lane when the row returns DOUBLE (the
            // math1/math2 families); Arg(0) rows (abs) keep their type.
            out.push(
                if matches!(p, ArgTy::Num) && sig.ret == Ret::Fixed(Ty::F64) {
                    promote_f64(e)
                } else {
                    e
                },
            );
        }
        let ret = match sig.ret {
            Ret::Fixed(t) => t,
            Ret::Arg(i) => out[i].ty,
            Ret::Widen | Ret::Unify => {
                unreachable!("no WholeCallNull table row uses these")
            }
        };
        Ok(SigArgs::Bound(out, ret))
    }

    fn function(&self, f: &sqlparser::ast::Function) -> Result<SExpr, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        // TASK-81: DuckDB refuses every call-node modifier on a scalar call
        // (OVER is a catalog error, FILTER invalid input, IGNORE NULLS a
        // parser error) while these fields silently fell on the floor here,
        // so the bare call was served where the oracle errors — the fuzz
        // campaign's largest class. Destructured EXHAUSTIVELY (no `..`) for
        // the TASK-69 reason: a modifier field added to sqlparser must break
        // this build, not the answers.
        let sqlparser::ast::Function {
            name: _,
            uses_odbc_syntax,
            parameters,
            args: _,
            filter,
            null_treatment,
            over,
            within_group,
        } = f;
        if *uses_odbc_syntax
            || !matches!(parameters, FunctionArguments::None)
            || filter.is_some()
            || null_treatment.is_some()
            || over.is_some()
            || !within_group.is_empty()
        {
            return Err(unsup(format!(
                "modifier on scalar call {} (FILTER, OVER, IGNORE NULLS and \
                 WITHIN GROUP apply to aggregates and window functions, \
                 which this engine does not serve)",
                f.name
            )));
        }
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
        // TASK-92: names with a WholeCallNull signature row resolve here
        // (sig.rs is the catalogue of what they accept and return); their
        // arms below only build nodes. Custom rows and CUSTOM_NAMES keep
        // every gate in their arm, verbatim.
        let resolved: Option<(Vec<SExpr>, Ty)> = match sig::lookup(&name) {
            Some(s) if s.null_arg == NullArg::WholeCallNull => {
                match self.sig_resolve(&name, s, &args)? {
                    SigArgs::Null(e) => return Ok(e),
                    SigArgs::Bound(bound, ret) => Some((bound, ret)),
                }
            }
            _ => None,
        };
        match name.as_str() {
            // ucase/lcase are alias-identical to upper/lower (wave-3 pins:
            // exhaustive all-codepoint sweep, zero mismatches).
            "upper" | "lower" | "ucase" | "lcase" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StrCase {
                        upper: matches!(name.as_str(), "upper" | "ucase"),
                        a: Box::new(inner),
                    },
                    ty,
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
            "instr" | "strpos" | "position" | "starts_with" | "prefix" | "ends_with"
            | "suffix" => {
                let op = match name.as_str() {
                    "instr" | "strpos" | "position" => StrOp2::Find,
                    "starts_with" | "prefix" => StrOp2::Starts,
                    _ => StrOp2::Ends,
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([h, n]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = h.nullable || n.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op,
                        a: Box::new(h),
                        b: Box::new(n),
                    },
                    ty,
                    nullable,
                })
            }
            // Custom row: the bare-NULL-needle ambiguity gate (MAP/LIST
            // overloads) lives in str2 and must see the raw args.
            "contains" => {
                let [h, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                self.str2(&name, StrOp2::Contains, h, n)
            }
            "length" | "len" | "char_length" | "character_length" | "strlen" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::SLen {
                        bytes: name == "strlen",
                        a: Box::new(inner),
                    },
                    ty,
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
                let (bound, _ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                Ok(math1_node(op, inner))
            }
            "log" => match args[..] {
                [x] => self.math1("log", NumOp1::Log10, x),
                [b, x] => self.math2("log", BinOp::Flogb, b, x),
                _ => Err(PrepareError::Bind("log takes 1 or 2 arguments".to_string())),
            },
            // One table row: the binary DOUBLE lane (math2's audited NULL
            // ordering now lives in the resolution head). fdiv/fmod are
            // the FLOOR pair — always DOUBLE, even for two int args.
            "pow" | "power" | "fdiv" | "fmod" | "nextafter" => {
                let op = match name.as_str() {
                    "pow" | "power" => BinOp::Fpow,
                    "fdiv" => BinOp::Ffloordiv,
                    "fmod" => BinOp::Ffloormod,
                    _ => BinOp::Fnextafter,
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([a, b]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::MathF2 {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty,
                    nullable,
                })
            }
            "pi" => {
                let _ = resolved.expect("signature row"); // arity 0 checked
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
                        // Measured: integer trunc is identity, WIDTH preserved.
                        t if t.is_int() => Ok(inner),
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
                // abs(NULL) binds to abs(BIGINT) in DuckDB — the head's
                // Arg(0)-row NULL rule.
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                // Width-polymorphic via the Arg(0) row: abs follows its
                // argument (measured).
                let nullable = inner.nullable;
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
                        // Measured: integer round is identity, WIDTH preserved.
                        t if t.is_int() => Ok(inner),
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
                // audit 2026-08-13: stricter than DuckDB twice — it binds
                // coalesce(NULL, NULL) as INTEGER and unifies BOOLEAN with
                // ints (coalesce(b, i) -> INTEGER); both refuse here
                // (BOOLEAN+DOUBLE refuses on both engines). Preserved.
                self.in_guarded.set(self.in_guarded.get() + 1);
                let _guard = GuardScope(&self.in_guarded);
                let mut bound = Vec::with_capacity(args.len());
                for arg in &args {
                    if let Some(e) = self.expr_or_null(arg)? {
                        // The hint rides with the arg's own SPELLING (never
                        // the bound node — fleet 2026-08-13).
                        bound.push((e, ast_int_literal(arg)));
                    } // literal NULL args never produce a value: drop them
                }
                if bound.is_empty() {
                    return Err(unsup("COALESCE of only NULL literals"));
                }
                // Seed-then-combine (DuckDB's fold, 0 errors on 19k
                // probes): the seed keeps its literal hint; every combine
                // makes the accumulator computed.
                let mut unified = bound[0].0.ty;
                let mut acc_lit = bound[0].1;
                for (e, new_lit) in &bound[1..] {
                    unified = match (unified, e.ty) {
                        (u, t) if u == t => u,
                        (u, t) if u.is_int() && t.is_int() => {
                            int_width_promote(u, acc_lit, t, *new_lit)
                        }
                        (u, t) if u.is_int() && t == Ty::F64 => Ty::F64,
                        (Ty::F64, t) if t.is_int() => Ty::F64,
                        (u, t) => {
                            return Err(PrepareError::Bind(format!(
                                "COALESCE arguments disagree: {} vs {}",
                                u.name(),
                                t.name()
                            )))
                        }
                    };
                    acc_lit = None;
                }
                let bound: Vec<SExpr> = bound.into_iter().map(|(e, _)| e).collect();
                let mut bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|mut e| {
                        if e.ty.is_int() && unified == Ty::F64 {
                            promote_f64(e)
                        } else {
                            // Width-only retype: fold may select this arm
                            // whole, and the OUTPUT width is the unified one.
                            e.ty = unified;
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
            // audit 2026-08-13: stricter than DuckDB twice — it binds
            // least(NULL, NULL) as INTEGER and unifies BOOLEAN with ints
            // (least(b, k) -> BIGINT); both refuse here (BOOLEAN+DOUBLE
            // refuses on both engines). Preserved.
            "least" | "greatest" => {
                if args.is_empty() {
                    return Err(PrepareError::Bind(format!(
                        "{name} needs at least 1 argument"
                    )));
                }
                let mut bound = Vec::new();
                for arg in &args {
                    // Literal NULL args contribute nothing (NULL-ignoring);
                    // hints ride with the SPELLING (fleet 2026-08-13).
                    if let Some(e) = self.expr_or_null(arg)? {
                        bound.push((e, ast_int_literal(arg)));
                    }
                }
                if bound.is_empty() {
                    return Err(unsup(format!("{name} of only NULL literals")));
                }
                // Seed-then-combine, same fold as COALESCE above.
                let mut unified = bound[0].0.ty;
                let mut acc_lit = bound[0].1;
                for (e, new_lit) in &bound[1..] {
                    unified = match (unified, e.ty) {
                        (u, t) if u == t => u,
                        (u, t) if u.is_int() && t.is_int() => {
                            int_width_promote(u, acc_lit, t, *new_lit)
                        }
                        (u, t) if u.is_int() && t == Ty::F64 => Ty::F64,
                        (Ty::F64, t) if t.is_int() => Ty::F64,
                        (u, t) => {
                            return Err(PrepareError::Bind(format!(
                                "{name} arguments disagree: {} vs {}",
                                u.name(),
                                t.name()
                            )))
                        }
                    };
                    acc_lit = None;
                }
                let bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|(e, _)| e)
                    .map(|mut e| {
                        if e.ty.is_int() && unified == Ty::F64 {
                            promote_f64(e)
                        } else {
                            // Width-only retype (see COALESCE above).
                            e.ty = unified;
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
                // audit 2026-08-13: stricter than DuckDB wherever cmp(Eq)
                // refuses a mix — nullif(s, i) -> VARCHAR and nullif(b, b)
                // -> BOOLEAN both bind there (compare at the promoted type,
                // result keeps arg 1's type). Preserved.
                let [a, b] = args[..] else {
                    return Err(PrepareError::Bind(
                        "nullif takes exactly 2 arguments".to_string(),
                    ));
                };
                match (self.expr_or_null(a)?, self.expr_or_null(b)?) {
                    // TASK-86 face closed by m-8 phase 2: DuckDB types the
                    // bare NULL first argument INTEGER, nullif's output
                    // takes the first argument's type, and NULL = b is
                    // never TRUE — the whole call IS an int32 NULL.
                    (None, _) => Ok(null_of(Ty::I32)),
                    // a = NULL is never TRUE, so nullif(a, NULL) is a.
                    (Some(a), None) => Ok(a),
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
                let (bound, ty) = resolved.expect("signature row");
                let Ok([a, b]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty,
                    nullable,
                })
            }
            "repeat" => {
                let [s, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                let (bs, bn) = (self.expr_or_null(s)?, self.expr_or_null(n)?);
                // TASK-86: a bare NULL string picks DuckDB's BLOB overload,
                // so the answer is BLOB there and string here — and every
                // OUTER call binding the result splits (strpos/ltrim/lower/
                // levenshtein/LIKE refuse BLOB on DuckDB while building
                // here). A NULL COUNT stays: both engines type that VARCHAR.
                let Some(bs) = bs else {
                    return Err(unsup(
                        "bare NULL as repeat's string (DuckDB picks the BLOB \
                         overload; spell it CAST(NULL AS VARCHAR))",
                    ));
                };
                let Some(bn) = bn else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str {
                    // No implicit numeric->VARCHAR cast (measured).
                    return Err(PrepareError::Bind(format!(
                        "no function matches repeat({})",
                        bs.ty.name()
                    )));
                }
                if !bn.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}(str, {})",
                        bn.ty.name()
                    )));
                }
                refuse_budget_breaking_count(&name, &bn)?;
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
                // audit 2026-08-13: a bare-NULL subject types Str here;
                // DuckDB's SQLNULL fallback types the slice INTEGER (while
                // array_extract(NULL, 2) agrees at VARCHAR). Value parity
                // holds. Preserved.
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
                // TASK-82: DuckDB's {l,r}pad count is INTEGER and its binder
                // does NOT downcast — a BIGINT count is a binder error there
                // (169 of the first campaign's 963 findings). Now that
                // widths are typed, the gate IS the type: INTEGER or
                // narrower binds, BIGINT refuses. The NULL short-circuit
                // below must not skip this check (certification seed 1589);
                // a bare-NULL count itself is fine: DuckDB types it INTEGER.
                let count_is_int32 =
                    |e: &SExpr| e.ty.is_int() && e.ty != Ty::I64;
                let bad_count = format!(
                    "no function matches {name}(VARCHAR, BIGINT, VARCHAR) — \
                     DuckDB's {name} count is INTEGER and a BIGINT does not \
                     implicitly narrow; spell a constant count as a plain \
                     literal or CAST(.. AS INTEGER)"
                );
                if bl.as_ref().is_some_and(|e| e.ty == Ty::I64)
                    && (bs.is_none() || bp.is_none())
                {
                    return Err(PrepareError::Bind(bad_count));
                }
                let (Some(bs), Some(bl), Some(bp)) = (bs, bl, bp) else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str || bp.ty != Ty::Str || !bl.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({}, {}, {})",
                        bs.ty.name(),
                        bl.ty.name(),
                        bp.ty.name()
                    )));
                }
                if !count_is_int32(&bl) {
                    return Err(PrepareError::Bind(bad_count));
                }
                refuse_budget_breaking_count(&name, &bl)?;
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
                let (bound, ty) = resolved.expect("signature row");
                let Ok([bs, bx, by]) = <[SExpr; 3]>::try_from(bound) else {
                    unreachable!("arity 3")
                };
                let nullable = bs.nullable || bx.nullable || by.nullable;
                Ok(SExpr {
                    kind: SKind::Str3 {
                        op,
                        a: Box::new(bs),
                        b: Box::new(bx),
                        c: Box::new(by),
                    },
                    ty,
                    nullable,
                })
            }
            // unicode('') = ord('') = -1, but ascii('') = 0 — the measured
            // sole divergence; all return the FIRST codepoint otherwise.
            "unicode" | "ord" | "ascii" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Sord {
                        empty_zero: name == "ascii",
                        a: Box::new(inner),
                    },
                    // Fixed(I32) row: a codepoint is INTEGER on DuckDB (the
                    // length family, by contrast, is BIGINT).
                    ty,
                    nullable,
                })
            }
            // bit_length = 8 * strlen exactly (measured) — pure desugar.
            "bit_length" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                let slen = SExpr {
                    kind: SKind::SLen {
                        bytes: true,
                        a: Box::new(inner),
                    },
                    ty,
                    nullable,
                };
                self.arith(ArithOp::Mul, lit_i64(8), slen, (None, None))
            }
            "strip_accents" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StripAccents(Box::new(inner)),
                    ty,
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
                self.arith(op, bx, by, (ast_int_literal(x), ast_int_literal(y)))
            }
            // Named rejects (wave-3 AC #3): each states WHY, not just what.
            "sum" | "count" | "avg" | "min" | "max" | "geomean" | "product" | "string_agg"
            | "first" | "last" | "any_value" => Err(unsup(format!(
                "aggregate function {name} (no aggregation in v0)"
            ))),
            // Wave-B regexp family (pins: 2026-07-27-waveB-regexp-pins.md).
            "regexp_matches" | "regexp_full_match" => {
                let (s, p, opts) = match args[..] {
                    [s, p] => (s, p, None),
                    [s, p, o] => (s, p, Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 2 or 3 arguments"
                        )))
                    }
                };
                let o = self.regex_options(opts, false)?;
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::I1));
                };
                let full = name == "regexp_full_match";
                let bs = str_only(&name, bs)?;
                match self.regex_pattern(p, o, full)? {
                    None => Ok(null_of(Ty::I1)),
                    Some(re) => Ok(SExpr {
                        nullable: bs.nullable,
                        kind: SKind::ReMatch {
                            re,
                            a: Box::new(bs),
                        },
                        ty: Ty::I1,
                    }),
                }
            }
            "regexp_extract" => {
                // audit 2026-08-13: stricter than DuckDB — its third arg
                // also accepts a constant NAME-LIST returning a STRUCT
                // (regexp_extract(s, '(a)(b)', ['x','y'])); absent here.
                let (s, p, group, opts) = match args[..] {
                    [s, p] => (s, p, None, None),
                    [s, p, g] => (s, p, Some(g), None),
                    [s, p, g, o] => (s, p, Some(g), Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 2 to 4 arguments"
                        )))
                    }
                };
                let o = self.regex_options(opts, false)?;
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                let bs = str_only(&name, bs)?;
                // Group index: constant, flat 0..9 range check unrelated to
                // the pattern; NULL group -> '' for non-NULL subjects.
                let group = match group {
                    None => 0u32,
                    Some(g) => match self.expr_or_null(g)? {
                        None => return Ok(empty_for_nonnull(bs)),
                        Some(bg) => match bg.kind {
                            SKind::Lit(Lit::I64(n)) if (0..=9).contains(&n) => n as u32,
                            SKind::Lit(Lit::I64(_)) => {
                                return Err(PrepareError::Bind(
                                    "Group index must be between 0 and 9!".into(),
                                ))
                            }
                            _ => {
                                return Err(unsup(
                                    "non-constant regexp_extract group index",
                                ))
                            }
                        },
                    },
                };
                match self.regex_pattern(p, o, false)? {
                    None => Ok(null_of(Ty::Str)),
                    Some(re) => Ok(SExpr {
                        nullable: bs.nullable,
                        kind: SKind::ReExtract {
                            re,
                            group,
                            a: Box::new(bs),
                        },
                        ty: Ty::Str,
                    }),
                }
            }
            "regexp_replace" => {
                let (s, p, r, opts) = match args[..] {
                    [s, p, r] => (s, p, r, None),
                    [s, p, r, o] => (s, p, r, Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 3 or 4 arguments"
                        )))
                    }
                };
                // Pinned asymmetry: for regexp_replace ANY NULL argument
                // (including the options string) -> NULL result.
                let Some(o) = self.regex_options_nullable(opts)? else {
                    return Ok(null_of(Ty::Str));
                };
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                let bs = str_only(&name, bs)?;
                let Some(br) = self.expr_or_null(r)? else {
                    return Ok(null_of(Ty::Str));
                };
                if matches!(br.kind, SKind::NullOf) && br.ty == Ty::Str {
                    // CAST(NULL AS VARCHAR) replacement — NULL result.
                    return Ok(null_of(Ty::Str));
                }
                let SKind::Lit(Lit::Str(rw)) = br.kind else {
                    return Err(unsup("non-constant regexp_replace replacement"));
                };
                let Some((re, group_count)) = self.regex_pattern_counted(p, o, false)? else {
                    return Ok(null_of(Ty::Str));
                };
                match super::retrans::translate_rewrite(&rw, group_count, o.global) {
                    // Invalid rewrites never error (measured RE2 quirks).
                    super::retrans::Rewrite::Identity => Ok(bs),
                    super::retrans::Rewrite::Template(t)
                    | super::retrans::Rewrite::ConsumeWithPrefix(t) => {
                        self.regexes.borrow_mut()[re as usize].rewrite = Some(t);
                        Ok(SExpr {
                            nullable: bs.nullable,
                            kind: SKind::ReReplace {
                                re,
                                global: o.global,
                                a: Box::new(bs),
                            },
                            ty: Ty::Str,
                        })
                    }
                }
            }
            "regexp_split_to_array" | "regexp_extract_all" => Err(unsup(format!(
                "function {name} (list-valued — non-scalar in v0)"
            ))),
            "reverse" => {
                // TASK-56 lifts the wave-3 descope: ASCII byte path +
                // UAX-29 extended grapheme path (pins-waveA). No implicit
                // casts — reverse(123) is a DuckDB binder error.
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Reverse(Box::new(inner)),
                    ty,
                    nullable,
                })
            }
            // The FUNCTION spelling of field access over a wide extern
            // (TASK-63) — DuckDB serializes it distinct from the dot form.
            // Over struct_pack it is the TASK-93 desugar instead.
            "struct_extract" => {
                if let Some(sub) = self.desugar_struct_extract(f)? {
                    return self.expr(&sub);
                }
                if let [target, SqlExpr::Value(v)] = args[..] {
                    if let SqlValue::SingleQuotedString(field) = &v.value {
                        let mut base = target;
                        while let SqlExpr::Nested(i) = base {
                            base = i;
                        }
                        if let SqlExpr::Function(func) = base {
                            if let Some(lane) = self.extern_field_lane(func, field)? {
                                return Ok(lane);
                            }
                        }
                    }
                }
                Err(unsup(format!(
                    "function {} (not in the v0 catalogue)",
                    f.name
                )))
            }
            _ => {
                // audit 2026-08-13: DuckDB 1.5.5 HAS if() and ifnull()
                // (iif/nvl are absent there too); neither is in the
                // catalogue, so a UDF/tree may claim those two names and
                // silently diverge from oracle semantics. Preserved.
                // A declared tree transform: same namespace as the ecall
                // UDFs, but it lowers to the native kernel rather than a
                // callback, so it is resolved before them.
                if let Some(cat) = self.find_tree(&f.name.to_string()) {
                    return self.tree_call(cat, &args);
                }
                // Declared UDF externs (DRAFT-22): width-1 is an ordinary
                // scalar expression; width-k is bare-item-only (handled in
                // the projection loop), so reaching it here is refused.
                if let Some((ext, spec)) = self.find_udf(&f.name.to_string()) {
                    if !spec.ret_names.is_empty() {
                        // Struct-valued at every width (the subtraction
                        // loop): a named extern MID-EXPRESSION has no
                        // scalar reading — DuckDB's struct registration
                        // would binder-error. Bare items take the struct
                        // boundary in the projection loop (slice 5).
                        return Err(unsup(format!(
                            "udf '{}' is struct-valued (declared field names) \
                             — serve it as its own SELECT item or address an \
                             output field ({}(...).name)",
                            spec.name, spec.name
                        )));
                    }
                    if spec.rets.len() != 1 {
                        return Err(unsup(format!(
                            "width-{} udf '{}' used as a scalar expression \
                             (multi-output transformer calls must be bare SELECT items)",
                            spec.rets.len(),
                            spec.name
                        )));
                    }
                    let args = self.bind_udf_args(f, spec)?;
                    return Ok(SExpr {
                        kind: SKind::ExternCall {
                            site: self.fresh_site(),
                            ext,
                            args,
                            ret: 0,
                            whole: false,
                        },
                        ty: spec.rets[0],
                        nullable: true,
                    });
                }
                Err(unsup(format!(
                    "function {} (not in the v0 catalogue)",
                    f.name
                )))
            }
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
    ///
    /// audit 2026-08-13: looser than DuckDB twice — its digits slot maxes
    /// at INTEGER (round(d, k::BIGINT) is a binder error there, any I64
    /// binds here), and a bare-NULL subject returns before the digits slot
    /// is even bound (round(NULL, s) is NULL here, a binder error there).
    /// This interleaved order is also why round/trunc stay Custom rows.
    /// Preserved.
    fn round2(&self, trunc: bool, x: &SqlExpr, n: &SqlExpr) -> Result<SExpr, PrepareError> {
        let name = if trunc { "trunc" } else { "round" };
        let Some(subject) = self.expr_or_null(x)? else {
            return Ok(null_of(Ty::I64));
        };
        if !subject.ty.is_int() && subject.ty != Ty::F64 {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}({}, digits)",
                subject.ty.name()
            )));
        }
        let ty = subject.ty;
        let Some(digits) = self.expr_or_null(n)? else {
            return Ok(null_of(ty));
        };
        if !digits.ty.is_int() {
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
            t if t.is_int() => promote_f64(inner),
            other => {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    other.name()
                )))
            }
        };
        Ok(math1_node(op, inner))
    }

    /// Wave-1 binary f64 math (now just log(base, x); the fixed-arity
    /// members read the signature table). A literal NULL in either slot
    /// pre-empts every domain check (measured: log(-2.0, NULL) is NULL,
    /// not an error).
    ///
    /// audit 2026-08-13: the NULL short-circuit preceding the type checks
    /// is looser than DuckDB — log(s, NULL) binds NULL::DOUBLE here where
    /// it refuses the VARCHAR sibling. Preserved (the head reproduces the
    /// same order for the table rows).
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
                t if t.is_int() => Ok(promote_f64(e)),
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
                Some(x) if x.ty.is_int() => Ok(Some(x)),
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
        t if t.is_int() || t == Ty::F64 => {
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

/// A bind-fold result value as a literal of the declared type (TASK-101).
fn scalar_lit(v: ScalarVal, ty: Ty) -> SExpr {
    let lit = match v {
        ScalarVal::I1(x) => Lit::I1(x),
        ScalarVal::I64(x) => Lit::I64(x),
        ScalarVal::F64(x) => Lit::F64(x),
        ScalarVal::Str(x) => Lit::Str(x),
    };
    SExpr {
        kind: SKind::Lit(lit),
        ty,
        nullable: false,
    }
}

fn null_of(ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::NullOf,
        ty,
        nullable: true,
    }
}

/// TASK-88: a pad/repeat COUNT literal that can exceed the engine's 1 GiB
/// string-builder budget refuses at build. DuckDB's behaviour past that
/// size is a coin flip between serving the multi-GB string and its own
/// builder error, so a giant literal count could never be a stable
/// bit-for-bit answer; refusal is the sanctioned mode, and the judgement
/// (no gigabyte allocations in a serving engine) is recorded on the
/// ticket. Data-driven counts keep the runtime cap, documented in
/// known-limitations.md.
fn refuse_budget_breaking_count(name: &str, count: &SExpr) -> Result<(), PrepareError> {
    const BUDGET: i64 = 1 << 30; // bytes; an n-char 1-byte result is n bytes
    if let SKind::Lit(Lit::I64(n)) = fold(count.clone()).kind {
        if n > BUDGET {
            return Err(PrepareError::Bind(format!(
                "{name} count {n} exceeds the 1 GiB string-builder budget — \
                 the result could never serve; DuckDB's own behaviour past \
                 this size is unstable (its builder error or a multi-GB \
                 string, spelling-dependent)"
            )));
        }
    }
    Ok(())
}

/// Checked-int32 evaluation of a LITERAL-shaped integer subtree, mirroring
/// DuckDB's typing: an int32-range number literal is INTEGER there, INTEGER
/// op INTEGER stays INTEGER, and overflow is a runtime ERROR. `NotShaped`
/// means some leaf isn't an int32 literal (column, cast, big literal, `/`
/// which is DOUBLE there) — 64-bit semantics apply and nothing refuses.
/// `Fine(None)` is a NULL-valued but trap-free subtree (INTEGER % 0 is NULL
/// on DuckDB, measured in the wave pins).
enum I32Fold {
    NotShaped,
    Traps,
    Fine(Option<i32>),
}

/// All-literal integer subtree evaluated in i128 — the wide arithmetic
/// DuckDB's range analysis folds comparisons through (TASK-87 face B).
/// None = not literal-shaped, or a rem-by-zero (NULL at runtime, no fold).
fn eval_i128_literal(e: &SqlExpr) -> Option<i128> {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => text.parse::<i128>().ok(),
            _ => None,
        },
        SqlExpr::Nested(inner) => eval_i128_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_i128_literal(expr),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => eval_i128_literal(expr)?.checked_neg(),
        SqlExpr::BinaryOp { left, op, right } => {
            let (x, y) = (eval_i128_literal(left)?, eval_i128_literal(right)?);
            match op {
                BinaryOperator::Plus => x.checked_add(y),
                BinaryOperator::Minus => x.checked_sub(y),
                BinaryOperator::Multiply => x.checked_mul(y),
                BinaryOperator::Modulo if y != 0 => x.checked_rem(y),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A literal-shaped numeric bound: a Number under parens/unary sign only —
/// binding one consumes no call site, so the Between dead-range probe may
/// bind it without breaking extern call-count parity (TASK-87 face C).
fn const_number_shaped(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Value(v) => matches!(v.value, SqlValue::Number(..)),
        SqlExpr::Nested(inner) => const_number_shaped(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus | UnaryOperator::Plus,
            expr,
        } => const_number_shaped(expr),
        _ => false,
    }
}

fn eval_i32_literal(e: &SqlExpr) -> I32Fold {
    use I32Fold::{Fine, NotShaped, Traps};
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => match text.parse::<i64>() {
                Ok(n) => match i32::try_from(n) {
                    Ok(n) => Fine(Some(n)),
                    Err(_) => NotShaped, // BIGINT literal there too
                },
                Err(_) => NotShaped,
            },
            _ => NotShaped,
        },
        SqlExpr::Nested(inner) => eval_i32_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_i32_literal(expr),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match eval_i32_literal(expr) {
            Fine(Some(n)) => n.checked_neg().map_or(Traps, |n| Fine(Some(n))),
            other => other,
        },
        SqlExpr::BinaryOp { left, op, right } => {
            let f = match op {
                BinaryOperator::Plus => i32::checked_add,
                BinaryOperator::Minus => i32::checked_sub,
                BinaryOperator::Multiply => i32::checked_mul,
                BinaryOperator::Modulo => i32::checked_rem,
                _ => return NotShaped,
            };
            match (eval_i32_literal(left), eval_i32_literal(right)) {
                (Fine(x), Fine(y)) => match (x, y) {
                    (Some(_), Some(0)) if matches!(op, BinaryOperator::Modulo) => {
                        Fine(None) // INTEGER % 0 is NULL, not a trap
                    }
                    (Some(x), Some(y)) => f(x, y).map_or(Traps, |n| Fine(Some(n))),
                    _ => Fine(None), // NULL propagates trap-free
                },
                (Traps, _) | (_, Traps) => Traps,
                _ => NotShaped,
            }
        }
        _ => NotShaped,
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
        // DuckDB's named widths. INT8 is BIGINT (eight BYTES); HUGEINT and
        // the unsigned family still collapse to i64 (range divergence noted
        // in the module docs; i128 is m-8 phase 4).
        Ok(match name.as_str() {
            "TINYINT" | "INT1" => Ty::I8,
            "SMALLINT" | "INT2" | "SHORT" => Ty::I16,
            "INTEGER" | "INT" | "INT4" | "SIGNED" => Ty::I32,
            _ => Ty::I64,
        })
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
                // DuckDB types a bare integer literal by magnitude: INTEGER
                // when it fits (never narrower), else BIGINT. `-2147483648`
                // parses as -(2147483648) and stays BIGINT there too — that
                // falls out of unary minus binding, not of this rule.
                let ty = if i32::try_from(i).is_ok() { Ty::I32 } else { Ty::I64 };
                (Lit::I64(i), ty)
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

/// The operator-table symbol for an [`ArithOp`] (TASK-92): the key into
/// `sig::OPS`, where the RESULT-TYPE rules live.
fn arith_sym(op: ArithOp) -> &'static str {
    match op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Div => "/",
        ArithOp::IDiv => "//",
        ArithOp::Rem => "%",
        ArithOp::Shl => "<<",
        ArithOp::Shr => ">>",
        ArithOp::BitAnd => "&",
        ArithOp::BitOr => "|",
        ArithOp::BitXor => "xor",
    }
}

/// The operator-table symbol for a [`CmpPred`] (TASK-92).
fn cmp_sym(pred: CmpPred) -> &'static str {
    match pred {
        CmpPred::Eq => "=",
        CmpPred::Ne => "<>",
        CmpPred::Lt => "<",
        CmpPred::Le => "<=",
        CmpPred::Gt => ">",
        CmpPred::Ge => ">=",
    }
}

/// DuckDB numeric promotion. The result-type RULE is the operator's
/// `sig::OPS` row — `/` is Fixed(F64), everything else Widens across the
/// integer width lattice (m-8 phase 2); this function is the rule's
/// consumer and owns the promotion nodes.
fn numeric_promote(
    op: ArithOp,
    a: SExpr,
    b: SExpr,
    lits: (Option<i64>, Option<i64>),
) -> Result<(SExpr, SExpr, Ty), PrepareError> {
    let numeric = |e: &SExpr| e.ty.is_int() || e.ty == Ty::F64;
    if !numeric(&a) || !numeric(&b) {
        return Err(PrepareError::Bind(format!(
            "arithmetic needs numeric operands, got {} and {}",
            a.ty.name(),
            b.ty.name()
        )));
    }
    let ty = match sig::op_ret(arith_sym(op)) {
        Ret::Fixed(t) => t,
        Ret::Widen => {
            if a.ty == Ty::F64 || b.ty == Ty::F64 {
                Ty::F64
            } else {
                int_width_promote(a.ty, lits.0, b.ty, lits.1)
            }
        }
        Ret::Arg(_) | Ret::Unify => unreachable!("not an operator rule"),
    };
    if ty == Ty::F64 {
        // promote_f64 is identity on an already-F64 operand.
        Ok((promote_f64(a), promote_f64(b), Ty::F64))
    } else {
        Ok((a, b, ty))
    }
}

fn width_rank(t: Ty) -> u8 {
    match t {
        Ty::I8 => 0,
        Ty::I16 => 1,
        Ty::I32 => 2,
        _ => 3,
    }
}

/// DuckDB's integer-width promotion (measured 2026-08-13): the wider side
/// wins — except a constant literal whose VALUE fits the narrower operand's
/// range adopts that width (c8 + 127 is TINYINT, c8 + 128 is INTEGER,
/// skipping SMALLINT). Family constructs (CASE/COALESCE/greatest) apply the
/// wider-side rule only; their literal-vs-narrower corner is reachable only
/// through explicit ::TINYINT/::SMALLINT casts mixed into multi-arm
/// unification.
/// One width-combine step — DuckDB's measured rule (2026-08-13 fleet,
/// scored over 19k queries): equal widths keep; a WIDER side that is a
/// syntactic literal narrows to a narrower NON-literal side, when its
/// value fits; otherwise the wider width wins. `unicode(s) % -2147483648`
/// is INTEGER; `2147483647 % -2147483648` is BIGINT.
fn int_width_promote(a_ty: Ty, a_lit: Option<i64>, b_ty: Ty, b_lit: Option<i64>) -> Ty {
    if a_ty == b_ty {
        return a_ty;
    }
    let (wide, narrow, wide_lit, narrow_lit) = if width_rank(a_ty) >= width_rank(b_ty) {
        (a_ty, b_ty, a_lit, b_lit)
    } else {
        (b_ty, a_ty, b_lit, a_lit)
    };
    if let (Some(v), None, Some((lo, hi))) = (wide_lit, narrow_lit, narrow.int_range()) {
        if (lo..=hi).contains(&v) {
            return narrow;
        }
    }
    wide
}

/// The value of a SYNTACTIC integer literal, from the SQL AST: a bare
/// Number, optionally under parentheses or unary MINUS. Never unary plus
/// (DuckDB's `+` is a real function that erases literal-ness), never a
/// function call, never a cast, never anything bound — the 2026-08-13
/// adversarial fleet proved every SExpr-shape heuristic leaks (verbatim
/// family returns, `0 - N` user spellings, retyped degenerations), so the
/// hint comes from the spelling alone. This is DuckDB's own notion for
/// its value-fits promotion.
/// Whether the SPELLING is a DECIMAL literal (a dot, no exponent —
/// `2.5`, `-2.681`; `1.5e0` is DOUBLE), optionally under parens or unary
/// minus. DuckDB folds a strict op over one of these and a bare NULL to
/// SQLNULL (INTEGER), discarding the decimal entirely.
fn ast_decimal_literal(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => {
                text.contains('.') && !text.to_ascii_lowercase().contains('e')
            }
            _ => false,
        },
        SqlExpr::Nested(inner) => ast_decimal_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => ast_decimal_literal(expr),
        _ => false,
    }
}

fn ast_int_literal(e: &SqlExpr) -> Option<i64> {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => text.parse::<i64>().ok(),
            _ => None,
        },
        SqlExpr::Nested(inner) => ast_int_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => ast_int_literal(expr).and_then(i64::checked_neg),
        _ => None,
    }
}

/// The wider of two integer widths — family unification's promotion rule.
fn wider_int(a: Ty, b: Ty) -> Ty {
    if width_rank(a) >= width_rank(b) {
        a
    } else {
        b
    }
}

/// Whether `v` is representable at width `t` (always true for lane types).
fn fits_width(t: Ty, v: i64) -> bool {
    t.int_range().map_or(true, |(lo, hi)| (lo..=hi).contains(&v))
}

/// DuckDB's name for an integer width, for refusal messages.
fn duck_int_name(t: Ty) -> &'static str {
    match t {
        Ty::I8 => "TINYINT",
        Ty::I16 => "SMALLINT",
        Ty::I32 => "INTEGER",
        Ty::F64 => "DOUBLE",
        _ => "BIGINT",
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

/// Turn a just-built `promote_f64` node into the f32-narrowing one, so an
/// integer `tree_predict` feature rounds ONCE the way sklearn does
/// (TASK-77). Anything else — an f64 expression, a typed NULL, a folded
/// literal — is already on the grid or has no integer to narrow, and passes
/// through untouched.
fn narrow_f32(e: SExpr) -> SExpr {
    match e.kind {
        SKind::IntToFloat(inner) => SExpr {
            kind: SKind::IntToFloat32(inner),
            ..e
        },
        _ => e,
    }
}
