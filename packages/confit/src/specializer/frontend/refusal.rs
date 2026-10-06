//! Refusal texts: the named `unsupported:` / `bind error:` messages.

use super::*;

pub(super) fn unsup(what: impl Into<String>) -> PrepareError {
    PrepareError::Unsupported(what.into())
}

/// The ONE refusal for what this build does not serve over a DECIMAL:
/// the builtins with DECIMAL overloads of their own (abs, round, ...),
/// which return a DECIMAL on DuckDB and would be silently wrong served
/// through DOUBLE, and the unifications docs/specs/decimal-expressions.md
/// names. Refusing is the severity ladder's own preference.
pub(super) fn refuse_dec(op: &str, ty: Ty, col: Option<&str>) -> PrepareError {
    let (p, s) = ty.dec().unwrap_or((38, 0));
    let where_ = match col {
        Some(c) => format!(" column '{c}'"),
        None => String::new(),
    };
    unsup(format!(
        "{op} over DECIMAL({p},{s}){where_} -- DECIMAL arithmetic, casts and \
         comparisons serve; this operation over a DECIMAL does not"
    ))
}

/// The DECIMAL-typed operand of `a`/`b`, if either is one. Nothing but a
/// static column produces a Dec, so the pair always names the offender.
pub(super) fn dec_operand<'a>(a: &'a SExpr, b: &'a SExpr) -> Option<&'a SExpr> {
    [a, b].into_iter().find(|e| e.ty.dec().is_some())
}

/// The ONLY place `TableFactor::Table` is destructured, and it is
/// destructured EXHAUSTIVELY.
///
/// The doctrine — no `..` in a sqlparser AST pattern, so a clause we have
/// never seen breaks the build instead of being silently dropped — applies
/// to the relation as to `Query` and `Select`: a `TableFactor::Table
/// { name, alias, .. }` pattern would swallow every modifier sqlparser can
/// hang off a table name. A dropped `TABLESAMPLE 3 ROWS` serves every row
/// where DuckDB serves 3, and `shape="map"`'s one-row-out-per-row-in
/// certificate does not catch it.
///
/// Every field other than `name`/`alias` refuses by name. Returns `None` for
/// a non-`Table` relation so each caller keeps its own wording for that.
/// The refusal for a FROM item that is not a plain table: names the kind
/// of relation instead of printing the SQL back.
pub(super) fn relation_refusal(tf: &TableFactor) -> String {
    let kind = match tf {
        TableFactor::Derived { .. } => "a derived table (subquery in FROM)",
        TableFactor::TableFunction { .. } | TableFactor::Function { .. } => "a table function",
        TableFactor::UNNEST { .. } => "UNNEST",
        TableFactor::NestedJoin { .. } => "a parenthesized join",
        TableFactor::Pivot { .. } => "PIVOT",
        TableFactor::Unpivot { .. } => "UNPIVOT",
        TableFactor::MatchRecognize { .. } => "MATCH_RECOGNIZE",
        TableFactor::JsonTable { .. } | TableFactor::OpenJsonTable { .. } => "a JSON table",
        TableFactor::XmlTable { .. } => "an XML table",
        _ => "this FROM item",
    };
    format!("FROM {kind} -- FROM and JOIN take named tables only")
}

/// The refusal for a join operator other than INNER or LEFT.
pub(super) fn join_refusal(op: &JoinOperator) -> String {
    let kind = match op {
        JoinOperator::Right(_) | JoinOperator::RightOuter(_) => "RIGHT JOIN",
        JoinOperator::FullOuter(_) => "FULL OUTER JOIN",
        JoinOperator::CrossJoin(_) => "CROSS JOIN",
        JoinOperator::Semi(_) | JoinOperator::LeftSemi(_) | JoinOperator::RightSemi(_) => {
            "SEMI JOIN"
        }
        JoinOperator::Anti(_) | JoinOperator::LeftAnti(_) | JoinOperator::RightAnti(_) => {
            "ANTI JOIN"
        }
        JoinOperator::CrossApply | JoinOperator::OuterApply => "APPLY",
        JoinOperator::AsOf { .. } => "ASOF JOIN",
        JoinOperator::StraightJoin(_) => "STRAIGHT_JOIN",
        _ => "this join type",
    };
    format!("join type {kind} -- served joins are [INNER] JOIN and LEFT [OUTER] JOIN")
}

/// The refusal for an expression form with no binding: names the form;
/// only an unrecognised form falls back to printing it.
pub(super) fn expr_refusal(e: &SqlExpr) -> String {
    let kind = match e {
        SqlExpr::Subquery(_) => "a scalar subquery",
        SqlExpr::InSubquery { .. } => "IN (SELECT ...)",
        SqlExpr::Exists { .. } => "EXISTS (SELECT ...)",
        SqlExpr::TypedString { .. } => "a typed literal (e.g. DATE '...')",
        SqlExpr::Struct { .. } | SqlExpr::Dictionary(_) => "a struct literal",
        SqlExpr::Map(_) => "a map literal",
        SqlExpr::Array(_) => "an array literal",
        SqlExpr::Tuple(_) => "a row/tuple value",
        SqlExpr::Interval(_) => "an INTERVAL literal",
        SqlExpr::Lambda(_) => "a lambda",
        SqlExpr::AtTimeZone { .. } => "AT TIME ZONE",
        SqlExpr::Extract { .. } => "EXTRACT",
        SqlExpr::Collate { .. } => "COLLATE",
        SqlExpr::InUnnest { .. } => "IN UNNEST(...)",
        SqlExpr::AnyOp { .. } | SqlExpr::AllOp { .. } => "ANY/ALL",
        SqlExpr::MatchAgainst { .. } => "MATCH ... AGAINST",
        SqlExpr::GroupingSets(_) | SqlExpr::Cube(_) | SqlExpr::Rollup(_) => "a grouping set",
        SqlExpr::Wildcard(_) | SqlExpr::QualifiedWildcard(..) => "a star outside the select list",
        _ => return format!("expression: {e}"),
    };
    format!("expression {kind}")
}

/// Refuse every `Query` clause this engine does not implement.
///
/// Destructured EXHAUSTIVELY on purpose — no `..` pattern. When sqlparser
/// grows a clause this stops compiling, instead of silently ignoring it. An
/// ignored clause is a wrong ANSWER, not a missing feature: a dropped
/// `FETCH FIRST n ROWS ONLY` serves every row.
pub(super) fn refuse_unhandled_query(query: &sqlparser::ast::Query) -> Result<(), PrepareError> {
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
/// destructured for the same reason as [`refuse_unhandled_query`] — a dropped
/// `QUALIFY` makes a dedupe-to-latest query emit every row, and
/// `shape='map'` would still certify it.
pub(super) fn refuse_unhandled_select(select: &sqlparser::ast::Select) -> Result<(), PrepareError> {
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

/// A static column the catalogue could not serve is PRESENT but unusable.
/// Saying "does not exist" about it sends the reader hunting a typo in a
/// correct query; name the type instead, the way the row path does.
pub(super) fn opaque_static_refusal(
    st: &StaticTable,
    name: &str,
    table: &str,
) -> Option<PrepareError> {
    // A struct refuses for a different reason than a timestamp does: its
    // fields are right there, and its whole value serves only where a value
    // of its own does (an output, a struct field, a CASE result, IS NULL).
    // Struct heads live in `structs`, everything else unservable in
    // `opaque`.
    if let Some(sc) = st
        .structs
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(name))
    {
        return Some(PrepareError::Unsupported(format!(
            "static table '{table}' column '{}' is a struct where a scalar \
             is needed (read one of its fields)",
            sc.name
        )));
    }
    st.opaque
        .iter()
        .find(|(c, _)| c.eq_ignore_ascii_case(name))
        .map(|(c, aty)| {
            PrepareError::Unsupported(format!(
                "static table '{table}' column '{c}' has type {aty}, which \
                 this engine does not serve — project a served column instead"
            ))
        })
}

/// A pad/repeat COUNT literal that can exceed the engine's 1 GiB
/// string-builder budget refuses at build.
///
/// The ground is OURS and it is a judgement: a serving engine does not
/// allocate a gigabyte per row. Refusing where DuckDB serves is the
/// sanctioned mode, and a build-time no beats meeting it per-row in
/// production. Data-driven counts keep the runtime cap, documented in
/// known-limitations.md.
///
/// DuckDB's behaviour past the budget is deterministic, not a reason to
/// refuse: repeat serves to 4294967295 bytes and errors above it, while
/// lpad/rpad refuse past INTEGER at the binder because their count
/// parameter is declared INTEGER (a separate rule).
pub(super) fn refuse_budget_breaking_count(name: &str, count: &SExpr) -> Result<(), PrepareError> {
    const BUDGET: i64 = 1 << 30; // bytes; an n-char 1-byte result is n bytes
    if let SKind::Lit(Lit::I64(n)) = fold(count.clone()).kind {
        if n > BUDGET {
            return Err(PrepareError::Bind(format!(
                "{name} count {n} exceeds the 1 GiB string-builder budget — \
                 this engine will not allocate a gigabyte per row, so the \
                 result could never serve. DuckDB does serve it; refusing at \
                 build is our deliberate limit, not a DuckDB restriction"
            )));
        }
    }
    Ok(())
}
