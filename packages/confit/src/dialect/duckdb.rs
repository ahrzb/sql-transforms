//! The DuckDB dialect: frontend (SQL → bound plan) and printer (plan →
//! DuckDB SQL). DuckDB is the reference engine (design D1), so this pair
//! carries laws L1 and L2: `parse(print(p)) == p` on PROJECTION-ROOTED
//! plans (every plan this frontend produces; a bare Scan prints as an
//! explicit SELECT list and reparses as the equivalent Project), and
//! parse→print must be invisible to the oracle bit-for-bit — the corpus
//! gate in tests/test_dialect_corpus_gate.py executes both.
//!
//! Frontend refusal discipline is the specializer frontend's: a clause
//! sqlparser parses and we ignore is a wrong ANSWER, so `Query` and
//! `Select` are destructured field-by-field — a new sqlparser clause
//! breaks the build, not the answers. Everything not lowered refuses by
//! name (`Unsupported`), wrong-against-catalog is `Bind`.
//!
//! v0 surface (grown corpus-first, per the design's phases): one base
//! table, WHERE, projection items that are bare columns, `*`, or ALIASED
//! expressions over the plan-core expression set. Unaliased complex items
//! refuse — reproducing DuckDB's auto-naming is unpinned. Functions,
//! joins, windows, aggregates: phase 2+.
//!
//! Printer discipline: every identifier quoted (DuckDB matches
//! case-insensitively regardless, spelling preserved), every compound
//! expression parenthesized (no precedence table to get wrong), literal
//! LEXEMES verbatim (never re-formatted). A plan whose intermediate schema
//! has duplicate column names refuses at print time — a name-addressed
//! subquery boundary cannot express it unambiguously.

use sqlparser::ast::{
    BinaryOperator, CastKind, Expr as SqlExpr, GroupByExpr, Select, SelectItem, SetExpr, Statement,
    TableFactor, UnaryOperator, Value as SqlValue,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Token, Tokenizer};

use super::plan::{BinOp, Catalog, Expr, Rel, UnOp};
use super::printer;
use super::ty::DTy;
use super::verify::verify;
use super::{unsup, DialectError};
use crate::specializer::rewrite;

// --- frontend ----------------------------------------------------------------

/// Parse one DuckDB-dialect SELECT into a verified plan.
pub fn parse_sql(sql: &str, cat: &Catalog) -> Result<Rel, DialectError> {
    let dialect = GenericDialect {};
    let located = Tokenizer::new(&dialect, sql)
        .tokenize_with_location()
        .map_err(|e| DialectError::Unsupported(format!("tokenize: {e}")))?;
    // DuckDB's numeric-underscore literals (1_000) tokenize as a Number
    // immediately followed by a Word starting with '_' - sqlparser would
    // bind that as `1 AS _000`, silently changing the VALUE. Refuse the
    // adjacency by name.
    for pair in located.windows(2) {
        let [a, b] = pair else { unreachable!() };
        if let (Token::Number(..), Token::Word(w)) = (&a.token, &b.token) {
            if w.value.starts_with('_') && w.quote_style.is_none() && a.span.end == b.span.start {
                return Err(unsup("numeric literal with underscores"));
            }
        }
    }
    let tokens: Vec<Token> = located.into_iter().map(|t| t.token).collect();
    // The oracle-grammar gap fixes that repair a SILENT misparse
    // (pins-wave5/sqlparser-spike.json): DuckDB's `k: expr` prefix aliases.
    let tokens = rewrite::rewrite_from_colon_aliases(rewrite::rewrite_colon_aliases(tokens));
    let statements = Parser::new(&dialect)
        .with_tokens(tokens)
        .parse_statements()
        .map_err(|e| DialectError::Unsupported(format!("parse: {e}")))?;
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
        return Err(unsup("top-level ORDER BY"));
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
    bind_select(select, cat)
}

/// Refuse every `Query` field we do not lower, by walking all of them.
fn refuse_unhandled_query(q: &sqlparser::ast::Query) -> Result<(), DialectError> {
    let sqlparser::ast::Query {
        with: _,     // checked above
        body: _,     // lowered
        order_by: _, // checked above
        limit_clause: _,
        fetch,
        locks,
        for_clause,
        settings,
        format_clause,
        pipe_operators,
    } = q;
    if fetch.is_some() {
        return Err(unsup("FETCH"));
    }
    if !locks.is_empty() {
        return Err(unsup("FOR UPDATE/SHARE"));
    }
    if for_clause.is_some() {
        return Err(unsup("FOR XML/JSON/BROWSE"));
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

struct Binder<'a> {
    /// (bound spelling, type) per in-scope column, plus the relation name
    /// and optional alias qualified references may use.
    cols: Vec<(String, DTy)>,
    qualifiers: Vec<String>,
    cat: &'a Catalog,
}

fn bind_select(select: &Select, cat: &Catalog) -> Result<Rel, DialectError> {
    let Select {
        select_token: _,
        distinct,
        top,
        top_before_distinct: _,
        projection,
        exclude,
        into,
        from,
        lateral_views,
        prewhere,
        selection,
        group_by,
        cluster_by,
        distribute_by,
        sort_by,
        having,
        named_window,
        optimizer_hints,
        select_modifiers,
        qualify,
        window_before_qualify: _,
        value_table_mode,
        connect_by,
        flavor: _,
    } = select;
    if distinct.is_some() {
        return Err(unsup("DISTINCT"));
    }
    if top.is_some() {
        return Err(unsup("TOP"));
    }
    if exclude.is_some() {
        return Err(unsup("SELECT EXCLUDE"));
    }
    if into.is_some() {
        return Err(unsup("SELECT INTO"));
    }
    if !lateral_views.is_empty() {
        return Err(unsup("LATERAL VIEW"));
    }
    if prewhere.is_some() {
        return Err(unsup("PREWHERE"));
    }
    let grouped = match group_by {
        GroupByExpr::Expressions(exprs, modifiers) => !exprs.is_empty() || !modifiers.is_empty(),
        GroupByExpr::All(_) => true,
    };
    if grouped || having.is_some() {
        return Err(unsup("GROUP BY / HAVING / aggregation"));
    }
    if !cluster_by.is_empty() || !distribute_by.is_empty() || !sort_by.is_empty() {
        return Err(unsup("CLUSTER/DISTRIBUTE/SORT BY"));
    }
    if !named_window.is_empty() {
        return Err(unsup("named WINDOW"));
    }
    if qualify.is_some() {
        return Err(unsup("QUALIFY"));
    }
    if value_table_mode.is_some() {
        return Err(unsup("value table SELECT"));
    }
    if !connect_by.is_empty() {
        return Err(unsup("CONNECT BY"));
    }
    if !optimizer_hints.is_empty() {
        return Err(unsup("optimizer hints"));
    }
    if select_modifiers.is_some() {
        return Err(unsup("SELECT modifiers"));
    }

    // FROM: exactly one base table, no joins (phase 2).
    let [table_with_joins] = from.as_slice() else {
        return Err(unsup(if from.is_empty() {
            "SELECT without FROM".to_string()
        } else {
            "multiple FROM relations".to_string()
        }));
    };
    if !table_with_joins.joins.is_empty() {
        return Err(unsup("JOIN"));
    }
    let (table_name, alias) = match &table_with_joins.relation {
        TableFactor::Table {
            name, alias, args, ..
        } => {
            if args.is_some() {
                return Err(unsup("table function FROM"));
            }
            let [part] = name.0.as_slice() else {
                return Err(unsup("schema-qualified table name"));
            };
            let Some(ident) = part.as_ident() else {
                return Err(unsup(format!("table name form: {part}")));
            };
            let alias = match alias {
                None => None,
                Some(a) if a.columns.is_empty() => Some(a.name.value.clone()),
                Some(_) => return Err(unsup("table alias with column list")),
            };
            (ident.value.clone(), alias)
        }
        other => return Err(unsup(format!("FROM relation: {other}"))),
    };
    let table = cat
        .table(&table_name)
        .ok_or_else(|| DialectError::Bind(format!("unknown table: {table_name}")))?;
    let mut qualifiers = vec![table.name.clone()];
    if let Some(a) = &alias {
        // An alias REPLACES the table name as a qualifier in DuckDB.
        qualifiers = vec![a.clone()];
    }
    let binder = Binder {
        cols: table
            .cols
            .iter()
            .map(|c| (c.name.clone(), c.ty.clone()))
            .collect(),
        qualifiers,
        cat,
    };

    let mut rel = Rel::Scan {
        table: table.name.clone(),
    };
    if let Some(pred) = selection {
        // DuckDB allows WHERE to reference SELECT aliases (lateral alias
        // references) — real semantics we don't lower yet, so an unknown
        // column that IS a projection alias refuses by name instead of
        // binding-erroring on valid SQL.
        let aliases: Vec<&str> = projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.as_str()),
                _ => None,
            })
            .collect();
        let pred = binder.expr(pred).map_err(|e| match &e {
            DialectError::Bind(m) => match m.strip_prefix("unknown column: ") {
                Some(name) if aliases.iter().any(|a| a.eq_ignore_ascii_case(name)) => {
                    unsup(format!("lateral alias reference in WHERE: {name}"))
                }
                _ => e,
            },
            _ => e,
        })?;
        if pred.ty()? != DTy::Bool {
            return Err(unsup(format!(
                "non-boolean WHERE (implicit cast unpinned): {}",
                pred.ty()?.name()
            )));
        }
        rel = Rel::Filter {
            input: Box::new(rel),
            pred,
        };
    }

    let mut items: Vec<(String, Expr)> = Vec::new();
    for item in projection {
        match item {
            SelectItem::Wildcard(opts) => {
                refuse_wildcard_opts(opts)?;
                for (ordinal, (name, ty)) in binder.cols.iter().enumerate() {
                    items.push((
                        name.clone(),
                        Expr::Col {
                            ordinal,
                            name: name.clone(),
                            ty: ty.clone(),
                        },
                    ));
                }
            }
            SelectItem::QualifiedWildcard(kind, opts) => {
                refuse_wildcard_opts(opts)?;
                let q = match kind {
                    sqlparser::ast::SelectItemQualifiedWildcardKind::ObjectName(name) => {
                        match name.0.as_slice() {
                            [part] => match part.as_ident() {
                                Some(id) => id.value.clone(),
                                None => return Err(unsup(format!("star qualifier form: {part}"))),
                            },
                            _ => return Err(unsup("multi-part star qualifier")),
                        }
                    }
                    other => return Err(unsup(format!("star qualifier: {other}"))),
                };
                if !binder
                    .qualifiers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(&q))
                {
                    return Err(DialectError::Bind(format!("unknown qualifier: {q}")));
                }
                for (ordinal, (name, ty)) in binder.cols.iter().enumerate() {
                    items.push((
                        name.clone(),
                        Expr::Col {
                            ordinal,
                            name: name.clone(),
                            ty: ty.clone(),
                        },
                    ));
                }
            }
            SelectItem::UnnamedExpr(e) => {
                let bound = binder.expr(e)?;
                // Output name: DuckDB preserves the QUERY's spelling for a
                // bare column; anything else needs an alias (auto-naming
                // is unpinned).
                let name = match e {
                    SqlExpr::Identifier(id) => id.value.clone(),
                    SqlExpr::CompoundIdentifier(parts) => {
                        parts.last().map(|p| p.value.clone()).unwrap_or_default()
                    }
                    _ => {
                        return Err(unsup(
                            "unaliased expression SELECT item (auto-naming unpinned)",
                        ));
                    }
                };
                items.push((name, bound));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                items.push((alias.value.clone(), binder.expr(expr)?));
            }
            SelectItem::ExprWithAliases { .. } => {
                return Err(unsup("multi-alias SELECT item"));
            }
        }
    }
    let rel = Rel::Project {
        input: Box::new(rel),
        items,
    };
    // A frontend that returns an unverifiable plan is broken — check here,
    // so every caller holds a plan every printer may trust.
    verify(&rel, cat)?;
    Ok(rel)
}

fn refuse_wildcard_opts(
    opts: &sqlparser::ast::WildcardAdditionalOptions,
) -> Result<(), DialectError> {
    let sqlparser::ast::WildcardAdditionalOptions {
        wildcard_token: _,
        opt_ilike,
        opt_alias,
        opt_exclude,
        opt_except,
        opt_replace,
        opt_rename,
    } = opts;
    if opt_ilike.is_some()
        || opt_alias.is_some()
        || opt_exclude.is_some()
        || opt_except.is_some()
        || opt_replace.is_some()
        || opt_rename.is_some()
    {
        return Err(unsup(
            "star modifiers (EXCLUDE/EXCEPT/REPLACE/RENAME/ILIKE)",
        ));
    }
    Ok(())
}

impl Binder<'_> {
    fn resolve(&self, spelled: &str) -> Result<Expr, DialectError> {
        let matches: Vec<usize> = self
            .cols
            .iter()
            .enumerate()
            .filter(|(_, (n, _))| n.eq_ignore_ascii_case(spelled))
            .map(|(i, _)| i)
            .collect();
        match matches.as_slice() {
            [ordinal] => {
                let (name, ty) = &self.cols[*ordinal];
                Ok(Expr::Col {
                    ordinal: *ordinal,
                    name: name.clone(),
                    ty: ty.clone(),
                })
            }
            [] if spelled.eq_ignore_ascii_case("rowid") => Err(unsup("rowid pseudo-column")),
            [] => Err(DialectError::Bind(format!("unknown column: {spelled}"))),
            _ => Err(DialectError::Bind(format!("ambiguous column: {spelled}"))),
        }
    }

    fn expr(&self, e: &SqlExpr) -> Result<Expr, DialectError> {
        match e {
            SqlExpr::Identifier(id) => self.resolve(&id.value),
            SqlExpr::CompoundIdentifier(parts) => {
                let [q, col] = parts.as_slice() else {
                    return Err(unsup("deep compound identifier"));
                };
                if !self
                    .qualifiers
                    .iter()
                    .any(|known| known.eq_ignore_ascii_case(&q.value))
                {
                    return Err(DialectError::Bind(format!(
                        "unknown qualifier: {}",
                        q.value
                    )));
                }
                self.resolve(&col.value)
            }
            SqlExpr::Nested(inner) => self.expr(inner),
            SqlExpr::Value(v) => literal(&v.value),
            SqlExpr::BinaryOp { left, op, right } => {
                let op = match op {
                    BinaryOperator::Plus => BinOp::Add,
                    BinaryOperator::Minus => BinOp::Sub,
                    BinaryOperator::Multiply => BinOp::Mul,
                    BinaryOperator::Divide => BinOp::FDiv,
                    BinaryOperator::DuckIntegerDivide => BinOp::IDiv,
                    BinaryOperator::Modulo => BinOp::Rem,
                    BinaryOperator::StringConcat => BinOp::Concat,
                    BinaryOperator::And => BinOp::And,
                    BinaryOperator::Or => BinOp::Or,
                    BinaryOperator::Eq => BinOp::Eq,
                    BinaryOperator::NotEq => BinOp::Neq,
                    BinaryOperator::Lt => BinOp::Lt,
                    BinaryOperator::LtEq => BinOp::Lte,
                    BinaryOperator::Gt => BinOp::Gt,
                    BinaryOperator::GtEq => BinOp::Gte,
                    other => return Err(unsup(format!("operator: {other}"))),
                };
                let l = Box::new(self.expr(left)?);
                let r = Box::new(self.expr(right)?);
                let bound = Expr::Bin { op, l, r };
                bound.ty()?; // surface Bind/Unsupported at the site
                Ok(bound)
            }
            SqlExpr::UnaryOp { op, expr } => match op {
                UnaryOperator::Minus => {
                    let bound = Expr::Un {
                        op: UnOp::Neg,
                        e: Box::new(self.expr(expr)?),
                    };
                    bound.ty()?;
                    Ok(bound)
                }
                UnaryOperator::Plus => self.expr(expr),
                UnaryOperator::Not => {
                    let bound = Expr::Un {
                        op: UnOp::Not,
                        e: Box::new(self.expr(expr)?),
                    };
                    bound.ty()?;
                    Ok(bound)
                }
                other => return Err(unsup(format!("unary operator: {other}"))),
            },
            SqlExpr::IsNull(inner) => Ok(Expr::IsNull {
                negated: false,
                e: Box::new(self.expr(inner)?),
            }),
            SqlExpr::IsNotNull(inner) => Ok(Expr::IsNull {
                negated: true,
                e: Box::new(self.expr(inner)?),
            }),
            SqlExpr::IsDistinctFrom(l, r) => {
                let bound = Expr::IsDistinct {
                    negated: false,
                    l: Box::new(self.expr(l)?),
                    r: Box::new(self.expr(r)?),
                };
                bound.ty()?;
                Ok(bound)
            }
            SqlExpr::IsNotDistinctFrom(l, r) => {
                let bound = Expr::IsDistinct {
                    negated: true,
                    l: Box::new(self.expr(l)?),
                    r: Box::new(self.expr(r)?),
                };
                bound.ty()?;
                Ok(bound)
            }
            SqlExpr::Cast {
                kind,
                expr,
                data_type,
                format,
                array,
            } => {
                if format.is_some() {
                    return Err(unsup("CAST ... FORMAT"));
                }
                if *array {
                    return Err(unsup("CAST ... ARRAY form"));
                }
                let strict = match kind {
                    CastKind::Cast | CastKind::DoubleColon => true,
                    CastKind::TryCast | CastKind::SafeCast => false,
                };
                let target = DTy::from_duckdb(&data_type.to_string())?;
                Ok(Expr::Cast {
                    strict,
                    e: Box::new(self.expr(expr)?),
                    target,
                })
            }
            SqlExpr::Case {
                case_token: _,
                end_token: _,
                operand,
                conditions,
                else_result,
            } => {
                if operand.is_some() {
                    return Err(unsup("simple CASE (CASE <expr> WHEN ...)"));
                }
                let mut whens = Vec::new();
                for w in conditions {
                    whens.push((self.expr(&w.condition)?, self.expr(&w.result)?));
                }
                let else_ = match else_result {
                    Some(e) => Some(Box::new(self.expr(e)?)),
                    None => None,
                };
                let bound = Expr::Case { whens, else_ };
                bound.ty()?;
                Ok(bound)
            }
            other => Err(unsup(format!("expression: {other}"))),
        }
    }
}

/// Type a literal token. Numbers follow the measured lattice: integers take
/// the smallest signed width that fits (INTEGER first), decimal-pointed
/// literals are DECIMAL(p,s) by digit count, exponent forms are DOUBLE.
fn literal(v: &SqlValue) -> Result<Expr, DialectError> {
    match v {
        SqlValue::Number(lexeme, _) => {
            let ty = number_type(lexeme)?;
            Ok(Expr::Lit {
                lexeme: lexeme.clone(),
                ty,
            })
        }
        SqlValue::SingleQuotedString(s) => Ok(Expr::Lit {
            lexeme: s.clone(),
            ty: DTy::Str,
        }),
        SqlValue::Boolean(b) => Ok(Expr::Lit {
            lexeme: if *b { "true".into() } else { "false".into() },
            ty: DTy::Bool,
        }),
        SqlValue::Null => Err(unsup(
            "NULL literal (typed by context; context typing unpinned)",
        )),
        other => Err(unsup(format!("literal: {other}"))),
    }
}

fn number_type(lexeme: &str) -> Result<DTy, DialectError> {
    if lexeme.contains(['e', 'E']) {
        return Ok(DTy::F64);
    }
    if let Some((int_part, frac_part)) = lexeme.split_once('.') {
        // Measured (pins-dialect): typeof(0.5) = DECIMAL(2,1) - a bare
        // zero integer part still counts one digit.
        let stripped = int_part.trim_start_matches('0');
        let int_digits = if stripped.is_empty() && !int_part.is_empty() {
            1
        } else {
            stripped.len()
        };
        let scale = frac_part.len();
        let p = int_digits + scale;
        let p = p.max(1);
        if p > 38 || scale > 38 {
            return Err(unsup(format!(
                "decimal literal beyond DECIMAL(38): {lexeme}"
            )));
        }
        return Ok(DTy::Dec(p as u8, scale as u8));
    }
    let n: i128 = lexeme
        .parse()
        .map_err(|_| unsup(format!("integer literal beyond HUGEINT: {lexeme}")))?;
    Ok(if i32::try_from(n).is_ok() {
        DTy::I32
    } else if i64::try_from(n).is_ok() {
        DTy::I64
    } else {
        DTy::I128
    })
}

// --- printer -----------------------------------------------------------------

/// Print a verified plan as DuckDB SQL. `cat` supplies scan schemas.
pub fn print_sql(rel: &Rel, cat: &Catalog) -> Result<String, DialectError> {
    verify(rel, cat)?;
    let (sql, _) = printer::query(&Duck, rel, cat, 0)?;
    Ok(sql)
}

struct Duck;

impl printer::ExprPrinter for Duck {
    fn quote_ident(&self, name: &str) -> String {
        format!("\"{}\"", name.replace('"', "\"\""))
    }

    fn expr(&self, e: &Expr, input: &[(String, DTy)]) -> Result<String, DialectError> {
        Ok(match e {
            Expr::Col { ordinal, name, .. } => printer::col_ref(self, *ordinal, name, input)?,
            Expr::Lit { lexeme, ty } => match ty {
                DTy::Str => format!("'{}'", lexeme.replace('\'', "''")),
                DTy::Bool | DTy::F64 => lexeme.clone(),
                t if t.is_numeric() => lexeme.clone(),
                t => format!("'{}'::{}", lexeme.replace('\'', "''"), t.duckdb_name()?),
            },
            Expr::Bin { op, l, r } => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::FDiv => "/",
                    BinOp::IDiv => "//",
                    BinOp::Rem => "%",
                    BinOp::Concat => "||",
                    BinOp::And => "AND",
                    BinOp::Or => "OR",
                    BinOp::Eq => "=",
                    BinOp::Neq => "<>",
                    BinOp::Lt => "<",
                    BinOp::Lte => "<=",
                    BinOp::Gt => ">",
                    BinOp::Gte => ">=",
                };
                format!("({} {sym} {})", self.expr(l, input)?, self.expr(r, input)?)
            }
            Expr::Un { op, e } => match op {
                UnOp::Neg => format!("(- {})", self.expr(e, input)?),
                UnOp::Not => format!("(NOT {})", self.expr(e, input)?),
            },
            Expr::Cast { strict, e, target } => format!(
                "{}({} AS {})",
                if *strict { "CAST" } else { "TRY_CAST" },
                self.expr(e, input)?,
                target.duckdb_name()?
            ),
            Expr::Case { whens, else_ } => {
                let mut s = String::from("(CASE");
                for (c, v) in whens {
                    s.push_str(&format!(
                        " WHEN {} THEN {}",
                        self.expr(c, input)?,
                        self.expr(v, input)?
                    ));
                }
                if let Some(el) = else_ {
                    s.push_str(&format!(" ELSE {}", self.expr(el, input)?));
                }
                s.push_str(" END)");
                s
            }
            Expr::IsNull { negated, e } => format!(
                "({} IS {}NULL)",
                self.expr(e, input)?,
                if *negated { "NOT " } else { "" }
            ),
            Expr::IsDistinct { negated, l, r } => format!(
                "({} IS {}DISTINCT FROM {})",
                self.expr(l, input)?,
                if *negated { "NOT " } else { "" },
                self.expr(r, input)?
            ),
        })
    }
}
