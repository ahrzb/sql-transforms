use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyErr;
use sqlparser::ast::{
    BinaryOperator, Expr as SqlExpr, Join, JoinConstraint, JoinOperator, SelectItem, SetExpr,
    Statement, TableFactor, TableWithJoins,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::expr::{Expr, Value};
use crate::types::Schema;

pub type Row = HashMap<String, HashMap<String, Value>>;

pub enum InterpError {
    Build(String),
    MissingKey(String),
    Eval(String),
}

impl From<InterpError> for PyErr {
    fn from(e: InterpError) -> PyErr {
        match e {
            InterpError::Build(msg) => PyValueError::new_err(msg),
            InterpError::MissingKey(msg) => PyKeyError::new_err(msg),
            InterpError::Eval(msg) => PyValueError::new_err(msg),
        }
    }
}

pub enum RelNode {
    TableScan {
        table: String,
    },
    Filter {
        input: Box<RelNode>,
        predicate: Expr,
    },
    CrossJoin {
        left: Box<RelNode>,
        right: Box<RelNode>,
    },
    Join {
        left: Box<RelNode>,
        right: Box<RelNode>,
        on: Vec<(Expr, Expr)>,
    },
    SubqueryAlias {
        input: Box<RelNode>,
        alias: String,
    },
    LookupJoin {
        input: Box<RelNode>,
        table: String,
        keys: Vec<Expr>,
    },
}

pub struct Plan {
    pub projection: Vec<(String, Expr)>,
    pub input: RelNode,
}

pub struct LookupSpec {
    pub static_table: String,
    pub key_columns: Vec<String>,
}

pub fn build_plan(sql: &str) -> Result<Plan, InterpError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|e| InterpError::Build(format!("SQL parse error: {e}")))?;

    if statements.len() != 1 {
        return Err(InterpError::Build(
            "Expected exactly one SQL statement".to_string(),
        ));
    }

    let select = match &statements[0] {
        Statement::Query(query) => match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            _ => {
                return Err(InterpError::Build(
                    "Only SELECT queries are supported".to_string(),
                ))
            }
        },
        _ => {
            return Err(InterpError::Build(
                "Only SELECT queries are supported".to_string(),
            ))
        }
    };

    let mut input = build_from(&select.from)?;
    if let Some(predicate) = &select.selection {
        input = RelNode::Filter {
            input: Box::new(input),
            predicate: crate::expr_build::convert_expr(predicate)?,
        };
    }
    let projection = build_projection(&select.projection)?;

    Ok(Plan { projection, input })
}

fn build_from(from: &[TableWithJoins]) -> Result<RelNode, InterpError> {
    if from.is_empty() {
        return Err(InterpError::Build("FROM clause is required".to_string()));
    }
    let mut seen_tables: HashSet<String> = HashSet::new();
    let mut node = build_table_with_joins(&from[0], &mut seen_tables)?;
    for twj in &from[1..] {
        let right = build_table_with_joins(twj, &mut seen_tables)?;
        node = RelNode::CrossJoin {
            left: Box::new(node),
            right: Box::new(right),
        };
    }
    Ok(node)
}

fn build_table_with_joins(
    twj: &TableWithJoins,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    let mut node = build_table_factor(&twj.relation, seen_tables)?;
    for join in &twj.joins {
        node = build_join(node, join, seen_tables)?;
    }
    Ok(node)
}

fn build_join(
    left: RelNode,
    join: &Join,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    let right = build_table_factor(&join.relation, seen_tables)?;
    match &join.join_operator {
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            let on_expr = require_on(constraint)?;
            let on = extract_equality_keys(on_expr)?;
            Ok(RelNode::Join {
                left: Box::new(left),
                right: Box::new(right),
                on,
            })
        }
        JoinOperator::CrossJoin(_) => Ok(RelNode::CrossJoin {
            left: Box::new(left),
            right: Box::new(right),
        }),
        other => Err(InterpError::Build(format!(
            "Unsupported JOIN type: {other:?} — only inner JOIN ... ON and CROSS JOIN are supported"
        ))),
    }
}

fn build_table_factor(
    factor: &TableFactor,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            let table = name.to_string();
            // Track the EFFECTIVE output name (alias if present, else the real
            // table name) — this is the key each relation's Row is stored
            // under, so a collision here (whether from a true self-join like
            // `FROM a JOIN a ON ...` or an alias collision like
            // `FROM a JOIN b AS a ON ...`) would cause one side's data to
            // silently overwrite the other's during row merging.
            let effective_name = match &alias {
                Some(a) => a.name.value.clone(),
                None => table.clone(),
            };
            if !seen_tables.insert(effective_name.clone()) {
                return Err(InterpError::Build(format!(
                    "table '{effective_name}' is referenced more than once in FROM/JOIN — \
                     self-joins and alias collisions are not supported"
                )));
            }
            let scan = RelNode::TableScan { table };
            Ok(match alias {
                Some(a) => RelNode::SubqueryAlias {
                    input: Box::new(scan),
                    alias: a.name.value.clone(),
                },
                None => scan,
            })
        }
        _ => Err(InterpError::Build("Unsupported FROM clause".to_string())),
    }
}

fn require_on(constraint: &JoinConstraint) -> Result<&SqlExpr, InterpError> {
    match constraint {
        JoinConstraint::On(e) => Ok(e),
        _ => Err(InterpError::Build(
            "JOIN requires an ON condition".to_string(),
        )),
    }
}

fn extract_equality_keys(expr: &SqlExpr) -> Result<Vec<(Expr, Expr)>, InterpError> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut pairs = extract_equality_keys(left)?;
            pairs.extend(extract_equality_keys(right)?);
            Ok(pairs)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => Ok(vec![(
            crate::expr_build::convert_expr(left)?,
            crate::expr_build::convert_expr(right)?,
        )]),
        _ => Err(InterpError::Build(
            "JOIN ON condition must be an equality, or an AND of equalities, between columns"
                .to_string(),
        )),
    }
}

fn build_projection(items: &[SelectItem]) -> Result<Vec<(String, Expr)>, InterpError> {
    let mut out = Vec::new();
    for item in items {
        match item {
            SelectItem::UnnamedExpr(e) => {
                let name = column_name(e)?;
                out.push((name, crate::expr_build::convert_expr(e)?));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                out.push((alias.value.clone(), crate::expr_build::convert_expr(expr)?));
            }
            _ => return Err(InterpError::Build("Unsupported SELECT item".to_string())),
        }
    }
    Ok(out)
}

fn column_name(e: &SqlExpr) -> Result<String, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            Ok(parts.last().map(|i| i.value.clone()).unwrap_or_default())
        }
        _ => Err(InterpError::Build(
            "Expression in SELECT list needs an alias (AS name)".to_string(),
        )),
    }
}

/// Walks a built `Plan`, rewriting any `Join` node where exactly one side is
/// a scan of a table named in `static_tables` into a `RelNode::LookupJoin`.
/// Returns an error if both sides of a `Join` are static tables (a
/// static-to-static join isn't a lookup and isn't supported).
pub fn optimize(
    plan: Plan,
    static_tables: &HashSet<String>,
) -> Result<(Plan, Vec<LookupSpec>), InterpError> {
    let mut specs = Vec::new();
    let input = optimize_rel(plan.input, static_tables, &mut specs)?;
    Ok((
        Plan {
            projection: plan.projection,
            input,
        },
        specs,
    ))
}

fn optimize_rel(
    node: RelNode,
    static_tables: &HashSet<String>,
    specs: &mut Vec<LookupSpec>,
) -> Result<RelNode, InterpError> {
    match node {
        RelNode::Join { left, right, on } => {
            let left = optimize_rel(*left, static_tables, specs)?;
            let right = optimize_rel(*right, static_tables, specs)?;
            let left_static = scan_table_name(&left).filter(|t| static_tables.contains(*t));
            let right_static = scan_table_name(&right).filter(|t| static_tables.contains(*t));
            match (left_static, right_static) {
                (Some(_), Some(_)) => Err(InterpError::Build(
                    "Joining two static tables together is not supported".to_string(),
                )),
                (None, Some(table)) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table)?;
                    specs.push(LookupSpec {
                        static_table: table.clone(),
                        key_columns,
                    });
                    Ok(RelNode::LookupJoin {
                        input: Box::new(left),
                        table,
                        keys,
                    })
                }
                (Some(table), None) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table)?;
                    specs.push(LookupSpec {
                        static_table: table.clone(),
                        key_columns,
                    });
                    Ok(RelNode::LookupJoin {
                        input: Box::new(right),
                        table,
                        keys,
                    })
                }
                (None, None) => Ok(RelNode::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    on,
                }),
            }
        }
        RelNode::CrossJoin { left, right } => Ok(RelNode::CrossJoin {
            left: Box::new(optimize_rel(*left, static_tables, specs)?),
            right: Box::new(optimize_rel(*right, static_tables, specs)?),
        }),
        RelNode::Filter { input, predicate } => Ok(RelNode::Filter {
            input: Box::new(optimize_rel(*input, static_tables, specs)?),
            predicate,
        }),
        RelNode::SubqueryAlias { input, alias } => Ok(RelNode::SubqueryAlias {
            input: Box::new(optimize_rel(*input, static_tables, specs)?),
            alias,
        }),
        other => Ok(other),
    }
}

fn scan_table_name(node: &RelNode) -> Option<&str> {
    match node {
        RelNode::TableScan { table } => Some(table),
        RelNode::SubqueryAlias { input, .. } => scan_table_name(input),
        _ => None,
    }
}

/// The qualifier (`table` part) of a plain `Expr::Column`, or `None` for
/// anything else (unqualified column, literal, expression, ...).
fn column_qualifier(e: &Expr) -> Option<&str> {
    match e {
        Expr::Column { table: Some(t), .. } => Some(t.as_str()),
        _ => None,
    }
}

/// Splits each ON-clause equality pair into (the static table's key column
/// name, the row-side expression to evaluate it against).
///
/// The ON clause's tuple order reflects how the equality was *written*
/// (`a = b` vs `b = a`), which is independent of which side of the JOIN is
/// structurally left/right in the FROM clause — so this identifies the
/// static side per-pair by matching each operand's column qualifier against
/// `static_table`, rather than assuming a fixed position.
fn split_keys(
    on: &[(Expr, Expr)],
    static_table: &str,
) -> Result<(Vec<Expr>, Vec<String>), InterpError> {
    let mut row_side_keys = Vec::new();
    let mut static_col_names = Vec::new();
    for (l, r) in on {
        let static_expr = match (column_qualifier(l), column_qualifier(r)) {
            (Some(t), _) if t == static_table => l,
            (_, Some(t)) if t == static_table => r,
            _ => {
                return Err(InterpError::Build(format!(
                    "JOIN ON keys against static table '{static_table}' must reference \
                     the static table's columns by name (e.g. {static_table}.col)"
                )))
            }
        };
        let row_expr = if std::ptr::eq(static_expr, l) { r } else { l };
        match static_expr {
            Expr::Column { name, .. } => static_col_names.push(name.clone()),
            _ => {
                return Err(InterpError::Build(format!(
                    "JOIN ON keys against static table '{static_table}' must be plain columns"
                )))
            }
        }
        row_side_keys.push(row_expr.clone());
    }
    Ok((row_side_keys, static_col_names))
}

pub fn execute(
    plan: &Plan,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
    lookups: &HashMap<String, crate::lookup::LookupIndex>,
) -> Result<Vec<HashMap<String, Value>>, InterpError> {
    let rows = execute_rel(&plan.input, tables, lookups)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut result = HashMap::new();
        for (alias, e) in &plan.projection {
            result.insert(alias.clone(), crate::expr::eval(e, row)?);
        }
        out.push(result);
    }
    Ok(out)
}

fn execute_rel(
    node: &RelNode,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
    lookups: &HashMap<String, crate::lookup::LookupIndex>,
) -> Result<Vec<Row>, InterpError> {
    match node {
        RelNode::TableScan { table } => {
            let flat_rows = tables.get(table).ok_or_else(|| {
                InterpError::Build(format!("Unknown table in FROM clause: {table}"))
            })?;
            Ok(flat_rows
                .iter()
                .map(|r| {
                    let mut row = Row::new();
                    row.insert(table.clone(), r.clone());
                    row
                })
                .collect())
        }
        RelNode::Filter { input, predicate } => {
            let rows = execute_rel(input, tables, lookups)?;
            let mut out = Vec::new();
            for row in rows {
                if let Value::Bool(true) = crate::expr::eval(predicate, &row)? {
                    out.push(row);
                }
            }
            Ok(out)
        }
        RelNode::CrossJoin { left, right } => {
            let left_rows = execute_rel(left, tables, lookups)?;
            let right_rows = execute_rel(right, tables, lookups)?;
            let mut out = Vec::with_capacity(left_rows.len() * right_rows.len());
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    out.push(merged);
                }
            }
            Ok(out)
        }
        RelNode::Join { left, right, on } => {
            let left_rows = execute_rel(left, tables, lookups)?;
            let right_rows = execute_rel(right, tables, lookups)?;
            let mut out = Vec::new();
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    let mut all_match = true;
                    for (le, re) in on {
                        let lv = crate::expr::eval(le, &merged)?;
                        let rv = crate::expr::eval(re, &merged)?;
                        if matches!(lv, Value::Null) || matches!(rv, Value::Null) || lv != rv {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        out.push(merged);
                    }
                }
            }
            Ok(out)
        }
        RelNode::SubqueryAlias { input, alias } => {
            let rows = execute_rel(input, tables, lookups)?;
            Ok(rows
                .into_iter()
                .map(|row| {
                    let inner = row.into_values().next().unwrap_or_default();
                    let mut renamed = Row::new();
                    renamed.insert(alias.clone(), inner);
                    renamed
                })
                .collect())
        }
        RelNode::LookupJoin { input, table, keys } => {
            let rows = execute_rel(input, tables, lookups)?;
            let index = lookups.get(table).ok_or_else(|| {
                InterpError::Build(format!("No lookup index built for table: {table}"))
            })?;
            let mut out = Vec::with_capacity(rows.len());
            for mut row in rows {
                let key: Vec<Value> = keys
                    .iter()
                    .map(|k| crate::expr::eval(k, &row))
                    .collect::<Result<_, _>>()?;
                let hit = index.index.get(&key).ok_or_else(|| {
                    let key_repr: Vec<String> =
                        key.iter().map(crate::expr::display_value).collect();
                    InterpError::MissingKey(format!(
                        "No row in static table '{table}' matches key ({})",
                        key_repr.join(", ")
                    ))
                })?;
                row.insert(table.clone(), hit.clone());
                out.push(row);
            }
            Ok(out)
        }
    }
}

pub struct ColumnValidation {
    pub row_table_columns: HashMap<String, Vec<String>>,
    pub effective_schemas: HashMap<String, Schema>,
}

/// Maps each relation's EFFECTIVE name (its alias if aliased, else its real
/// table name — the qualifier `Expr::Column` references use after
/// `SubqueryAlias` renaming) to its real table name and whether it's a row
/// table (vs. static). Walks the already-optimized Plan, so any `Join` with
/// a static side has already become a `LookupJoin`.
fn resolve_tables(
    node: &RelNode,
    row_table_names: &HashSet<String>,
    out: &mut HashMap<String, (String, bool)>,
) {
    match node {
        RelNode::TableScan { table } => {
            let is_row = row_table_names.contains(table);
            out.insert(table.clone(), (table.clone(), is_row));
        }
        RelNode::SubqueryAlias { input, alias } => {
            if let Some(real) = scan_table_name(input) {
                let is_row = row_table_names.contains(real);
                out.insert(alias.clone(), (real.to_string(), is_row));
            }
        }
        RelNode::Filter { input, .. } => resolve_tables(input, row_table_names, out),
        RelNode::CrossJoin { left, right } | RelNode::Join { left, right, .. } => {
            resolve_tables(left, row_table_names, out);
            resolve_tables(right, row_table_names, out);
        }
        RelNode::LookupJoin { input, table, .. } => {
            resolve_tables(input, row_table_names, out);
            out.insert(table.clone(), (table.clone(), false));
        }
    }
}

/// Validates every `Expr::Column` reference in the plan (projection, WHERE,
/// JOIN ON) against the resolved table schemas, and collects — per row
/// table's REAL name — the set of columns the query actually references.
/// Also returns the effective-name -> Schema map (aliases resolved), reused
/// by the output type-inference pass.
pub fn validate_columns(
    plan: &Plan,
    row_table_names: &HashSet<String>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
) -> Result<ColumnValidation, InterpError> {
    let mut resolved = HashMap::new();
    resolve_tables(&plan.input, row_table_names, &mut resolved);

    let mut effective_schemas = HashMap::new();
    for (effective_name, (real_name, is_row)) in &resolved {
        let schema = if *is_row {
            row_schemas.get(real_name)
        } else {
            static_schemas.get(real_name)
        };
        if let Some(s) = schema {
            effective_schemas.insert(effective_name.clone(), s.clone());
        }
    }

    let mut used_columns: HashMap<String, HashSet<String>> = HashMap::new();
    for (_, e) in &plan.projection {
        validate_expr(e, &resolved, row_schemas, static_schemas, &mut used_columns)?;
    }
    validate_rel(
        &plan.input,
        &resolved,
        row_schemas,
        static_schemas,
        &mut used_columns,
    )?;

    Ok(ColumnValidation {
        row_table_columns: used_columns
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect(),
        effective_schemas,
    })
}

fn validate_rel(
    node: &RelNode,
    resolved: &HashMap<String, (String, bool)>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
    used_columns: &mut HashMap<String, HashSet<String>>,
) -> Result<(), InterpError> {
    match node {
        RelNode::TableScan { .. } => Ok(()),
        RelNode::Filter { input, predicate } => {
            validate_expr(
                predicate,
                resolved,
                row_schemas,
                static_schemas,
                used_columns,
            )?;
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::CrossJoin { left, right } => {
            validate_rel(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_rel(right, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::Join { left, right, on } => {
            for (l, r) in on {
                validate_expr(l, resolved, row_schemas, static_schemas, used_columns)?;
                validate_expr(r, resolved, row_schemas, static_schemas, used_columns)?;
            }
            validate_rel(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_rel(right, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::SubqueryAlias { input, .. } => {
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::LookupJoin { input, keys, .. } => {
            for k in keys {
                validate_expr(k, resolved, row_schemas, static_schemas, used_columns)?;
            }
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
    }
}

fn validate_expr(
    e: &Expr,
    resolved: &HashMap<String, (String, bool)>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
    used_columns: &mut HashMap<String, HashSet<String>>,
) -> Result<(), InterpError> {
    match e {
        Expr::Column {
            table: Some(t),
            name,
        } => {
            let (real, is_row) = resolved
                .get(t)
                .ok_or_else(|| InterpError::Build(format!("Unknown table qualifier: {t}")))?;
            check_column(real, *is_row, name, row_schemas, static_schemas)?;
            if *is_row {
                used_columns
                    .entry(real.clone())
                    .or_default()
                    .insert(name.clone());
            }
            Ok(())
        }
        Expr::Column { table: None, name } => {
            let mut matches: Vec<(&String, bool)> = Vec::new();
            for (real, is_row) in resolved.values() {
                let schema = if *is_row {
                    row_schemas.get(real)
                } else {
                    static_schemas.get(real)
                };
                if schema.is_some_and(|s| s.contains_key(name)) {
                    matches.push((real, *is_row));
                }
            }
            match matches.as_slice() {
                [] => Err(InterpError::Build(format!("Unknown column: {name}"))),
                [(real, is_row)] => {
                    if *is_row {
                        used_columns
                            .entry((*real).clone())
                            .or_default()
                            .insert(name.clone());
                    }
                    Ok(())
                }
                _ => Err(InterpError::Build(format!(
                    "Ambiguous column reference: {name}"
                ))),
            }
        }
        Expr::Literal(_) => Ok(()),
        Expr::BinaryOp { left, right, .. } => {
            validate_expr(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_expr(right, resolved, row_schemas, static_schemas, used_columns)
        }
        Expr::Not(inner) | Expr::Cast { expr: inner, .. } => {
            validate_expr(inner, resolved, row_schemas, static_schemas, used_columns)
        }
        Expr::Function { args, .. } => {
            for a in args {
                validate_expr(a, resolved, row_schemas, static_schemas, used_columns)?;
            }
            Ok(())
        }
    }
}

fn check_column(
    real_table: &str,
    is_row: bool,
    name: &str,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
) -> Result<(), InterpError> {
    let schema = if is_row {
        row_schemas.get(real_table)
    } else {
        static_schemas.get(real_table)
    };
    match schema {
        Some(s) if s.contains_key(name) => Ok(()),
        Some(_) => Err(InterpError::Build(format!(
            "Unknown column: {real_table}.{name}"
        ))),
        None => Err(InterpError::Build(format!("Unknown table: {real_table}"))),
    }
}
