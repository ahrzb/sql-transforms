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
}

pub struct Plan {
    pub projection: Vec<(String, Expr)>,
    pub input: RelNode,
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
            if !seen_tables.insert(table.clone()) {
                return Err(InterpError::Build(format!(
                    "Self-joins are not supported: table '{table}' is referenced more than once"
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

pub fn execute(
    plan: &Plan,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
) -> Result<Vec<HashMap<String, Value>>, InterpError> {
    let rows = execute_rel(&plan.input, tables)?;
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
            let rows = execute_rel(input, tables)?;
            let mut out = Vec::new();
            for row in rows {
                if let Value::Bool(true) = crate::expr::eval(predicate, &row)? {
                    out.push(row);
                }
            }
            Ok(out)
        }
        RelNode::CrossJoin { left, right } => {
            let left_rows = execute_rel(left, tables)?;
            let right_rows = execute_rel(right, tables)?;
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
            let left_rows = execute_rel(left, tables)?;
            let right_rows = execute_rel(right, tables)?;
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
            let rows = execute_rel(input, tables)?;
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
    }
}
