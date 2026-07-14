use std::collections::HashMap;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyErr;
use sqlparser::ast::{
    Expr as SqlExpr, SelectItem, SetExpr, Statement, TableFactor, TableWithJoins,
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
    if from.len() != 1 {
        return Err(InterpError::Build(
            "Multiple FROM tables are not yet supported".to_string(),
        ));
    }
    let twj = &from[0];
    if !twj.joins.is_empty() {
        return Err(InterpError::Build("JOIN is not yet supported".to_string()));
    }
    build_table_factor(&twj.relation)
}

fn build_table_factor(factor: &TableFactor) -> Result<RelNode, InterpError> {
    match factor {
        TableFactor::Table { name, .. } => Ok(RelNode::TableScan {
            table: name.to_string(),
        }),
        _ => Err(InterpError::Build("Unsupported FROM clause".to_string())),
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
    }
}
