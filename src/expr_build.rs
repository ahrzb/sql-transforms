use sqlparser::ast::{BinaryOperator, Expr as SqlExpr, UnaryOperator, Value as SqlValue};

use crate::expr::{BinOp, Expr, Value};
use crate::plan::InterpError;

pub fn convert_expr(e: &SqlExpr) -> Result<Expr, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column {
            table: None,
            name: ident.value.clone(),
        }),
        SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 => Ok(Expr::Column {
            table: Some(parts[0].value.clone()),
            name: parts[1].value.clone(),
        }),
        SqlExpr::Value(vws) => Ok(Expr::Literal(convert_literal(&vws.value)?)),
        SqlExpr::Nested(inner) => convert_expr(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr,
        } => Ok(Expr::Not(Box::new(convert_expr(expr)?))),
        SqlExpr::BinaryOp { left, op, right } => {
            let bin_op = convert_binary_operator(op)?;
            Ok(Expr::BinaryOp {
                op: bin_op,
                left: Box::new(convert_expr(left)?),
                right: Box::new(convert_expr(right)?),
            })
        }
        _ => Err(InterpError::Build(format!("Unsupported expression: {e}"))),
    }
}

fn convert_literal(v: &SqlValue) -> Result<Value, InterpError> {
    match v {
        SqlValue::Null => Ok(Value::Null),
        SqlValue::Boolean(b) => Ok(Value::Bool(*b)),
        SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
            Ok(Value::Str(s.clone()))
        }
        SqlValue::Number(text, _) => {
            if text.contains('.') || text.to_lowercase().contains('e') {
                text.parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| InterpError::Build(format!("Invalid numeric literal: {text}")))
            } else {
                text.parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| InterpError::Build(format!("Invalid numeric literal: {text}")))
            }
        }
        other => Err(InterpError::Build(format!("Unsupported literal: {other}"))),
    }
}

fn convert_binary_operator(op: &BinaryOperator) -> Result<BinOp, InterpError> {
    match op {
        BinaryOperator::Plus => Ok(BinOp::Add),
        BinaryOperator::Minus => Ok(BinOp::Sub),
        BinaryOperator::Multiply => Ok(BinOp::Mul),
        BinaryOperator::Divide => Ok(BinOp::Div),
        BinaryOperator::Modulo => Ok(BinOp::Mod),
        BinaryOperator::Eq => Ok(BinOp::Eq),
        BinaryOperator::NotEq => Ok(BinOp::NotEq),
        BinaryOperator::Lt => Ok(BinOp::Lt),
        BinaryOperator::Gt => Ok(BinOp::Gt),
        BinaryOperator::LtEq => Ok(BinOp::LtEq),
        BinaryOperator::GtEq => Ok(BinOp::GtEq),
        BinaryOperator::And => Ok(BinOp::And),
        BinaryOperator::Or => Ok(BinOp::Or),
        other => Err(InterpError::Build(format!("Unsupported operator: {other}"))),
    }
}
