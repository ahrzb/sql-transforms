use sqlparser::ast::Expr as SqlExpr;

use crate::expr::Expr;
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
        _ => Err(InterpError::Build(format!("Unsupported expression: {e}"))),
    }
}
