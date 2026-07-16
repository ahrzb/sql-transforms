use sqlparser::ast::{
    Array, BinaryOperator, DataType, Expr as SqlExpr, Function, FunctionArg, FunctionArgExpr,
    FunctionArguments, UnaryOperator, Value as SqlValue,
};

use crate::expr::{BinOp, CastType, Expr, Value};
use crate::plan::InterpError;

pub fn convert_expr(e: &SqlExpr) -> Result<Expr, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column {
            table: None,
            name: ident.value.clone(),
        }),
        // `a.b` is ambiguous at parse time -- it's either `table.column` or
        // `struct_col.field`, and we don't know the relation-alias set here
        // (that only exists once the row/static tables are supplied, in
        // plan::validate_columns). Parse the first two parts as
        // Column{table,name} (today's `table.column` shape) and layer any
        // further dotted parts on top as FieldAccess (so `s.a.b` parses as
        // field `b` of field `a` of column-or-table `s`). validate_expr
        // rewrites the Column node into a FieldAccess when its `table` part
        // turns out not to be a relation alias -- see plan.rs.
        SqlExpr::CompoundIdentifier(parts) if parts.len() >= 2 => {
            let mut expr = Expr::Column {
                table: Some(parts[0].value.clone()),
                name: parts[1].value.clone(),
            };
            for part in &parts[2..] {
                expr = Expr::FieldAccess {
                    base: Box::new(expr),
                    field: part.value.clone(),
                };
            }
            Ok(expr)
        }
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
        SqlExpr::Function(func) => convert_function(func),
        SqlExpr::Array(Array { elem, .. }) => Ok(Expr::List(
            elem.iter().map(convert_expr).collect::<Result<_, _>>()?,
        )),
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(Expr::Cast {
            expr: Box::new(convert_expr(expr)?),
            target: convert_cast_type(data_type)?,
        }),
        // sqlparser 0.62 parses SUBSTR/SUBSTRING into a dedicated AST node
        // rather than a generic Function call; normalize it to
        // Expr::Function("substr", ...) so eval_builtin's dispatch handles
        // both call syntaxes uniformly.
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            if substring_from.is_none() && substring_for.is_none() {
                return Err(InterpError::Build(
                    "SUBSTRING requires FROM and/or FOR".to_string(),
                ));
            }
            let mut args = vec![convert_expr(expr)?];
            match substring_from {
                Some(from) => args.push(convert_expr(from)?),
                // SQL-92: SUBSTRING(expr FOR n) with no FROM means "the
                // first n characters", equivalent to
                // SUBSTRING(expr FROM 1 FOR n).
                None => args.push(Expr::Literal(Value::Int(1))),
            }
            if let Some(for_) = substring_for {
                args.push(convert_expr(for_)?);
            }
            Ok(Expr::Function {
                name: "substr".to_string(),
                args,
            })
        }
        // Likewise TRIM(expr) is a dedicated AST node. Only the plain form
        // (no BOTH/LEADING/TRAILING side, no explicit trim characters) maps
        // onto eval_builtin's "trim" (Rust's str::trim, whitespace only).
        SqlExpr::Trim {
            expr,
            trim_where: None,
            trim_what: None,
            trim_characters: None,
            ..
        } => Ok(Expr::Function {
            name: "trim".to_string(),
            args: vec![convert_expr(expr)?],
        }),
        _ => Err(InterpError::Build(format!("Unsupported expression: {e}"))),
    }
}

fn convert_function(func: &Function) -> Result<Expr, InterpError> {
    let name = func.name.to_string().to_lowercase();
    let args = match &func.args {
        FunctionArguments::List(list) => list
            .args
            .iter()
            .map(convert_function_arg)
            .collect::<Result<Vec<_>, _>>()?,
        FunctionArguments::None => Vec::new(),
        FunctionArguments::Subquery(_) => {
            return Err(InterpError::Build(format!(
                "Subquery arguments are not supported in function: {name}"
            )))
        }
    };
    if name == "named_struct" {
        if args.len() % 2 != 0 {
            return Err(InterpError::Build(
                "named_struct expects an even number of arguments (key, value, ...)".to_string(),
            ));
        }
        let mut fields = Vec::with_capacity(args.len() / 2);
        let mut it = args.into_iter();
        while let (Some(key), Some(value)) = (it.next(), it.next()) {
            let Expr::Literal(Value::Str(key)) = key else {
                return Err(InterpError::Build(
                    "named_struct field names must be string literals".to_string(),
                ));
            };
            fields.push((key, value));
        }
        return Ok(Expr::Struct(fields));
    }
    if name == "struct" {
        let fields = args
            .into_iter()
            .enumerate()
            .map(|(i, e)| (format!("c{i}"), e))
            .collect();
        return Ok(Expr::Struct(fields));
    }
    Ok(Expr::Function { name, args })
}

fn convert_function_arg(arg: &FunctionArg) -> Result<Expr, InterpError> {
    match arg {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => convert_expr(e),
        _ => Err(InterpError::Build(
            "Only plain positional function arguments are supported".to_string(),
        )),
    }
}

fn convert_cast_type(dt: &DataType) -> Result<CastType, InterpError> {
    let name = dt.to_string().to_uppercase();
    if name.starts_with("VARCHAR")
        || name.starts_with("TEXT")
        || name.starts_with("STRING")
        || name.starts_with("CHAR")
    {
        Ok(CastType::Str)
    } else if name.starts_with("BIGINT") || name.starts_with("INT") {
        Ok(CastType::Int)
    } else if name.starts_with("DOUBLE") || name.starts_with("FLOAT") || name.starts_with("REAL") {
        Ok(CastType::Float)
    } else if name.starts_with("BOOL") {
        Ok(CastType::Bool)
    } else {
        Err(InterpError::Build(format!(
            "Unsupported CAST target type: {name}"
        )))
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
