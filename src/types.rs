use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Base {
    Int,
    Float,
    Str,
    Bool,
    /// Unresolvable — a passthrough column, a multi-type union, an
    /// unsupported generic annotation, etc. Maps to Python `Any`.
    Other,
}

#[derive(Clone, Copy, Debug)]
pub struct FieldType {
    pub base: Base,
    pub nullable: bool,
}

pub type Schema = HashMap<String, FieldType>;

use crate::expr::{BinOp, CastType, Expr, Value};
use crate::plan::InterpError;

/// Statically infers the FieldType of a projection expression, mirroring
/// `crate::expr::eval()`'s structure but computing a type instead of a
/// value. Sound but not tight on nullability: `nullable: true` means
/// "cannot prove this can't be NULL," not "will be NULL."
pub fn infer_type(
    expr: &Expr,
    schemas: &HashMap<String, Schema>,
) -> Result<FieldType, InterpError> {
    match expr {
        Expr::Column { table, name } => resolve_column_type(table.as_deref(), name, schemas),
        Expr::Literal(v) => Ok(literal_type(v)),
        Expr::BinaryOp { op, left, right } => {
            let l = infer_type(left, schemas)?;
            let r = infer_type(right, schemas)?;
            Ok(binary_op_type(*op, l, r))
        }
        Expr::Not(inner) => {
            let inner_ty = infer_type(inner, schemas)?;
            Ok(FieldType {
                base: Base::Bool,
                nullable: inner_ty.nullable,
            })
        }
        Expr::Cast { expr, target } => {
            let inner_ty = infer_type(expr, schemas)?;
            Ok(FieldType {
                base: cast_target_base(*target),
                nullable: inner_ty.nullable,
            })
        }
        Expr::Function { name, args } => {
            let arg_types: Vec<FieldType> = args
                .iter()
                .map(|a| infer_type(a, schemas))
                .collect::<Result<_, _>>()?;
            Ok(function_type(name, &arg_types))
        }
    }
}

fn resolve_column_type(
    table: Option<&str>,
    name: &str,
    schemas: &HashMap<String, Schema>,
) -> Result<FieldType, InterpError> {
    if let Some(t) = table {
        return schemas
            .get(t)
            .and_then(|s| s.get(name))
            .copied()
            .ok_or_else(|| InterpError::Build(format!("Unknown column: {t}.{name}")));
    }
    let mut found = None;
    for s in schemas.values() {
        if let Some(ft) = s.get(name) {
            if found.is_some() {
                return Err(InterpError::Build(format!(
                    "Ambiguous column reference: {name}"
                )));
            }
            found = Some(*ft);
        }
    }
    found.ok_or_else(|| InterpError::Build(format!("Unknown column: {name}")))
}

fn literal_type(v: &Value) -> FieldType {
    match v {
        Value::Int(_) => FieldType {
            base: Base::Int,
            nullable: false,
        },
        Value::Float(_) => FieldType {
            base: Base::Float,
            nullable: false,
        },
        Value::Str(_) => FieldType {
            base: Base::Str,
            nullable: false,
        },
        Value::Bool(_) => FieldType {
            base: Base::Bool,
            nullable: false,
        },
        Value::Null | Value::Object(_) => FieldType {
            base: Base::Other,
            nullable: true,
        },
    }
}

fn binary_op_type(op: BinOp, l: FieldType, r: FieldType) -> FieldType {
    let nullable = l.nullable || r.nullable;
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
            let base = if l.base == Base::Int && r.base == Base::Int {
                Base::Int
            } else {
                Base::Float
            };
            FieldType { base, nullable }
        }
        BinOp::Eq
        | BinOp::NotEq
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::LtEq
        | BinOp::GtEq
        | BinOp::And
        | BinOp::Or => FieldType {
            base: Base::Bool,
            nullable,
        },
    }
}

fn cast_target_base(target: CastType) -> Base {
    match target {
        CastType::Str => Base::Str,
        CastType::Int => Base::Int,
        CastType::Float => Base::Float,
        CastType::Bool => Base::Bool,
    }
}

fn function_type(name: &str, args: &[FieldType]) -> FieldType {
    let any_nullable = args.iter().any(|a| a.nullable);
    match name {
        "upper" | "lower" | "trim" | "substr" | "substring" => FieldType {
            base: Base::Str,
            nullable: any_nullable,
        },
        "abs" | "round" => {
            let base = args.first().map(|a| a.base).unwrap_or(Base::Other);
            FieldType {
                base,
                nullable: any_nullable,
            }
        }
        "concat" => FieldType {
            base: Base::Str,
            nullable: false,
        },
        "coalesce" | "nullif" => {
            let base = args.first().map(|a| a.base).unwrap_or(Base::Other);
            FieldType {
                base,
                nullable: true,
            }
        }
        _ => FieldType {
            base: Base::Other,
            nullable: true,
        },
    }
}
