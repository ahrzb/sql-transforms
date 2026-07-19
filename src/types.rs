use std::collections::{HashMap, HashSet};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Base {
    Int,
    Float,
    Str,
    Bool,
    /// Unresolvable — a passthrough column, a multi-type union, an
    /// unsupported generic annotation, etc. Maps to Python `Any`.
    Other,
    /// Ordered field list (name, type). Not `Copy` — Task 1 spine only,
    /// no SQL construction surface yet.
    Struct(Vec<(String, FieldType)>),
    /// Element type.
    List(Box<FieldType>),
}

#[derive(Clone, PartialEq, Eq, Debug)]
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
        Expr::Struct(fields) => {
            let field_types = fields
                .iter()
                .map(|(name, e)| Ok((name.clone(), infer_type(e, schemas)?)))
                .collect::<Result<_, InterpError>>()?;
            Ok(FieldType {
                base: Base::Struct(field_types),
                nullable: false,
            })
        }
        Expr::List(items) => {
            let item_types = items
                .iter()
                .map(|e| infer_type(e, schemas))
                .collect::<Result<Vec<_>, _>>()?;
            let elem = unify_list_element_types(&item_types);
            Ok(FieldType {
                base: Base::List(Box::new(elem)),
                nullable: false,
            })
        }
        Expr::FieldAccess { base, field } => {
            let base_ty = infer_type(base, schemas)?;
            match &base_ty.base {
                Base::Struct(fields) => fields
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, ft)| FieldType {
                        base: ft.base.clone(),
                        nullable: ft.nullable || base_ty.nullable,
                    })
                    .ok_or_else(|| InterpError::Build(format!("Unknown struct field: {field}"))),
                _ => Err(InterpError::Build(format!(
                    "Cannot access field '{field}' on a non-struct column"
                ))),
            }
        }
        Expr::Transform {
            input_features,
            output_fields,
            arg,
            ..
        } => {
            let arg_ty = infer_type(arg, schemas)?;
            match &arg_ty.base {
                Base::Struct(fields) => {
                    let got: HashSet<&String> = fields.iter().map(|(n, _)| n).collect();
                    let want: HashSet<&String> = input_features.iter().collect();
                    if got != want {
                        return Err(InterpError::Build(format!(
                            "transformer input struct fields {:?} do not match \
                             feature_names_in_ {:?}",
                            fields.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                            input_features
                        )));
                    }
                }
                _ => {
                    return Err(InterpError::Build(
                        "transformer argument must be a struct (e.g. named_struct(...))"
                            .to_string(),
                    ))
                }
            }
            Ok(FieldType {
                base: Base::Struct(output_fields.clone()),
                nullable: false,
            })
        }
        Expr::Case { arms, default } => {
            let mut branch_types: Vec<FieldType> = arms
                .iter()
                .map(|(_, result)| infer_type(result, schemas))
                .collect::<Result<_, _>>()?;
            let has_else = default.is_some();
            if let Some(d) = default {
                branch_types.push(infer_type(d, schemas)?);
            }
            // No explicit ELSE => an unmatched row yields NULL, so nullable
            // regardless of the branch types.
            let nullable = !has_else || branch_types.iter().any(|t| t.nullable);
            Ok(FieldType {
                base: common_base(&branch_types),
                nullable,
            })
        }
    }
}

/// Unify element types for a list literal: identical types (by FieldType
/// equality, so nullability must also agree) collapse to that type;
/// anything else (including an empty list) is unresolvable.
fn unify_list_element_types(item_types: &[FieldType]) -> FieldType {
    let mut iter = item_types.iter();
    let Some(first) = iter.next() else {
        return FieldType {
            base: Base::Other,
            nullable: true,
        };
    };
    if iter.all(|t| t == first) {
        first.clone()
    } else {
        FieldType {
            base: Base::Other,
            nullable: true,
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
            .cloned()
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
            found = Some(ft.clone());
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
        Value::Struct(fields) => FieldType {
            base: Base::Struct(
                fields
                    .iter()
                    .map(|(name, v)| (name.clone(), literal_type(v)))
                    .collect(),
            ),
            nullable: false,
        },
        Value::List(items) => {
            let inner = items.first().map(literal_type).unwrap_or(FieldType {
                base: Base::Other,
                nullable: true,
            });
            FieldType {
                base: Base::List(Box::new(inner)),
                nullable: false,
            }
        }
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
        BinOp::Concat => FieldType {
            base: Base::Str,
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

/// Is `inferred` provably safe to store in a field declared as `declared`?
/// Anything not provably wrong is allowed through — Pydantic's own
/// `model_validate()` is the real authority at `.infer()` time for
/// anything this can't rule out.
pub fn compatible(inferred: &Base, declared: &Base) -> bool {
    match (inferred, declared) {
        (a, b) if a == b => true,
        // Every valid int is a valid float; Pydantic's default lax mode
        // coerces this without loss.
        (Base::Int, Base::Float) => true,
        // We have no basis to say an unresolvable inferred type is wrong.
        (Base::Other, _) => true,
        // Struct compatible iff same set of field names (order-independent)
        // with compatible field types per name; list iff compatible element
        // type.
        (Base::Struct(a_fields), Base::Struct(b_fields)) => {
            a_fields.len() == b_fields.len()
                && a_fields.iter().all(|(a_name, a_ft)| {
                    b_fields
                        .iter()
                        .find(|(b_name, _)| b_name == a_name)
                        .is_some_and(|(_, b_ft)| compatible(&a_ft.base, &b_ft.base))
                })
        }
        (Base::List(a_inner), Base::List(b_inner)) => compatible(&a_inner.base, &b_inner.base),
        _ => false,
    }
}

/// Common result base for variadic same-shape functions (COALESCE/NULLIF):
/// all-equal keeps that base; a mix of Int/Float widens to Float (DataFusion's
/// numeric supertype); anything else is left unresolved for Pydantic to judge.
fn common_base(args: &[FieldType]) -> Base {
    match args.first() {
        None => Base::Other,
        Some(first) if args.iter().all(|a| a.base == first.base) => first.base.clone(),
        _ if args.iter().all(|a| matches!(a.base, Base::Int | Base::Float)) => Base::Float,
        _ => Base::Other,
    }
}

fn function_type(name: &str, args: &[FieldType]) -> FieldType {
    let any_nullable = args.iter().any(|a| a.nullable);
    match name {
        "upper" | "lower" | "trim" | "substr" | "substring" => FieldType {
            base: Base::Str,
            nullable: any_nullable,
        },
        // ABS preserves its argument's type; ROUND always yields Float (DataFusion
        // returns Float64 even for an integer argument).
        "abs" => {
            let base = args.first().map(|a| a.base.clone()).unwrap_or(Base::Other);
            FieldType {
                base,
                nullable: any_nullable,
            }
        }
        "round" => FieldType {
            base: Base::Float,
            nullable: any_nullable,
        },
        "concat" => FieldType {
            base: Base::Str,
            nullable: false,
        },
        // DataFusion types these as the common supertype of the arguments
        // (COALESCE(int, float) -> float), not args[0]'s type.
        "coalesce" | "nullif" => FieldType {
            base: common_base(args),
            nullable: true,
        },
        _ => FieldType {
            base: Base::Other,
            nullable: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ft(base: Base) -> FieldType {
        FieldType {
            base,
            nullable: false,
        }
    }

    #[test]
    fn struct_compatibility_is_name_keyed_not_positional() {
        let xy = Base::Struct(vec![("x".into(), ft(Base::Int)), ("y".into(), ft(Base::Str))]);
        let yx = Base::Struct(vec![("y".into(), ft(Base::Str)), ("x".into(), ft(Base::Int))]);
        assert!(compatible(&xy, &yx), "same names+types, reordered, should be compatible");

        let different_names =
            Base::Struct(vec![("x".into(), ft(Base::Int)), ("z".into(), ft(Base::Str))]);
        assert!(!compatible(&xy, &different_names));

        let different_types =
            Base::Struct(vec![("x".into(), ft(Base::Str)), ("y".into(), ft(Base::Str))]);
        assert!(!compatible(&xy, &different_types));
    }
}
