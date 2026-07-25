//! The type vocabulary shared by every interpreter engine.
//!
//! `Base`/`FieldType`/`Schema` describe the shape of a column; they carry no
//! SQL semantics. Which type a given *expression* produces is engine-specific
//! and lives in each engine's own `types::infer_type`.

use std::collections::HashMap;

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
