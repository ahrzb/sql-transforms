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
    /// Ordered field list (name, type).
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
