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
