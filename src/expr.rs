use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

pub enum Value {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
    /// Opaque passthrough for row values that aren't a SQL primitive
    /// (e.g. a nested dict). Round-trips unchanged through column refs;
    /// arithmetic/comparison on it is a runtime error.
    Object(Py<PyAny>),
}

impl Clone for Value {
    // Py<PyAny> isn't Clone (cloning it requires a GIL token to bump the
    // refcount safely), so this can't be derived; Python::attach supplies
    // the token for the Object case.
    fn clone(&self) -> Self {
        match self {
            Value::Int(i) => Value::Int(*i),
            Value::Float(f) => Value::Float(*f),
            Value::Str(s) => Value::Str(s.clone()),
            Value::Bool(b) => Value::Bool(*b),
            Value::Null => Value::Null,
            Value::Object(o) => Python::attach(|py| Value::Object(o.clone_ref(py))),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Null, Value::Null) => true,
            (Value::Object(a), Value::Object(b)) => a.as_ptr() == b.as_ptr(),
            _ => false,
        }
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Int(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            Value::Float(f) => {
                1u8.hash(state);
                f.to_bits().hash(state);
            }
            Value::Str(s) => {
                2u8.hash(state);
                s.hash(state);
            }
            Value::Bool(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            Value::Null => 4u8.hash(state),
            Value::Object(o) => {
                5u8.hash(state);
                (o.as_ptr() as usize).hash(state);
            }
        }
    }
}

/// Human-readable type name for error messages (Value has no Debug impl
/// because Py<PyAny> can't derive one without a GIL token).
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "string",
        Value::Bool(_) => "bool",
        Value::Null => "null",
        Value::Object(_) => "object",
    }
}

/// String form used by CONCAT and CAST(.. AS VARCHAR).
pub fn display_value(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Str(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        Value::Object(_) => "<object>".to_string(),
    }
}

impl Value {
    pub fn from_pyobject(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
        if obj.is_none() {
            return Ok(Value::Null);
        }
        if let Ok(b) = obj.cast::<PyBool>() {
            return Ok(Value::Bool(b.is_true()));
        }
        if let Ok(i) = obj.cast::<PyInt>() {
            return Ok(Value::Int(i.extract::<i64>()?));
        }
        if let Ok(f) = obj.cast::<PyFloat>() {
            return Ok(Value::Float(f.extract::<f64>()?));
        }
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Value::Str(s.extract::<String>()?));
        }
        Ok(Value::Object(obj.clone().unbind()))
    }

    pub fn to_pyobject(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
            Value::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
            Value::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
            Value::Null => py.None(),
            Value::Object(o) => o.clone_ref(py),
        })
    }
}

#[derive(Clone)]
pub enum Expr {
    Column { table: Option<String>, name: String },
}

pub fn eval(expr: &Expr, row: &crate::plan::Row) -> Result<Value, crate::plan::InterpError> {
    match expr {
        Expr::Column { table, name } => resolve_column(row, table.as_deref(), name),
    }
}

fn resolve_column(
    row: &crate::plan::Row,
    table: Option<&str>,
    name: &str,
) -> Result<Value, crate::plan::InterpError> {
    use crate::plan::InterpError;

    if let Some(t) = table {
        return row
            .get(t)
            .and_then(|cols| cols.get(name))
            .cloned()
            .ok_or_else(|| InterpError::Build(format!("Unknown column: {t}.{name}")));
    }
    let mut found: Option<&Value> = None;
    for cols in row.values() {
        if let Some(v) = cols.get(name) {
            if found.is_some() {
                return Err(InterpError::Build(format!(
                    "Ambiguous column reference: {name}"
                )));
            }
            found = Some(v);
        }
    }
    found
        .cloned()
        .ok_or_else(|| InterpError::Build(format!("Unknown column: {name}")))
}
