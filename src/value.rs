//! Runtime scalar values shared by every interpreter engine.
//!
//! The representation and its Python marshalling are engine-agnostic;
//! anything with SQL semantics attached (comparison, arithmetic, display)
//! lives in the per-engine `expr` module instead.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

#[derive(Debug)]
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
    /// Ordered field list (name, value). Field order is significant for
    /// equality/hash, mirroring `Base::Struct`.
    Struct(Vec<(String, Value)>),
    /// Ordered element list.
    List(Vec<Value>),
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
            Value::Struct(fields) => {
                Value::Struct(fields.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            }
            Value::List(items) => Value::List(items.iter().map(|v| v.clone()).collect()),
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
            (Value::Struct(a), Value::Struct(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
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
            Value::Struct(fields) => {
                6u8.hash(state);
                fields.hash(state);
            }
            Value::List(items) => {
                7u8.hash(state);
                items.hash(state);
            }
        }
    }
}

/// Human-readable type name for error messages.
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Str(_) => "string",
        Value::Bool(_) => "bool",
        Value::Null => "null",
        Value::Object(_) => "object",
        Value::Struct(_) => "struct",
        Value::List(_) => "list",
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

    /// Schema-driven read: converts a Python value into a `Value` per the
    /// field's declared `Base`, recursing into `Struct`/`List` so a nested
    /// dict/list is marshalled by its declared shape rather than falling
    /// through to an opaque `Value::Object`. Scalars behave exactly like
    /// `from_pyobject`. `obj` may be a raw `dict`/`list` OR (when the row
    /// model declares a nested pydantic submodel / `list[X]`) an already-
    /// validated nested `BaseModel` instance / `list` — struct field access
    /// falls back from dict indexing to attribute access to cover both.
    pub fn from_pyobject_typed(
        obj: &Bound<'_, PyAny>,
        base: &crate::types::Base,
    ) -> PyResult<Value> {
        use crate::types::Base;
        if obj.is_none() {
            return Ok(Value::Null);
        }
        match base {
            Base::Struct(fields) => {
                // Accept a dict (read by key) or a pydantic-model-like object
                // (read by attr) as struct-shaped input; anything else (e.g.
                // a bare scalar) is a genuine type mismatch and must error,
                // not silently marshal into an all-null struct.
                let dict = obj.cast::<PyDict>().ok();
                // Probe `model_fields` on the CLASS, not the instance: instance
                // access is deprecated in Pydantic 2.11 (PydanticDeprecatedSince211)
                // and removed in 3.0, where the instance probe would return false
                // and misclassify a struct value. Class access stays valid.
                if dict.is_none() && !obj.get_type().hasattr("model_fields").unwrap_or(false) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Expected a struct/dict value for a struct-typed field: got {}",
                        obj.get_type().name()?
                    )));
                }
                let mut out = Vec::with_capacity(fields.len());
                for (name, field_ft) in fields {
                    let field_val = if let Some(dict) = &dict {
                        dict.get_item(name)?
                    } else {
                        obj.getattr(name.as_str()).ok()
                    };
                    let v = match field_val {
                        Some(item) => Value::from_pyobject_typed(&item, &field_ft.base)?,
                        None => Value::Null,
                    };
                    out.push((name.clone(), v));
                }
                Ok(Value::Struct(out))
            }
            Base::List(inner) => {
                let list = obj.cast::<PyList>().map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Expected a list value for a list-typed field: {e}"
                    ))
                })?;
                let mut out = Vec::with_capacity(list.len());
                for item in list.iter() {
                    out.push(Value::from_pyobject_typed(&item, &inner.base)?);
                }
                Ok(Value::List(out))
            }
            _ => Value::from_pyobject(obj),
        }
    }

    pub fn to_pyobject(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
            Value::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
            Value::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
            Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
            Value::Null => py.None(),
            Value::Object(o) => o.clone_ref(py),
            Value::Struct(fields) => {
                let dict = PyDict::new(py);
                for (k, v) in fields {
                    dict.set_item(k, v.to_pyobject(py)?)?;
                }
                dict.into_any().unbind()
            }
            Value::List(items) => {
                let elements = items
                    .iter()
                    .map(|v| v.to_pyobject(py))
                    .collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, elements)?.into_any().unbind()
            }
        })
    }
}
