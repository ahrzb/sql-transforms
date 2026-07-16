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

/// String form used by CONCAT and CAST(.. AS VARCHAR).
pub fn display_value(v: &Value) -> String {
    match v {
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Str(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        Value::Object(_) => "<object>".to_string(),
        Value::Struct(fields) => {
            let inner = fields
                .iter()
                .map(|(k, v)| format!("{k}: {}", display_value(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
        Value::List(items) => {
            let inner = items
                .iter()
                .map(display_value)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
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
                if dict.is_none() && !obj.hasattr("model_fields").unwrap_or(false) {
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

#[derive(Clone)]
pub enum Expr {
    Column {
        table: Option<String>,
        name: String,
    },
    Literal(Value),
    BinaryOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Not(Box<Expr>),
    Function {
        name: String,
        args: Vec<Expr>,
    },
    Cast {
        expr: Box<Expr>,
        target: CastType,
    },
    Struct(Vec<(String, Expr)>),
    List(Vec<Expr>),
    FieldAccess {
        base: Box<Expr>,
        field: String,
    },
}

#[derive(Clone, Copy)]
pub enum CastType {
    Str,
    Int,
    Float,
    Bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    And,
    Or,
}

pub fn eval(expr: &Expr, row: &crate::plan::Row) -> Result<Value, crate::plan::InterpError> {
    match expr {
        Expr::Column { table, name } => resolve_column(row, table.as_deref(), name),
        Expr::Literal(v) => Ok(v.clone()),
        Expr::BinaryOp { op, left, right } => {
            let l = eval(left, row)?;
            let r = eval(right, row)?;
            eval_binary_op(*op, l, r)
        }
        Expr::Not(inner) => {
            let v = eval(inner, row)?;
            match as_tribool(&v)? {
                Some(b) => Ok(Value::Bool(!b)),
                None => Ok(Value::Null),
            }
        }
        Expr::Function { name, args } => {
            let values: Vec<Value> = args
                .iter()
                .map(|a| eval(a, row))
                .collect::<Result<_, _>>()?;
            eval_builtin(name, values)
        }
        Expr::Cast { expr, target } => {
            let v = eval(expr, row)?;
            eval_cast(v, *target)
        }
        Expr::Struct(fields) => {
            let values = fields
                .iter()
                .map(|(k, e)| Ok((k.clone(), eval(e, row)?)))
                .collect::<Result<_, crate::plan::InterpError>>()?;
            Ok(Value::Struct(values))
        }
        Expr::List(items) => {
            let values = items
                .iter()
                .map(|e| eval(e, row))
                .collect::<Result<_, _>>()?;
            Ok(Value::List(values))
        }
        Expr::FieldAccess { base, field } => {
            let v = eval(base, row)?;
            match v {
                Value::Null => Ok(Value::Null),
                Value::Struct(fields) => fields
                    .into_iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, v)| v)
                    .ok_or_else(|| {
                        crate::plan::InterpError::Eval(format!("Unknown struct field: {field}"))
                    }),
                other => Err(crate::plan::InterpError::Eval(format!(
                    "Cannot access field '{field}' on a {} value",
                    type_name(&other)
                ))),
            }
        }
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

fn eval_binary_op(op: BinOp, l: Value, r: Value) -> Result<Value, crate::plan::InterpError> {
    match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => arithmetic(op, l, r),
        BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
            comparison(op, l, r)
        }
        BinOp::And | BinOp::Or => logic(op, l, r),
    }
}

fn arithmetic(op: BinOp, l: Value, r: Value) -> Result<Value, crate::plan::InterpError> {
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => {
            if matches!(op, BinOp::Div | BinOp::Mod) && b == 0 {
                return Err(crate::plan::InterpError::Eval(
                    "division by zero".to_string(),
                ));
            }
            Ok(match op {
                BinOp::Add => Value::Int(a + b),
                BinOp::Sub => Value::Int(a - b),
                BinOp::Mul => Value::Int(a * b),
                BinOp::Div => Value::Int(a / b),
                BinOp::Mod => Value::Int(a % b),
                _ => unreachable!(),
            })
        }
        (a, b) => {
            let af = as_f64(&a)?;
            let bf = as_f64(&b)?;
            Ok(match op {
                BinOp::Add => Value::Float(af + bf),
                BinOp::Sub => Value::Float(af - bf),
                BinOp::Mul => Value::Float(af * bf),
                BinOp::Div => Value::Float(af / bf),
                BinOp::Mod => Value::Float(af % bf),
                _ => unreachable!(),
            })
        }
    }
}

fn as_f64(v: &Value) -> Result<f64, crate::plan::InterpError> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        other => Err(crate::plan::InterpError::Eval(format!(
            "Cannot use a {} value in an arithmetic expression",
            type_name(other)
        ))),
    }
}

fn comparison(op: BinOp, l: Value, r: Value) -> Result<Value, crate::plan::InterpError> {
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    let ordering = compare_values(&l, &r)?;
    Ok(Value::Bool(match op {
        BinOp::Eq => ordering == std::cmp::Ordering::Equal,
        BinOp::NotEq => ordering != std::cmp::Ordering::Equal,
        BinOp::Lt => ordering == std::cmp::Ordering::Less,
        BinOp::Gt => ordering == std::cmp::Ordering::Greater,
        BinOp::LtEq => ordering != std::cmp::Ordering::Greater,
        BinOp::GtEq => ordering != std::cmp::Ordering::Less,
        _ => unreachable!(),
    }))
}

fn compare_values(l: &Value, r: &Value) -> Result<std::cmp::Ordering, crate::plan::InterpError> {
    match (l, r) {
        (Value::Int(a), Value::Int(b)) => Ok(a.cmp(b)),
        (Value::Str(a), Value::Str(b)) => Ok(a.cmp(b)),
        (Value::Bool(a), Value::Bool(b)) => Ok(a.cmp(b)),
        (a, b) => {
            let af = as_f64(a)?;
            let bf = as_f64(b)?;
            af.partial_cmp(&bf)
                .ok_or_else(|| crate::plan::InterpError::Eval("Cannot compare NaN".to_string()))
        }
    }
}

fn logic(op: BinOp, l: Value, r: Value) -> Result<Value, crate::plan::InterpError> {
    let lb = as_tribool(&l)?;
    let rb = as_tribool(&r)?;
    Ok(match op {
        BinOp::And => match (lb, rb) {
            (Some(false), _) | (_, Some(false)) => Value::Bool(false),
            (Some(true), Some(true)) => Value::Bool(true),
            _ => Value::Null,
        },
        BinOp::Or => match (lb, rb) {
            (Some(true), _) | (_, Some(true)) => Value::Bool(true),
            (Some(false), Some(false)) => Value::Bool(false),
            _ => Value::Null,
        },
        _ => unreachable!(),
    })
}

fn as_tribool(v: &Value) -> Result<Option<bool>, crate::plan::InterpError> {
    match v {
        Value::Bool(b) => Ok(Some(*b)),
        Value::Null => Ok(None),
        other => Err(crate::plan::InterpError::Eval(format!(
            "Expected a boolean expression, got a {} value",
            type_name(other)
        ))),
    }
}

fn eval_builtin(name: &str, args: Vec<Value>) -> Result<Value, crate::plan::InterpError> {
    use crate::plan::InterpError;

    if matches!(name, "upper" | "lower" | "trim" | "substr" | "substring")
        && args.iter().any(|a| matches!(a, Value::Null))
    {
        return Ok(Value::Null);
    }

    match name {
        "upper" => Ok(Value::Str(as_str(&args, 0)?.to_uppercase())),
        "lower" => Ok(Value::Str(as_str(&args, 0)?.to_lowercase())),
        "trim" => Ok(Value::Str(as_str(&args, 0)?.trim().to_string())),
        "concat" => {
            let mut s = String::new();
            for a in &args {
                if !matches!(a, Value::Null) {
                    s.push_str(&display_value(a));
                }
            }
            Ok(Value::Str(s))
        }
        "abs" => match &args[0] {
            Value::Int(i) => Ok(Value::Int(i.abs())),
            Value::Float(f) => Ok(Value::Float(f.abs())),
            Value::Null => Ok(Value::Null),
            other => Err(InterpError::Eval(format!(
                "ABS expects a number, got a {} value",
                type_name(other)
            ))),
        },
        "round" => match &args[0] {
            Value::Float(f) => Ok(Value::Float(f.round())),
            Value::Int(i) => Ok(Value::Int(*i)),
            Value::Null => Ok(Value::Null),
            other => Err(InterpError::Eval(format!(
                "ROUND expects a number, got a {} value",
                type_name(other)
            ))),
        },
        "substr" | "substring" => {
            let s = as_str(&args, 0)?;
            let start = as_i64(&args, 1)?;
            let length = if args.len() > 2 {
                Some(as_i64(&args, 2)?)
            } else {
                None
            };
            Ok(Value::Str(substr(s, start, length)))
        }
        "coalesce" => Ok(args
            .into_iter()
            .find(|v| !matches!(v, Value::Null))
            .unwrap_or(Value::Null)),
        "nullif" => {
            if args.len() != 2 {
                return Err(InterpError::Eval("NULLIF expects 2 arguments".to_string()));
            }
            if args[0] == args[1] {
                Ok(Value::Null)
            } else {
                Ok(args[0].clone())
            }
        }
        other => Err(InterpError::Eval(format!("Unknown function: {other}"))),
    }
}

fn as_str(args: &[Value], idx: usize) -> Result<&str, crate::plan::InterpError> {
    match args.get(idx) {
        Some(Value::Str(s)) => Ok(s.as_str()),
        other => Err(crate::plan::InterpError::Eval(format!(
            "Expected a string argument at position {idx}, got {:?}",
            other.map(type_name)
        ))),
    }
}

fn as_i64(args: &[Value], idx: usize) -> Result<i64, crate::plan::InterpError> {
    match args.get(idx) {
        Some(Value::Int(i)) => Ok(*i),
        other => Err(crate::plan::InterpError::Eval(format!(
            "Expected an integer argument at position {idx}, got {:?}",
            other.map(type_name)
        ))),
    }
}

fn substr(s: &str, start: i64, length: Option<i64>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let idx = if start > 0 { (start - 1) as usize } else { 0 };
    let idx = idx.min(chars.len());
    let end = match length {
        Some(len) => (idx + len.max(0) as usize).min(chars.len()),
        None => chars.len(),
    };
    chars[idx..end].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_and_list_value_equality() {
        let a = Value::Struct(vec![("x".into(), Value::Int(1))]);
        let b = Value::Struct(vec![("x".into(), Value::Int(1))]);
        let c = Value::List(vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(c.clone(), c);
    }
}

fn eval_cast(v: Value, target: CastType) -> Result<Value, crate::plan::InterpError> {
    use crate::plan::InterpError;

    if matches!(v, Value::Null) {
        return Ok(Value::Null);
    }
    Ok(match target {
        CastType::Str => Value::Str(display_value(&v)),
        CastType::Int => match v {
            Value::Int(i) => Value::Int(i),
            Value::Float(f) => Value::Int(f.trunc() as i64),
            Value::Str(s) => Value::Int(
                s.trim()
                    .parse::<i64>()
                    .map_err(|_| InterpError::Eval(format!("Cannot cast '{s}' to INT")))?,
            ),
            Value::Bool(b) => Value::Int(b as i64),
            Value::Null | Value::Object(_) | Value::Struct(_) | Value::List(_) => {
                return Err(InterpError::Eval(
                    "Cannot cast this value to INT".to_string(),
                ))
            }
        },
        CastType::Float => match v {
            Value::Int(i) => Value::Float(i as f64),
            Value::Float(f) => Value::Float(f),
            Value::Str(s) => Value::Float(
                s.trim()
                    .parse::<f64>()
                    .map_err(|_| InterpError::Eval(format!("Cannot cast '{s}' to FLOAT")))?,
            ),
            Value::Bool(b) => Value::Float(if b { 1.0 } else { 0.0 }),
            Value::Null | Value::Object(_) | Value::Struct(_) | Value::List(_) => {
                return Err(InterpError::Eval(
                    "Cannot cast this value to FLOAT".to_string(),
                ))
            }
        },
        CastType::Bool => match v {
            Value::Bool(b) => Value::Bool(b),
            Value::Int(i) => Value::Bool(i != 0),
            Value::Float(f) => Value::Bool(f != 0.0),
            Value::Str(s) => Value::Bool(s.eq_ignore_ascii_case("true")),
            Value::Null | Value::Object(_) | Value::Struct(_) | Value::List(_) => {
                return Err(InterpError::Eval(
                    "Cannot cast this value to BOOLEAN".to_string(),
                ))
            }
        },
    })
}
