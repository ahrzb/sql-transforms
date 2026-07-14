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
        (Value::Int(a), Value::Int(b)) => Ok(match op {
            BinOp::Add => Value::Int(a + b),
            BinOp::Sub => Value::Int(a - b),
            BinOp::Mul => Value::Int(a * b),
            BinOp::Div => Value::Int(a / b),
            BinOp::Mod => Value::Int(a % b),
            _ => unreachable!(),
        }),
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
