# Phase 2: Rust SQL Interpreter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust/PyO3 extension module `sql_transform._interpreter` exposing `InferFn(sql, row_tables, static_tables)` that parses a SQL query once, pre-indexes static tables into lookup structures, and evaluates the query row-by-row for dict inputs — output matching DataFusion batch semantics exactly.

**Architecture:** Parse SQL with the `sqlparser` crate (the parser DataFusion itself is built on) directly into a custom lightweight `Plan`/`RelNode`/`Expr` tree — bypassing DataFusion's logical planner entirely, because that planner requires a full `Schema` for every table and `row_tables` are given only as bare names (no schema) at build time; schemas only become known per-row at `infer()` time. `static_tables` (real `pyarrow.Table` data) are converted at build time into columnar `HashMap` lookup indices keyed by the JOIN's ON columns. `InferFn::infer()` walks the plan tree once per call, evaluating `Expr` nodes against a nested per-table row map (`{table_name: {col: value}}`) so that same-named columns from different tables never collide.

**Tech Stack:** Rust 1.96, `pyo3` 0.29 (`extension-module`, `abi3-py313`), `sqlparser` 0.62, `maturin` 1.14 (build backend, replaces `hatchling`), Python 3.13, `datafusion` (Python) 54.0.0 — used only in tests, to produce the expected output every interpreter test is compared against.

## Global Constraints

- `pyo3 = { version = "0.29", features = ["extension-module", "abi3-py313"] }` in `Cargo.toml`
- `sqlparser = "0.62"` in `Cargo.toml`
- `[build-system] requires = ["maturin>=1.14,<2.0"]`, `build-backend = "maturin"` in `pyproject.toml`
- `maturin>=1.14,<2.0` added to `[dependency-groups].dev` so `uv run maturin develop` works
- Compiled extension module name: `sql_transform._interpreter` (`[lib] name = "_interpreter"` in `Cargo.toml`, `#[pymodule] fn _interpreter(...)` in `src/lib.rs`, `module-name = "sql_transform._interpreter"` in `[tool.maturin]`)
- Every interpreter test compares `InferFn(...).infer(...)` output against real DataFusion batch output for the same SQL + data (per spec's Testing Strategy) — never hand-computed "expected" values for arithmetic/function results
- Arithmetic/comparison/CAST semantics must match **actual DataFusion behavior**, verified empirically in this plan (int/int division truncates toward zero, matches Rust's native `i64` `/` and `%` — see Task 3), not assumed from prose
- No placeholders: `ValueError` at `InferFn()` build time for every unsupported pattern in the spec's Join/Expression tables; `KeyError` at `infer()` time for a missing lookup key, message includes the key value and table name
- Ruff lint + format passes on Python files before each commit (existing `[tool.ruff]` config unchanged); run `cargo fmt` on Rust files before each commit
- Pytest discovers both `*_test.py` (existing convention) and `test_*.py` (this feature's `tests/` dir, per spec's Repo Layout) — `pyproject.toml`'s `python_files` must list both patterns

---

### Task 1: Rust/maturin build scaffolding

**Files:**
- Modify: `pyproject.toml` (switch build backend from `hatchling` to `maturin`, add pytest file pattern)
- Create: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable `sql_transform._interpreter.InferFn` class with a `#[new]` constructor `(sql: str, row_tables: list[str], static_tables: dict[str, Any]) -> InferFn` and a stub `.infer(tables: dict[str, list[dict]]) -> list[dict]` method (always returns `[]` in this task — later tasks replace the body). This exact constructor and method signature is what every later task builds on.

- [ ] **Step 1: Switch `pyproject.toml` to the maturin build backend**

Read `pyproject.toml`. Replace the `[build-system]`, `[tool.hatch.build]`, and `[tool.hatch.build.targets.wheel]` sections with:

```toml
[build-system]
requires = ["maturin>=1.14,<2.0"]
build-backend = "maturin"

[tool.maturin]
module-name = "sql_transform._interpreter"
python-source = "."
exclude = ["**/*_test.py", "tests/**"]
```

Add `"maturin>=1.14,<2.0"` to `[dependency-groups].dev`:

```toml
[dependency-groups]
dev = [
    "ipdb>=0.13.13",
    "ipython>=9.2.0",
    "maturin>=1.14,<2.0",
    "pytest>=8.3.5",
    "ruff>=0.11.7",
]
```

Update `[tool.pytest.ini_options]` to discover both naming conventions:

```toml
[tool.pytest.ini_options]
python_files = ["*_test.py", "test_*.py"]
```

- [ ] **Step 2: Create `Cargo.toml`**

Create `Cargo.toml` at the repo root:

```toml
[package]
name = "sql-transform-interpreter"
version = "0.1.0"
edition = "2021"

[lib]
name = "_interpreter"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.29", features = ["extension-module", "abi3-py313"] }
sqlparser = "0.62"
```

- [ ] **Step 3: Create the PyO3 module stub — `src/lib.rs`**

Create `src/lib.rs`:

```rust
use std::collections::HashMap;

use pyo3::prelude::*;

#[pyclass]
struct InferFn {
    sql: String,
}

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        sql: String,
        row_tables: Vec<String>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = (&row_tables, &static_tables);
        Ok(InferFn { sql })
    }

    fn infer(&self, tables: HashMap<String, Vec<Py<PyAny>>>) -> PyResult<Vec<Py<PyAny>>> {
        let _ = (&self.sql, &tables);
        Ok(Vec::new())
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
```

- [ ] **Step 4: Write the smoke test**

Create `tests/test_interpreter.py`:

```python
"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Every behavioral test compares InferFn output against real DataFusion
batch output for the same SQL + data, per the Phase 2 spec's testing
strategy.
"""

from sql_transform._interpreter import InferFn


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    assert fn is not None
```

- [ ] **Step 5: Build and verify the smoke test passes**

```
uv sync
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: `uv sync` installs `maturin`; `maturin develop` compiles the Rust crate and installs `sql_transform._interpreter` into the venv; `test_module_imports_and_constructs` PASSES.

If `cargo build` reports type errors, run `cargo build 2>&1 | tail -50` to see them directly (faster than `maturin develop`) and fix against the installed crate docs: `cargo doc --open -p pyo3 --no-deps`.

- [ ] **Step 6: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add pyproject.toml Cargo.toml src/lib.rs tests/test_interpreter.py
git commit -m "chore: scaffold Rust/PyO3 interpreter module via maturin"
```

Note: `Cargo.lock` and `target/` will appear after the build — add a `.gitignore` entry for `/target` if one doesn't already exist, and commit `Cargo.lock` (binary/cdylib crates should commit their lockfile).

---

### Task 2: Value type + column pass-through projection

**Files:**
- Create: `src/expr.rs`
- Create: `src/plan.rs`
- Create: `src/expr_build.rs`
- Modify: `src/lib.rs` (wire `InferFn` to `plan::build_plan` / `plan::execute`)
- Modify: `tests/test_interpreter.py` (add behavioral tests)

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `expr::Value` enum: `Int(i64) | Float(f64) | Str(String) | Bool(bool) | Null | Object(Py<PyAny>)` — the interpreter's internal representation of any row/literal value. `Object` is an opaque passthrough for row values that aren't one of the four SQL-primitive types (e.g. a nested dict) — arithmetic/comparison on it is a runtime error, but it round-trips unchanged through column references.
  - `expr::Value::from_pyobject(&Bound<PyAny>) -> PyResult<Value>` and `Value::to_pyobject(Python) -> PyResult<Py<PyAny>>`
  - `plan::Row = HashMap<String, HashMap<String, Value>>` — table/alias name → column name → value. Keeping table as the outer key (instead of flattening) is what lets a later JOIN merge two tables with same-named columns without collision.
  - `plan::InterpError` enum (`Build(String) | MissingKey(String) | Eval(String)`) with `impl From<InterpError> for PyErr` — `Build` → `ValueError` (raised from `InferFn::new`), `MissingKey` → `KeyError` (raised from `infer`), `Eval` → `ValueError` (raised from `infer`)
  - `plan::Plan { projection: Vec<(String, Expr)>, input: RelNode }`, `plan::RelNode::TableScan { table: String }`
  - `plan::build_plan(sql: &str) -> Result<Plan, InterpError>`
  - `plan::execute(plan: &Plan, tables: &HashMap<String, Vec<HashMap<String, Value>>>) -> Result<Vec<HashMap<String, Value>>, InterpError>`
  - `expr_build::convert_expr(&sqlparser::ast::Expr) -> Result<Expr, InterpError>`

- [ ] **Step 1: Write failing tests for column pass-through**

Append to `tests/test_interpreter.py`:

```python
import datafusion


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


def test_column_pass_through():
    sql = "SELECT age FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_multiple_columns():
    sql = "SELECT a, b FROM data"
    data = {"a": [1], "b": ["x"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 1, "b": "x"}]})
    assert actual == _expected(sql, data)
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "column_pass_through or multiple_columns"
```

Expected: both FAIL — `infer()` currently always returns `[]`.

- [ ] **Step 3: Implement `src/expr.rs`**

Create `src/expr.rs`:

```rust
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

#[derive(Clone)]
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
        if let Ok(b) = obj.downcast::<PyBool>() {
            return Ok(Value::Bool(b.is_true()));
        }
        if let Ok(i) = obj.downcast::<PyInt>() {
            return Ok(Value::Int(i.extract::<i64>()?));
        }
        if let Ok(f) = obj.downcast::<PyFloat>() {
            return Ok(Value::Float(f.extract::<f64>()?));
        }
        if let Ok(s) = obj.downcast::<PyString>() {
            return Ok(Value::Str(s.extract::<String>()?));
        }
        Ok(Value::Object(obj.clone().unbind()))
    }

    pub fn to_pyobject(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Value::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
            Value::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
            Value::Str(s) => s.into_pyobject(py)?.into_any().unbind(),
            Value::Bool(b) => b.into_pyobject(py)?.into_any().unbind(),
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
```

- [ ] **Step 4: Implement `src/plan.rs`**

Create `src/plan.rs`:

```rust
use std::collections::HashMap;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyErr;
use sqlparser::ast::{
    Expr as SqlExpr, SelectItem, SetExpr, Statement, TableFactor, TableWithJoins,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::expr::{Expr, Value};

pub type Row = HashMap<String, HashMap<String, Value>>;

pub enum InterpError {
    Build(String),
    MissingKey(String),
    Eval(String),
}

impl From<InterpError> for PyErr {
    fn from(e: InterpError) -> PyErr {
        match e {
            InterpError::Build(msg) => PyValueError::new_err(msg),
            InterpError::MissingKey(msg) => PyKeyError::new_err(msg),
            InterpError::Eval(msg) => PyValueError::new_err(msg),
        }
    }
}

pub enum RelNode {
    TableScan { table: String },
}

pub struct Plan {
    pub projection: Vec<(String, Expr)>,
    pub input: RelNode,
}

pub fn build_plan(sql: &str) -> Result<Plan, InterpError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|e| InterpError::Build(format!("SQL parse error: {e}")))?;

    if statements.len() != 1 {
        return Err(InterpError::Build(
            "Expected exactly one SQL statement".to_string(),
        ));
    }

    let select = match &statements[0] {
        Statement::Query(query) => match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            _ => return Err(InterpError::Build("Only SELECT queries are supported".to_string())),
        },
        _ => return Err(InterpError::Build("Only SELECT queries are supported".to_string())),
    };

    let input = build_from(&select.from)?;
    let projection = build_projection(&select.projection)?;

    Ok(Plan { projection, input })
}

fn build_from(from: &[TableWithJoins]) -> Result<RelNode, InterpError> {
    if from.len() != 1 {
        return Err(InterpError::Build(
            "Multiple FROM tables are not yet supported".to_string(),
        ));
    }
    let twj = &from[0];
    if !twj.joins.is_empty() {
        return Err(InterpError::Build("JOIN is not yet supported".to_string()));
    }
    build_table_factor(&twj.relation)
}

fn build_table_factor(factor: &TableFactor) -> Result<RelNode, InterpError> {
    match factor {
        TableFactor::Table { name, .. } => Ok(RelNode::TableScan {
            table: name.to_string(),
        }),
        _ => Err(InterpError::Build("Unsupported FROM clause".to_string())),
    }
}

fn build_projection(items: &[SelectItem]) -> Result<Vec<(String, Expr)>, InterpError> {
    let mut out = Vec::new();
    for item in items {
        match item {
            SelectItem::UnnamedExpr(e) => {
                let name = column_name(e)?;
                out.push((name, crate::expr_build::convert_expr(e)?));
            }
            SelectItem::ExprWithAlias { expr, alias } => {
                out.push((alias.value.clone(), crate::expr_build::convert_expr(expr)?));
            }
            _ => return Err(InterpError::Build("Unsupported SELECT item".to_string())),
        }
    }
    Ok(out)
}

fn column_name(e: &SqlExpr) -> Result<String, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(ident.value.clone()),
        SqlExpr::CompoundIdentifier(parts) => {
            Ok(parts.last().map(|i| i.value.clone()).unwrap_or_default())
        }
        _ => Err(InterpError::Build(
            "Expression in SELECT list needs an alias (AS name)".to_string(),
        )),
    }
}

pub fn execute(
    plan: &Plan,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
) -> Result<Vec<HashMap<String, Value>>, InterpError> {
    let rows = execute_rel(&plan.input, tables)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut result = HashMap::new();
        for (alias, e) in &plan.projection {
            result.insert(alias.clone(), crate::expr::eval(e, row)?);
        }
        out.push(result);
    }
    Ok(out)
}

fn execute_rel(
    node: &RelNode,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
) -> Result<Vec<Row>, InterpError> {
    match node {
        RelNode::TableScan { table } => {
            let flat_rows = tables.get(table).ok_or_else(|| {
                InterpError::Build(format!("Unknown table in FROM clause: {table}"))
            })?;
            Ok(flat_rows
                .iter()
                .map(|r| {
                    let mut row = Row::new();
                    row.insert(table.clone(), r.clone());
                    row
                })
                .collect())
        }
    }
}
```

- [ ] **Step 5: Implement `src/expr_build.rs`**

Create `src/expr_build.rs`:

```rust
use sqlparser::ast::Expr as SqlExpr;

use crate::expr::Expr;
use crate::plan::InterpError;

pub fn convert_expr(e: &SqlExpr) -> Result<Expr, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column {
            table: None,
            name: ident.value.clone(),
        }),
        SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 => Ok(Expr::Column {
            table: Some(parts[0].value.clone()),
            name: parts[1].value.clone(),
        }),
        _ => Err(InterpError::Build(format!(
            "Unsupported expression: {e}"
        ))),
    }
}
```

(This module grows a lot over the next few tasks as more SQL expression forms are supported — literals, arithmetic, functions, CAST — which is why it's kept separate from the pure evaluator in `src/expr.rs`.)

- [ ] **Step 6: Wire `src/lib.rs` to the real plan/execute path**

Replace `src/lib.rs`:

```rust
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod expr;
mod expr_build;
mod plan;

use expr::Value;
use plan::Plan;

#[pyclass]
struct InferFn {
    plan: Plan,
}

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        sql: String,
        row_tables: Vec<String>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = (&row_tables, &static_tables);
        let plan = plan::build_plan(&sql)?;
        Ok(InferFn { plan })
    }

    fn infer(
        &self,
        py: Python<'_>,
        tables: HashMap<String, Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &tables {
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let dict = bound.downcast::<PyDict>()?;
                let mut row: HashMap<String, Value> = HashMap::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    row.insert(key, Value::from_pyobject(&v)?);
                }
                out_rows.push(row);
            }
            value_tables.insert(table.clone(), out_rows);
        }

        let result_rows = plan::execute(&self.plan, &value_tables)?;

        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            out.push(dict.unbind());
        }
        Ok(out)
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
```

- [ ] **Step 7: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS, including the Task 1 smoke test.

- [ ] **Step 8: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/expr.rs src/expr_build.rs src/plan.rs src/lib.rs tests/test_interpreter.py
git commit -m "feat: column pass-through projection via plan interpreter"
```

---

### Task 3: Literals, arithmetic, comparison, logic (three-valued NULL)

Empirically verified against the installed `datafusion` 54.0.0 — this is what the implementation must match, not general "SQL rules" intuition:

```
age=30: age/2 → 15 (int!)         a=7,b=2: a/b → 3 (truncate toward zero)
                                    c=-7,b=2: c/b → -3 (also truncates toward zero)
                                    a%b → 1, c%b → -1 (sign follows dividend)
7/2.5 → 2.8 (mixed int/float promotes to float)
age > 100 → False; NULL = NULL → None; age > NULL → None (three-valued logic)
```

Int/int arithmetic in DataFusion matches Rust's native `i64` `/` and `%` operators exactly (both truncate toward zero) — so `Value::Int` arithmetic can use plain Rust operators with no extra logic.

**Files:**
- Modify: `src/expr.rs` (extend `Expr`, `eval`)
- Modify: `src/expr_build.rs` (extend `convert_expr`)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `Value`, `InterpError` (Task 2)
- Produces: `Expr::Literal(Value)`, `Expr::BinaryOp { op: BinOp, left: Box<Expr>, right: Box<Expr> }`, `Expr::Not(Box<Expr>)`, and the `BinOp` enum — later tasks (`Expr::Function`, `Expr::Cast`) are added alongside these, not replacing them

- [ ] **Step 1: Write failing tests**

Append to `tests/test_interpreter.py`:

```python
def test_literal():
    sql = "SELECT 42 AS x FROM data"
    data = {"age": [1]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 1}]})
    assert actual == _expected(sql, data)


def test_arithmetic():
    sql = "SELECT age / 2 AS half FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)


def test_arithmetic_precedence():
    sql = "SELECT (a + b) * c AS x FROM data"
    data = {"a": [2], "b": [3], "c": [4]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 2, "b": 3, "c": 4}]})
    assert actual == _expected(sql, data)


def test_negative_integer_division_truncates():
    sql = "SELECT c / b AS x, c % b AS y FROM data"
    data = {"c": [-7], "b": [2]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"c": -7, "b": 2}]})
    assert actual == _expected(sql, data)


def test_mixed_int_float_division():
    sql = "SELECT a / f AS x FROM data"
    data = {"a": [7], "f": [2.5]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": 7, "f": 2.5}]})
    assert actual == _expected(sql, data)


def test_null_comparison_is_null():
    sql = "SELECT age > 100 AS gt, age > NULL AS gtnull FROM data"
    data = {"age": [30]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 30}]})
    assert actual == _expected(sql, data)
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "literal or arithmetic or null_comparison"
```

Expected: FAIL — `convert_expr` errors with "Unsupported expression" for literals/binary ops.

- [ ] **Step 3: Extend `src/expr.rs` — `Expr` variants and `eval`**

In `src/expr.rs`, replace the `Expr` enum and everything below it (from `#[derive(Clone)]\npub enum Expr` to the end of the file) with:

```rust
#[derive(Clone)]
pub enum Expr {
    Column { table: Option<String>, name: String },
    Literal(Value),
    BinaryOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
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

fn eval_binary_op(
    op: BinOp,
    l: Value,
    r: Value,
) -> Result<Value, crate::plan::InterpError> {
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
```

Keep everything above `#[derive(Clone)]\npub enum Expr` in the file unchanged (the `Value` type, `PartialEq`/`Eq`/`Hash` impls, `type_name`, `display_value`, `from_pyobject`/`to_pyobject`).

- [ ] **Step 4: Extend `src/expr_build.rs` — literals and binary/unary operators**

Replace `src/expr_build.rs`:

```rust
use sqlparser::ast::{BinaryOperator, Expr as SqlExpr, UnaryOperator, Value as SqlValue};

use crate::expr::{BinOp, Expr, Value};
use crate::plan::InterpError;

pub fn convert_expr(e: &SqlExpr) -> Result<Expr, InterpError> {
    match e {
        SqlExpr::Identifier(ident) => Ok(Expr::Column {
            table: None,
            name: ident.value.clone(),
        }),
        SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 => Ok(Expr::Column {
            table: Some(parts[0].value.clone()),
            name: parts[1].value.clone(),
        }),
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
        _ => Err(InterpError::Build(format!("Unsupported expression: {e}"))),
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
```

Note: `sqlparser::ast::Value` (SQL literal) is imported as `SqlValue` to avoid colliding with `crate::expr::Value` (the interpreter's runtime value) — keep this alias in every file that needs both.

- [ ] **Step 5: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 6: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/expr.rs src/expr_build.rs tests/test_interpreter.py
git commit -m "feat: literals, arithmetic, comparison, and three-valued logic"
```

---

### Task 4: WHERE filter + multi-row inputs

**Files:**
- Modify: `src/plan.rs` (add `RelNode::Filter`, wire `select.selection`)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `Expr`, `eval`, `BinOp` (Task 3)
- Produces: `RelNode::Filter { input: Box<RelNode>, predicate: Expr }` — a row survives iff `eval(predicate, row) == Value::Bool(true)` (both `Value::Null` and `Value::Bool(false)` exclude the row, matching SQL `WHERE` semantics)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_interpreter.py`:

```python
def test_where_filter():
    sql = "SELECT x FROM data WHERE x > 5"
    data = {"x": [3, 7, 10]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"x": 3}, {"x": 7}, {"x": 10}]})
    assert actual == _expected(sql, data)


def test_multi_row():
    sql = "SELECT age FROM data"
    data = {"age": [1, 2, 3]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 1}, {"age": 2}, {"age": 3}]})
    assert actual == _expected(sql, data)
```

- [ ] **Step 2: Run tests — verify `test_where_filter` fails**

```
uv run pytest tests/test_interpreter.py -v -k "where_filter or multi_row"
```

Expected: `test_multi_row` already PASSES (TableScan iterates all rows since Task 2). `test_where_filter` FAILS — `build_from` doesn't look at `select.selection`.

- [ ] **Step 3: Add `RelNode::Filter` to `src/plan.rs`**

In `src/plan.rs`, add `Filter` to the `RelNode` enum:

```rust
pub enum RelNode {
    TableScan { table: String },
    Filter { input: Box<RelNode>, predicate: Expr },
}
```

Replace `build_plan` to apply the WHERE clause after building the FROM:

```rust
pub fn build_plan(sql: &str) -> Result<Plan, InterpError> {
    let dialect = GenericDialect {};
    let statements = Parser::parse_sql(&dialect, sql)
        .map_err(|e| InterpError::Build(format!("SQL parse error: {e}")))?;

    if statements.len() != 1 {
        return Err(InterpError::Build(
            "Expected exactly one SQL statement".to_string(),
        ));
    }

    let select = match &statements[0] {
        Statement::Query(query) => match query.body.as_ref() {
            SetExpr::Select(select) => select.as_ref(),
            _ => return Err(InterpError::Build("Only SELECT queries are supported".to_string())),
        },
        _ => return Err(InterpError::Build("Only SELECT queries are supported".to_string())),
    };

    let mut input = build_from(&select.from)?;
    if let Some(predicate) = &select.selection {
        input = RelNode::Filter {
            input: Box::new(input),
            predicate: crate::expr_build::convert_expr(predicate)?,
        };
    }
    let projection = build_projection(&select.projection)?;

    Ok(Plan { projection, input })
}
```

Add the `Filter` arm to `execute_rel`:

```rust
fn execute_rel(
    node: &RelNode,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
) -> Result<Vec<Row>, InterpError> {
    match node {
        RelNode::TableScan { table } => {
            let flat_rows = tables.get(table).ok_or_else(|| {
                InterpError::Build(format!("Unknown table in FROM clause: {table}"))
            })?;
            Ok(flat_rows
                .iter()
                .map(|r| {
                    let mut row = Row::new();
                    row.insert(table.clone(), r.clone());
                    row
                })
                .collect())
        }
        RelNode::Filter { input, predicate } => {
            let rows = execute_rel(input, tables)?;
            let mut out = Vec::new();
            for row in rows {
                if let Value::Bool(true) = crate::expr::eval(predicate, &row)? {
                    out.push(row);
                }
            }
            Ok(out)
        }
    }
}
```

- [ ] **Step 4: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 5: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/plan.rs tests/test_interpreter.py
git commit -m "feat: WHERE filter with three-valued NULL semantics"
```

---

### Task 5: Built-in functions + CAST

Empirically verified against `datafusion` 54.0.0:

```
UPPER('hi')→'HI'  CONCAT('a','-','b')→'a-b'  SUBSTR('hello',2,3)→'ell'  TRIM('  x  ')→'x'
ABS(-5)→5  ROUND(3.6)→4.0  COALESCE(NULL,NULL,5)→5  NULLIF(3,3)→None
CAST(3.7 AS BIGINT)→3 (truncates toward zero)  CAST('5' AS BIGINT)→5  CAST(3 AS DOUBLE)→3.0
```

**Files:**
- Modify: `src/expr.rs` (add `Expr::Function`, `Expr::Cast`, `CastType`, `eval` dispatch)
- Modify: `src/expr_build.rs` (convert `SqlExpr::Function` / `SqlExpr::Cast`)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `Expr`, `eval`, `type_name`, `display_value` (Tasks 2-3)
- Produces: `Expr::Function { name: String, args: Vec<Expr> }` (lowercased function name — `upper`, `lower`, `concat`, `substr`/`substring`, `trim`, `abs`, `round`, `coalesce`, `nullif`), `Expr::Cast { expr: Box<Expr>, target: CastType }`, `CastType { Str, Int, Float, Bool }`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_interpreter.py`:

```python
def test_builtin_upper():
    sql = "SELECT UPPER(name) AS up FROM data"
    data = {"name": ["hello"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"name": "hello"}]})
    assert actual == _expected(sql, data)


def test_builtin_concat():
    sql = "SELECT CONCAT(a, '-', b) AS combo FROM data"
    data = {"a": ["x"], "b": ["y"]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": "x", "b": "y"}]})
    assert actual == _expected(sql, data)


def test_builtin_substr_trim_abs_round():
    sql = (
        "SELECT SUBSTR(s, 2, 3) AS sub, TRIM(pad) AS t, "
        "ABS(neg) AS ab, ROUND(pi) AS ro FROM data"
    )
    data = {"s": ["hello"], "pad": ["  x  "], "neg": [-5], "pi": [3.6]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"s": "hello", "pad": "  x  ", "neg": -5, "pi": 3.6}]})
    assert actual == _expected(sql, data)


def test_builtin_coalesce_nullif():
    sql = "SELECT COALESCE(a, b, 5) AS co, NULLIF(x, x) AS ni FROM data"
    data = {"a": [None], "b": [None], "x": [3]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": None, "b": None, "x": 3}]})
    assert actual == _expected(sql, data)


def test_cast():
    sql = "SELECT CAST(age AS VARCHAR) AS s FROM data"
    data = {"age": [42]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"age": 42}]})
    assert actual == _expected(sql, data)
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "builtin or cast"
```

Expected: FAIL — `convert_expr` doesn't handle `SqlExpr::Function` or `SqlExpr::Cast`.

- [ ] **Step 3: Extend `src/expr.rs` — add `Function`/`Cast` variants and evaluators**

In `src/expr.rs`, extend the `Expr` enum (add to the existing variants, keep `Column`/`Literal`/`BinaryOp`/`Not`):

```rust
#[derive(Clone)]
pub enum Expr {
    Column { table: Option<String>, name: String },
    Literal(Value),
    BinaryOp { op: BinOp, left: Box<Expr>, right: Box<Expr> },
    Not(Box<Expr>),
    Function { name: String, args: Vec<Expr> },
    Cast { expr: Box<Expr>, target: CastType },
}

#[derive(Clone, Copy)]
pub enum CastType {
    Str,
    Int,
    Float,
    Bool,
}
```

Add two arms to `eval`'s `match expr` (alongside the existing `Column`/`Literal`/`BinaryOp`/`Not` arms):

```rust
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
```

Add the function/cast evaluators (anywhere after `eval`, alongside `arithmetic`/`comparison`/`logic`):

```rust
fn eval_builtin(name: &str, args: Vec<Value>) -> Result<Value, crate::plan::InterpError> {
    use crate::plan::InterpError;

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
            Value::Null | Value::Object(_) => {
                return Err(InterpError::Eval("Cannot cast this value to INT".to_string()))
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
            Value::Null | Value::Object(_) => {
                return Err(InterpError::Eval("Cannot cast this value to FLOAT".to_string()))
            }
        },
        CastType::Bool => match v {
            Value::Bool(b) => Value::Bool(b),
            Value::Int(i) => Value::Bool(i != 0),
            Value::Float(f) => Value::Bool(f != 0.0),
            Value::Str(s) => Value::Bool(s.eq_ignore_ascii_case("true")),
            Value::Null | Value::Object(_) => {
                return Err(InterpError::Eval("Cannot cast this value to BOOLEAN".to_string()))
            }
        },
    })
}
```

- [ ] **Step 4: Extend `src/expr_build.rs` — convert `Function`/`Cast`**

In `src/expr_build.rs`, replace the imports at the top with:

```rust
use sqlparser::ast::{
    BinaryOperator, DataType, Expr as SqlExpr, Function, FunctionArg, FunctionArgExpr,
    FunctionArguments, UnaryOperator, Value as SqlValue,
};

use crate::expr::{BinOp, CastType, Expr, Value};
use crate::plan::InterpError;
```

Add two arms to `convert_expr`'s `match e` (alongside the existing arms, before the final `_ =>`):

```rust
        SqlExpr::Function(func) => convert_function(func),
        SqlExpr::Cast {
            expr, data_type, ..
        } => Ok(Expr::Cast {
            expr: Box::new(convert_expr(expr)?),
            target: convert_cast_type(data_type)?,
        }),
```

Add the conversion helpers:

```rust
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
    if name.starts_with("VARCHAR") || name.starts_with("TEXT") || name.starts_with("STRING")
        || name.starts_with("CHAR")
    {
        Ok(CastType::Str)
    } else if name.starts_with("BIGINT") || name.starts_with("INT") {
        Ok(CastType::Int)
    } else if name.starts_with("DOUBLE") || name.starts_with("FLOAT") || name.starts_with("REAL")
    {
        Ok(CastType::Float)
    } else if name.starts_with("BOOL") {
        Ok(CastType::Bool)
    } else {
        Err(InterpError::Build(format!(
            "Unsupported CAST target type: {name}"
        )))
    }
}
```

- [ ] **Step 5: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 6: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/expr.rs src/expr_build.rs tests/test_interpreter.py
git commit -m "feat: built-in functions and CAST"
```

---

### Task 6: CrossJoin + inner Join (row × row) + build-time JOIN validation

**Files:**
- Modify: `src/plan.rs` (`RelNode::CrossJoin`, `RelNode::Join`, `RelNode::SubqueryAlias`, rewrite `build_from`)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `Expr`, `eval`, `BinOp` (Tasks 3, 5), `expr_build::convert_expr`
- Produces:
  - `RelNode::CrossJoin { left: Box<RelNode>, right: Box<RelNode> }` — nested-loop cartesian product, merging each pair of nested `Row`s (outer keys are table names, so two tables never collide)
  - `RelNode::Join { left: Box<RelNode>, right: Box<RelNode>, on: Vec<(Expr, Expr)> }` — cartesian product filtered to rows where every `(left_expr, right_expr)` pair is equal (NULL never matches)
  - `RelNode::SubqueryAlias { input: Box<RelNode>, alias: String }` — renames a single-relation input's one outer key to `alias`
  - Build-time `ValueError` for: `LEFT`/`RIGHT`/`FULL OUTER` JOIN, non-equality ON condition, self-join (same table name referenced twice in FROM)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_interpreter.py`:

```python
import pytest


def test_cross_join():
    sql = "SELECT a.x, b.y FROM a, b"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"x": [1]}, name="a")
    ctx.from_pydict({"y": [2]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer({"a": [{"x": 1}], "b": [{"y": 2}]})
    assert actual == expected


def test_inner_join_two_row_tables():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [10, 20]}, name="a")
    ctx.from_pydict({"id": [1, 2], "y": [100, 200]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer(
        {
            "a": [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            "b": [{"id": 1, "y": 100}, {"id": 2, "y": 200}],
        }
    )
    assert actual == expected


def test_inner_join_multi_key():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"k1": [1], "k2": ["p"], "x": [10]}, name="a")
    ctx.from_pydict({"k1": [1], "k2": ["p"], "y": [100]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["a", "b"], static_tables={})
    actual = fn.infer(
        {
            "a": [{"k1": 1, "k2": "p", "x": 10}],
            "b": [{"k1": 1, "k2": "p", "y": 100}],
        }
    )
    assert actual == expected


def test_error_left_join():
    sql = "SELECT a.x FROM a LEFT JOIN b ON a.id = b.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a", "b"], static_tables={})


def test_error_non_equality_on():
    sql = "SELECT a.x FROM a JOIN b ON a.id > b.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a", "b"], static_tables={})


def test_error_self_join():
    sql = "SELECT a.x FROM a JOIN a ON a.id = a.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=["a"], static_tables={})
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "join"
```

Expected: FAIL — `build_from` currently rejects any FROM with more than one table or with joins.

- [ ] **Step 3: Rewrite `build_from` and add JOIN plan nodes in `src/plan.rs`**

Replace the `use` block at the top of `src/plan.rs` with:

```rust
use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyErr;
use sqlparser::ast::{
    BinaryOperator, Expr as SqlExpr, Join, JoinConstraint, JoinOperator, SelectItem, SetExpr,
    Statement, TableFactor, TableWithJoins,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::expr::{Expr, Value};
```

Add variants to `RelNode`:

```rust
pub enum RelNode {
    TableScan { table: String },
    Filter { input: Box<RelNode>, predicate: Expr },
    CrossJoin { left: Box<RelNode>, right: Box<RelNode> },
    Join { left: Box<RelNode>, right: Box<RelNode>, on: Vec<(Expr, Expr)> },
    SubqueryAlias { input: Box<RelNode>, alias: String },
}
```

Replace `build_from` and `build_table_factor` with:

```rust
fn build_from(from: &[TableWithJoins]) -> Result<RelNode, InterpError> {
    if from.is_empty() {
        return Err(InterpError::Build("FROM clause is required".to_string()));
    }
    let mut seen_tables: HashSet<String> = HashSet::new();
    let mut node = build_table_with_joins(&from[0], &mut seen_tables)?;
    for twj in &from[1..] {
        let right = build_table_with_joins(twj, &mut seen_tables)?;
        node = RelNode::CrossJoin {
            left: Box::new(node),
            right: Box::new(right),
        };
    }
    Ok(node)
}

fn build_table_with_joins(
    twj: &TableWithJoins,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    let mut node = build_table_factor(&twj.relation, seen_tables)?;
    for join in &twj.joins {
        node = build_join(node, join, seen_tables)?;
    }
    Ok(node)
}

fn build_join(
    left: RelNode,
    join: &Join,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    let right = build_table_factor(&join.relation, seen_tables)?;
    match &join.join_operator {
        JoinOperator::Join(constraint) | JoinOperator::Inner(constraint) => {
            let on_expr = require_on(constraint)?;
            let on = extract_equality_keys(on_expr)?;
            Ok(RelNode::Join {
                left: Box::new(left),
                right: Box::new(right),
                on,
            })
        }
        JoinOperator::CrossJoin(_) => Ok(RelNode::CrossJoin {
            left: Box::new(left),
            right: Box::new(right),
        }),
        other => Err(InterpError::Build(format!(
            "Unsupported JOIN type: {other:?} — only inner JOIN ... ON and CROSS JOIN are supported"
        ))),
    }
}

fn build_table_factor(
    factor: &TableFactor,
    seen_tables: &mut HashSet<String>,
) -> Result<RelNode, InterpError> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            let table = name.to_string();
            if !seen_tables.insert(table.clone()) {
                return Err(InterpError::Build(format!(
                    "Self-joins are not supported: table '{table}' is referenced more than once"
                )));
            }
            let scan = RelNode::TableScan { table };
            Ok(match alias {
                Some(a) => RelNode::SubqueryAlias {
                    input: Box::new(scan),
                    alias: a.name.value.clone(),
                },
                None => scan,
            })
        }
        _ => Err(InterpError::Build("Unsupported FROM clause".to_string())),
    }
}

fn require_on(constraint: &JoinConstraint) -> Result<&SqlExpr, InterpError> {
    match constraint {
        JoinConstraint::On(e) => Ok(e),
        _ => Err(InterpError::Build(
            "JOIN requires an ON condition".to_string(),
        )),
    }
}

fn extract_equality_keys(expr: &SqlExpr) -> Result<Vec<(Expr, Expr)>, InterpError> {
    match expr {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            let mut pairs = extract_equality_keys(left)?;
            pairs.extend(extract_equality_keys(right)?);
            Ok(pairs)
        }
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => Ok(vec![(
            crate::expr_build::convert_expr(left)?,
            crate::expr_build::convert_expr(right)?,
        )]),
        _ => Err(InterpError::Build(
            "JOIN ON condition must be an equality, or an AND of equalities, between columns"
                .to_string(),
        )),
    }
}
```

Add `CrossJoin`/`Join`/`SubqueryAlias` arms to `execute_rel`:

```rust
        RelNode::CrossJoin { left, right } => {
            let left_rows = execute_rel(left, tables)?;
            let right_rows = execute_rel(right, tables)?;
            let mut out = Vec::with_capacity(left_rows.len() * right_rows.len());
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    out.push(merged);
                }
            }
            Ok(out)
        }
        RelNode::Join { left, right, on } => {
            let left_rows = execute_rel(left, tables)?;
            let right_rows = execute_rel(right, tables)?;
            let mut out = Vec::new();
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    let mut all_match = true;
                    for (le, re) in on {
                        let lv = crate::expr::eval(le, &merged)?;
                        let rv = crate::expr::eval(re, &merged)?;
                        if matches!(lv, Value::Null) || matches!(rv, Value::Null) || lv != rv {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        out.push(merged);
                    }
                }
            }
            Ok(out)
        }
        RelNode::SubqueryAlias { input, alias } => {
            let rows = execute_rel(input, tables)?;
            Ok(rows
                .into_iter()
                .map(|row| {
                    let inner = row.into_values().next().unwrap_or_default();
                    let mut renamed = Row::new();
                    renamed.insert(alias.clone(), inner);
                    renamed
                })
                .collect())
        }
```

(Add these three arms inside the existing `match node` in `execute_rel`, alongside `TableScan` and `Filter`.)

- [ ] **Step 4: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 5: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/plan.rs tests/test_interpreter.py
git commit -m "feat: CROSS JOIN and inner row-to-row JOIN with build-time validation"
```

---

### Task 7: Static table lookup index + LookupJoin optimization

**Files:**
- Create: `src/lookup.rs`
- Modify: `src/plan.rs` (add `RelNode::LookupJoin`, `optimize()` pass, thread `lookups` through `execute`)
- Modify: `src/lib.rs` (build lookup indices from `static_tables` in `InferFn::new`, call `optimize`)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `Row`, `Value`, `Expr`, `eval`, `RelNode::Join`/`TableScan`/`SubqueryAlias` (Tasks 2, 6)
- Produces:
  - `lookup::LookupIndex { index: HashMap<Vec<Value>, HashMap<String, Value>> }` — key tuple → the static row's non-key columns
  - `lookup::build_index(py: Python, table: &Py<PyAny>, key_columns: &[String]) -> Result<LookupIndex, InterpError>` — calls `.to_pylist()` on the `pyarrow.Table` via PyO3 (no `arrow` crate dependency needed)
  - `plan::LookupSpec { static_table: String, key_columns: Vec<String> }` and `plan::optimize(plan: Plan, static_tables: &HashSet<String>) -> Result<(Plan, Vec<LookupSpec>), InterpError>` — rewrites `Join` nodes where exactly one side is a static-table scan into `RelNode::LookupJoin`; returns an error if both sides are static
  - `RelNode::LookupJoin { input: Box<RelNode>, table: String, keys: Vec<Expr> }`
  - `plan::execute` and `execute_rel` now take an extra `lookups: &HashMap<String, LookupIndex>` parameter
  - Build-time `ValueError` for joining two static tables together; runtime `KeyError` (message includes the key value and table name) for a missing lookup key

- [ ] **Step 1: Write failing tests**

Append to `tests/test_interpreter.py`:

```python
import pyarrow as pa


def test_join_row_and_static_table():
    ref_table = pa.table({"id": [1, 2], "y": [10, 20]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [5, 6]}, name="data")
    ctx.from_arrow(ref_table, name="ref")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables=["data"], static_tables={"ref": ref_table})
    actual = fn.infer({"data": [{"id": 1, "x": 5}, {"id": 2, "x": 6}]})
    assert actual == expected


def test_join_row_and_static_table_single_row():
    ref_table = pa.table({"id": [1, 2], "y": [10, 20]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    fn = InferFn(sql, row_tables=["data"], static_tables={"ref": ref_table})
    result = fn.infer({"data": [{"id": 1, "x": 5}]})
    assert result == [{"x": 5, "y": 10}]


def test_missing_lookup_key_raises_key_error():
    ref_table = pa.table({"id": [1], "y": [10]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    fn = InferFn(sql, row_tables=["data"], static_tables={"ref": ref_table})
    with pytest.raises(KeyError) as exc_info:
        fn.infer({"data": [{"id": 999, "x": 5}]})
    message = str(exc_info.value)
    assert "999" in message
    assert "ref" in message


def test_error_static_static_join():
    ref1 = pa.table({"id": [1], "y": [10]})
    ref2 = pa.table({"id": [1], "z": [20]})
    sql = "SELECT ref1.y, ref2.z FROM ref1 JOIN ref2 ON ref1.id = ref2.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables=[], static_tables={"ref1": ref1, "ref2": ref2})
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k "static_table or key_error"
```

Expected: FAIL — `data`/`ref` currently both build as row `TableScan`s and `infer()` never receives `ref`'s rows (`static_tables` is still ignored), so `Unknown table` or missing-data errors occur.

- [ ] **Step 3: Implement `src/lookup.rs`**

Create `src/lookup.rs`:

```rust
use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::expr::Value;
use crate::plan::InterpError;

pub struct LookupIndex {
    pub index: HashMap<Vec<Value>, HashMap<String, Value>>,
}

pub fn build_index(
    py: Python<'_>,
    table: &Py<PyAny>,
    key_columns: &[String],
) -> Result<LookupIndex, InterpError> {
    let bound = table.bind(py);
    let rows_obj = bound
        .call_method0("to_pylist")
        .map_err(|e| InterpError::Build(format!("Failed to read static table: {e}")))?;
    let rows: Vec<Py<PyAny>> = rows_obj.extract().map_err(|e| {
        InterpError::Build(format!("Static table must convert to a list of rows: {e}"))
    })?;

    let mut index = HashMap::new();
    for row_obj in rows {
        let row_bound = row_obj.bind(py);
        let dict = row_bound
            .downcast::<PyDict>()
            .map_err(|e| InterpError::Build(format!("Static table row must be a dict: {e}")))?;

        let mut rest: HashMap<String, Value> = HashMap::new();
        for (k, v) in dict.iter() {
            let col: String = k
                .extract()
                .map_err(|e| InterpError::Build(format!("Static table column name error: {e}")))?;
            if key_columns.contains(&col) {
                continue;
            }
            let value = Value::from_pyobject(&v)
                .map_err(|e| InterpError::Build(format!("Static table value error: {e}")))?;
            rest.insert(col, value);
        }

        let mut key = Vec::with_capacity(key_columns.len());
        for col in key_columns {
            let v = dict
                .get_item(col)
                .map_err(|e| {
                    InterpError::Build(format!("Static table missing key column '{col}': {e}"))
                })?
                .ok_or_else(|| {
                    InterpError::Build(format!("Static table missing key column '{col}'"))
                })?;
            key.push(
                Value::from_pyobject(&v)
                    .map_err(|e| InterpError::Build(format!("Static table key value error: {e}")))?,
            );
        }

        index.insert(key, rest);
    }

    Ok(LookupIndex { index })
}
```

- [ ] **Step 4: Add `LookupJoin` + `optimize()` to `src/plan.rs`**

Add to `RelNode`:

```rust
    LookupJoin { input: Box<RelNode>, table: String, keys: Vec<Expr> },
```

Add near the top of `src/plan.rs` (with the other type defs):

```rust
pub struct LookupSpec {
    pub static_table: String,
    pub key_columns: Vec<String>,
}
```

Add the optimizer pass (new functions in `src/plan.rs`):

```rust
pub fn optimize(
    plan: Plan,
    static_tables: &HashSet<String>,
) -> Result<(Plan, Vec<LookupSpec>), InterpError> {
    let mut specs = Vec::new();
    let input = optimize_rel(plan.input, static_tables, &mut specs)?;
    Ok((
        Plan {
            projection: plan.projection,
            input,
        },
        specs,
    ))
}

fn optimize_rel(
    node: RelNode,
    static_tables: &HashSet<String>,
    specs: &mut Vec<LookupSpec>,
) -> Result<RelNode, InterpError> {
    match node {
        RelNode::Join { left, right, on } => {
            let left = optimize_rel(*left, static_tables, specs)?;
            let right = optimize_rel(*right, static_tables, specs)?;
            let left_static = scan_table_name(&left).filter(|t| static_tables.contains(*t));
            let right_static = scan_table_name(&right).filter(|t| static_tables.contains(*t));
            match (left_static, right_static) {
                (Some(_), Some(_)) => Err(InterpError::Build(
                    "Joining two static tables together is not supported".to_string(),
                )),
                (None, Some(table)) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table, false)?;
                    specs.push(LookupSpec {
                        static_table: table.clone(),
                        key_columns,
                    });
                    Ok(RelNode::LookupJoin {
                        input: Box::new(left),
                        table,
                        keys,
                    })
                }
                (Some(table), None) => {
                    let table = table.to_string();
                    let (keys, key_columns) = split_keys(&on, &table, true)?;
                    specs.push(LookupSpec {
                        static_table: table.clone(),
                        key_columns,
                    });
                    Ok(RelNode::LookupJoin {
                        input: Box::new(right),
                        table,
                        keys,
                    })
                }
                (None, None) => Ok(RelNode::Join {
                    left: Box::new(left),
                    right: Box::new(right),
                    on,
                }),
            }
        }
        RelNode::CrossJoin { left, right } => Ok(RelNode::CrossJoin {
            left: Box::new(optimize_rel(*left, static_tables, specs)?),
            right: Box::new(optimize_rel(*right, static_tables, specs)?),
        }),
        RelNode::Filter { input, predicate } => Ok(RelNode::Filter {
            input: Box::new(optimize_rel(*input, static_tables, specs)?),
            predicate,
        }),
        RelNode::SubqueryAlias { input, alias } => Ok(RelNode::SubqueryAlias {
            input: Box::new(optimize_rel(*input, static_tables, specs)?),
            alias,
        }),
        other => Ok(other),
    }
}

fn scan_table_name(node: &RelNode) -> Option<&str> {
    match node {
        RelNode::TableScan { table } => Some(table),
        RelNode::SubqueryAlias { input, .. } => scan_table_name(input),
        _ => None,
    }
}

fn split_keys(
    on: &[(Expr, Expr)],
    static_table: &str,
    static_is_left: bool,
) -> Result<(Vec<Expr>, Vec<String>), InterpError> {
    let mut row_side_keys = Vec::new();
    let mut static_col_names = Vec::new();
    for (l, r) in on {
        let (static_expr, row_expr) = if static_is_left { (l, r) } else { (r, l) };
        match static_expr {
            Expr::Column { name, .. } => static_col_names.push(name.clone()),
            _ => {
                return Err(InterpError::Build(format!(
                    "JOIN ON keys against static table '{static_table}' must be plain columns"
                )))
            }
        }
        row_side_keys.push(row_expr.clone());
    }
    Ok((row_side_keys, static_col_names))
}
```

Thread `lookups` through `execute`/`execute_rel` — replace both functions:

```rust
pub fn execute(
    plan: &Plan,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
    lookups: &HashMap<String, crate::lookup::LookupIndex>,
) -> Result<Vec<HashMap<String, Value>>, InterpError> {
    let rows = execute_rel(&plan.input, tables, lookups)?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let mut result = HashMap::new();
        for (alias, e) in &plan.projection {
            result.insert(alias.clone(), crate::expr::eval(e, row)?);
        }
        out.push(result);
    }
    Ok(out)
}

fn execute_rel(
    node: &RelNode,
    tables: &HashMap<String, Vec<HashMap<String, Value>>>,
    lookups: &HashMap<String, crate::lookup::LookupIndex>,
) -> Result<Vec<Row>, InterpError> {
    match node {
        RelNode::TableScan { table } => {
            let flat_rows = tables.get(table).ok_or_else(|| {
                InterpError::Build(format!("Unknown table in FROM clause: {table}"))
            })?;
            Ok(flat_rows
                .iter()
                .map(|r| {
                    let mut row = Row::new();
                    row.insert(table.clone(), r.clone());
                    row
                })
                .collect())
        }
        RelNode::Filter { input, predicate } => {
            let rows = execute_rel(input, tables, lookups)?;
            let mut out = Vec::new();
            for row in rows {
                if let Value::Bool(true) = crate::expr::eval(predicate, &row)? {
                    out.push(row);
                }
            }
            Ok(out)
        }
        RelNode::CrossJoin { left, right } => {
            let left_rows = execute_rel(left, tables, lookups)?;
            let right_rows = execute_rel(right, tables, lookups)?;
            let mut out = Vec::with_capacity(left_rows.len() * right_rows.len());
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    out.push(merged);
                }
            }
            Ok(out)
        }
        RelNode::Join { left, right, on } => {
            let left_rows = execute_rel(left, tables, lookups)?;
            let right_rows = execute_rel(right, tables, lookups)?;
            let mut out = Vec::new();
            for l in &left_rows {
                for r in &right_rows {
                    let mut merged = l.clone();
                    merged.extend(r.clone());
                    let mut all_match = true;
                    for (le, re) in on {
                        let lv = crate::expr::eval(le, &merged)?;
                        let rv = crate::expr::eval(re, &merged)?;
                        if matches!(lv, Value::Null) || matches!(rv, Value::Null) || lv != rv {
                            all_match = false;
                            break;
                        }
                    }
                    if all_match {
                        out.push(merged);
                    }
                }
            }
            Ok(out)
        }
        RelNode::SubqueryAlias { input, alias } => {
            let rows = execute_rel(input, tables, lookups)?;
            Ok(rows
                .into_iter()
                .map(|row| {
                    let inner = row.into_values().next().unwrap_or_default();
                    let mut renamed = Row::new();
                    renamed.insert(alias.clone(), inner);
                    renamed
                })
                .collect())
        }
        RelNode::LookupJoin { input, table, keys } => {
            let rows = execute_rel(input, tables, lookups)?;
            let index = lookups.get(table).ok_or_else(|| {
                InterpError::Build(format!("No lookup index built for table: {table}"))
            })?;
            let mut out = Vec::with_capacity(rows.len());
            for mut row in rows {
                let key: Vec<Value> = keys
                    .iter()
                    .map(|k| crate::expr::eval(k, &row))
                    .collect::<Result<_, _>>()?;
                let hit = index.index.get(&key).ok_or_else(|| {
                    let key_repr: Vec<String> =
                        key.iter().map(crate::expr::display_value).collect();
                    InterpError::MissingKey(format!(
                        "No row in static table '{table}' matches key ({})",
                        key_repr.join(", ")
                    ))
                })?;
                row.insert(table.clone(), hit.clone());
                out.push(row);
            }
            Ok(out)
        }
    }
}
```

- [ ] **Step 5: Wire `src/lib.rs` — build lookup indices, call `optimize`**

Replace `src/lib.rs`:

```rust
use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod expr;
mod expr_build;
mod lookup;
mod plan;

use expr::Value;
use lookup::LookupIndex;
use plan::Plan;

#[pyclass]
struct InferFn {
    plan: Plan,
    lookups: HashMap<String, LookupIndex>,
}

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: Vec<String>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let _ = &row_tables;

        let raw_plan = plan::build_plan(&sql)?;
        let static_table_names: HashSet<String> = static_tables.keys().cloned().collect();
        let (plan, specs) = plan::optimize(raw_plan, &static_table_names)?;

        let mut lookups = HashMap::new();
        for spec in specs {
            let table_obj = static_tables.get(&spec.static_table).ok_or_else(|| {
                plan::InterpError::Build(format!(
                    "SQL references static table '{}' that was not provided",
                    spec.static_table
                ))
            })?;
            let index = lookup::build_index(py, table_obj, &spec.key_columns)?;
            lookups.insert(spec.static_table, index);
        }

        Ok(InferFn { plan, lookups })
    }

    fn infer(
        &self,
        py: Python<'_>,
        tables: HashMap<String, Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &tables {
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let dict = bound.downcast::<PyDict>()?;
                let mut row: HashMap<String, Value> = HashMap::new();
                for (k, v) in dict.iter() {
                    let key: String = k.extract()?;
                    row.insert(key, Value::from_pyobject(&v)?);
                }
                out_rows.push(row);
            }
            value_tables.insert(table.clone(), out_rows);
        }

        let result_rows = plan::execute(&self.plan, &value_tables, &self.lookups)?;

        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            out.push(dict.unbind());
        }
        Ok(out)
    }
}

#[pymodule]
fn _interpreter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<InferFn>()?;
    Ok(())
}
```

Note the `#[new]` constructor now takes `py: Python<'_>` as its first parameter — PyO3 injects this automatically; it does not become part of the Python-visible signature (`InferFn(sql, row_tables, static_tables)` is unchanged for callers).

- [ ] **Step 6: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 7: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add src/lookup.rs src/plan.rs src/lib.rs tests/test_interpreter.py
git commit -m "feat: static table lookup index and row-to-static LookupJoin"
```

---

### Task 8: Edge cases — empty input, reusable fn, nested-object passthrough, NULL functions

**Files:**
- Modify: `tests/test_interpreter.py` — every code path exercised here already exists from Tasks 2-7; this task closes the one real gap (`Value::Object` round-tripping was defined in Task 2 but never exercised by a test) and locks down the rest of the spec's Edge Cases section with explicit regression tests

**Interfaces:**
- Consumes: everything from Tasks 1-7
- Produces: no new Rust code paths

- [ ] **Step 1: Write the edge-case tests**

Append to `tests/test_interpreter.py`:

```python
def test_empty_row_list_returns_empty():
    fn = InferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    assert fn.infer({"data": []}) == []


def test_reusable_fn_different_inputs():
    fn = InferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    first = fn.infer({"data": [{"age": 1}]})
    second = fn.infer({"data": [{"age": 2}]})
    assert first == [{"age": 1}]
    assert second == [{"age": 2}]


def test_nested_object_passthrough():
    sql = "SELECT payload FROM data"
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    payload = {"nested": [1, 2, 3]}
    result = fn.infer({"data": [{"payload": payload}]})
    assert result == [{"payload": payload}]


def test_coalesce_all_null():
    sql = "SELECT COALESCE(a, b) AS co FROM data"
    data = {"a": [None], "b": [None]}
    fn = InferFn(sql, row_tables=["data"], static_tables={})
    actual = fn.infer({"data": [{"a": None, "b": None}]})
    assert actual == _expected(sql, data)
```

- [ ] **Step 2: Run the full test suite**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS. If `test_nested_object_passthrough` fails, check `Value::from_pyobject` in `src/expr.rs` — it must fall through to `Value::Object(obj.clone().unbind())` for a Python `dict` (it doesn't match `PyBool`/`PyInt`/`PyFloat`/`PyString`, so it should already work per Task 2's implementation; if it errors instead, the `downcast` chain has an ordering bug to fix).

- [ ] **Step 3: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add tests/test_interpreter.py
git commit -m "test: verify edge cases — empty input, reuse, nested objects, NULL functions"
```

---

### Task 9: Python packaging — re-export, type stubs, final verification

**Files:**
- Modify: `sql_transform/__init__.py` (re-export `InferFn`)
- Create: `sql_transform/_interpreter.pyi`
- Modify: `tests/test_interpreter.py` (import from the public path)

**Interfaces:**
- Consumes: `sql_transform._interpreter.InferFn` (Rust extension, Tasks 1-7)
- Produces: `from sql_transform import InferFn` — the public import path from the spec's Public API section

- [ ] **Step 1: Write a failing test for the public import path**

Append to `tests/test_interpreter.py`:

```python
def test_public_import_path():
    from sql_transform import InferFn as PublicInferFn

    fn = PublicInferFn("SELECT age FROM data", row_tables=["data"], static_tables={})
    assert fn.infer({"data": [{"age": 5}]}) == [{"age": 5}]
```

- [ ] **Step 2: Run — verify it fails**

```
uv run pytest tests/test_interpreter.py -v -k public_import
```

Expected: FAIL — `sql_transform/__init__.py` doesn't export `InferFn` yet (it still exports the Phase-1 `SQLTransform`; leave those imports untouched, only add the new export).

- [ ] **Step 3: Add the re-export**

Read `sql_transform/__init__.py`. Add near the top, alongside the existing imports:

```python
from sql_transform._interpreter import InferFn

__all__ = ["InferFn", "SQLTransform"]
```

(If `__all__` doesn't already exist in the file, add it as shown; if it exists, add `"InferFn"` to it.)

- [ ] **Step 4: Add type stubs**

Create `sql_transform/_interpreter.pyi`:

```python
from typing import Any

class InferFn:
    def __init__(
        self,
        sql: str,
        row_tables: list[str],
        static_tables: dict[str, Any],
    ) -> None: ...
    def infer(self, tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]: ...
```

- [ ] **Step 5: Full verification pass**

```
uv run maturin develop
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
cargo fmt --check
cargo build
```

Expected: every test across the whole repo passes (both the Phase 1 `sql_transform/*_test.py` files and `tests/test_interpreter.py`), ruff reports no issues, `cargo fmt --check` and `cargo build` are clean.

- [ ] **Step 6: Final commit**

```
git add sql_transform/__init__.py sql_transform/_interpreter.pyi tests/test_interpreter.py
git commit -m "feat: export InferFn from sql_transform, add type stubs"
```
