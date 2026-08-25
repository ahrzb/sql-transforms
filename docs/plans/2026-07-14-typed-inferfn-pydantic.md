# Typed InferFn (Pydantic v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `InferFn`'s dict-based row API with a Pydantic-v2-typed one — row tables are declared as model classes at construction time (real schema before any row exists), unknown columns and provably-wrong output types are rejected at construction (`ValueError`), and the output is either a caller-supplied model or one `InferFn` synthesizes from the query itself.

**Architecture:** Two new pure-logic modules sit alongside the existing dict-based interpreter (`src/expr.rs`, `src/plan.rs`, `src/expr_build.rs`, `src/lookup.rs`, `src/lib.rs`), whose internals (`Value`, `Row`, `Expr`, `eval()`, `execute()`) are **untouched**: `src/types.rs` (a `FieldType` lattice + a static type-inference pass mirroring `eval()`'s structure) and `src/schema.rs` (extracts that same `FieldType` shape from a Pydantic model's `model_fields` or a `pyarrow.Table`'s `.schema`). `src/plan.rs` gains one build-time tree walk that validates every column reference against the right schema and, as a side effect, collects which columns each row table's query actually uses. `src/lib.rs`'s `InferFn` uses that to pull only the referenced attributes off each row via `getattr` (not a full `model_dump()`), and to either synthesize an output model via `pydantic.create_model()` or validate a caller-supplied one against the query's inferred output shape.

**Tech Stack:** Same Rust/pyo3/sqlparser stack as the merged Phase 2 interpreter, plus `pydantic>=2.0,<3.0` as a **Python** dependency only — no new Cargo dependency; Pydantic and Arrow schemas are read purely via PyO3 attribute/method calls on the Python objects passed in (`model_fields`, `.schema`, `create_model`, `model_validate`), the same pattern `src/lookup.rs` already uses for `pyarrow.Table.to_pylist()`.

## Global Constraints

- `pydantic>=2.0,<3.0` added to `[project].dependencies` in `pyproject.toml` — Pydantic v2 API only (`model_fields`, `model_validate`, `create_model`); v1 is out of scope
- **This is a breaking change** to the just-merged dict-based `InferFn` — `row_tables` changes from `list[str]` to `dict[str, type[BaseModel]]`, `.infer()`'s `tables` argument takes model instances instead of dicts, and `.infer()` returns model instances instead of dicts. `tests/test_interpreter.py` is rewritten around this, not extended.
- `static_tables` stays `dict[str, pyarrow.Table]` — **unchanged**. Arrow already carries a schema (`pyarrow.Table.schema`); nothing here needs to become Pydantic.
- No new Cargo dependency for this feature — everything Pydantic/Arrow-schema related is read via PyO3 calls on the Python objects passed in, never a Rust-side crate
- **Nullability inference is sound but not tight** (see the design spec's Non-Goals): "nullable: true" means "cannot prove non-null," never "will be null." A tighter analysis is explicitly out of scope.
- **Output-model validation, when the caller supplies one:** missing/extra projected fields, or a base-type mismatch we can *prove* wrong (see the `compatible()` table below), is a build-time `ValueError`. A declared-non-nullable-but-inferred-possibly-nullable field is **never** a build-time error — it's deferred to Pydantic's own `model_validate()` raising `ValidationError` at `.infer()` time if a `None` actually shows up there.
- Known pyo3 0.29 API drift already hit repeatedly in the merged code (same crate, same version, applies here too): `Bound::downcast` doesn't exist, use `.cast()`; `Python::with_gil` doesn't exist, use `Python::attach`; `Py<T>` isn't `Clone` (the `py-clone` Cargo feature is off) — don't add a `#[derive(Clone)]` anywhere it would need one. If a task's exact code doesn't compile against the installed pyo3/sqlparser due to a small API-shape difference, investigate the actual installed crate (`cargo doc --open -p <crate> --no-deps`, or read the vendored source under `~/.cargo/registry/src/`) and fix minimally, preserving the design — don't guess.
- Ruff lint + format passes on Python files before each commit; `cargo fmt`/`cargo clippy` on Rust files
- Every behavioral Python test that involves computed values compares against real DataFusion output via the existing `_expected()` helper in `tests/test_interpreter.py`, same convention as the merged interpreter

---

### Task 1: Typed row input — schema extraction, build-time column validation, `getattr` row conversion

**Files:**
- Create: `src/types.rs`
- Create: `src/schema.rs`
- Modify: `src/plan.rs` (add `resolve_tables`/`validate_columns`/`ColumnValidation`)
- Modify: `src/lib.rs` (full rewrite: typed `row_tables`, `getattr`-based row conversion; **output stays a raw dict for now** — Task 2 changes that)
- Modify: `pyproject.toml` (add `pydantic` dependency)
- Modify: `tests/test_interpreter.py` (full rewrite around Pydantic row models)

**Interfaces:**
- Consumes: `crate::expr::{Expr, Value, BinOp, CastType}` (unchanged), `crate::plan::{Plan, RelNode, InterpError, Row}` (unchanged apart from this task's additions)
- Produces:
  - `types::Base` enum (`Int | Float | Str | Bool | Other`), `types::FieldType { base: Base, nullable: bool }` (both `Clone, Copy, PartialEq, Eq, Debug`), `types::Schema = HashMap<String, FieldType>`
  - `schema::from_pydantic_model(py: Python<'_>, model_class: &Py<PyAny>) -> Result<types::Schema, plan::InterpError>`
  - `schema::from_arrow_table(py: Python<'_>, table: &Py<PyAny>) -> Result<types::Schema, plan::InterpError>`
  - `plan::ColumnValidation { row_table_columns: HashMap<String, Vec<String>>, effective_schemas: HashMap<String, types::Schema> }`
  - `plan::validate_columns(plan: &Plan, row_table_names: &HashSet<String>, row_schemas: &HashMap<String, types::Schema>, static_schemas: &HashMap<String, types::Schema>) -> Result<ColumnValidation, InterpError>` — Task 2/3 reuse `effective_schemas` for output-type inference
  - `InferFn(sql, row_tables: dict[str, type[BaseModel]], static_tables: dict[str, pa.Table])` — new Python constructor shape (no `output_model` yet — that's Task 2/3)
  - `.infer(tables: dict[str, list[BaseModel instances]]) -> list[dict]` — output is still a plain dict this task; row *input* is fully typed

- [ ] **Step 1: Add the Pydantic dependency**

Read `pyproject.toml`. Add `"pydantic>=2.0,<3.0",` to `[project].dependencies`, alongside the existing `datafusion`/`pyarrow` entries:

```toml
[project]
dependencies = [
    "datafusion>=46.0.0",
    "pyarrow>=19.0",
    "pydantic>=2.0,<3.0",
]
```

Run:

```
uv sync
```

Expected: `pydantic`, `pydantic-core`, `annotated-types`, `typing-extensions`, `typing-inspection` installed.

- [ ] **Step 2: Write the failing tests**

Replace `tests/test_interpreter.py` entirely with:

```python
"""Tests for the Rust SQL interpreter (sql_transform._interpreter).

Row tables are declared as Pydantic v2 model classes at InferFn construction
time; row inputs to .infer() are instances of those models. Every
behavioral test that involves computed values compares InferFn output
against real DataFusion batch output for the same SQL + data.
"""

import datafusion
import pyarrow as pa
import pytest
from pydantic import BaseModel

from sql_transform._interpreter import InferFn


def _expected(sql: str, data: dict) -> list[dict]:
    ctx = datafusion.SessionContext()
    ctx.from_pydict(data, name="data")
    return ctx.sql(sql).collect()[0].to_pylist()


class Data(BaseModel):
    age: int
    name: str | None = None


def test_module_imports_and_constructs():
    fn = InferFn("SELECT age FROM data", row_tables={"data": Data}, static_tables={})
    assert fn is not None


def test_column_pass_through():
    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30)]})
    assert actual == _expected(sql, {"age": [30]})


def test_arithmetic_and_where():
    sql = "SELECT age, age * 2 AS doubled FROM data WHERE age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25), Data(age=40)]})
    assert actual == _expected(sql, {"age": [15, 25, 40]})


def test_builtin_function_and_cast():
    sql = "SELECT UPPER(name) AS n, CAST(age AS VARCHAR) AS s FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=30, name="alice")]})
    assert actual == _expected(sql, {"age": [30], "name": ["alice"]})


class A(BaseModel):
    id: int
    x: int


class B(BaseModel):
    id: int
    y: int


def test_cross_join():
    sql = "SELECT a.x, b.y FROM a, b"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1], "x": [10]}, name="a")
    ctx.from_pydict({"id": [1], "y": [20]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer({"a": [A(id=1, x=10)], "b": [B(id=1, y=20)]})
    assert actual == expected


def test_inner_join():
    sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [10, 20]}, name="a")
    ctx.from_pydict({"id": [1, 2], "y": [100, 200]}, name="b")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(sql, row_tables={"a": A, "b": B}, static_tables={})
    actual = fn.infer(
        {
            "a": [A(id=1, x=10), A(id=2, x=20)],
            "b": [B(id=1, y=100), B(id=2, y=200)],
        }
    )
    assert actual == expected


def test_aliased_row_table():
    sql = "SELECT d.age FROM data AS d WHERE d.age > 18"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [Data(age=15), Data(age=25)]})
    assert actual == _expected(sql, {"age": [15, 25]})


def test_join_row_and_static_table():
    ref_table = pa.table({"id": [1, 2], "y": [10, 20]})
    sql = "SELECT data.x, ref.y FROM data JOIN ref ON data.id = ref.id"

    class RowWithId(BaseModel):
        id: int
        x: int

    ctx = datafusion.SessionContext()
    ctx.from_pydict({"id": [1, 2], "x": [5, 6]}, name="data")
    ctx.from_arrow(ref_table, name="ref")
    expected = ctx.sql(sql).collect()[0].to_pylist()

    fn = InferFn(
        sql, row_tables={"data": RowWithId}, static_tables={"ref": ref_table}
    )
    actual = fn.infer(
        {"data": [RowWithId(id=1, x=5), RowWithId(id=2, x=6)]}
    )
    assert actual == expected


def test_error_unknown_row_column():
    sql = "SELECT nonexistent FROM data"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={})


def test_error_unknown_static_column():
    ref_table = pa.table({"id": [1], "y": [10]})
    sql = "SELECT data.age, ref.nonexistent FROM data JOIN ref ON data.age = ref.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"data": Data}, static_tables={"ref": ref_table})


def test_error_self_join_still_rejected():
    sql = "SELECT a.x FROM a JOIN a ON a.id = a.id"
    with pytest.raises(ValueError):
        InferFn(sql, row_tables={"a": A}, static_tables={})
```

- [ ] **Step 3: Run tests — verify they fail**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: `TypeError` from the old `#[new]` signature rejecting a `dict` where it expects a `list` (or similar) — `row_tables` is still `Vec<String>` in the compiled extension at this point.

- [ ] **Step 4: Create `src/types.rs`**

```rust
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
```

- [ ] **Step 5: Create `src/schema.rs`**

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::plan::InterpError;
use crate::types::{Base, FieldType, Schema};

/// Extract a Schema from a Pydantic v2 model class's `model_fields`.
pub fn from_pydantic_model(
    py: Python<'_>,
    model_class: &Py<PyAny>,
) -> Result<Schema, InterpError> {
    let bound = model_class.bind(py);
    let fields = bound
        .getattr("model_fields")
        .map_err(|e| InterpError::Build(format!("Not a Pydantic v2 model class: {e}")))?;
    let dict = fields
        .cast::<PyDict>()
        .map_err(|e| InterpError::Build(format!("model_fields must be a dict: {e}")))?;

    let typing = PyModule::import(py, "typing")
        .map_err(|e| InterpError::Build(format!("Failed to import typing: {e}")))?;
    let types_module = PyModule::import(py, "types")
        .map_err(|e| InterpError::Build(format!("Failed to import types: {e}")))?;

    let mut schema = Schema::new();
    for (name, field_info) in dict.iter() {
        let name: String = name
            .extract()
            .map_err(|e| InterpError::Build(format!("Invalid field name: {e}")))?;
        let annotation = field_info.getattr("annotation").map_err(|e| {
            InterpError::Build(format!("Field '{name}' has no annotation: {e}"))
        })?;
        let field_type = annotation_to_field_type(py, &typing, &types_module, &annotation)?;
        schema.insert(name, field_type);
    }
    Ok(schema)
}

/// Extract a Schema from a `pyarrow.Table`'s `.schema`.
pub fn from_arrow_table(py: Python<'_>, table: &Py<PyAny>) -> Result<Schema, InterpError> {
    let bound = table.bind(py);
    let arrow_schema = bound
        .getattr("schema")
        .map_err(|e| InterpError::Build(format!("Not a pyarrow.Table: {e}")))?;
    let names: Vec<String> = arrow_schema
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| InterpError::Build(format!("Failed to read static table schema: {e}")))?;

    let mut schema = Schema::new();
    for name in names {
        let field = arrow_schema
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read field '{name}': {e}")))?;
        let nullable: bool = field.getattr("nullable").and_then(|n| n.extract()).map_err(|e| {
            InterpError::Build(format!("Failed to read nullability of '{name}': {e}"))
        })?;
        let type_str: String = field
            .getattr("type")
            .and_then(|t| t.str())
            .map(|s| s.to_string())
            .map_err(|e| InterpError::Build(format!("Failed to read type of '{name}': {e}")))?;
        schema.insert(
            name,
            FieldType {
                base: arrow_type_to_base(&type_str),
                nullable,
            },
        );
    }
    Ok(schema)
}

fn arrow_type_to_base(type_str: &str) -> Base {
    if type_str.starts_with("int") || type_str.starts_with("uint") {
        Base::Int
    } else if type_str.starts_with("float")
        || type_str.starts_with("double")
        || type_str.starts_with("decimal")
    {
        Base::Float
    } else if type_str.starts_with("string")
        || type_str.starts_with("utf8")
        || type_str.starts_with("large_string")
    {
        Base::Str
    } else if type_str == "bool" {
        Base::Bool
    } else {
        Base::Other
    }
}

/// Convert a Python type annotation (`int`, `str | None`,
/// `typing.Optional[int]`, ...) to a FieldType. Nullability is detected via
/// `typing.get_origin`/`get_args`, which works uniformly for both `X | None`
/// (`types.UnionType`) and `Optional[X]`/`Union[X, None]` (`typing.Union`) —
/// both produce args containing `NoneType` when nullable. A non-Union
/// generic (`list[int]`, `dict[str, int]`, ...) maps to `Base::Other`.
fn annotation_to_field_type(
    py: Python<'_>,
    typing: &Bound<'_, PyModule>,
    types_module: &Bound<'_, PyModule>,
    annotation: &Bound<'_, PyAny>,
) -> Result<FieldType, InterpError> {
    let origin = typing
        .call_method1("get_origin", (annotation,))
        .map_err(|e| InterpError::Build(format!("Failed to inspect type annotation: {e}")))?;

    if origin.is_none() {
        return Ok(FieldType {
            base: python_type_to_base(annotation),
            nullable: false,
        });
    }

    let is_union = origin
        .is(&typing
            .getattr("Union")
            .map_err(|e| InterpError::Build(format!("typing.Union missing: {e}")))?)
        || origin.is(&types_module
            .getattr("UnionType")
            .map_err(|e| InterpError::Build(format!("types.UnionType missing: {e}")))?);
    if !is_union {
        return Ok(FieldType {
            base: Base::Other,
            nullable: false,
        });
    }

    let args: Vec<Py<PyAny>> = typing
        .call_method1("get_args", (annotation,))
        .and_then(|a| a.extract())
        .map_err(|e| InterpError::Build(format!("Failed to inspect union args: {e}")))?;

    let none_type = py.None().bind(py).get_type();
    let mut non_none: Vec<Py<PyAny>> = Vec::new();
    let mut nullable = false;
    for arg in args {
        if arg.bind(py).is(&none_type) {
            nullable = true;
        } else {
            non_none.push(arg);
        }
    }

    let base = if non_none.len() == 1 {
        python_type_to_base(non_none[0].bind(py))
    } else {
        Base::Other
    };
    Ok(FieldType { base, nullable })
}

fn python_type_to_base(t: &Bound<'_, PyAny>) -> Base {
    match t
        .getattr("__name__")
        .ok()
        .and_then(|n| n.extract::<String>().ok())
    {
        Some(name) => match name.as_str() {
            "int" => Base::Int,
            "float" => Base::Float,
            "str" => Base::Str,
            "bool" => Base::Bool,
            _ => Base::Other,
        },
        None => Base::Other,
    }
}
```

- [ ] **Step 6: Add column validation to `src/plan.rs`**

Add `use crate::types::Schema;` to the `use` block at the top of `src/plan.rs` (alongside the existing `use crate::expr::{Expr, Value};`).

Append to `src/plan.rs`:

```rust
pub struct ColumnValidation {
    pub row_table_columns: HashMap<String, Vec<String>>,
    pub effective_schemas: HashMap<String, Schema>,
}

/// Maps each relation's EFFECTIVE name (its alias if aliased, else its real
/// table name — the qualifier `Expr::Column` references use after
/// `SubqueryAlias` renaming) to its real table name and whether it's a row
/// table (vs. static). Walks the already-optimized Plan, so any `Join` with
/// a static side has already become a `LookupJoin`.
fn resolve_tables(
    node: &RelNode,
    row_table_names: &HashSet<String>,
    out: &mut HashMap<String, (String, bool)>,
) {
    match node {
        RelNode::TableScan { table } => {
            let is_row = row_table_names.contains(table);
            out.insert(table.clone(), (table.clone(), is_row));
        }
        RelNode::SubqueryAlias { input, alias } => {
            if let Some(real) = scan_table_name(input) {
                let is_row = row_table_names.contains(real);
                out.insert(alias.clone(), (real.to_string(), is_row));
            }
        }
        RelNode::Filter { input, .. } => resolve_tables(input, row_table_names, out),
        RelNode::CrossJoin { left, right } | RelNode::Join { left, right, .. } => {
            resolve_tables(left, row_table_names, out);
            resolve_tables(right, row_table_names, out);
        }
        RelNode::LookupJoin { input, table, .. } => {
            resolve_tables(input, row_table_names, out);
            out.insert(table.clone(), (table.clone(), false));
        }
    }
}

/// Validates every `Expr::Column` reference in the plan (projection, WHERE,
/// JOIN ON) against the resolved table schemas, and collects — per row
/// table's REAL name — the set of columns the query actually references.
/// Also returns the effective-name -> Schema map (aliases resolved), reused
/// by the output type-inference pass.
pub fn validate_columns(
    plan: &Plan,
    row_table_names: &HashSet<String>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
) -> Result<ColumnValidation, InterpError> {
    let mut resolved = HashMap::new();
    resolve_tables(&plan.input, row_table_names, &mut resolved);

    let mut effective_schemas = HashMap::new();
    for (effective_name, (real_name, is_row)) in &resolved {
        let schema = if *is_row {
            row_schemas.get(real_name)
        } else {
            static_schemas.get(real_name)
        };
        if let Some(s) = schema {
            effective_schemas.insert(effective_name.clone(), s.clone());
        }
    }

    let mut used_columns: HashMap<String, HashSet<String>> = HashMap::new();
    for (_, e) in &plan.projection {
        validate_expr(e, &resolved, row_schemas, static_schemas, &mut used_columns)?;
    }
    validate_rel(
        &plan.input,
        &resolved,
        row_schemas,
        static_schemas,
        &mut used_columns,
    )?;

    Ok(ColumnValidation {
        row_table_columns: used_columns
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect(),
        effective_schemas,
    })
}

fn validate_rel(
    node: &RelNode,
    resolved: &HashMap<String, (String, bool)>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
    used_columns: &mut HashMap<String, HashSet<String>>,
) -> Result<(), InterpError> {
    match node {
        RelNode::TableScan { .. } => Ok(()),
        RelNode::Filter { input, predicate } => {
            validate_expr(predicate, resolved, row_schemas, static_schemas, used_columns)?;
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::CrossJoin { left, right } => {
            validate_rel(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_rel(right, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::Join { left, right, on } => {
            for (l, r) in on {
                validate_expr(l, resolved, row_schemas, static_schemas, used_columns)?;
                validate_expr(r, resolved, row_schemas, static_schemas, used_columns)?;
            }
            validate_rel(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_rel(right, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::SubqueryAlias { input, .. } => {
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
        RelNode::LookupJoin { input, keys, .. } => {
            for k in keys {
                validate_expr(k, resolved, row_schemas, static_schemas, used_columns)?;
            }
            validate_rel(input, resolved, row_schemas, static_schemas, used_columns)
        }
    }
}

fn validate_expr(
    e: &Expr,
    resolved: &HashMap<String, (String, bool)>,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
    used_columns: &mut HashMap<String, HashSet<String>>,
) -> Result<(), InterpError> {
    match e {
        Expr::Column {
            table: Some(t),
            name,
        } => {
            let (real, is_row) = resolved
                .get(t)
                .ok_or_else(|| InterpError::Build(format!("Unknown table qualifier: {t}")))?;
            check_column(real, *is_row, name, row_schemas, static_schemas)?;
            if *is_row {
                used_columns
                    .entry(real.clone())
                    .or_default()
                    .insert(name.clone());
            }
            Ok(())
        }
        Expr::Column { table: None, name } => {
            let mut matches: Vec<(&String, bool)> = Vec::new();
            for (real, is_row) in resolved.values() {
                let schema = if *is_row {
                    row_schemas.get(real)
                } else {
                    static_schemas.get(real)
                };
                if schema.is_some_and(|s| s.contains_key(name)) {
                    matches.push((real, *is_row));
                }
            }
            match matches.as_slice() {
                [] => Err(InterpError::Build(format!("Unknown column: {name}"))),
                [(real, is_row)] => {
                    if *is_row {
                        used_columns
                            .entry((*real).clone())
                            .or_default()
                            .insert(name.clone());
                    }
                    Ok(())
                }
                _ => Err(InterpError::Build(format!("Ambiguous column reference: {name}"))),
            }
        }
        Expr::Literal(_) => Ok(()),
        Expr::BinaryOp { left, right, .. } => {
            validate_expr(left, resolved, row_schemas, static_schemas, used_columns)?;
            validate_expr(right, resolved, row_schemas, static_schemas, used_columns)
        }
        Expr::Not(inner) | Expr::Cast { expr: inner, .. } => {
            validate_expr(inner, resolved, row_schemas, static_schemas, used_columns)
        }
        Expr::Function { args, .. } => {
            for a in args {
                validate_expr(a, resolved, row_schemas, static_schemas, used_columns)?;
            }
            Ok(())
        }
    }
}

fn check_column(
    real_table: &str,
    is_row: bool,
    name: &str,
    row_schemas: &HashMap<String, Schema>,
    static_schemas: &HashMap<String, Schema>,
) -> Result<(), InterpError> {
    let schema = if is_row {
        row_schemas.get(real_table)
    } else {
        static_schemas.get(real_table)
    };
    match schema {
        Some(s) if s.contains_key(name) => Ok(()),
        Some(_) => Err(InterpError::Build(format!("Unknown column: {real_table}.{name}"))),
        None => Err(InterpError::Build(format!("Unknown table: {real_table}"))),
    }
}
```

- [ ] **Step 7: Rewrite `src/lib.rs`**

```rust
use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod expr;
mod expr_build;
mod lookup;
mod plan;
mod schema;
mod types;

use expr::Value;
use lookup::LookupIndex;
use plan::Plan;

#[pyclass]
struct InferFn {
    plan: Plan,
    lookups: HashMap<String, LookupIndex>,
    row_table_columns: HashMap<String, Vec<String>>,
}

#[pymethods]
impl InferFn {
    #[new]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
    ) -> PyResult<Self> {
        let raw_plan = plan::build_plan(&sql)?;
        let row_table_names: HashSet<String> = row_tables.keys().cloned().collect();
        let static_table_names: HashSet<String> = static_tables.keys().cloned().collect();
        let (optimized_plan, specs) = plan::optimize(raw_plan, &static_table_names)?;

        let mut row_schemas = HashMap::new();
        for (name, model_class) in &row_tables {
            row_schemas.insert(name.clone(), schema::from_pydantic_model(py, model_class)?);
        }
        let mut static_schemas = HashMap::new();
        for (name, table_obj) in &static_tables {
            static_schemas.insert(name.clone(), schema::from_arrow_table(py, table_obj)?);
        }

        let column_validation = plan::validate_columns(
            &optimized_plan,
            &row_table_names,
            &row_schemas,
            &static_schemas,
        )?;

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

        Ok(InferFn {
            plan: optimized_plan,
            lookups,
            row_table_columns: column_validation.row_table_columns,
        })
    }

    fn infer(
        &self,
        py: Python<'_>,
        tables: HashMap<String, Vec<Py<PyAny>>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let empty: Vec<String> = Vec::new();
        let mut value_tables: HashMap<String, Vec<HashMap<String, Value>>> = HashMap::new();
        for (table, rows) in &tables {
            let columns = self.row_table_columns.get(table).unwrap_or(&empty);
            let mut out_rows = Vec::with_capacity(rows.len());
            for row_obj in rows {
                let bound = row_obj.bind(py);
                let mut row: HashMap<String, Value> = HashMap::new();
                for col in columns {
                    let attr = bound.getattr(col.as_str()).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Row for table '{table}' is missing attribute '{col}': {e}"
                        ))
                    })?;
                    row.insert(col.clone(), Value::from_pyobject(&attr)?);
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

Note: `RelNode::CrossJoin { left, right } | RelNode::Join { left, right, .. }` (used in `resolve_tables`, Step 6) is an or-pattern binding — this requires the bound variable names/types to match across both arms, which they do here (`left`/`right`: `Box<RelNode>` in both). If `cargo build` rejects the or-pattern for any reason, split it into two identical match arms instead.

- [ ] **Step 8: Build and run tests**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS.

- [ ] **Step 9: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
cargo fmt
git add pyproject.toml uv.lock src/types.rs src/schema.rs src/plan.rs src/lib.rs tests/test_interpreter.py
git commit -m "feat: typed row_tables via Pydantic models, build-time column validation"
```

---

### Task 2: Output type inference + synthesized `output_model`

**Files:**
- Modify: `src/types.rs` (add `infer_type`)
- Modify: `src/schema.rs` (add `field_type_to_python`)
- Modify: `src/lib.rs` (synthesize `output_model`, expose it, typed output conversion)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `types::{Base, FieldType, Schema}`, `plan::ColumnValidation.effective_schemas` (Task 1), `expr::{Expr, BinOp, CastType, Value}` (unchanged)
- Produces:
  - `types::infer_type(expr: &Expr, schemas: &HashMap<String, Schema>) -> Result<FieldType, InterpError>`
  - `schema::field_type_to_python(py: Python<'_>, ft: FieldType) -> PyResult<Py<PyAny>>`
  - `InferFn.output_model` — readable Python attribute (`#[pyo3(get)]`), the synthesized Pydantic model class
  - `.infer()` now returns `list[fn.output_model instances]` instead of `list[dict]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interpreter.py`:

```python
def test_output_model_is_synthesized_and_typed():
    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})

    assert set(fn.output_model.model_fields) == {"age", "doubled"}
    assert fn.output_model.model_fields["age"].annotation is int
    assert fn.output_model.model_fields["doubled"].annotation is int

    results = fn.infer({"data": [Data(age=30)]})
    assert len(results) == 1
    assert isinstance(results[0], fn.output_model)
    assert results[0].age == 30
    assert results[0].doubled == 60


def test_output_model_nullable_column():
    sql = "SELECT name FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["name"].annotation == (str | None)

    results = fn.infer({"data": [Data(age=1, name=None)]})
    assert results[0].name is None


def test_output_model_cast_type_is_exact():
    sql = "SELECT CAST(age AS VARCHAR) AS s FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["s"].annotation is str


def test_output_model_division_promotes_to_float():
    sql = "SELECT age / 2 AS half FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    # age (int) / 2 (int literal) -> Int per the truncating-int-division rule
    assert fn.output_model.model_fields["half"].annotation is int


def test_output_model_comparison_is_bool():
    sql = "SELECT age > 18 AS is_adult FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    assert fn.output_model.model_fields["is_adult"].annotation is bool
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k output_model
```

Expected: FAIL — `InferFn` has no `output_model` attribute yet, and `.infer()` still returns dicts.

- [ ] **Step 3: Add `infer_type` to `src/types.rs`**

Append to `src/types.rs`:

```rust
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
                return Err(InterpError::Build(format!("Ambiguous column reference: {name}")));
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
```

- [ ] **Step 4: Add `field_type_to_python` to `src/schema.rs`**

Append to `src/schema.rs`:

```rust
/// Convert a FieldType into the Python type object `create_model` needs —
/// the inverse of `annotation_to_field_type`: `Optional[T]` if nullable,
/// else `T` directly. `Base::Other` maps to `typing.Any`.
pub fn field_type_to_python(py: Python<'_>, ft: FieldType) -> PyResult<Py<PyAny>> {
    let builtins = PyModule::import(py, "builtins")?;
    let typing = PyModule::import(py, "typing")?;
    let base_type: Py<PyAny> = match ft.base {
        Base::Int => builtins.getattr("int")?.unbind(),
        Base::Float => builtins.getattr("float")?.unbind(),
        Base::Str => builtins.getattr("str")?.unbind(),
        Base::Bool => builtins.getattr("bool")?.unbind(),
        Base::Other => typing.getattr("Any")?.unbind(),
    };
    if !ft.nullable {
        return Ok(base_type);
    }
    let none_type = py.None().bind(py).get_type().unbind();
    let union = typing.getattr("Union")?;
    let optional = union.call_method1("__getitem__", ((base_type, none_type),))?;
    Ok(optional.unbind())
}
```

- [ ] **Step 5: Wire output-model synthesis into `src/lib.rs`**

Add `use plan::Plan;` stays; add `use expr::Expr;` to the `use` block at the top (alongside `use expr::Value;`).

Add the `output_model` field to the `InferFn` struct:

```rust
#[pyclass]
struct InferFn {
    plan: Plan,
    lookups: HashMap<String, LookupIndex>,
    row_table_columns: HashMap<String, Vec<String>>,
    #[pyo3(get)]
    output_model: Py<PyAny>,
}
```

Add this function (module-level, alongside `InferFn`'s impl block):

```rust
fn synthesize_output_model(
    py: Python<'_>,
    projection: &[(String, Expr)],
    schemas: &HashMap<String, types::Schema>,
) -> PyResult<Py<PyAny>> {
    let pydantic = PyModule::import(py, "pydantic")?;
    let create_model = pydantic.getattr("create_model")?;
    let builtins = PyModule::import(py, "builtins")?;
    let ellipsis = builtins.getattr("Ellipsis")?;

    let kwargs = PyDict::new(py);
    for (alias, expr) in projection {
        let ft = types::infer_type(expr, schemas)?;
        let py_type = schema::field_type_to_python(py, ft)?;
        kwargs.set_item(alias, (py_type, &ellipsis))?;
    }
    let model = create_model.call(("OutputRow",), Some(&kwargs))?;
    Ok(model.unbind())
}
```

In `InferFn::new`, after computing `column_validation` (and before building `lookups`, order doesn't matter as long as it's after `column_validation` exists), add:

```rust
        let output_model =
            synthesize_output_model(py, &optimized_plan.projection, &column_validation.effective_schemas)?;
```

Update the final `Ok(InferFn { ... })` to include it:

```rust
        Ok(InferFn {
            plan: optimized_plan,
            lookups,
            row_table_columns: column_validation.row_table_columns,
            output_model,
        })
```

Replace `infer`'s output-building loop (the part after `let result_rows = plan::execute(...)?;`) with:

```rust
        let output_model = self.output_model.bind(py);
        let mut out = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            let dict = PyDict::new(py);
            for (k, v) in row {
                dict.set_item(k, v.to_pyobject(py)?)?;
            }
            let instance = output_model.call_method1("model_validate", (dict,))?;
            out.push(instance.unbind());
        }
        Ok(out)
```

Change `infer`'s return type from `PyResult<Vec<Py<PyDict>>>` to `PyResult<Vec<Py<PyAny>>>`.

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
git add src/types.rs src/schema.rs src/lib.rs tests/test_interpreter.py
git commit -m "feat: static output type inference, synthesized output_model"
```

---

### Task 3: User-supplied `output_model`

**Files:**
- Modify: `src/types.rs` (add `compatible`)
- Modify: `src/lib.rs` (optional `output_model` constructor param, validation)
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: `types::{Base, FieldType, Schema, infer_type}` (Task 2), `schema::from_pydantic_model` (Task 1)
- Produces:
  - `types::compatible(inferred: Base, declared: Base) -> bool`
  - `InferFn(sql, row_tables, static_tables, output_model: type[BaseModel] | None = None)` — when supplied, validated against the query; when omitted, synthesized as in Task 2

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interpreter.py`:

```python
class Result(BaseModel):
    age: int
    doubled: int


def test_user_supplied_output_model_compatible():
    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(
        sql, row_tables={"data": Data}, static_tables={}, output_model=Result
    )
    assert fn.output_model is Result
    results = fn.infer({"data": [Data(age=30)]})
    assert isinstance(results[0], Result)
    assert results[0] == Result(age=30, doubled=60)


def test_user_supplied_output_model_widening_int_to_float_ok():
    class WideResult(BaseModel):
        age: float
        doubled: int

    sql = "SELECT age, age * 2 AS doubled FROM data"
    fn = InferFn(
        sql, row_tables={"data": Data}, static_tables={}, output_model=WideResult
    )
    results = fn.infer({"data": [Data(age=30)]})
    assert results[0].age == 30.0


def test_user_supplied_output_model_missing_field():
    class Incomplete(BaseModel):
        age: int
        # missing "doubled"

    sql = "SELECT age, age * 2 AS doubled FROM data"
    with pytest.raises(ValueError):
        InferFn(
            sql, row_tables={"data": Data}, static_tables={}, output_model=Incomplete
        )


def test_user_supplied_output_model_extra_field():
    class TooMany(BaseModel):
        age: int
        doubled: int
        unrelated: str

    sql = "SELECT age, age * 2 AS doubled FROM data"
    with pytest.raises(ValueError):
        InferFn(
            sql, row_tables={"data": Data}, static_tables={}, output_model=TooMany
        )


def test_user_supplied_output_model_provably_wrong_base_type():
    class WrongType(BaseModel):
        age: str  # query produces Int, str is not provably compatible

    sql = "SELECT age FROM data"
    with pytest.raises(ValueError):
        InferFn(
            sql, row_tables={"data": Data}, static_tables={}, output_model=WrongType
        )


def test_user_supplied_output_model_non_nullable_but_inferred_nullable_defers_to_runtime():
    class NonNullResult(BaseModel):
        name: str  # declared non-nullable; `name` is Optional[str] on Data

    sql = "SELECT name FROM data"
    # Build succeeds — we can't PROVE `name` will be null, only that we
    # can't prove it won't.
    fn = InferFn(
        sql, row_tables={"data": Data}, static_tables={}, output_model=NonNullResult
    )
    # A row that actually produces None fails at infer() time, not build time.
    with pytest.raises(Exception):  # pydantic.ValidationError
        fn.infer({"data": [Data(age=1, name=None)]})
    # A row with a real value works fine.
    result = fn.infer({"data": [Data(age=1, name="alice")]})
    assert result[0].name == "alice"
```

- [ ] **Step 2: Run tests — verify they fail**

```
uv run pytest tests/test_interpreter.py -v -k user_supplied
```

Expected: FAIL — `InferFn` doesn't accept an `output_model` keyword argument yet.

- [ ] **Step 3: Add `compatible` to `src/types.rs`**

Append to `src/types.rs`:

```rust
/// Is `inferred` provably safe to store in a field declared as `declared`?
/// Anything not provably wrong is allowed through — Pydantic's own
/// `model_validate()` is the real authority at `.infer()` time for
/// anything this can't rule out.
pub fn compatible(inferred: Base, declared: Base) -> bool {
    match (inferred, declared) {
        (a, b) if a == b => true,
        // Every valid int is a valid float; Pydantic's default lax mode
        // coerces this without loss.
        (Base::Int, Base::Float) => true,
        // We have no basis to say an unresolvable inferred type is wrong.
        (Base::Other, _) => true,
        _ => false,
    }
}
```

- [ ] **Step 4: Accept and validate a user-supplied `output_model` in `src/lib.rs`**

Add this function (module-level):

```rust
fn validate_output_model(
    py: Python<'_>,
    model: &Py<PyAny>,
    projection: &[(String, Expr)],
    schemas: &HashMap<String, types::Schema>,
) -> PyResult<()> {
    let declared_schema = schema::from_pydantic_model(py, model)?;

    let mut projected_aliases: HashSet<String> = HashSet::new();
    for (alias, expr) in projection {
        projected_aliases.insert(alias.clone());
        let declared = declared_schema.get(alias).ok_or_else(|| {
            plan::InterpError::Build(format!(
                "output_model is missing field '{alias}' produced by the query"
            ))
        })?;
        let inferred = types::infer_type(expr, schemas)?;
        if !types::compatible(inferred.base, declared.base) {
            return Err(plan::InterpError::Build(format!(
                "output_model field '{alias}' is declared as a type incompatible with the \
                 query's inferred output ({:?} vs declared {:?})",
                inferred.base, declared.base
            ))
            .into());
        }
    }

    let declared_fields: HashSet<String> = declared_schema.keys().cloned().collect();
    let extra: Vec<&String> = declared_fields.difference(&projected_aliases).collect();
    if !extra.is_empty() {
        return Err(plan::InterpError::Build(format!(
            "output_model declares fields not produced by the query: {extra:?}"
        ))
        .into());
    }
    Ok(())
}
```

Change `InferFn::new`'s signature to accept an optional `output_model`:

```rust
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, output_model=None))]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        output_model: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
```

Replace the Task 2 line that unconditionally synthesizes `output_model` with:

```rust
        let output_model = match output_model {
            Some(supplied) => {
                validate_output_model(
                    py,
                    &supplied,
                    &optimized_plan.projection,
                    &column_validation.effective_schemas,
                )?;
                supplied
            }
            None => synthesize_output_model(
                py,
                &optimized_plan.projection,
                &column_validation.effective_schemas,
            )?,
        };
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
git add src/types.rs src/lib.rs tests/test_interpreter.py
git commit -m "feat: accept and validate a user-supplied output_model"
```

---

### Task 4: Duck-typing and edge-case regression tests

**Files:**
- Modify: `tests/test_interpreter.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3
- Produces: no new Rust code — closes the remaining rows of the design spec's Testing Strategy table (duck-typed row instances, missing-attribute errors, unused-field pass-through)

- [ ] **Step 1: Write the tests**

Append to `tests/test_interpreter.py`:

```python
def test_duck_typed_row_instance_works():
    """A structurally-compatible instance of a DIFFERENT class than the one
    declared in row_tables still works — no isinstance check, just getattr."""

    class NotData:
        def __init__(self, age):
            self.age = age

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    actual = fn.infer({"data": [NotData(age=42)]})
    assert actual[0].age == 42


def test_row_instance_missing_referenced_attribute_raises_clear_error():
    class Incomplete:
        pass

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": Data}, static_tables={})
    with pytest.raises(ValueError, match="age"):
        fn.infer({"data": [Incomplete()]})


def test_unused_model_fields_are_never_touched():
    """A row model can have fields the query doesn't reference — getattr
    only pulls the columns actually used, so an unrelated/broken field on
    the instance is never even accessed."""

    class Poison:
        @property
        def unused(self):
            raise RuntimeError("should never be accessed")

    class WithExtra(BaseModel, arbitrary_types_allowed=True):
        age: int
        unused: object = None

    sql = "SELECT age FROM data"
    fn = InferFn(sql, row_tables={"data": WithExtra}, static_tables={})
    row = WithExtra(age=5, unused=Poison())
    actual = fn.infer({"data": [row]})
    assert actual[0].age == 5
```

- [ ] **Step 2: Run tests — verify they pass**

```
uv run maturin develop
uv run pytest tests/test_interpreter.py -v
```

Expected: all PASS (these exercise behavior already implemented by Tasks 1-3 — no Rust changes expected here). If `test_unused_model_fields_are_never_touched` fails with `RuntimeError`, that means `row_table_columns` isn't correctly limiting `getattr` to only the referenced columns — go back to Task 1's `validate_columns`/`infer`'s row-conversion loop and fix the column-collection logic, don't add a workaround here.

- [ ] **Step 3: Ruff + cargo fmt + commit**

```
uv run ruff check --fix .
uv run ruff format .
git add tests/test_interpreter.py
git commit -m "test: duck-typing and unused-field edge cases for typed InferFn"
```

---

### Task 5: Type stubs + final verification

**Files:**
- Modify: `sql_transform/_interpreter.pyi`
- Modify: `sql_transform/__init__.py` (verify unaffected)

**Interfaces:**
- Consumes: everything from Tasks 1-4
- Produces: accurate `.pyi` stub for the new `InferFn` signature

- [ ] **Step 1: Update the type stub**

Replace `sql_transform/_interpreter.pyi` entirely:

```python
from typing import Any

from pydantic import BaseModel

class InferFn:
    output_model: type[BaseModel]

    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, Any],
        output_model: type[BaseModel] | None = None,
    ) -> None: ...
    def infer(self, tables: dict[str, list[BaseModel]]) -> list[BaseModel]: ...
```

- [ ] **Step 2: Verify `sql_transform/__init__.py` needs no change**

Read `sql_transform/__init__.py`. It re-exports `InferFn` via `from sql_transform._interpreter import InferFn` — this line is unaffected by the constructor signature change (Python doesn't check signatures at import time). Confirm this by re-running `test_public_import_path`-equivalent behavior below; no edit expected.

- [ ] **Step 3: Full verification pass**

```
uv run maturin develop
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
cargo fmt --check
cargo clippy --all-targets
cargo build
```

Expected: every test passes (interpreter tests + the unrelated Phase-1 `sql_transform/*_test.py` suite), ruff/cargo fmt/clippy/build all clean.

- [ ] **Step 4: Final commit**

```
git add sql_transform/_interpreter.pyi
git commit -m "docs: update _interpreter.pyi for typed InferFn"
```
