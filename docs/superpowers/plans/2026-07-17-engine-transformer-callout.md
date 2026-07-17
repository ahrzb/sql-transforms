# Engine transformer callout (Part 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Both engines can invoke an opaque, already-fitted Python transformer (an sklearn transformer or a whole fitted `Pipeline`) during one query — struct in, struct out — with differential parity between `transform` (DataFusion) and `infer` (Rust `InferFn`).

**Architecture:** `transform` registers the transformer as a **vectorized** DataFusion struct-in/struct-out scalar UDF (`sql_transform/_transformer_udf.py`). `infer` gains a new `Expr::Transform` node: `InferFn.__new__` takes a `transformers` registry, reads each object's `feature_names_in_` at build time, rewrites the matching `Expr::Function` call in the projection into `Expr::Transform`, and evaluates it row-at-a-time via `Python::attach` (so `expr::eval` stays pure Rust). Both engines align input by `feature_names_in_` and marshal output through a caller-declared pyarrow schema, so the oracle matches by construction.

**Tech Stack:** Rust (pyo3 0.29, sqlparser 0.62), Python 3.13/3.14, datafusion 54, pyarrow, pydantic v2, scikit-learn, numpy, pandas (dev/test).

**Design doc:** `docs/superpowers/specs/2026-07-17-engine-transformer-callout-design.md`

## Global Constraints

- **v0, no backward-compat.** Add the `transformers` parameter directly; no compat shims.
- **DataFusion is the oracle — no exceptions.** Every parity test compares `transform` (DataFusion, UDF-registered) against `infer` (Rust). Never assert against a Python-only computation as the sole oracle.
- **No numpy in Rust.** The Rust eval path builds a Python `list`-of-`list`s and lets sklearn's `check_array` coerce it. numpy is used **only** in `sql_transform/_transformer_udf.py` (the oracle side).
- **GIL entry is `Python::attach(|py| …)`**, matching the existing codebase idiom (`src/expr.rs` `Value::clone`). Do **not** use `Python::with_gil`.
- **`InterpError` has no `From<PyErr>`.** Every `PyResult`/`PyErr` inside a `Result<_, InterpError>` context must be mapped explicitly (`.map_err(|e| InterpError::Eval(format!("…: {e}")))` for eval-time, `InterpError::Build(...)` for build-time).
- **Output struct field order is significant** and comes from the **declared** output schema (an ordered `Vec<(String, FieldType)>`), never from a `HashMap`.
- **`feature_names_in_` is required.** An object without it is a clear **build-time** `ValueError` telling the caller to fit on named data.
- **Reserved-name convention.** Transformers are called in SQL by a reserved identifier (tests use `__tfm_0__`); Part 2 generates these.
- **Scope is Part 1 only.** No `{ref}` surface, no `SQLTransform`/t-string wiring, no `unnest`/derived-table lowering, no multi-output native refs, no fitting. Struct in / struct out, one transformer per node, resolved in **projection** expressions.

---

## File Structure

- **Create** `sql_transform/_transformer_udf.py` — `_transformer_udf(obj, in_schema, out_schema, name)` building the vectorized DataFusion UDF (oracle side). Pure Python.
- **Create** `tests/test_transformer_udf.py` — standalone oracle test (DataFusion only).
- **Create** `tests/test_diff_transformer_callout.py` — differential parity `transform == infer`.
- **Modify** `src/expr.rs` — add `Expr::Transform` variant (+ `use std::sync::Arc;`, `use crate::types::FieldType;`) and its `eval` arm.
- **Modify** `src/types.rs` — add the `Expr::Transform` arm to `infer_type` (+ `use std::collections::HashSet;`).
- **Modify** `src/schema.rs` — add `arrow_schema_to_ordered_fields(py, schema_obj)`.
- **Modify** `src/lib.rs` — add the `transformers` parameter, the `ResolvedTransformer` build step (reading `feature_names_in_`), and the projection resolution pass (`resolve_transformers`).
- **Modify** `sql_transform/_interpreter.pyi` — document the new `transformers` parameter.
- **Modify** `pyproject.toml` — add `numpy`, `scikit-learn`, `pandas` to the `dev` dependency group.

**Build/test commands** (run from the worktree root `.claude/worktrees/opaque-transform-refs`):
- Rust type-check: `cargo build`
- Rust unit tests: `cargo test`
- Rebuild the Python extension after any Rust change: `uv run maturin develop`
- Python tests: `uv run pytest tests/<file> -v`

---

## Task 1: DataFusion transformer UDF (the oracle)

**Files:**
- Modify: `pyproject.toml` (dev deps)
- Create: `sql_transform/_transformer_udf.py`
- Test: `tests/test_transformer_udf.py`

**Interfaces:**
- Produces: `_transformer_udf(obj, in_schema: pa.Schema, out_schema: pa.Schema, name: str) -> datafusion.ScalarUDF`. `obj` is a fitted sklearn transformer/Pipeline exposing `.transform` and `feature_names_in_`. The returned UDF is registered via `ctx.register_udf(...)` and called in SQL as `name(<struct arg>)`, returning a struct of `out_schema`'s fields.

> **Note (spec refinement):** the design doc wrote the signature as `_transformer_udf(obj, in_schema, out_schema)`. A `name` argument is added because DataFusion registers a UDF by its `.name`, and the SQL must call it by the reserved name (`__tfm_0__`). This is the only deviation.

- [ ] **Step 1: Add dev dependencies**

Edit `pyproject.toml`, extending the `dev` group under `[dependency-groups]`:

```toml
[dependency-groups]
dev = [
    "ipdb>=0.13.13",
    "ipython>=9.2.0",
    "maturin>=1.14,<2.0",
    "numpy>=2.0",
    "pandas>=2.0",
    "pytest>=8.3.5",
    "ruff>=0.11.7",
    "scikit-learn>=1.5",
]
```

Then run: `uv sync`
Expected: resolves and installs numpy, pandas, scikit-learn (sklearn pulls numpy/scipy). `pandas` is required because `feature_names_in_` only exists when the transformer was fit on **named** data (a DataFrame).

- [ ] **Step 2: Write the failing oracle test**

Create `tests/test_transformer_udf.py`:

```python
"""Standalone oracle test: the DataFusion transformer UDF must match sklearn."""

import pandas as pd
import pyarrow as pa
import pytest
from datafusion import SessionContext
from sklearn.preprocessing import StandardScaler

from sql_transform._transformer_udf import _transformer_udf

# Both engines deliberately feed sklearn positionally-aligned nameless arrays
# (we reorder to feature_names_in_ ourselves), so sklearn's redundant
# "X does not have valid feature names" warning is a known false positive here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _collect(df) -> list[dict]:
    return pa.Table.from_batches(df.collect(), schema=df.schema()).to_pylist()


def test_standard_scaler_udf_matches_sklearn():
    train_df = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train_df)  # fit on NAMED data -> feature_names_in_
    in_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    out_schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])

    table = pa.Table.from_pandas(train_df)
    ctx = SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.register_udf(_transformer_udf(sc, in_schema, out_schema, "__tfm_0__"))

    q = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
    got = _collect(ctx.sql(q))

    expected = sc.transform(train_df)
    assert len(got) == 4
    for i, r in enumerate(got):
        assert abs(r["s"]["age"] - expected[i][0]) < 1e-9
        assert abs(r["s"]["income"] - expected[i][1]) < 1e-9
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_transformer_udf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sql_transform._transformer_udf'`.

- [ ] **Step 4: Implement `_transformer_udf`**

Create `sql_transform/_transformer_udf.py`:

```python
"""Wrap a fitted sklearn transformer as a vectorized DataFusion UDF.

The oracle side of the engine transformer-callout capability: `transform`
registers this UDF so a query can call a fitted transformer by name, struct in
/ struct out. The row engine (`InferFn`) performs the identical alignment and
marshalling in Rust; the differential harness proves the two agree.

Input is aligned to the object's `feature_names_in_` order; output is built
from the caller-declared `out_schema` (no introspection). numpy lives here and
here only -- the Rust engine imports no numpy.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from datafusion import ScalarUDF, udf


def _transformer_udf(
    obj: object,
    in_schema: pa.Schema,
    out_schema: pa.Schema,
    name: str,
) -> ScalarUDF:
    """Build a vectorized struct-in/struct-out DataFusion scalar UDF.

    obj: a fitted sklearn transformer/Pipeline exposing `.transform` and
        `feature_names_in_`.
    in_schema: names+types of the struct the SQL feeds in (the authored
        `named_struct`). Its field-name set must equal `feature_names_in_`.
    out_schema: names+types of the returned struct (declared, not introspected).
    name: the reserved SQL identifier the UDF is registered and called under.
    """
    feature_names = [str(n) for n in obj.feature_names_in_]
    in_type = pa.struct(list(in_schema))
    out_type = pa.struct(list(out_schema))
    out_fields = list(out_schema)

    def _apply(struct_array: pa.Array) -> pa.Array:
        # DataFusion hands the whole batch's StructArray in one call.
        cols = [
            struct_array.field(fname).to_numpy(zero_copy_only=False)
            for fname in feature_names
        ]
        x = np.column_stack(cols)
        y = np.asarray(obj.transform(x))
        out_cols = [
            pa.array(y[:, i], type=out_fields[i].type)
            for i in range(len(out_fields))
        ]
        return pa.StructArray.from_arrays(out_cols, fields=out_fields)

    return udf(_apply, [in_type], out_type, "immutable", name=name)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_transformer_udf.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock sql_transform/_transformer_udf.py tests/test_transformer_udf.py
git commit -m "feat: DataFusion transformer UDF (oracle side)"
```

---

## Task 2: Rust `Expr::Transform` engine capability

**Files:**
- Modify: `src/expr.rs` (variant + eval arm)
- Modify: `src/types.rs` (`infer_type` arm)
- Modify: `src/schema.rs` (`arrow_schema_to_ordered_fields`)
- Modify: `src/lib.rs` (`transformers` param + resolution)
- Modify: `sql_transform/_interpreter.pyi`
- Test: `tests/test_transformer_callout_infer.py`

**Interfaces:**
- Consumes: `Value::Struct`, `Value::to_pyobject`, `Value::from_pyobject_typed`, `types::{Base, FieldType, Schema}`, `schema::from_arrow_table` internals.
- Produces:
  - Rust `Expr::Transform { obj: std::sync::Arc<Py<PyAny>>, input_features: Vec<String>, output_fields: Vec<(String, crate::types::FieldType)>, arg: Box<Expr> }`.
  - `schema::arrow_schema_to_ordered_fields(py, schema_obj: &Py<PyAny>) -> Result<Vec<(String, FieldType)>, InterpError>`.
  - `InferFn.__new__(sql, row_tables, static_tables, output_model=None, transformers=None)` where `transformers: dict[str, tuple[object, pa.Schema]]` maps a reserved name to `(fitted_obj, out_schema)`.

- [ ] **Step 1: Add the `Expr::Transform` variant**

In `src/expr.rs`, add these imports near the top (after the existing `use` lines):

```rust
use std::sync::Arc;

use crate::types::FieldType;
```

Add a variant to the `pub enum Expr { … }` block (after `FieldAccess`):

```rust
    /// Callout to an opaque, already-fitted Python transformer. `obj` is the
    /// fitted sklearn object; `input_features` is its `feature_names_in_`
    /// (the struct arg is reordered to this before `.transform`);
    /// `output_fields` is the caller-declared output schema (marshalling
    /// target, order significant); `arg` evaluates to the input `Value::Struct`.
    Transform {
        obj: Arc<Py<PyAny>>,
        input_features: Vec<String>,
        output_fields: Vec<(String, FieldType)>,
        arg: Box<Expr>,
    },
```

(`Expr` keeps `#[derive(Clone)]`: `Arc` clones without a GIL token, `Vec`/`FieldType`/`Box` are already `Clone`.)

- [ ] **Step 2: Verify it compiles as non-exhaustive failures**

Run: `cargo build`
Expected: FAIL — `eval` (src/expr.rs) and `infer_type` (src/types.rs) `match` on `Expr` are now non-exhaustive (`error[E0004]: non-exhaustive patterns: \`&Transform { … }\` not covered`). This confirms exactly which sites need arms (validate_expr in plan.rs is **not** among them because resolution runs after it — see Step 6).

- [ ] **Step 3: Add the `eval` arm**

In `src/expr.rs`, in `pub fn eval(...)`, add this arm to the `match expr { … }` (after the `Expr::FieldAccess` arm):

```rust
        Expr::Transform {
            obj,
            input_features,
            output_fields,
            arg,
        } => {
            let fields = match eval(arg, row)? {
                Value::Struct(f) => f,
                Value::Null => return Ok(Value::Null),
                other => {
                    return Err(crate::plan::InterpError::Eval(format!(
                        "transformer argument must be a struct, got a {} value",
                        type_name(&other)
                    )))
                }
            };
            // infer() already holds the GIL; attach is cheap and re-entrant, so
            // eval() stays a pure-Rust signature (no py token threaded through).
            Python::attach(|py| -> Result<Value, crate::plan::InterpError> {
                // Reorder the struct's fields to feature_names_in_ order, then
                // build a 1-row Python list-of-lists. sklearn's check_array
                // coerces it -- no numpy on the Rust side.
                let mut ordered: Vec<Py<PyAny>> = Vec::with_capacity(input_features.len());
                for feat in input_features {
                    let value = fields
                        .iter()
                        .find(|(n, _)| n == feat)
                        .map(|(_, v)| v)
                        .ok_or_else(|| {
                            crate::plan::InterpError::Eval(format!(
                                "transformer input struct is missing field '{feat}'"
                            ))
                        })?;
                    let py_val = value.to_pyobject(py).map_err(|e| {
                        crate::plan::InterpError::Eval(format!(
                            "marshalling transformer input failed: {e}"
                        ))
                    })?;
                    ordered.push(py_val);
                }
                let row_list = PyList::new(py, ordered).map_err(|e| {
                    crate::plan::InterpError::Eval(format!(
                        "building transformer input row failed: {e}"
                    ))
                })?;
                let x = PyList::new(py, [row_list]).map_err(|e| {
                    crate::plan::InterpError::Eval(format!(
                        "building transformer input matrix failed: {e}"
                    ))
                })?;
                let y = obj
                    .bind(py)
                    .call_method1("transform", (x,))
                    .map_err(|e| {
                        crate::plan::InterpError::Eval(format!("transformer.transform failed: {e}"))
                    })?;
                // .tolist() turns numpy scalars into Python builtins so
                // from_pyobject sees float/int/str, not opaque numpy objects.
                let y_list = y.call_method0("tolist").map_err(|e| {
                    crate::plan::InterpError::Eval(format!(
                        "transformer output .tolist() failed: {e}"
                    ))
                })?;
                let y0 = y_list.get_item(0).map_err(|e| {
                    crate::plan::InterpError::Eval(format!(
                        "transformer produced no output row: {e}"
                    ))
                })?;
                let mut out = Vec::with_capacity(output_fields.len());
                for (i, (fname, ft)) in output_fields.iter().enumerate() {
                    let elem = y0.get_item(i).map_err(|e| {
                        crate::plan::InterpError::Eval(format!(
                            "transformer output missing position {i} for field '{fname}': {e}"
                        ))
                    })?;
                    let val = Value::from_pyobject_typed(&elem, &ft.base).map_err(|e| {
                        crate::plan::InterpError::Eval(format!(
                            "marshalling transformer output field '{fname}' failed: {e}"
                        ))
                    })?;
                    out.push((fname.clone(), val));
                }
                Ok(Value::Struct(out))
            })
        }
```

- [ ] **Step 4: Add the `infer_type` arm**

In `src/types.rs`, add `HashSet` to the collections import at the top:

```rust
use std::collections::{HashMap, HashSet};
```

In `pub fn infer_type(...)`, add this arm to the `match expr { … }` (after the `Expr::FieldAccess` arm):

```rust
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
```

- [ ] **Step 5: Add `arrow_schema_to_ordered_fields` to schema.rs**

In `src/schema.rs`, add this public function (it reuses the module-private `arrow_field_to_field_type`):

```rust
/// Parse a `pyarrow.Schema` object into an order-preserving `Vec` of
/// `(name, FieldType)`. Unlike `from_arrow_table` (which reads a `pa.Table`
/// and returns an unordered `Schema` HashMap), this takes a bare `pa.Schema`
/// and preserves field order -- required because a transformer's declared
/// output becomes a `Base::Struct`/`Value::Struct` whose field order is
/// semantically significant.
pub fn arrow_schema_to_ordered_fields(
    py: Python<'_>,
    schema_obj: &Py<PyAny>,
) -> Result<Vec<(String, FieldType)>, InterpError> {
    let bound = schema_obj.bind(py);
    let names: Vec<String> = bound
        .getattr("names")
        .and_then(|n| n.extract())
        .map_err(|e| {
            InterpError::Build(format!("transformer output schema is not a pyarrow.Schema: {e}"))
        })?;
    let pa_types = PyModule::import(py, "pyarrow.types")
        .map_err(|e| InterpError::Build(format!("Failed to import pyarrow.types: {e}")))?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let field = bound
            .call_method1("field", (name.as_str(),))
            .map_err(|e| InterpError::Build(format!("Failed to read output field '{name}': {e}")))?;
        let ft = arrow_field_to_field_type(&pa_types, &field)
            .map_err(|e| InterpError::Build(format!("Failed to read type of output field '{name}': {e}")))?;
        out.push((name, ft));
    }
    Ok(out)
}
```

- [ ] **Step 6: Add the `transformers` param and resolution pass to lib.rs**

In `src/lib.rs`, add this struct and two free functions (place them above `impl InferFn` or near `synthesize_output_model`):

```rust
/// A `transformers` registry entry resolved at build time: the fitted object,
/// its `feature_names_in_` (input alignment order), and the caller-declared
/// output field list (marshalling target, order significant).
struct ResolvedTransformer {
    obj: std::sync::Arc<Py<PyAny>>,
    input_features: Vec<String>,
    output_fields: Vec<(String, types::FieldType)>,
}

/// Reads `obj.feature_names_in_.tolist()`. Absence is a clear build error:
/// the object was fit on bare arrays, so we cannot align inputs by name.
fn read_feature_names_in(py: Python<'_>, obj: &Py<PyAny>) -> Result<Vec<String>, plan::InterpError> {
    let bound = obj.bind(py);
    let attr = bound.getattr("feature_names_in_").map_err(|_| {
        plan::InterpError::Build(
            "transformer has no `feature_names_in_`; fit it on named data \
             (e.g. a pandas DataFrame) so input columns can be aligned by name"
                .to_string(),
        )
    })?;
    attr.call_method0("tolist")
        .and_then(|l| l.extract::<Vec<String>>())
        .map_err(|e| plan::InterpError::Build(format!("could not read feature_names_in_: {e}")))
}

/// Rewrites every `Expr::Function` whose name is a registered transformer into
/// an `Expr::Transform`, recursing through the whole expression tree so a
/// transformer call nested inside arithmetic is still resolved. A transformer
/// call must have exactly one argument.
fn resolve_transformers(
    expr: Expr,
    resolved: &HashMap<String, ResolvedTransformer>,
) -> Result<Expr, plan::InterpError> {
    match expr {
        Expr::Function { name, args } => {
            let mut new_args = Vec::with_capacity(args.len());
            for a in args {
                new_args.push(resolve_transformers(a, resolved)?);
            }
            if let Some(rt) = resolved.get(&name) {
                if new_args.len() != 1 {
                    return Err(plan::InterpError::Build(format!(
                        "transformer '{name}' takes exactly one argument, got {}",
                        new_args.len()
                    )));
                }
                let arg = new_args.into_iter().next().unwrap();
                return Ok(Expr::Transform {
                    obj: rt.obj.clone(),
                    input_features: rt.input_features.clone(),
                    output_fields: rt.output_fields.clone(),
                    arg: Box::new(arg),
                });
            }
            Ok(Expr::Function { name, args: new_args })
        }
        Expr::BinaryOp { op, left, right } => Ok(Expr::BinaryOp {
            op,
            left: Box::new(resolve_transformers(*left, resolved)?),
            right: Box::new(resolve_transformers(*right, resolved)?),
        }),
        Expr::Not(inner) => Ok(Expr::Not(Box::new(resolve_transformers(*inner, resolved)?))),
        Expr::Cast { expr, target } => Ok(Expr::Cast {
            expr: Box::new(resolve_transformers(*expr, resolved)?),
            target,
        }),
        Expr::Struct(fields) => {
            let mut out = Vec::with_capacity(fields.len());
            for (k, v) in fields {
                out.push((k, resolve_transformers(v, resolved)?));
            }
            Ok(Expr::Struct(out))
        }
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for e in items {
                out.push(resolve_transformers(e, resolved)?);
            }
            Ok(Expr::List(out))
        }
        Expr::FieldAccess { base, field } => Ok(Expr::FieldAccess {
            base: Box::new(resolve_transformers(*base, resolved)?),
            field,
        }),
        // Column, Literal, and an already-built Transform pass through.
        other => Ok(other),
    }
}
```

Add `use expr::Value;` if not already imported (it is, per current lib.rs). Change the `#[new]` signature and body. Replace the current signature block:

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

with:

```rust
    #[new]
    #[pyo3(signature = (sql, row_tables, static_tables, output_model=None, transformers=None))]
    fn new(
        py: Python<'_>,
        sql: String,
        row_tables: HashMap<String, Py<PyAny>>,
        static_tables: HashMap<String, Py<PyAny>>,
        output_model: Option<Py<PyAny>>,
        transformers: Option<HashMap<String, (Py<PyAny>, Py<PyAny>)>>,
    ) -> PyResult<Self> {
```

Then, immediately **after** the `validate_columns(...)` call (which produces `column_validation`) and **before** the `let output_model = match output_model { … }` block, insert:

```rust
        // Resolve registered transformers AFTER column validation (so
        // validate_columns sees the plain Expr::Function and its named_struct
        // arg -- no Transform arm needed there) and BEFORE output-model
        // synthesis (so infer_type sees Expr::Transform and returns the
        // declared output struct type). Reads feature_names_in_ here (with py).
        let transformers = transformers.unwrap_or_default();
        if !transformers.is_empty() {
            let mut resolved: HashMap<String, ResolvedTransformer> = HashMap::new();
            for (name, (obj, out_schema_obj)) in &transformers {
                let input_features = read_feature_names_in(py, obj)?;
                let output_fields = schema::arrow_schema_to_ordered_fields(py, out_schema_obj)?;
                resolved.insert(
                    name.clone(),
                    ResolvedTransformer {
                        obj: std::sync::Arc::new(obj.clone_ref(py)),
                        input_features,
                        output_fields,
                    },
                );
            }
            let projection = std::mem::take(&mut optimized_plan.projection);
            let mut new_projection = Vec::with_capacity(projection.len());
            for (alias, expr) in projection {
                new_projection.push((alias, resolve_transformers(expr, &resolved)?));
            }
            optimized_plan.projection = new_projection;
        }
```

- [ ] **Step 7: Update the type stub**

In `sql_transform/_interpreter.pyi`, change the `__init__` signature to:

```python
    def __init__(
        self,
        sql: str,
        row_tables: dict[str, type[BaseModel]],
        static_tables: dict[str, pa.Table],
        output_model: type[BaseModel] | None = None,
        transformers: dict[str, tuple[object, pa.Schema]] | None = None,
    ) -> None: ...
```

- [ ] **Step 8: Build and run existing Rust + differential tests (no regressions)**

Run: `cargo build` — Expected: PASS (all `Expr` matches now exhaustive).
Run: `cargo test` — Expected: PASS.
Run: `uv run maturin develop` — Expected: builds the extension.
Run: `uv run pytest tests/ sql_transform/ -q` — Expected: PASS (the new `transformers` param defaults to `None`, so all existing callers are unaffected).

- [ ] **Step 9: Write the failing direct-infer test**

Create `tests/test_transformer_callout_infer.py`:

```python
"""Direct `infer` tests for the Expr::Transform callout (no oracle).

Parity with DataFusion is proven separately in test_diff_transformer_callout.py;
these pin the Rust marshalling to hand-computed values and the build-time errors.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.preprocessing import StandardScaler

from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model

# See test_transformer_udf.py: the nameless-input warning is a known false positive.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)

_THIS = pa.schema([("age", pa.float64()), ("income", pa.float64())])
_OUT = pa.schema([("age", pa.float64()), ("income", pa.float64())])
_SQL = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"


def _fitted_scaler():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    return StandardScaler().fit(train), train


def _infer(sql, transformers, rows):
    model = synthesize_this_model(_THIS)
    fn = InferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables={},
        transformers=transformers,
    )
    return [r.model_dump() for r in fn.infer({"__THIS__": [model(**r) for r in rows]})]


def test_standard_scaler_infer_hand_computed():
    sc, _ = _fitted_scaler()
    out = _infer(_SQL, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])
    # population std (ddof=0): age mean 25, std 11.18034; income mean 2.5, std 1.11803
    assert abs(out[0]["s"]["age"] - ((10.0 - 25.0) / 11.180339887498949)) < 1e-9
    assert abs(out[0]["s"]["income"] - ((1.0 - 2.5) / 1.1180339887498949)) < 1e-9


def test_missing_feature_names_in_is_build_error():
    sc = StandardScaler().fit(np.array([[10.0, 1.0], [20.0, 2.0]]))  # bare array
    with pytest.raises(ValueError, match="feature_names_in_"):
        _infer(_SQL, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])


def test_non_struct_argument_is_build_error():
    sc, _ = _fitted_scaler()
    sql = "SELECT __tfm_0__(age) AS s FROM __THIS__"
    with pytest.raises(ValueError, match="must be a struct"):
        _infer(sql, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])


def test_field_name_mismatch_is_build_error():
    sc, _ = _fitted_scaler()
    sql = "SELECT __tfm_0__(named_struct('age', age, 'wrong', income)) AS s FROM __THIS__"
    with pytest.raises(ValueError, match="feature_names_in_"):
        _infer(sql, {"__tfm_0__": (sc, _OUT)}, [{"age": 10.0, "income": 1.0}])
```

- [ ] **Step 10: Run the direct-infer test**

Run: `uv run pytest tests/test_transformer_callout_infer.py -v`
Expected: PASS (the extension was rebuilt in Step 8; if you edited Rust since, re-run `uv run maturin develop` first).

- [ ] **Step 11: Commit**

```bash
git add src/expr.rs src/types.rs src/schema.rs src/lib.rs \
        sql_transform/_interpreter.pyi tests/test_transformer_callout_infer.py
git commit -m "feat: Expr::Transform -- Rust engine transformer callout"
```

---

## Task 3: Differential parity (`transform == infer`)

**Files:**
- Test: `tests/test_diff_transformer_callout.py`

**Interfaces:**
- Consumes: `_transformer_udf` (Task 1), `InferFn(..., transformers=...)` (Task 2), `differential._rows_equal`, `sql_transform._schema.synthesize_this_model`.

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_diff_transformer_callout.py`:

```python
"""Differential parity: transform (DataFusion UDF) == infer (Rust Expr::Transform)."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from datafusion import SessionContext
from differential import _rows_equal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, StandardScaler

from sql_transform._interpreter import InferFn
from sql_transform._schema import synthesize_this_model
from sql_transform._transformer_udf import _transformer_udf

# See test_transformer_udf.py: the nameless-input warning is a known false
# positive; both the oracle UDF and the Rust infer path emit it.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _parity(sql, table, obj, in_schema, out_schema, name="__tfm_0__"):
    # Oracle: DataFusion with the transformer registered as a UDF.
    ctx = SessionContext()
    ctx.from_arrow(table, name="__THIS__")
    ctx.register_udf(_transformer_udf(obj, in_schema, out_schema, name))
    df = ctx.sql(sql)
    oracle = pa.Table.from_batches(df.collect(), schema=df.schema()).to_pylist()

    # Rust infer with the same object in the transformers registry.
    model = synthesize_this_model(table.schema)
    rows = [model(**r) for r in table.to_pylist()]
    fn = InferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables={},
        transformers={name: (obj, out_schema)},
    )
    actual = [r.model_dump() for r in fn.infer({"__THIS__": rows})]

    assert _rows_equal(actual, oracle), (sql, actual, oracle)
    return oracle


def test_standard_scaler_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    sc = StandardScaler().fit(train)
    schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    sql = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), sc, schema, schema)


def test_whole_pipeline_parity():
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline([("sc", StandardScaler()), ("pt", PowerTransformer())]).fit(train)
    schema = pa.schema([("age", pa.float64()), ("income", pa.float64())])
    sql = "SELECT __tfm_0__(named_struct('age', age, 'income', income)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), pipe, schema, schema)


def test_ordinal_encoder_non_float_in_and_out_parity():
    train = pd.DataFrame(
        {"color": ["red", "green", "blue", "red"], "size": ["S", "M", "L", "M"]}
    )
    enc = OrdinalEncoder(dtype=np.int64).fit(train)  # string in, int out
    in_schema = pa.schema([("color", pa.string()), ("size", pa.string())])
    out_schema = pa.schema([("color", pa.int64()), ("size", pa.int64())])
    sql = "SELECT __tfm_0__(named_struct('color', color, 'size', size)) AS s FROM __THIS__"
    _parity(sql, pa.Table.from_pandas(train), enc, in_schema, out_schema)
```

- [ ] **Step 2: Run the parity test**

Run: `uv run pytest tests/test_diff_transformer_callout.py -v`
Expected: PASS — all three cases agree between DataFusion and Rust `infer`.

- [ ] **Step 3: Run the whole suite for regressions**

Run: `cargo test && uv run maturin develop && uv run pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_diff_transformer_callout.py
git commit -m "test: differential parity for engine transformer callout"
```

---

## Self-Review

**Spec coverage:**
- Registry boundary `transformers: dict[str, tuple[obj, pa.Schema]]` → Task 2 Step 6 (param) + `.pyi`.
- Input alignment by `feature_names_in_`, required → Task 2 Step 6 (`read_feature_names_in`, build error) + eval reorder (Step 3) + `test_missing_feature_names_in_is_build_error`.
- Declared output schema, order significant → `arrow_schema_to_ordered_fields` (Step 5) + `output_fields: Vec<(String, FieldType)>`.
- `transform` vectorized UDF oracle → Task 1.
- `infer` `Expr::Transform` + `Python::attach` eval, no numpy in Rust → Task 2 Steps 1, 3.
- Build-time resolution after build, one-arg requirement, struct-field check → Steps 6, 4; `test_non_struct_argument_is_build_error`, `test_field_name_mismatch_is_build_error`.
- Differential parity, struct column compared directly (no unnest) → Task 3; cases StandardScaler, whole Pipeline, OrdinalEncoder, one hand-computed value (Task 2 Step 9).

**Placeholder scan:** none — every code step carries complete code.

**Type consistency:** `Expr::Transform` fields (`obj: Arc<Py<PyAny>>`, `input_features: Vec<String>`, `output_fields: Vec<(String, FieldType)>`, `arg: Box<Expr>`) are identical in the variant definition (Step 1), the `eval` arm (Step 3), the `infer_type` arm (Step 4, using `..` for `obj`), and `resolve_transformers` (Step 6). `arrow_schema_to_ordered_fields` returns `Vec<(String, FieldType)>`, matching `ResolvedTransformer.output_fields` and the variant. `_transformer_udf(obj, in_schema, out_schema, name)` signature matches all call sites in Tasks 1 and 3.

**Note on resolution ordering:** the spec said "before validate_columns"; this plan runs it **after** (before output-model synthesis). Equivalent validation (the un-rewritten `Expr::Function`'s `named_struct` arg validates the same columns), and it avoids adding a `Transform` arm to `validate_expr` — a strictly smaller change. Flagged so the reviewer treats it as a deliberate, justified deviation rather than a miss.

---

## Execution Handoff

Plan complete. Recommended: **subagent-driven-development** — three tasks, each with its own test cycle; Task 1 (Python oracle) and Task 2 (Rust) are independently reviewable, Task 3 ties them together.
